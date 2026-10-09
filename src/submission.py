"""Сборка полной конкурсной сетки прогнозов с жёсткой валидацией."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_PERIODS = pd.date_range("2024-07-01", "2024-12-01", freq="MS")
DEFAULT_ENTITY_COUNT = 2_094
DEFAULT_ROWS = DEFAULT_ENTITY_COUNT * len(DEFAULT_PERIODS)


class SubmissionValidator:
    """Выбирает финальный forecast arm и сохраняет полную сетку МО × месяц."""

    def __init__(
        self,
        output: Path = Path("reports/submission.csv"),
        root_output: Path = Path("submission.csv"),
    ) -> None:
        self.output = output
        self.root_output = root_output

    @staticmethod
    def _prediction_column(predictions: pd.DataFrame, requested: str | None) -> str:
        target_col = requested or "pred_regime_aware"
        if target_col not in predictions.columns:
            raise KeyError(
                f"Критическая ошибка: колонка {target_col} отсутствует в прогнозах! "
                "Скрытый фоллбэк запрещён."
            )
        LOGGER.info("Для финального submission выбран %s", target_col)
        return target_col

    @staticmethod
    def _entities(
        predictions: pd.DataFrame, entity_column: str, expected_entities: int | None
    ) -> pd.Index:
        entities = pd.Index(
            predictions[entity_column].dropna().astype("string").unique(), name=entity_column
        ).sort_values()
        if expected_entities is not None and len(entities) != expected_entities:
            raise ValueError(
                f"Ожидалось {expected_entities} уникальных МО, получено {len(entities)}"
            )
        if entities.empty:
            raise ValueError("Список МО пуст")
        return entities

    @staticmethod
    def _fill_predictions(frame: pd.DataFrame, entity_column: str) -> pd.DataFrame:
        result = frame.sort_values([entity_column, "period"], kind="stable").copy()
        result["pred"] = result.groupby(entity_column, observed=True)["pred"].transform(
            lambda group: group.ffill().bfill()
        )
        result["pred"] = result["pred"].fillna(
            result.groupby("period", observed=True)["pred"].transform("median")
        )
        if result["pred"].isna().any():
            global_median = float(result["pred"].median())
            if not np.isfinite(global_median):
                raise ValueError("Невозможно заполнить прогнозы: нет конечных значений")
            result["pred"] = result["pred"].fillna(global_median)
        return result

    @staticmethod
    def _fill_from_history(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        required = {"mo", "period", "y"}
        if not required.issubset(history):
            raise ValueError(f"В истории отсутствуют колонки: {sorted(required - set(history))}")
        past = history[["mo", "period", "y"]].copy()
        past["mo"] = past["mo"].astype("string")
        past["period"] = pd.to_datetime(past["period"], errors="raise").dt.tz_localize(None)
        past["y"] = pd.to_numeric(past["y"], errors="coerce")
        past = past.loc[
            past["mo"].notna() & past["period"].notna() & np.isfinite(past["y"]) & past["y"].gt(0)
        ]
        if past.duplicated(["mo", "period"]).any():
            raise ValueError("История содержит дубликаты ключей mo/period")
        missing = frame.loc[frame["pred"].isna(), ["mo", "period"]].copy()
        if missing.empty:
            return frame
        missing["_row"] = missing.index
        # На origin недоступны target-month и последующие наблюдения.
        fallback = pd.merge_asof(
            missing.sort_values("period"),
            past.sort_values("period"),
            on="period",
            by="mo",
            direction="backward",
            allow_exact_matches=False,
        )
        cold_start = int(fallback["y"].isna().sum())
        for period, subset in fallback.loc[fallback["y"].isna()].groupby("period"):
            available = past.loc[past["period"].lt(period), "y"]
            if available.empty:
                raise ValueError(f"Нет истории до {period} для причинного fallback submission")
            fallback.loc[subset.index, "y"] = float(available.median())
        result = frame.copy()
        result.loc[fallback["_row"].to_numpy(), "pred"] = fallback["y"].to_numpy(dtype=float)
        LOGGER.warning(
            "Submission fallback: %d строк восстановлены по истории до целевого месяца "
            "(last-positive-value=%d, cold-start median=%d)",
            len(fallback),
            len(fallback) - cold_start,
            cold_start,
        )
        return result

    @staticmethod
    def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)

    def build(
        self,
        predictions: pd.DataFrame,
        *,
        expected_rows: int | None = DEFAULT_ROWS,
        expected_entities: int | None = DEFAULT_ENTITY_COUNT,
        periods: pd.DatetimeIndex = DEFAULT_PERIODS,
        prediction_column: str | None = None,
        id_columns: tuple[str, ...] = ("mo", "period"),
        entity_universe: Sequence[str] | None = None,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if id_columns != ("mo", "period"):
            raise ValueError("Конкурсная схема требует id_columns=('mo', 'period')")
        required = set(id_columns)
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"В прогнозах отсутствуют колонки: {sorted(missing)}")
        source_column = self._prediction_column(predictions, prediction_column)
        source = (
            predictions[["mo", "period", source_column]]
            .rename(columns={source_column: "pred"})
            .copy()
        )
        source["mo"] = source["mo"].astype("string")
        source["period"] = (
            pd.to_datetime(source["period"], errors="raise")
            .dt.tz_localize(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        source["pred"] = pd.to_numeric(source["pred"], errors="coerce")
        source = source.loc[source["period"].isin(periods)].copy()
        if source.duplicated(["mo", "period"]).any():
            raise ValueError("Прогнозы содержат дубликаты ключей mo/period")
        root_exists = self.root_output.exists()
        existing = pd.read_csv(self.root_output) if root_exists else None
        if entity_universe is not None:
            roster = pd.DataFrame({"mo": list(entity_universe)})
        else:
            # OOF покрывает оценённые МО; фиксированный файл задаёт roster поставки.
            roster = existing if existing is not None else predictions
        entities = self._entities(roster, "mo", expected_entities)
        unknown = set(predictions["mo"].dropna().astype("string")) - set(entities)
        if unknown:
            raise ValueError(f"В прогнозах есть МО вне списка submission: {sorted(unknown)[:5]}")
        full_grid = pd.MultiIndex.from_product(
            [entities, periods], names=["mo", "period"]
        ).to_frame(index=False)
        result = full_grid.merge(source, on=["mo", "period"], how="left", validate="one_to_one")
        missing_before = int(result["pred"].isna().sum())
        if source_column == "pred_regime_aware" and missing_before:
            raise ValueError(
                f"Критическая ошибка: отсутствуют {missing_before} прогнозов pred_regime_aware в полной конкурсной сетке"
            )
        if history is not None:
            result = self._fill_from_history(result, history)
        result = self._fill_predictions(result, "mo")
        result = result.sort_values(["mo", "period"], kind="stable").reset_index(drop=True)
        expected = expected_rows if expected_rows is not None else len(entities) * len(periods)
        assert len(result) == expected, (
            f"Ожидалось {expected} строк submission, получено {len(result)}"
        )
        assert result["pred"].isna().sum() == 0, "Submission содержит NaN"
        values = result["pred"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Submission содержит Inf")
        assert int((result["pred"] <= 0).sum()) == 0, (
            "Все прогнозы должны быть строго положительными"
        )
        assert not result.duplicated(["mo", "period"]).any(), "Submission содержит дубликаты"
        # Корневой submission и проверенная копия должны относиться к одному запуску.
        same_path = self.root_output.resolve() == self.output.resolve()
        self._write_atomic(result, self.output)
        if not same_path:
            self._write_atomic(result, self.root_output)
        digest = hashlib.md5(self.output.read_bytes()).hexdigest()
        quantiles = result["pred"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        LOGGER.info(
            "Submission rows=%d filled=%d md5=%s quantiles=%s",
            len(result),
            missing_before,
            digest,
            quantiles,
        )
        return result

    def build_from_parquet(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        return self.build(pd.read_parquet(path), **kwargs)


def build_submission(
    predictions_path: Path,
    *,
    output: Path = Path("reports/submission.csv"),
    root_output: Path = Path("submission.csv"),
    entity_universe: Sequence[str] | None = None,
    history: pd.DataFrame | None = None,
    prediction_column: str | None = None,
) -> pd.DataFrame:
    """Публичная функция для CLI и программного запуска."""
    return SubmissionValidator(output, root_output).build_from_parquet(
        predictions_path,
        entity_universe=entity_universe,
        history=history,
        prediction_column=prediction_column,
    )
