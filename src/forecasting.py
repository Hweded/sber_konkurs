"""Общие OOF-origin для трёх моделей; обучение заморожено на фолд."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from src.device import resolve_device

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.configuration import AppConfig
from src.data_loader import load_target
from src.features import to_validation_panel
from src.ensemble import OOFBlender
from src.models_forecast import (
    cached_pipeline_factory,
    predict_chronos_resilient,
    predict_prophet_parallel,
)
from src.validation import ValidationConfig, prepare_panel, expanding_window_splits, _r2

LOGGER = logging.getLogger(__name__)
MODELS = ("prophet", "catboost", "chronos", "ensemble", "regime_aware")
PREDICTION_COLUMNS = {
    "prophet": "pred_prophet",
    "catboost": "pred_catboost",
    "chronos": "pred_chronos",
    "ensemble": "pred_ensemble",
    "regime_aware": "pred_regime_aware",
}


class RegimeAwareEnsemble:
    """Причинное переключение на CatBoost при известном на origin шоке."""

    def __init__(self, calm_alpha: float = 0.40) -> None:
        if not 0.35 <= calm_alpha <= 0.50:
            raise ValueError("Вес Chronos в спокойном режиме должен быть 0.35–0.50")
        self.calm_alpha = calm_alpha

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        news_threshold: float = 2.0,
    ) -> tuple[pd.Series, pd.Series]:
        required = {"pred_catboost", "pred_chronos", "period", "mo"}
        if not required.issubset(frame):
            raise ValueError(f"Нет колонок ансамбля: {sorted(required - set(frame))}")
        ordered = frame.sort_values(["mo", "period"], kind="stable")
        # Детекторы по y/residual доступны только после закрытия прошлого месяца.
        alerts = pd.Series(False, index=ordered.index)
        for column in ("cusum_shock", "consensus_shock"):
            if column in ordered:
                lagged = ordered.groupby("mo", observed=True)[column].shift(1)
                alerts |= lagged.eq(True).fillna(False)
        # Признак TDA матрицы уже сдвинут до t-1 при построении.
        if "tda_shock" in ordered:
            alerts |= ordered["tda_shock"].fillna(False).astype(bool)
        # NLP-колонки матрицы уже относятся к t-1 (публикация -> target month).
        for column in ("news_shock_score", "telegram_shock_score"):
            if column in ordered:
                alerts |= pd.to_numeric(ordered[column], errors="coerce").abs().ge(news_threshold)
        alpha = pd.Series(np.where(alerts, 0.0, self.calm_alpha), index=ordered.index)
        cat = pd.to_numeric(ordered["pred_catboost"], errors="raise")
        chronos = pd.to_numeric(ordered["pred_chronos"], errors="raise")
        prediction = (1.0 - alpha) * cat + alpha * chronos
        if not np.isfinite(prediction.to_numpy(dtype=float)).all():
            raise ValueError("Ансамбль получил неконечный прогноз")
        return prediction.reindex(frame.index), alpha.reindex(frame.index)


def forecast(config: AppConfig) -> pd.DataFrame:
    """Строит общую OOF-панель CatBoost, Prophet и zero-shot Chronos."""
    from catboost import CatBoostRegressor
    from chronos import BaseChronosPipeline
    import torch

    device_info = resolve_device(config.device.mode)
    LOGGER.info(
        "Forecast device=%s GPU=%s CUDA=%s",
        device_info.device,
        device_info.gpu_name or "none",
        device_info.cuda_version or "none",
    )

    if config.optimization.enabled:
        raise ValueError("Nested tuning не реализован: optimization.enabled должен быть false")
    import pyarrow.parquet as parquet
    schema_columns = set(parquet.ParquetFile(config.paths.supervised).schema_arrow.names)
    requested = {"period", "mo", "y", *config.validation.feature_columns}
    requested.update(c for c in schema_columns if c.startswith(("macro_", "rosstat_")))
    requested.update({
        "sentiment_index", "news_volume", "news_shock_score", "telegram_sentiment",
        "telegram_volume", "telegram_shock_score", "tda_entropy",
        "tda_wasserstein_dist", "tda_shock",
    })
    data = pd.read_parquet(config.paths.supervised, columns=sorted(requested & schema_columns))
    external_news = {
        "sentiment_index",
        "news_volume",
        "news_shock_score",
        "telegram_sentiment",
        "telegram_volume",
        "telegram_shock_score",
        "tda_entropy",
        "tda_wasserstein_dist",
        "tda_shock",
    }
    extra = [
        c
        for c in data
        if str(c).startswith(("macro_", "rosstat_")) or c in external_news
    ]
    settings = config.validation.model_dump()
    settings["feature_columns"] = tuple(dict.fromkeys([*config.validation.feature_columns, *extra]))
    validation = ValidationConfig.model_validate(settings)
    panel = prepare_panel(to_validation_panel(data, validation), validation)
    folds = expanding_window_splits(panel, validation)
    # NOTE: берём сырой таргет, иначе warmup искусственно обрежет контекст Chronos.
    target_history = load_target(config.data.directory, config.data.target, config.data.separator)
    target_history["period"] = pd.to_datetime(target_history["period"], errors="raise")
    cc = config.models.chronos
    chronos_info = resolve_device(cc.device)
    chronos_device = chronos_info.device
    LOGGER.info(
        "Chronos device=%s GPU=%s CUDA=%s",
        chronos_device,
        chronos_info.gpu_name or "none",
        chronos_info.cuda_version or "none",
    )

    def load_chronos(device: str, dtype: Any) -> Any:
        LOGGER.info("Chronos: загрузка модели %s на %s, dtype=%s", cc.model_id, device, dtype)
        return BaseChronosPipeline.from_pretrained(
            cc.model_id,
            revision=cc.revision,
            device_map=device,
            torch_dtype=dtype,
        )

    # NOTE: CPU-копию грузим лениво только после реального CUDA OOM.
    make_chronos = cached_pipeline_factory(load_chronos)
    chronos_torch_device = torch.device(chronos_device)
    outputs: list[pd.DataFrame] = []
    blender = OOFBlender(config.ensemble)
    for fold in folds:
        train = panel.iloc[list(fold.train_positions)]
        test = panel.iloc[list(fold.test_positions)].copy()
        columns = list(validation.feature_columns)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        parameters = dict(config.models.catboost)
        parameters.update(random_seed=config.reproducibility.seed, thread_count=config.reproducibility.threads)
        boosting = CatBoostRegressor(**parameters)
        boosting.fit(imputer.fit_transform(train[columns]), train.y)
        test["catboost_prediction"] = boosting.predict(imputer.transform(test[columns]))
        prophet_predictions = predict_prophet_parallel(
            train,
            test,
            config.models.prophet,
            seed=config.reproducibility.seed,
        )
        test["prophet_prediction"] = prophet_predictions["prophet_prediction"].to_numpy(
            dtype=np.float64
        )
        train_entities = set(train["mo"])
        context_list: list[torch.Tensor] = []
        context_indices: list[Any] = []
        fallback_by_index: dict[Any, float] = {}
        fallback_entities: set[Any] = set()
        fallback_empty = 0
        fallback_missing_periods = 0
        fallback_nonfinite = 0
        for entity, subset in test.groupby("mo", observed=True, sort=False):
            entity_history = target_history.loc[target_history.mo.eq(entity)].sort_values("period")
            for index, row in subset.iterrows():
                context = entity_history.loc[entity_history.period.lt(row.period)].tail(cc.context_length)
                expected = (
                    pd.date_range(context.period.min(), row.period - pd.offsets.MonthBegin(1), freq="MS")
                    if not context.empty else pd.DatetimeIndex([])
                )
                values = pd.to_numeric(context.y, errors="coerce").to_numpy(dtype=np.float32)
                complete = (
                    not context.empty
                    and pd.DatetimeIndex(context.period).equals(expected)
                    and np.isfinite(values).all()
                )
                if not complete:
                    finite = values[np.isfinite(values)]
                    if finite.size == 0:
                        raise ValueError(f"Нет конечной истории таргета для МО {entity} до {row.period}")
                    fallback_by_index[index] = float(finite[-1])
                    fallback_entities.add(entity)
                    if context.empty:
                        fallback_empty += 1
                    elif not np.isfinite(values).all():
                        fallback_nonfinite += 1
                    else:
                        fallback_missing_periods += 1
                    continue
                context_list.append(torch.as_tensor(values, dtype=torch.float32))
                context_indices.append(index)
        test["chronos_prediction"] = np.nan
        if context_list:
            forecasts, applied_mode = predict_chronos_resilient(
                make_chronos,
                context_list,
                prediction_length=cc.prediction_length,
                batch_size=config.device.batch_size,
                device=chronos_torch_device,
                dtype=getattr(torch, cc.dtype),
                fallback_to_cpu=config.device.fallback_to_cpu_on_oom,
                fallback_values=[float(context[-1].item()) for context in context_list],
            )
            LOGGER.info("Chronos fold=%d режим=%s", fold.number, applied_mode)
            for index, prediction in zip(context_indices, forecasts[:, 0], strict=True):
                test.loc[index, "chronos_prediction"] = float(prediction)
        for index, prediction in fallback_by_index.items():
            test.loc[index, "chronos_prediction"] = prediction
        # NOTE: cold-start тянем от последнего y и медианного train-темпа, без будущих строк.
        cold_start_indices = test.index[~test["mo"].isin(train_entities)]
        if len(cold_start_indices):
            train_values = pd.to_numeric(train["y"], errors="coerce")
            train_growth = train_values.groupby(train["mo"], observed=True).pct_change()
            growth = float(train_growth.replace([np.inf, -np.inf], np.nan).median())
            if not np.isfinite(growth):
                growth = 0.0
            for index in cold_start_indices:
                entity = test.loc[index, "mo"]
                history = target_history.loc[
                    target_history.mo.eq(entity) & target_history.period.lt(test.loc[index, "period"]), "y"
                ]
                finite = pd.to_numeric(history, errors="coerce").dropna()
                if finite.empty:
                    base = float(pd.to_numeric(train["y"], errors="coerce").median())
                else:
                    base = float(finite.iloc[-1])
                test.loc[index, "chronos_prediction"] = max(0.0, base * (1.0 + growth))
            LOGGER.warning("Фолд %d: %d cold-start строк получили hierarchical fallback", fold.number, len(cold_start_indices))
        if fallback_by_index:
            LOGGER.warning(
                "Chronos fold=%d: context fallback для %d строк (%d МО): "
                "empty=%d missing_periods=%d nonfinite=%d",
                fold.number,
                len(fallback_by_index),
                len(fallback_entities),
                fallback_empty,
                fallback_missing_periods,
                fallback_nonfinite,
            )
        LOGGER.info(
            "Chronos fold=%d: model=%d fallback_context=%d",
            fold.number, len(context_indices), len(fallback_by_index),
        )
        if config.ensemble.enabled and outputs:
            previous = pd.concat(outputs, ignore_index=True)
            blender.fit_weights(
                previous["target"], previous["pred_catboost"], previous["pred_chronos"],
            )
        else:
            blender.alpha_ = config.ensemble.default_alpha
        test["ensemble_prediction"] = blender.predict(
            test["catboost_prediction"], test["chronos_prediction"],
        )
        gate_input = test.rename(columns={
            "catboost_prediction": "pred_catboost",
            "chronos_prediction": "pred_chronos",
        })
        test["regime_aware_prediction"], test["regime_alpha"] = (
            RegimeAwareEnsemble().predict(gate_input)
        )
        result = test[["period", "mo", "y", *(m + "_prediction" for m in MODELS), "regime_alpha"]].copy()
        result = result.rename(columns={
            "y": "target",
            **{model + "_prediction": column for model, column in PREDICTION_COLUMNS.items()},
        })
        result["ensemble_alpha"] = blender.alpha_
        for model, prediction_column in PREDICTION_COLUMNS.items():
            if not np.isfinite(result[prediction_column]).all():
                raise ValueError(f"Неконечные прогнозы {model}")
            result["residual_" + model] = result.target - result[prediction_column]
        result["fold"] = fold.number
        result["actual_available_at"] = result.period + pd.offsets.MonthBegin(1)
        outputs.append(result)
        LOGGER.info("Фолд %d: %d OOF-строк, все три модели", fold.number, len(result))
    oof = pd.concat(outputs, ignore_index=True)
    destination = Path(config.changepoint_detection.predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(destination, index=False)
    return oof


def metric_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Micro-MAE и pooled R², без усреднения R² между фолдами."""
    rows: list[dict[str, Any]] = []
    groups = [(str(k), v) for k, v in oof.groupby("fold", sort=True)] + [("pooled", oof)]
    for fold, frame in groups:
        for model in MODELS:
            target_column = "target" if "target" in frame else "y"
            modern_column = PREDICTION_COLUMNS.get(model, model + "_prediction")
            legacy_column = model + "_prediction"
            if modern_column in frame:
                prediction_column = modern_column
            elif legacy_column in frame:
                prediction_column = legacy_column
            else:
                continue
            actual = frame[target_column].to_numpy(dtype=float)
            predicted = frame[prediction_column].to_numpy(dtype=float)
            mae = float(mean_absolute_error(actual, predicted))
            denominator = float(np.abs(actual).sum())
            rows.append({
                "fold": fold,
                "model": (
                    "Ensemble (CatBoost + Chronos)" if model == "ensemble"
                    else "RegimeAware (CatBoost + Chronos)" if model == "regime_aware"
                    else model
                ),
                "n": len(frame),
                "MAE": mae,
                "WAPE": float(np.abs(actual - predicted).sum() / denominator) if denominator > 0 else np.nan,
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "R2": _r2(actual, predicted),
            })
    table = pd.DataFrame(rows)
    for fold, group in table.groupby("fold", sort=False):
        mae_by_model = group.set_index("model")["MAE"]
        ensemble_mae = mae_by_model.get("Ensemble (CatBoost + Chronos)")
        if ensemble_mae is None:
            continue
        prophet_mae = mae_by_model.get("prophet")
        catboost_mae = mae_by_model.get("catboost")
        mask = table["fold"].eq(fold) & table["model"].eq("Ensemble (CatBoost + Chronos)")
        table.loc[mask, "delta_mae_vs_prophet_pct"] = (
            100.0 * (prophet_mae - ensemble_mae) / prophet_mae if prophet_mae and prophet_mae > 0 else np.nan
        )
        table.loc[mask, "delta_mae_vs_catboost_pct"] = (
            100.0 * (catboost_mae - ensemble_mae) / catboost_mae if catboost_mae and catboost_mae > 0 else np.nan
        )
    return table


def horizon_metric_table(
    oof: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Оценивает только прогнозы, выданные на origin без будущих наблюдений.

    Для старой одношаговой OOF-панели period — месяц цели, origin — предыдущий
    месяц. Отсутствующие горизонты отмечаются явно, а не подменяются ошибкой
    последнего месяца фолда (это нарушило бы информационный контракт).
    """
    if not horizons or any(h < 1 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("Горизонты должны быть уникальными положительными месяцами")
    required = {"period", "target", "fold"}
    if not required.issubset(oof):
        raise ValueError(f"OOF не содержит {sorted(required - set(oof))}")
    frame = oof.copy()
    frame["period"] = pd.to_datetime(frame["period"], errors="raise", utc=True).dt.tz_localize(None)
    if "origin" in frame:
        frame["origin"] = pd.to_datetime(frame["origin"], errors="raise", utc=True).dt.tz_localize(None)
    else:
        # Совместимость только с исходным строго одношаговым OOF.
        frame["origin"] = frame["period"] - pd.offsets.MonthBegin(1)
        frame["lead_months"] = 1
    if "lead_months" not in frame:
        frame["lead_months"] = (
            (frame["period"].dt.year - frame["origin"].dt.year) * 12
            + frame["period"].dt.month - frame["origin"].dt.month
        )
    observed_lead = (
        (frame["period"].dt.year - frame["origin"].dt.year) * 12
        + frame["period"].dt.month - frame["origin"].dt.month
    )
    if (frame["lead_months"] < 1).any() or not observed_lead.eq(frame["lead_months"]).all():
        raise ValueError("Период прогноза должен совпадать с origin + lead_months")
    if {"mo", "origin", "period"}.issubset(frame) and frame.duplicated(["mo", "origin", "period"]).any():
        raise ValueError("Повторяются прогнозы одной территории на одном origin")
    rows: list[dict[str, Any]] = []
    folds: list[str] = [*(str(value) for value in sorted(frame["fold"].unique())), "pooled"]
    for fold in folds:
        subset = frame if fold == "pooled" else frame.loc[frame["fold"].astype(str).eq(fold)]
        for horizon in horizons:
            for scope, selected in (
                ("exact", subset.loc[subset["lead_months"].eq(horizon)]),
                ("cumulative", subset.loc[subset["lead_months"].between(1, horizon)]),
            ):
                # Среднее 1..h допустимо лишь для origin с полным наблюдаемым
                # набором лидов, иначе короткие origin получат больший вес.
                if scope == "cumulative" and not selected.empty:
                    counts = selected.groupby(["mo", "origin"], observed=True)["lead_months"].nunique()
                    complete = counts.loc[counts.eq(horizon)].index
                    selected = selected.set_index(["mo", "origin"]).loc[
                        lambda value: value.index.isin(complete)
                    ].reset_index()
                for model in ("prophet", "catboost", "chronos", "ensemble"):
                    column = PREDICTION_COLUMNS[model]
                    row: dict[str, Any] = {
                        "fold": fold, "horizon": horizon, "scope": scope,
                        "model": model, "n": 0, "origins": 0,
                        "MAE": np.nan, "R2": np.nan, "WAPE": np.nan,
                        "RMSE": np.nan, "status": "not_evaluated",
                    }
                    if column in selected and not selected.empty:
                        actual = selected["target"].to_numpy(dtype=float)
                        predicted = selected[column].to_numpy(dtype=float)
                        if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
                            raise ValueError(f"Неконечные значения {model}, горизонт {horizon}")
                        error = actual - predicted
                        denominator = float(np.abs(actual).sum())
                        row.update({
                            "n": len(selected), "origins": selected["origin"].nunique(),
                            "MAE": float(mean_absolute_error(actual, predicted)),
                            "R2": _r2(actual, predicted),
                            "WAPE": float(np.abs(error).sum() / denominator) if denominator else np.nan,
                            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                            "status": "measured",
                        })
                    rows.append(row)
    return pd.DataFrame(rows)


def save_horizon_metrics(
    oof: pd.DataFrame,
    artifacts: Path,
    figures: Path,
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Экспортирует длинную таблицу и честный график только измеренных точек."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = horizon_metric_table(oof, horizons)
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifacts / "forecast_metrics_by_horizon.csv", index=False)
    axis_figure, axis = plt.subplots(figsize=(10, 5))
    measured = table.loc[
        table["fold"].eq("pooled") & table["scope"].eq("exact")
        & table["status"].eq("measured")
    ]
    for model in ("prophet", "catboost", "chronos"):
        values = measured.loc[measured["model"].eq(model)].sort_values("horizon")
        if not values.empty:
            axis.plot(values["horizon"], values["MAE"], marker="o", label=model)
    axis.set_xticks(list(horizons))
    axis.set_xlabel("Месяцев от origin")
    axis.set_ylabel("OOF MAE")
    axis.set_title("MAE по горизонту: пропуски = нет честной оценки")
    axis.grid(alpha=0.3)
    axis.legend()
    axis_figure.tight_layout()
    axis_figure.savefig(figures / "horizons_mae_comparison.png", dpi=300)
    plt.close(axis_figure)
    export_metrics_artifacts(table, artifacts)
    return table


def export_metrics_artifacts(table: pd.DataFrame, artifacts: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and a compact Markdown metric table."""
    import json

    artifacts.mkdir(parents=True, exist_ok=True)
    # ``to_json`` converts NumPy NaN values to JSON null (``where`` cannot do
    # that reliably for float columns without changing their dtype).
    records = json.loads(table.to_json(orient="records"))
    payload = {
        "protocol": "expanding_window_frozen_origin",
        "horizons": sorted({int(value) for value in table["horizon"].dropna().unique()}),
        "metrics": records,
    }
    cpd_payload_path = artifacts / "changepoint_validation.csv"
    if cpd_payload_path.exists():
        cpd_payload = json.loads(pd.read_csv(cpd_payload_path).to_json(orient="records"))
        payload["changepoint_metrics"] = cpd_payload
    json_path = artifacts / "metrics.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Keep the competition root artifact byte-identical to the audited copy.
    Path("metrics.json").write_bytes(json_path.read_bytes())
    measured = table.loc[
        table["fold"].eq("pooled") & table["scope"].eq("exact")
    ].copy()
    lines = [
        "# Forecast metrics",
        "",
        "| Horizon | Model | N | MAE | R² | RMSE | WAPE | Status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in measured.sort_values(["horizon", "model"]).itertuples(index=False):
        def fmt(value: Any) -> str:
            return "n/a" if pd.isna(value) else f"{float(value):.6f}"
        lines.append(
            f"| {int(row.horizon)} | {row.model} | {int(row.n)} | {fmt(row.MAE)} | "
            f"{fmt(row.R2)} | {fmt(row.RMSE)} | {fmt(row.WAPE)} | {row.status} |"
        )
    cpd_path = artifacts / "changepoint_validation.csv"
    if cpd_path.exists():
        cpd = pd.read_csv(cpd_path)
        lines.extend([
            "",
            "## Changepoint metrics",
            "",
            "| Method | Precision | Recall | F1 | Detection delay, months |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in cpd.itertuples(index=False):
            delay = getattr(row, "detection_delay_months", np.nan)
            delay_text = "n/a" if pd.isna(delay) else f"{float(delay):.6f}"
            lines.append(
                f"| {row.method} | {float(row.precision):.6f} | {float(row.recall):.6f} | "
                f"{float(row.f1):.6f} | {delay_text} |"
            )
    md_path = artifacts / "tables.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # A root-level copy is convenient for competition submission packaging.
    root_metrics = artifacts.parent.parent / "metrics.json"
    root_metrics.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path


def build_direct_panel(
    frame: pd.DataFrame,
    horizon: int,
    config: ValidationConfig,
) -> pd.DataFrame:
    """Build a causal direct-forecast panel for one monthly horizon.

    A row has ``origin=t`` and target ``y[t+h]``.  Feature values are copied
    from the origin month, so lagged, rolling, macro and news columns cannot
    accidentally expose the target-month information at long horizons.
    """
    if horizon < 1:
        raise ValueError("horizon должен быть положительным")
    required = {"period", "mo", "y", *config.feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Для direct panel отсутствуют колонки: {sorted(missing)}")
    source = frame.copy()
    source["period"] = (
        pd.to_datetime(source["period"], utc=True, errors="raise")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    source = source.sort_values(["mo", "period"], kind="stable")
    features = source[["mo", "period", *config.feature_columns]].rename(columns={"period": "origin"})
    features["period"] = features["origin"] + pd.DateOffset(months=horizon)
    targets = source[["mo", "period", "y"]].rename(columns={"period": "period", "y": "target"})
    result = features.merge(targets, on=["mo", "period"], how="inner", validate="many_to_one")
    result["target_end"] = result["origin"] + pd.DateOffset(months=horizon)
    result["available_at"] = result["origin"] - pd.Timedelta(nanoseconds=1)
    result["region_id"] = result["mo"]
    result["y"] = pd.to_numeric(result.pop("target"), errors="raise")
    result["target"] = result["y"]
    result["lead_months"] = horizon
    return result[["period", "origin", "target_end", "available_at", "mo", "region_id", "y", "target", "lead_months", *config.feature_columns]]


def _direct_validation_config(config: ValidationConfig, horizon: int) -> ValidationConfig:
    """Return validation settings compatible with a direct h-step panel."""
    return config.model_copy(update={
        "horizon": horizon,
        "gap": max(config.gap, horizon),
    })


def _recursive_feature_row(history: Sequence[float], period: pd.Timestamp, columns: Sequence[str]) -> dict[str, float]:
    """Build causal lag/rolling features for a recursively predicted month."""
    values = np.asarray(history, dtype=float)
    def lag(number: int) -> float:
        return float(values[-number]) if len(values) >= number else np.nan
    row: dict[str, float] = {
        "month": float(period.month),
        "quarter": float(period.quarter),
        "sin_month": float(np.sin(2.0 * np.pi * (period.month - 1) / 12.0)),
        "cos_month": float(np.cos(2.0 * np.pi * (period.month - 1) / 12.0)),
        "y_lag_1": lag(1), "y_lag_2": lag(2), "y_lag_3": lag(3), "y_lag_12": lag(12),
        "growth_mom": (lag(1) / lag(2) - 1.0) if np.isfinite(lag(1)) and np.isfinite(lag(2)) and lag(2) else np.nan,
        "growth_yoy": (lag(1) / lag(13) - 1.0) if np.isfinite(lag(1)) and np.isfinite(lag(13)) and lag(13) else np.nan,
    }
    for window in (3, 12):
        recent = values[-window:]
        recent = recent[np.isfinite(recent)]
        row[f"y_rolling_mean_{window}"] = float(np.mean(recent)) if len(recent) == window else np.nan
        row[f"y_rolling_std_{window}"] = float(np.std(recent, ddof=1)) if len(recent) == window else np.nan
        row[f"y_rolling_min_{window}"] = float(np.min(recent)) if len(recent) == window else np.nan
        row[f"y_rolling_max_{window}"] = float(np.max(recent)) if len(recent) == window else np.nan
    mean = float(np.mean(values[-12:])) if len(values) else np.nan
    std = float(np.std(values[-12:], ddof=1)) if len(values) > 1 else np.nan
    row["spending_volatility"] = std / abs(mean) if np.isfinite(mean) and mean else np.nan
    return {column: row.get(column, np.nan) for column in columns}


def recursive_catboost_path(
    model: Any, imputer: Any, history: pd.DataFrame, origin: pd.Timestamp,
    periods: Sequence[pd.Timestamp], columns: Sequence[str],
    *, entities: Sequence[str] | None = None, frozen_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict one batch per month, never appending observations after origin."""
    past = history.loc[pd.to_datetime(history["period"]).lt(origin)].copy()
    contexts: dict[str, list[float]] = {}
    for entity, group in past.groupby("mo", observed=True):
        series = group.sort_values("period").set_index("period")["y"]
        calendar = pd.date_range(series.index.min(), origin - pd.offsets.MonthBegin(1), freq="MS")
        contexts[str(entity)] = pd.to_numeric(series.reindex(calendar), errors="coerce").dropna().astype(float).tolist()
    roster = sorted(str(entity) for entity in (entities if entities is not None else contexts))
    if not roster or past.empty:
        raise ValueError("Нет истории до recursive origin")
    cold_start = float(past["y"].median())
    for entity in roster:
        contexts.setdefault(entity, [cold_start])
    frozen = frozen_features.set_index("mo") if frozen_features is not None else pd.DataFrame()
    outputs = []
    for period in periods:
        features = []
        for entity in roster:
            row = _recursive_feature_row(contexts[entity], pd.Timestamp(period), columns)
            if not frozen.empty and entity in frozen.index:
                for column in columns:
                    if column in frozen.columns and column.startswith(("macro_", "rosstat_", "news_", "telegram_", "sentiment_")):
                        row[column] = frozen.loc[entity, column]
            features.append(row)
        feature_frame = pd.DataFrame(features, columns=columns).apply(pd.to_numeric, errors="coerce").astype(float)
        predictions = np.asarray(model.predict(imputer.transform(feature_frame)), dtype=float)
        predictions = np.maximum(predictions, 1e-6)
        for entity, prediction in zip(roster, predictions, strict=True):
            contexts[entity].append(float(prediction))
        outputs.append(pd.DataFrame({"mo": roster, "period": period, "pred_catboost": predictions}))
    return pd.concat(outputs, ignore_index=True)


def forecast_multi_horizon(
    config: AppConfig,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Run causal expanding-window OOF for Prophet, CatBoost and Chronos.

    The returned long panel contains one frozen-origin prediction per
    ``(municipality, origin, horizon)``.  Chronos is asked for the complete
    path and the h-th point is scored, while Prophet and CatBoost are fitted
    independently for each horizon.  This keeps the four competition
    horizons comparable and avoids reusing an h=1 forecast as a proxy.
    """
    from pathlib import Path

    from sklearn.impute import SimpleImputer

    from src.data_loader import load_target
    from src.features import to_validation_panel

    requested_horizons = tuple(int(value) for value in (horizons or config.validation.horizons))
    if not requested_horizons or any(value < 1 for value in requested_horizons):
        raise ValueError("horizons должен содержать положительные месяцы")
    data = pd.read_parquet(config.paths.supervised)
    data["period"] = pd.to_datetime(data["period"], utc=True, errors="raise")
    extra = [
        column for column in data.columns
        if str(column).startswith(("macro_", "rosstat_"))
        or column in {"sentiment_index", "news_volume", "news_shock_score", "telegram_sentiment",
                      "telegram_volume", "telegram_shock_score"}
    ]
    validation = config.validation.model_copy(update={
        "feature_columns": tuple(dict.fromkeys([*config.validation.feature_columns, *extra]))
    })
    feature_columns = [column for column in validation.feature_columns if column in data.columns]
    validation = validation.model_copy(update={"feature_columns": tuple(feature_columns)})
    target_history = load_target(config.data.directory, config.data.target, config.data.separator)
    # ``prepare_panel`` validates origins in UTC; keep Chronos history on the
    # same timezone-aware dtype before applying the strict ``period < origin``
    # causal filter.
    target_history["period"] = pd.to_datetime(target_history["period"], utc=True, errors="raise")

    # Optional imports remain lazy: a metrics-only run can execute without
    # downloading a foundation-model checkpoint.
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("CatBoost необходим для multi-horizon forecast") from exc
    try:
        from chronos import BaseChronosPipeline
        import torch
    except ImportError:
        BaseChronosPipeline = None  # type: ignore[assignment,misc]
        torch = None  # type: ignore[assignment]

    chronos_factory = None
    chronos_device = "cpu"
    if BaseChronosPipeline is not None and torch is not None:
        cc = config.models.chronos
        info = resolve_device(cc.device)
        chronos_device = info.device

        def load_pipeline(device: str, dtype: Any) -> Any:
            return BaseChronosPipeline.from_pretrained(
                cc.model_id, revision=cc.revision, device_map=device, torch_dtype=dtype,
            )

        chronos_factory = cached_pipeline_factory(load_pipeline)

    outputs: list[pd.DataFrame] = []
    for horizon in requested_horizons:
        direct = build_direct_panel(data, horizon, validation)
        # A short source history can make a direct horizon unevaluable after
        # warmup.  Keep the run alive so horizon_metric_table can report an
        # explicit ``not_evaluated`` row instead of aborting all later stages.
        if direct.empty:
            LOGGER.warning(
                "Горизонт h=%d пропущен: после warmup нет наблюдаемых target/origin пар",
                horizon,
            )
            continue
        direct_config = _direct_validation_config(validation, horizon)
        n_periods = direct["origin"].nunique()
        available_for_folds = n_periods - direct_config.gap - direct_config.min_train_periods
        max_splits = available_for_folds // direct_config.fold_size
        if max_splits < 2:
            # A purged direct h=12 panel cannot have a training origin on a
            # two-year source.  Evaluate the long horizon with a causal
            # recursive one-step CatBoost instead: every model is fitted only
            # on observations available at its own origin, then rolled forward
            # using its previous predictions.  This keeps h=12 measurable
            # without leaking the future target into training.
            if horizon >= 12:
                periods = sorted(pd.to_datetime(data["period"], utc=True).dropna().unique())
                candidate_origins = [
                    period for period in periods
                    if period + pd.DateOffset(months=horizon) in periods
                ][direct_config.min_train_periods:]
                # Keep two explicitly reported origins: the source has only
                # two years, and running six recursive Prophet fits adds cost
                # without increasing independent temporal coverage.
                needed = 2
                candidate_origins = candidate_origins[-needed:]
                if len(candidate_origins) >= 2:
                    recursive_columns = [
                        column for column in direct_config.feature_columns
                        if not str(column).startswith(("macro_", "rosstat_", "news_", "telegram_", "local_news", "local_telegram"))
                        and column not in {"sentiment_index", "news_volume", "news_shock_score", "telegram_sentiment", "telegram_volume", "telegram_shock_score"}
                    ]
                    target_lookup = data.set_index(["mo", "period"])["y"]
                    entities = sorted(data["mo"].dropna().unique())
                    params = dict(config.models.catboost)
                    params.update(random_seed=config.reproducibility.seed, thread_count=config.reproducibility.threads)
                    for fold_number, start in enumerate(range(0, len(candidate_origins), direct_config.fold_size), start=1):
                        for origin in candidate_origins[start:start + direct_config.fold_size]:
                            train = data.loc[data["period"].lt(origin)].copy()
                            train = train.loc[train["y"].notna()]
                            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                            model = CatBoostRegressor(**params)
                            model.fit(imputer.fit_transform(train[recursive_columns]), train["y"])
                            target_period = origin + pd.DateOffset(months=horizon)
                            eligible_entities = train.groupby("mo", observed=True).size()
                            eligible_entities = eligible_entities.index[eligible_entities.ge(direct_config.min_train_periods)]
                            path = recursive_catboost_path(model, imputer, target_history, origin,
                                                          pd.date_range(origin, target_period, freq="MS"),
                                                          recursive_columns, entities=eligible_entities)
                            test = path.loc[path["period"].eq(target_period)].merge(
                                data.loc[data["period"].eq(target_period), ["mo", "period", "y"]],
                                on=["mo", "period"], validate="one_to_one",
                            ).rename(columns={"y": "target"})
                            test["origin"] = origin
                            if test.empty:
                                continue
                            train_prophet = train[["period", "y", "mo"]]
                            prophet = predict_prophet_parallel(train_prophet, test[["period", "mo"]], config.models.prophet, seed=config.reproducibility.seed)
                            test["pred_prophet"] = prophet["prophet_prediction"].to_numpy(dtype=float)
                            test["pred_chronos"] = np.nan
                            if chronos_factory is not None and torch is not None:
                                cc = config.models.chronos
                                contexts: list[Any] = []
                                indices: list[Any] = []
                                fallback: dict[Any, float] = {}
                                for index, row in test.iterrows():
                                    history = target_history.loc[(target_history.mo.eq(row.mo)) & target_history.period.lt(row.origin)].sort_values("period").tail(cc.context_length)
                                    values = pd.to_numeric(history.y, errors="coerce").to_numpy(dtype=np.float32)
                                    finite = values[np.isfinite(values)]
                                    if finite.size < 2 or not np.isfinite(values).all():
                                        fallback[index] = float(finite[-1]) if finite.size else 0.0
                                    else:
                                        contexts.append(torch.as_tensor(values, dtype=torch.float32))
                                        indices.append(index)
                                if contexts:
                                    predictions, _ = predict_chronos_resilient(chronos_factory, contexts, prediction_length=horizon + 1, batch_size=config.device.batch_size, device=torch.device(chronos_device), dtype=getattr(torch, config.models.chronos.dtype), fallback_to_cpu=config.device.fallback_to_cpu_on_oom, fallback_values=[float(context[-1].item()) for context in contexts])
                                    for index, values in zip(indices, predictions, strict=True):
                                        test.loc[index, "pred_chronos"] = float(values[horizon])
                                for index, value in fallback.items():
                                    test.loc[index, "pred_chronos"] = value
                            else:
                                test["pred_chronos"] = test["mo"].map(train.groupby("mo", observed=True)["y"].last()).fillna(train["y"].median())
                            # This recursive fallback uses target features only;
                            # it is excluded from the paired news experiment.
                            test["pred_catboost_no_news"] = np.nan
                            test["pred_catboost_news"] = np.nan
                            test["pred_ensemble"] = (1.0 - config.ensemble.default_alpha) * test["pred_catboost"] + config.ensemble.default_alpha * test["pred_chronos"]
                            test["lead_months"] = horizon
                            test["fold"] = fold_number
                            test["actual_available_at"] = test["period"] + pd.offsets.MonthBegin(1)
                            test["news_available_at"] = test["origin"] - pd.offsets.MonthBegin(1)
                            outputs.append(test[["period", "origin", "mo", "target", "pred_prophet", "pred_catboost", "pred_catboost_news", "pred_catboost_no_news", "pred_chronos", "pred_ensemble", "lead_months", "fold", "actual_available_at", "news_available_at"]])
                    continue
            LOGGER.warning(
                "Горизонт h=%d пропущен: %d origin-периодов недостаточно для "
                "min_train_periods=%d, gap=%d и двух фолдов размера %d",
                horizon, n_periods, direct_config.min_train_periods,
                direct_config.gap, direct_config.fold_size,
            )
            continue
        if max_splits < direct_config.n_splits:
            LOGGER.warning(
                "Горизонт h=%d: число фолдов уменьшено с %d до %d из-за длины истории",
                horizon, direct_config.n_splits, max_splits,
            )
            direct_config = direct_config.model_copy(update={"n_splits": max_splits})
        prepared = prepare_panel(direct, direct_config)
        folds = expanding_window_splits(prepared, direct_config)
        for fold in folds:
            train = prepared.iloc[list(fold.train_positions)]
            test = prepared.iloc[list(fold.test_positions)].copy()
            columns = list(direct_config.feature_columns)
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            params = dict(config.models.catboost)
            params.update(random_seed=config.reproducibility.seed, thread_count=config.reproducibility.threads)
            model = CatBoostRegressor(**params)
            model.fit(imputer.fit_transform(train[columns]), train.y)
            test["pred_catboost"] = model.predict(imputer.transform(test[columns]))
            # Train a second OOF arm without point-in-time news columns.  The
            # two predictions share the fold but come from separate models,
            # making the news ablation a real experiment rather than a
            # post-hoc copy of one prediction.
            news_columns = {
                column for column in columns
                if str(column).startswith(("news_", "telegram_", "local_news", "local_telegram"))
                or column in {"sentiment_index", "news_volume", "news_shock_score", "telegram_sentiment",
                              "telegram_volume", "telegram_shock_score"}
            }
            no_news_columns = [column for column in columns if column not in news_columns]
            if no_news_columns:
                no_news_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                no_news_model = CatBoostRegressor(**params)
                no_news_model.fit(no_news_imputer.fit_transform(train[no_news_columns]), train.y)
                test["pred_catboost_no_news"] = no_news_model.predict(
                    no_news_imputer.transform(test[no_news_columns]),
                )
            else:
                test["pred_catboost_no_news"] = test["pred_catboost"]
            test["pred_catboost_news"] = test["pred_catboost"]

            train_prophet = train.rename(columns={"period": "period"})[["period", "y", "mo"]]
            test_prophet = test[["period", "mo"]].copy()
            prophet = predict_prophet_parallel(train_prophet, test_prophet, config.models.prophet, seed=config.reproducibility.seed)
            test["pred_prophet"] = prophet["prophet_prediction"].to_numpy(dtype=float)

            test["pred_chronos"] = np.nan
            if chronos_factory is not None and torch is not None:
                cc = config.models.chronos
                contexts: list[Any] = []
                indices: list[Any] = []
                fallback: dict[Any, float] = {}
                for index, row in test.iterrows():
                    history = target_history.loc[
                        target_history.mo.eq(row.mo) & target_history.period.lt(row.origin),
                    ].sort_values("period").tail(cc.context_length)
                    values = pd.to_numeric(history.y, errors="coerce").to_numpy(dtype=np.float32)
                    finite = values[np.isfinite(values)]
                    if finite.size == 0:
                        fallback[index] = 0.0
                    elif len(values) < 2 or not np.isfinite(values).all():
                        fallback[index] = float(finite[-1])
                    else:
                        contexts.append(torch.as_tensor(values, dtype=torch.float32))
                        indices.append(index)
                if contexts:
                    predictions, _ = predict_chronos_resilient(
                        chronos_factory, contexts, prediction_length=horizon + 1,
                        batch_size=config.device.batch_size, device=torch.device(chronos_device),
                        dtype=getattr(torch, cc.dtype), fallback_to_cpu=config.device.fallback_to_cpu_on_oom,
                        fallback_values=[float(context[-1].item()) for context in contexts],
                    )
                    for index, values in zip(indices, predictions, strict=True):
                        test.loc[index, "pred_chronos"] = float(values[horizon])
                for index, value in fallback.items():
                    test.loc[index, "pred_chronos"] = value
            else:
                last_values = train.sort_values("origin").groupby("mo", observed=True)["y"].last()
                test["pred_chronos"] = test["mo"].map(last_values).fillna(train["y"].median())

            test["pred_ensemble"] = (
                (1.0 - config.ensemble.default_alpha) * test["pred_catboost"]
                + config.ensemble.default_alpha * test["pred_chronos"]
            )
            result = test[["period", "origin", "mo", "y", "pred_prophet", "pred_catboost",
                           "pred_catboost_news", "pred_catboost_no_news", "pred_chronos",
                           "pred_ensemble", "lead_months"]].copy()
            result = result.rename(columns={"y": "target"})
            result["fold"] = fold.number
            result["actual_available_at"] = result["period"] + pd.offsets.MonthBegin(1)
            result["news_available_at"] = result["origin"] - pd.offsets.MonthBegin(1)
            outputs.append(result)
    if not outputs:
        raise ValueError("Multi-horizon CV не сформировал ни одного прогноза")
    result = pd.concat(outputs, ignore_index=True)
    destination = Path(config.changepoint_detection.predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    return result


def forecast_submission(config: AppConfig) -> Path:
    """Create July-December forecasts from the single July 2024 origin."""
    from catboost import CatBoostRegressor

    origin = pd.Timestamp("2024-07-01", tz="UTC")
    data = pd.read_parquet(config.paths.supervised)
    data["period"] = pd.to_datetime(data["period"], utc=True)
    history = load_target(config.data.directory, config.data.target, config.data.separator)
    history["period"] = pd.to_datetime(history["period"], utc=True)
    roster_path = config.paths.artifacts / "submission_roster.csv"
    if roster_path.exists():
        roster = pd.read_csv(roster_path)["mo"].astype(str).tolist()
    elif Path("submission.csv").exists():
        roster = sorted(pd.read_csv("submission.csv")["mo"].astype(str).unique())
        pd.DataFrame({"mo": roster}).to_csv(roster_path, index=False)
    else:
        raise FileNotFoundError("Нужен официальный submission roster или исходный submission.csv")
    columns = list(dict.fromkeys([
        *config.validation.feature_columns,
        *(column for column in data if column.startswith(("macro_", "rosstat_", "news_", "telegram_"))
          or column == "sentiment_index"),
    ]))
    columns = [column for column in columns if column in data]
    train = data.loc[data["period"].lt(origin)]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    params = dict(config.models.catboost)
    params.update(random_seed=config.reproducibility.seed, thread_count=config.reproducibility.threads)
    model = CatBoostRegressor(**params)
    model.fit(imputer.fit_transform(train[columns]), train["y"])
    frozen = data.loc[data["period"].le(origin)].sort_values("period").groupby("mo", observed=True).tail(1)
    periods = pd.date_range(origin, pd.Timestamp("2024-12-01", tz="UTC"), freq="MS")
    predictions = recursive_catboost_path(model, imputer, history, origin, periods, columns,
                                         entities=roster, frozen_features=frozen)
    prophet_train = history.loc[history["period"].lt(origin), ["period", "y", "mo"]]
    prophet = predict_prophet_parallel(prophet_train, predictions[["period", "mo"]],
                                       config.models.prophet, seed=config.reproducibility.seed)
    predictions["pred_prophet"] = prophet["prophet_prediction"].to_numpy(dtype=float)
    predictions["origin"] = origin
    predictions["protocol"] = "recursive_frozen_origin"
    destination = Path(config.changepoint_detection.predictions).with_name("predictions_submission.parquet")
    predictions.to_parquet(destination, index=False)
    import hashlib
    import json
    summary = {
        "origin": origin.isoformat(), "training_last_period": train["period"].max().isoformat(),
        "periods": [period.isoformat() for period in periods], "rows": len(predictions),
        "entities": len(roster), "roster_source": "initial_submission.csv; official code mapping not supplied",
        "roster_sha256": hashlib.sha256(roster_path.read_bytes()).hexdigest(),
        "protocol": "recursive_frozen_origin", "future_external_features": "frozen at origin",
        "submission_model": "prophet", "model_selection": "lowest pooled h=1 MAE and lower h=6 MAE on measured OOF",
        "observed_values_after_origin_used": False,
        "cold_start_entities": len(set(roster) - set(history.loc[history["period"].lt(origin), "mo"])),
    }
    (config.paths.artifacts / "submission_protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return destination
