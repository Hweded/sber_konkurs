"""Rolling-origin CV региональной панели с purging по доступности цели."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class ValidationConfig(BaseModel):
    """Размеры задаются в уникальных датах, а не строках панели."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    n_splits: Annotated[int, Field(strict=True, ge=2)] = 5
    horizons: tuple[PositiveInt, ...] = (1, 3, 6, 12)
    horizon: PositiveInt = 4
    fold_size: PositiveInt = 4
    gap: NonNegativeInt = 3
    min_train_periods: PositiveInt = 52
    frequency: str = "W-MON"
    time_column: str = "origin"
    target_end_column: str = "target_end"
    availability_column: str = "available_at"
    entity_column: str = "region_id"
    target_column: str = "target"
    feature_columns: tuple[str, ...]

    @model_validator(mode="after")
    def check_contract(self) -> ValidationConfig:
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Горизонты должны быть уникальными")
        if self.gap < self.horizon - 1:
            raise ValueError("gap должен быть >= horizon - 1 для direct-прогноза")
        metadata = (
            self.time_column,
            self.target_end_column,
            self.availability_column,
            self.entity_column,
            self.target_column,
        )
        if len(set(metadata)) != len(metadata):
            raise ValueError("Имена метаданных должны быть различны")
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Нужен непустой список уникальных признаков")
        if set(metadata).intersection(self.feature_columns):
            raise ValueError("Цель и метаданные нельзя включать в признаки")
        return self


class TemporalFold(BaseModel):
    model_config = ConfigDict(frozen=True)
    number: int
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class FoldMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    fold: int
    n_train: int
    n_test: int
    mae: float
    r2: float | None


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    folds: tuple[FoldMetrics, ...]
    oof_mae: float
    oof_r2: float | None
    n_predictions: int
    protocol: str = "rolling_origin_direct_h_step_frozen_model_per_fold"


class Regressor(Protocol):
    def fit(self, X: Any, y: Any, /) -> Any: ...
    def predict(self, X: Any, /) -> Any: ...
    def get_params(self, *, deep: bool = True) -> dict[str, Any]: ...


def prepare_panel(frame: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    """Проверяет временной контракт панели без импутации и масштабирования."""
    required = {
        config.time_column,
        config.target_end_column,
        config.availability_column,
        config.entity_column,
        config.target_column,
        *config.feature_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {sorted(missing)}")
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Пустая таблица или повторяющиеся имена колонок")
    panel = frame.copy(deep=True)
    for column in (config.time_column, config.target_end_column, config.availability_column):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="raise")
        if panel[column].isna().any():
            raise ValueError(f"Пропущенные даты: {column}")
    if panel[config.entity_column].isna().any():
        raise ValueError("Пропущен идентификатор территории")
    if panel.duplicated([config.entity_column, config.time_column]).any():
        raise ValueError("Повторяется пара территория/origin")
    panel = panel.sort_values(config.time_column, kind="stable").reset_index(drop=True)
    dates = pd.DatetimeIndex(panel[config.time_column].unique()).sort_values()
    expected = pd.date_range(dates[0], dates[-1], freq=config.frequency)
    if not dates.equals(expected):
        raise ValueError("Даты не образуют регулярную сетку заданной частоты")
    extended = pd.date_range(dates[0], periods=len(dates) + config.horizon, freq=config.frequency)
    target_dates = pd.Series(extended[config.horizon :], index=dates)
    if not panel[config.target_end_column].equals(panel[config.time_column].map(target_dates)):
        raise ValueError("target_end не соответствует origin + horizon")
    if (panel[config.availability_column] >= panel[config.time_column]).any():
        raise ValueError("Утечка: признаки должны быть доступны строго до origin")
    target = panel[config.target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Цель должна быть конечной; импутация цели запрещена")
    features = panel.loc[:, list(config.feature_columns)].to_numpy(dtype=np.float64)
    if np.isinf(features).any():
        raise ValueError("Бесконечности в признаках запрещены")
    return panel


def _folds(panel: pd.DataFrame, config: ValidationConfig) -> Iterator[TemporalFold]:
    dates = pd.DatetimeIndex(panel[config.time_column].unique()).sort_values()
    initial_size = len(dates) - config.n_splits * config.fold_size - config.gap
    if initial_size < config.min_train_periods:
        raise ValueError(
            f"Первый train содержит {initial_size} периодов; нужно {config.min_train_periods}"
        )
    splitter = TimeSeriesSplit(
        n_splits=config.n_splits,
        test_size=config.fold_size,
        gap=config.gap,
        max_train_size=None,
    )
    for number, (train, test) in enumerate(splitter.split(dates), start=1):
        train_dates, test_dates = dates[train], dates[test]
        # Равенство допустимо — origin трактуем как конец закрытого периода.
        train_mask = panel[config.time_column].isin(train_dates) & (
            panel[config.target_end_column] <= test_dates[0]
        )
        test_mask = panel[config.time_column].isin(test_dates)
        if panel.loc[train_mask, config.time_column].nunique() < config.min_train_periods:
            raise ValueError("После purging недостаточно обучающих периодов")
        train_entities = set(panel.loc[train_mask, config.entity_column])
        test_entities = set(panel.loc[test_mask, config.entity_column])
        cold_start = test_entities - train_entities
        if cold_start:
            LOGGER.warning(
                "Фолд %d: %d cold-start территорий сохранены в test; требуется fallback модели",
                number,
                len(cold_start),
            )
        yield TemporalFold(
            number=number,
            train_positions=tuple(int(i) for i in np.flatnonzero(train_mask.to_numpy())),
            test_positions=tuple(int(i) for i in np.flatnonzero(test_mask.to_numpy())),
            train_start=train_dates[0].isoformat(),
            train_end=train_dates[-1].isoformat(),
            test_start=test_dates[0].isoformat(),
            test_end=test_dates[-1].isoformat(),
        )


def expanding_window_splits(
    frame: pd.DataFrame, config: ValidationConfig
) -> tuple[TemporalFold, ...]:
    """Позиции относятся к результату prepare_panel, не к исходному frame."""
    return tuple(_folds(prepare_panel(frame, config), config))


def _r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float | None:
    # Не подменяем неопределённый R² искусственным 0/1.
    if len(y_true) < 2 or np.all(y_true == y_true[0]):
        return None
    value = float(r2_score(y_true, y_pred, force_finite=False))
    return value if np.isfinite(value) else None


def cross_validate(
    frame: pd.DataFrame,
    config: ValidationConfig,
    estimator: Regressor,
    *,
    scale: bool = False,
) -> ValidationReport:
    """Обучает новый sklearn Pipeline внутри каждого train-фолда."""
    panel = prepare_panel(frame, config)
    results: list[FoldMetrics] = []
    truths: list[NDArray[np.float64]] = []
    predictions: list[NDArray[np.float64]] = []
    for fold in _folds(panel, config):
        train = panel.iloc[list(fold.train_positions)]
        test = panel.iloc[list(fold.test_positions)]
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", clone(estimator)))
        pipeline = Pipeline(steps)
        columns = list(config.feature_columns)
        y_train = train[config.target_column].to_numpy(dtype=np.float64)
        y_test = test[config.target_column].to_numpy(dtype=np.float64)
        try:
            pipeline.fit(train[columns], y_train)
            predicted = np.asarray(pipeline.predict(test[columns]), dtype=np.float64)
            if predicted.shape != y_test.shape or not np.isfinite(predicted).all():
                raise ValueError("Некорректная форма или неконечные значения прогноза")
        except Exception:
            LOGGER.exception("Ошибка обучения/прогноза в фолде %s", fold.number)
            raise
        metric = FoldMetrics(
            fold=fold.number,
            n_train=len(train),
            n_test=len(test),
            mae=float(mean_absolute_error(y_test, predicted)),
            r2=_r2(y_test, predicted),
        )
        LOGGER.info("Фолд %s: MAE=%.6f R²=%s", fold.number, metric.mae, metric.r2)
        results.append(metric)
        truths.append(y_test)
        predictions.append(predicted)
    all_true, all_pred = np.concatenate(truths), np.concatenate(predictions)
    return ValidationReport(
        folds=tuple(results),
        oof_mae=float(mean_absolute_error(all_true, all_pred)),
        oof_r2=_r2(all_true, all_pred),
        n_predictions=len(all_true),
    )
