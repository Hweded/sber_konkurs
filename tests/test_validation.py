"""Автоматические проверки временных границ и fold-local обучения."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor

from src.validation import ValidationConfig, cross_validate, expanding_window_splits, prepare_panel


class RecordingRegressor(RegressorMixin, BaseEstimator):
    seen: ClassVar[list[NDArray[np.float64]]] = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> RecordingRegressor:
        self.seen.append(X.copy())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full(X.shape[0], self.mean_, dtype=np.float64)


@pytest.fixture
def config() -> ValidationConfig:
    return ValidationConfig(n_splits=3, horizon=2, fold_size=2, gap=1,
                            min_train_periods=4, frequency="D", feature_columns=("x",))


@pytest.fixture
def panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=18, tz="UTC")
    origin = dates.repeat(2)
    return pd.DataFrame({
        "origin": origin, "target_end": origin + pd.Timedelta(days=2),
        "available_at": origin - pd.Timedelta(days=1),
        "region_id": ["a", "b"] * 18, "x": np.arange(36, dtype=float),
        "target": np.arange(36, dtype=float) * 2,
    })


def test_panel_dates_and_expansion(panel: pd.DataFrame, config: ValidationConfig) -> None:
    prepared = prepare_panel(panel.sample(frac=1, random_state=42), config)
    folds = expanding_window_splits(prepared, config)
    previous: set[int] = set()
    for fold in folds:
        train, test = prepared.iloc[list(fold.train_positions)], prepared.iloc[list(fold.test_positions)]
        assert previous.issubset(fold.train_positions)
        assert set(train.origin).isdisjoint(test.origin)
        assert train.target_end.max() <= test.origin.min()
        assert test.origin.nunique() == config.fold_size
        assert len(test) == config.fold_size * 2
        previous = set(fold.train_positions)


def test_fold_local_scaling(panel: pd.DataFrame, config: ValidationConfig) -> None:
    RecordingRegressor.seen.clear()
    panel.loc[0, "x"] = np.nan
    panel.loc[panel.index[-4:], "x"] = 1e9
    model = RecordingRegressor()
    report = cross_validate(panel, config, model, scale=True)
    assert report.n_predictions == 12
    assert len(RecordingRegressor.seen) == 3
    for matrix in RecordingRegressor.seen:
        assert np.isfinite(matrix).all()
        assert np.allclose(matrix.mean(axis=0), 0, atol=1e-10)
    assert not hasattr(model, "mean_")


def test_future_values_do_not_change_first_training(panel: pd.DataFrame, config: ValidationConfig) -> None:
    RecordingRegressor.seen.clear()
    cross_validate(panel, config, RecordingRegressor(), scale=True)
    original = RecordingRegressor.seen[0].copy()
    first = expanding_window_splits(panel, config)[0]
    panel.loc[list(first.test_positions), "x"] = 1e12
    RecordingRegressor.seen.clear()
    cross_validate(panel, config, RecordingRegressor(), scale=True)
    np.testing.assert_array_equal(original, RecordingRegressor.seen[0])


def test_reject_feature_leakage(panel: pd.DataFrame, config: ValidationConfig) -> None:
    panel.loc[0, "available_at"] = panel.loc[0, "origin"]
    with pytest.raises(ValueError, match="Утечка"):
        prepare_panel(panel, config)


def test_reject_wrong_horizon(panel: pd.DataFrame, config: ValidationConfig) -> None:
    panel.loc[0, "target_end"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="target_end"):
        prepare_panel(panel, config)


def test_reject_duplicates(panel: pd.DataFrame, config: ValidationConfig) -> None:
    with pytest.raises(ValueError, match="Повторяется"):
        prepare_panel(pd.concat([panel, panel.iloc[:1]]), config)


def test_reject_irregular_grid(panel: pd.DataFrame, config: ValidationConfig) -> None:
    with pytest.raises(ValueError, match="сетку"):
        prepare_panel(panel.iloc[2:].drop(index=[10, 11]), config)


def test_reject_insufficient_history(panel: pd.DataFrame, config: ValidationConfig) -> None:
    with pytest.raises(ValueError, match="Первый train"):
        expanding_window_splits(panel.iloc[:12], config)


def test_undefined_r2(panel: pd.DataFrame, config: ValidationConfig) -> None:
    panel["target"] = 5.0
    report = cross_validate(panel, config, DummyRegressor())
    assert report.oof_mae == 0.0
    assert report.oof_r2 is None


def test_reject_target_feature() -> None:
    with pytest.raises(ValueError, match="метаданные"):
        ValidationConfig(feature_columns=("target",))


def test_reject_short_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        ValidationConfig(feature_columns=("x",), horizon=4, gap=0)
