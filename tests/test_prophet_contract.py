"""Регрессии схемы Prophet и контроля массового fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import models_forecast as models


class RecordingProphet:
    histories = []

    def fit(self, history, **kwargs):
        self.histories.append(history.copy())
        return self

    def predict(self, future):
        return pd.DataFrame({"yhat": np.full(len(future), 17.0)})


class SequentialParallel:
    def __init__(self, **kwargs):
        pass

    def __call__(self, tasks):
        return [func(*args, **kwargs) for func, args, kwargs in tasks]


@pytest.fixture
def prophet_double(monkeypatch):
    RecordingProphet.histories.clear()
    monkeypatch.setattr(models, "_create_prophet", lambda params: RecordingProphet())
    monkeypatch.setattr(models, "Parallel", SequentialParallel)


@pytest.mark.parametrize("date_col", ["period", "ds", "date"])
@pytest.mark.parametrize("target_col", ["y", "target", "value"])
def test_training_aliases_reach_prophet_fit(prophet_double, date_col, target_col):
    train = pd.DataFrame(
        {
            date_col: pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC"),
            target_col: [1.0, 2.0, 3.0],
            "mo": "a",
        }
    )
    test = pd.DataFrame({"period": [pd.Timestamp("2024-04-01", tz="UTC")], "mo": "a"})
    result = models.predict_prophet_parallel(train, test)
    assert result.prophet_prediction.tolist() == [17.0]
    assert RecordingProphet.histories[0].columns.tolist() == ["ds", "y"]
    assert RecordingProphet.histories[0].ds.dt.tz is None
    assert RecordingProphet.histories[0].y.tolist() == [1.0, 2.0, 3.0]


def test_missing_training_schema_stops_before_workers(prophet_double, monkeypatch):
    def unexpected_parallel(**kwargs):
        pytest.fail("Некорректная схема не должна запускать воркеры")

    monkeypatch.setattr(models, "Parallel", unexpected_parallel)
    train = pd.DataFrame({"unknown": [1.0], "mo": "a"})
    test = pd.DataFrame({"period": [pd.Timestamp("2024-04-01")], "mo": "a"})
    with pytest.raises(ValueError, match="Prophet.*колон"):
        models.predict_prophet_parallel(train, test)


@pytest.mark.parametrize("failures,raises", [(1, False), (2, True)])
def test_fallback_limit_counts_municipalities_not_rows(
    prophet_double, monkeypatch, failures, raises
):
    entities = [str(i) for i in range(20)]
    train = pd.DataFrame(
        [
            {"period": period, "y": float(i + 1), "mo": entity}
            for i, entity in enumerate(entities)
            for period in pd.date_range("2024-01-01", periods=3, freq="MS")
        ]
    )
    test = pd.DataFrame(
        [
            {"period": pd.Timestamp("2024-04-01"), "mo": entity}
            for entity in entities
            for _ in range(3 if int(entity) < failures else 1)
        ]
    )
    original = models._fit_predict_single_mo

    def fit_or_fallback(mo, history, future, params, seed):
        if int(mo) < failures:
            return models._constant_predictions(mo, future, 1.0)
        return original(mo, history, future, params, seed)

    monkeypatch.setattr(models, "_fit_predict_single_mo", fit_or_fallback)
    if raises:
        with pytest.raises(RuntimeError, match="Более 5% МО"):
            models.predict_prophet_parallel(train, test, batch_size=5)
    else:
        result = models.predict_prophet_parallel(train, test, batch_size=5)
        assert result.attrs["prophet_diagnostics"]["fallback_entities"] == failures
        assert result.attrs["prophet_diagnostics"]["fitted_entities"] == 20 - failures
        assert result.columns.tolist() == ["period", "mo", "prophet_prediction"]


def test_successful_fit_preserves_unsorted_dates(prophet_double, monkeypatch):
    def sorted_prediction(self, future):
        ordered = future.sort_values("ds").reset_index(drop=True)
        return pd.DataFrame({"ds": ordered.ds, "yhat": ordered.ds.dt.month.astype(float)})

    monkeypatch.setattr(RecordingProphet, "predict", sorted_prediction)
    train = pd.DataFrame(
        {
            "period": pd.date_range("2024-01-01", periods=3, freq="MS"),
            "y": [1.0, 2.0, 3.0],
            "mo": "a",
        }
    )
    test = pd.DataFrame(
        {"period": pd.to_datetime(["2024-06-01", "2024-04-01", "2024-05-01"]), "mo": "a"}
    )
    result = models.predict_prophet_parallel(train, test)
    assert result.prophet_prediction.tolist() == [6.0, 4.0, 5.0]
    assert result.period.tolist() == test.period.tolist()


@pytest.mark.parametrize("release", [True, False])
def test_worker_cleanup_on_parallel_error(prophet_double, monkeypatch, release):
    cleanup = []

    class BrokenParallel(SequentialParallel):
        def __call__(self, tasks):
            raise RuntimeError("worker failure")

    monkeypatch.setattr(models, "Parallel", BrokenParallel)
    monkeypatch.setattr(models, "release_prophet_workers", lambda: cleanup.append(True))
    train = pd.DataFrame(
        {
            "period": pd.date_range("2024-01-01", periods=3, freq="MS"),
            "y": [1.0, 2.0, 3.0],
            "mo": "a",
        }
    )
    test = pd.DataFrame({"period": [pd.Timestamp("2024-04-01")], "mo": "a"})
    with pytest.raises(RuntimeError, match="worker failure"):
        models.predict_prophet_parallel(train, test, release_workers=release)
    assert cleanup == ([True] if release else [])
