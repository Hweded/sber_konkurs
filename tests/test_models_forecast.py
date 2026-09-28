"""Быстрые проверки параллельного Prophet без запуска Stan."""
from __future__ import annotations

import sys
import subprocess
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import models_forecast
from src.models_forecast import predict_prophet_parallel


class FakeProphet:
    """Минимальный Prophet-double, записывающий параметры и входы."""

    constructor_calls: list[dict[str, Any]] = []
    fit_calls: list[pd.DataFrame] = []
    predict_calls: list[pd.DataFrame] = []

    def __init__(self, **parameters: Any) -> None:
        self.constructor_calls.append(parameters)

    def fit(self, history: pd.DataFrame, **_: Any) -> FakeProphet:
        self.fit_calls.append(history.copy())
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        self.predict_calls.append(future.copy())
        return pd.DataFrame({"yhat": np.arange(len(future), dtype=np.float64) + 10.0})


def _install_fake_prophet(monkeypatch: Any) -> None:
    module = ModuleType("prophet")
    module.Prophet = FakeProphet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prophet", module)


@pytest.mark.parametrize("returncode", [0, 1])
def test_windows_tbb_probe_does_not_decode_command_output(monkeypatch: Any, returncode: int) -> None:
    cmdstan = ModuleType("cmdstanpy")
    model = ModuleType("cmdstanpy.model")
    calls: list[dict[str, Any]] = []

    def original_command(command: list[str], **kwargs: Any) -> None:
        raise UnicodeDecodeError("utf-8", b"\x88", 0, 1, "invalid start byte")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(kwargs)
        if kwargs.get("check") and returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return subprocess.CompletedProcess(command, returncode)

    model.do_command = original_command  # type: ignore[attr-defined]
    cmdstan.model = model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cmdstanpy", cmdstan)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(models_forecast.subprocess, "run", fake_run)

    with models_forecast._cmdstan_windows_probe():
        if returncode:
            with pytest.raises(RuntimeError, match="Windows TBB lookup failed"):
                model.do_command(["where.exe", "tbb.dll"], fd_out=None)  # type: ignore[attr-defined]
        else:
            model.do_command(["where.exe", "tbb.dll"], fd_out=None)  # type: ignore[attr-defined]

    assert calls[0]["stdout"] == subprocess.DEVNULL
    assert calls[0]["stderr"] == subprocess.DEVNULL
    assert calls[0]["check"] is True
    assert model.do_command is original_command  # type: ignore[attr-defined]


def test_windows_tbb_probe_restores_command_after_error(monkeypatch: Any) -> None:
    cmdstan = ModuleType("cmdstanpy")
    model = ModuleType("cmdstanpy.model")
    calls: list[list[str]] = []

    def original_command(command: list[str], **kwargs: Any) -> None:
        calls.append(command)
        raise ValueError("Other command failed")

    model.do_command = original_command  # type: ignore[attr-defined]
    cmdstan.model = model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cmdstanpy", cmdstan)
    monkeypatch.setattr(sys, "platform", "win32")

    with pytest.raises(ValueError, match="Other command failed"):
        with models_forecast._cmdstan_windows_probe():
            model.do_command(["stanc.exe", "--version"])  # type: ignore[attr-defined]

    assert calls == [["stanc.exe", "--version"]]
    assert model.do_command is original_command  # type: ignore[attr-defined]


def test_short_history_uses_constant_without_prophet(monkeypatch: Any) -> None:
    """При <3 точках модель не создаётся, длина и fallback корректны."""
    _install_fake_prophet(monkeypatch)
    FakeProphet.constructor_calls.clear()
    train = pd.DataFrame(
        {
            "period": ["2024-01-01", "2024-02-01", "bad-date"],
            "y": [1.0, 7.5, "bad-target"],
            "mo": ["a", "a", "a"],
        }
    )
    test = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-03-01", "2024-04-01"]),
            "mo": ["a", "a"],
        }
    )

    result = models_forecast._fit_predict_single_mo("a", train, test)

    assert FakeProphet.constructor_calls == []
    assert result["prophet_prediction"].tolist() == [7.5, 7.5]
    assert len(result) == len(test)


def test_prophet_receives_required_parameters_and_clean_history(monkeypatch: Any) -> None:
    """Сезонности выключены, changepoints фиксированы, history очищена."""
    _install_fake_prophet(monkeypatch)
    FakeProphet.constructor_calls.clear()
    FakeProphet.fit_calls.clear()
    train = pd.DataFrame(
        {
            "period": ["2024-03-01", "bad", "2024-01-01", "2024-02-01"],
            "y": [3.0, 999.0, 1.0, 2.0],
            "mo": ["a"] * 4,
        }
    )
    test = pd.DataFrame({"period": pd.to_datetime(["2024-04-01"]), "mo": ["a"]})

    result = models_forecast._fit_predict_single_mo(
        "a",
        train,
        test,
        {"yearly_seasonality": True, "n_changepoints": 99},
        seed=17,
    )

    parameters = FakeProphet.constructor_calls[0]
    assert parameters["yearly_seasonality"] is False
    assert parameters["weekly_seasonality"] is False
    assert parameters["daily_seasonality"] is False
    assert parameters["n_changepoints"] == 2
    assert parameters["changepoint_prior_scale"] == 0.05
    assert parameters["stan_backend"] == "CMDSTANPY"
    assert list(FakeProphet.fit_calls[0].columns) == ["ds", "y"]
    assert FakeProphet.fit_calls[0]["y"].tolist() == [1.0, 2.0, 3.0]
    assert result["prophet_prediction"].tolist() == [10.0]


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Moscow"])
def test_prophet_gets_naive_dates_and_preserves_output_timezone(monkeypatch: Any, timezone: str) -> None:
    _install_fake_prophet(monkeypatch)
    FakeProphet.fit_calls.clear()
    FakeProphet.predict_calls.clear()
    dates = pd.date_range("2024-01-01", periods=4, freq="MS", tz=timezone)
    train = pd.DataFrame({"period": dates[:3], "mo": ["a"] * 3, "y": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"period": dates[3:], "mo": ["a"]})

    result = models_forecast._fit_predict_single_mo("a", train, test)

    assert FakeProphet.fit_calls[0]["ds"].tolist() == dates[:3].tz_localize(None).tolist()
    assert FakeProphet.predict_calls[0]["ds"].tolist() == dates[3:].tz_localize(None).tolist()
    assert result["period"].tolist() == test["period"].tolist()
    assert result["prophet_prediction"].tolist() == [10.0]


def test_backend_failure_stops_before_parallel_fallback(monkeypatch: Any) -> None:
    module = ModuleType("prophet")
    calls: list[dict[str, Any]] = []

    def broken_prophet(**parameters: Any) -> None:
        calls.append(parameters)
        raise ValueError("CmdStan installation missing binaries")

    def unexpected_parallel(**_: Any) -> None:
        pytest.fail("Workers must not start when Stan is unavailable")

    module.Prophet = broken_prophet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prophet", module)
    monkeypatch.setattr(models_forecast, "Parallel", unexpected_parallel)
    train = pd.DataFrame({
        "period": list(pd.date_range("2024-01-01", periods=3, freq="MS")) * 2,
        "mo": ["a"] * 3 + ["b"] * 3,
        "y": [1.0, 2.0, 3.0] * 2,
    })
    test = pd.DataFrame({"period": pd.to_datetime(["2024-04-01"] * 2), "mo": ["a", "b"]})

    with pytest.raises(RuntimeError, match="CmdStan installation missing binaries") as caught:
        predict_prophet_parallel(train, test)

    assert len(calls) == 1
    assert calls[0]["stan_backend"] == "CMDSTANPY"
    assert isinstance(caught.value.__cause__, ValueError)


def test_individual_fit_error_still_uses_fallback(monkeypatch: Any) -> None:
    _install_fake_prophet(monkeypatch)

    def failed_fit(self: Any, history: pd.DataFrame, **_: Any) -> None:
        raise ValueError("Optimization failed")

    monkeypatch.setattr(FakeProphet, "fit", failed_fit)
    train = pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "mo": ["a"] * 3, "y": [1.0, 2.0, 3.0],
    })
    test = pd.DataFrame({"period": pd.to_datetime(["2024-04-01"]), "mo": ["a"]})

    result = models_forecast._fit_predict_single_mo("a", train, test)

    assert result["prophet_prediction"].tolist() == [3.0]


def test_parallel_aggregation_preserves_schema_and_test_order(monkeypatch: Any) -> None:
    """Агрегация восстанавливает исходный interleaved-порядок строк test."""

    class SequentialParallel:
        def __init__(self, **_: Any) -> None:
            pass

        def __call__(self, tasks: Any) -> list[pd.DataFrame]:
            return [task[0](*task[1], **task[2]) for task in tasks]

    monkeypatch.setattr(models_forecast, "Parallel", SequentialParallel)
    train = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "mo": ["b", "a"],
            "y": [20.0, 10.0],
        }
    )
    test = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-02-01", "2024-02-01", "2024-03-01"]),
            "mo": ["b", "a", "b"],
        }
    )

    result = predict_prophet_parallel(train, test)

    assert result.columns.tolist() == ["period", "mo", "prophet_prediction"]
    assert result["mo"].tolist() == ["b", "a", "b"]
    assert result["prophet_prediction"].tolist() == [20.0, 10.0, 20.0]
