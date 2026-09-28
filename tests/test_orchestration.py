"""Проверка конфигурации, экспорта и настоящих детекторов без загрузки весов."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.configuration import load_config
from src.forecasting import metric_table
from src.changepoints import detect_changepoints, DetectionConfig
from src.evaluate_changepoints import evaluate, run_benchmark
from src.reporting import generate_report
from src.evaluate_changepoints import validate_against_reference_events
from src.submission import SubmissionValidator


def test_config_and_help() -> None:
    config = load_config(Path("configs/config.yaml"))
    assert config.changepoint_detection.min_history == 4
    process = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True)
    assert process.returncode == 0
    assert b"--train-forecast" in process.stdout


def test_actual_detectors_prefix_invariance() -> None:
    rng = np.random.default_rng(8)
    y = np.r_[rng.normal(100, 1, 24), rng.normal(170, 1, 24)]
    frame = pd.DataFrame({"period": pd.date_range("2020-01-01", periods=48, freq="MS"), "y": y, "prediction": 100.0})
    config = DetectionConfig(model="l2", pen=2)
    whole = detect_changepoints(frame, config)
    prefix = detect_changepoints(frame.iloc[:32], config)
    pd.testing.assert_frame_equal(whole.iloc[:32], prefix)
    assert whole.pelt_shock.sum() > 0
    assert whole.binseg_shock.sum() > 0


def test_export_smoke(tmp_path: Path) -> None:
    config = load_config(Path("configs/config.yaml"))
    rng = np.random.default_rng(10)
    frame = pd.DataFrame({"period": pd.date_range("2020-01-01", periods=36, freq="MS"), "mo": "synthetic", "y": rng.normal(100, 1, 36), "fold": np.repeat([1, 2, 3], 12)})
    for model, error in (("prophet", 2), ("catboost", 1), ("chronos", 3)):
        frame[model + "_prediction"] = frame.y + error
    predictions = tmp_path / "oof.parquet"
    frame.to_parquet(predictions)
    detection = config.changepoint_detection.model_copy(update={"predictions": str(predictions), "news": str(tmp_path / "absent.parquet"), "output": str(tmp_path / "shocks.parquet"), "figures": str(tmp_path / "figures"), "max_figures": 1})
    settings = config.model_copy(update={"changepoint_detection": detection, "paths": config.paths.model_copy(update={"artifacts": tmp_path / "artifacts"})})
    run_benchmark(detection)
    generate_report(settings)
    result = json.loads((tmp_path / "artifacts/final_summary.json").read_text(encoding="utf-8"))
    assert result["catboost_relative_mae_improvement"] == 0.5
    assert len(result["figures"]) == 1
    assert Path(result["figures"][0]).exists()
    assert len(metric_table(frame)) == 12


def test_changepoint_evaluation_splits_gapped_oof_series() -> None:
    periods = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-04-01", "2023-05-01"])
    frame = pd.DataFrame(
        {
            "period": periods,
            "mo": "synthetic",
            "target": [100.0, 101.0, 103.0, 104.0],
            "pred_catboost": [99.0, 100.0, 102.0, 103.0],
        }
    )

    detected, _, summary = evaluate(
        frame,
        DetectionConfig(model="l2", min_history=3, window=6),
    )

    local = detected.loc[detected["mo"].eq("synthetic")]
    assert local["period"].tolist() == periods.tolist()
    assert local["cusum_eligible"].eq(False).all()
    assert summary["rows"] == len(detected)


def test_changepoint_evaluation_accepts_modern_oof_columns() -> None:
    periods = pd.date_range("2023-01-01", periods=12, freq="MS")
    frame = pd.DataFrame(
        {
            "period": periods,
            "mo": "synthetic",
            "target": np.linspace(100.0, 111.0, len(periods)),
            "pred_catboost": np.linspace(99.0, 110.0, len(periods)),
            "actual_available_at": periods + pd.offsets.MonthBegin(1),
        }
    )

    detected, metrics, summary = evaluate(
        frame,
        DetectionConfig(model="l2", min_history=6, window=6, national_history=None),
    )

    assert len(detected) == len(frame)
    assert detected["y"].equals(frame["target"])
    assert detected["prediction"].equals(frame["pred_catboost"])
    assert not metrics.empty
    assert summary["entities"] == 1


def test_submission_expands_grid_and_prefers_catboost(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "mo": ["a", "a", "b", "b"],
            "period": pd.to_datetime(["2024-07-01", "2024-12-01", "2024-07-01", "2024-12-01"]),
            "pred_catboost": [10.0, 12.0, 20.0, 22.0],
            "pred_ensemble": [100.0, 120.0, 200.0, 220.0],
        }
    )
    validator = SubmissionValidator(tmp_path / "reports/submission.csv", tmp_path / "submission.csv")
    result = validator.build(predictions, expected_rows=12, expected_entities=2)
    assert len(result) == 12
    assert result["pred"].notna().all()
    assert result.loc[result.mo.eq("a"), "pred"].tolist() == [10.0] * 5 + [12.0]
    assert (tmp_path / "reports/submission.csv").read_bytes() == (tmp_path / "submission.csv").read_bytes()


def test_submission_rejects_nonpositive_predictions(tmp_path: Path) -> None:
    predictions = pd.DataFrame({"mo": ["a"], "period": ["2024-07-01"], "pred_catboost": [0.0]})
    validator = SubmissionValidator(tmp_path / "report.csv", tmp_path / "root.csv")
    with pytest.raises(AssertionError, match="положительными"):
        validator.build(predictions, expected_rows=6, expected_entities=1)


def test_changepoint_reference_metrics_use_day_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "period": pd.to_datetime(["2023-07-15", "2023-08-15", "2023-12-01", "2024-07-01", "2024-10-01"]),
            "pelt_shock": [1, 1, 1, 1, 1],
            "cusum_shock": [0, 1, 0, 0, 0],
            "residual_shock": [0, 0, 0, 0, 0],
            "tda_shock": [0, 0, 1, 1, 1],
        }
    )
    result = validate_against_reference_events(frame).set_index("method")
    assert result.loc["Pelt", "recall"] == 1.0
    assert result.loc["Pelt", "precision"] == 0.8
    assert result.loc["CUSUM", "coverage_pct"] == 25.0
    assert result.loc["Residual", "f1"] == 0.0


def test_reference_metrics_prefer_monthly_macro_alerts() -> None:
    frame = pd.DataFrame(
        {
            "period": pd.to_datetime(["2023-08-20", "2023-12-20", "2024-07-20", "2024-10-20", "2024-11-20"]),
            "pelt_shock": [0, 0, 0, 0, 0],
            "pelt_macro_shock": [1, 1, 1, 1, 1],
            "cusum_shock": [0, 0, 0, 0, 0],
            "residual_shock": [0, 0, 0, 0, 0],
            "tda_shock": [0, 0, 0, 0, 0],
        },
    )
    result = validate_against_reference_events(frame).set_index("method")
    assert result.loc["Pelt", "recall"] == 1.0
    assert result.loc["Pelt", "precision"] == 0.8
    assert result.loc["Pelt", "f1"] == pytest.approx(8 / 9)
