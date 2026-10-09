"""Tests for the memory-safe helpers introduced for the from-scratch retrain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ensemble import EnsembleConfig, blender_from_config, load_blend_weights
from src.memory import MemoryMonitor, downcast_frame, frame_footprint_mib, rss_mib


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "mo": ["А", "Б", "А"],
            "y": pd.array([1.5, None, 3.25], dtype="Float64"),
            "month": np.array([1, 2, 3], dtype=np.int64),
            "big_counter": np.array([10**9, 10**9 + 1, 10**9 + 2], dtype=np.int64),
            "y_lag_1": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    )


def test_downcast_shrinks_types_but_protects_target() -> None:
    frame = _frame()
    lean = downcast_frame(frame)
    assert str(lean["mo"].dtype) == "category"
    assert lean["y_lag_1"].dtype == np.float32
    assert lean["month"].dtype == np.int16
    # 10**9 does not fit in int16 but does fit in int32
    assert lean["big_counter"].dtype == np.int32
    # target and period keep full precision
    assert str(lean["y"].dtype) == str(frame["y"].dtype)
    assert lean["period"].dtype == frame["period"].dtype


def test_downcast_preserves_values_and_missingness() -> None:
    frame = _frame()
    lean = downcast_frame(frame)
    assert np.array_equal(frame["y"].to_numpy(float), lean["y"].to_numpy(float), equal_nan=True)
    assert lean.isna().sum().sum() == frame.isna().sum().sum()
    assert frame_footprint_mib(lean) <= frame_footprint_mib(frame)


def test_downcast_is_idempotent() -> None:
    once = downcast_frame(_frame())
    twice = downcast_frame(once)
    assert once.dtypes.astype(str).tolist() == twice.dtypes.astype(str).tolist()


def test_memory_monitor_reports_peak() -> None:
    monitor = MemoryMonitor("test")
    monitor.sample().sample()
    if rss_mib() is None:
        assert monitor.samples == 0
        assert "не измерено" in monitor.describe()
    else:
        assert monitor.samples == 2
        assert monitor.peak_rss_mib > 0
        assert "пиковое RSS" in monitor.describe()


def test_load_blend_weights_reads_normalised_horizons(tmp_path: Path) -> None:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"1": [2, 2, 0], "6": [0, 1, 0], "7": [1, 0, 0]}), encoding="utf-8")
    loaded = load_blend_weights(path)
    assert loaded is not None
    assert loaded[1] == pytest.approx((0.5, 0.5, 0.0))
    assert loaded[6] == (0.0, 1.0, 0.0)
    assert 7 not in loaded  # outside the configured horizons


def test_load_blend_weights_missing_file_falls_back(tmp_path: Path) -> None:
    assert load_blend_weights(tmp_path / "absent.json") is None
    assert load_blend_weights(None) is None


def test_load_blend_weights_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"1": [1.0, 0.0]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_blend_weights(path)


def test_blender_from_config_uses_weight_file(tmp_path: Path) -> None:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"1": [0.0, 1.0, 0.0]}), encoding="utf-8")
    blender = blender_from_config(EnsembleConfig(weights_path=path))
    assert blender.weights_for(1) == (0.0, 1.0, 0.0)
    # horizons absent from the file keep the factory values
    assert blender.weights_for(6) == pytest.approx((0.8220, 0.1779, 0.0001), abs=1e-6)


def test_blender_from_config_without_file_is_factory() -> None:
    blender = blender_from_config(EnsembleConfig(weights_path=None))
    assert blender.weights_for(1) == pytest.approx((0.5741, 0.4258, 0.0001), abs=1e-6)