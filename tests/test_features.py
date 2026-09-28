"""Тесты календарных лагов, разделения МО, макроизмерений и NLP."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_config import DataConfig, FeatureConfig, MacroSourceConfig, TargetConfig
from src.data_loader import discover_file, load_macro, load_target
from src.features import generate_features, merge_news_features, to_validation_panel
from src.validation import ValidationConfig, expanding_window_splits


def panel() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=30, freq="MS")
    return pd.concat([pd.DataFrame({"period": dates, "mo": mo, "y": np.arange(1, 31, dtype=float) * factor}) for mo, factor in (("a", 1), ("b", 100))], ignore_index=True)


def source(periods: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"period": periods, "value": values, "obs_status": "A", "freq": "Месяц", "unit_mult": 0, "unit_measure": "руб."})


def test_exact_features() -> None:
    result = generate_features(panel(), FeatureConfig())
    first = result.loc[result.mo.eq("a")].iloc[0]
    assert first.y == 13
    assert first.y_lag_1 == 12
    assert first.y_lag_12 == 1
    assert first.y_rolling_mean_3 == 11
    assert first.y_rolling_std_3 == 1
    assert first.y_rolling_min_12 == 1
    assert first.y_rolling_max_12 == 12
    assert np.isnan(first.growth_yoy)
    second = result.loc[result.mo.eq("a")].iloc[1]
    assert second.growth_yoy == 12
    assert result.loc[result.mo.eq("b")].iloc[0].y_lag_1 == 1200
    assert str(result.period.dtype) == "datetime64[ns]"


def test_future_invariance() -> None:
    original = panel()
    changed = original.copy()
    cutoff = pd.Timestamp("2023-06-01")
    changed.loc[changed.period.ge(cutoff), "y"] = 1e12
    left = generate_features(original, FeatureConfig()).drop(columns="y")
    right = generate_features(changed, FeatureConfig()).drop(columns="y")
    pd.testing.assert_frame_equal(left.loc[left.period.le(cutoff)], right.loc[right.period.le(cutoff)])


def test_missing_month_is_not_previous_observation() -> None:
    data = panel()
    data = data.loc[~(data.mo.eq("a") & data.period.eq("2023-02-01"))]
    result = generate_features(data, FeatureConfig())
    march = result.loc[result.mo.eq("a") & result.period.eq("2023-03-01")].iloc[0]
    assert np.isnan(march.y_lag_1)
    assert march.y_lag_2 == 13
    assert np.isnan(march.y_rolling_mean_3)


def test_zero_denominator() -> None:
    data = panel()
    data.loc[data.period.eq("2022-12-01"), "y"] = 0
    result = generate_features(data, FeatureConfig())
    assert result.loc[result.period.eq("2023-02-01"), "growth_mom"].isna().all()
    assert not np.isinf(result.select_dtypes(include="number").to_numpy()).any()


def test_macro_calendar_shift_and_segments(tmp_path: Path) -> None:
    data = source(["2023-01-01", "2023-03-01"] * 2, [10, 30, 100, 300])
    data["type"] = ["a", "a", "b", "b"]
    data.to_csv(tmp_path / "macro.csv", sep=";", index=False, encoding="utf-8")
    config = DataConfig(macro_sources=(MacroSourceConfig(name="test", pattern="macro.csv", dimensions=("type",), frequency="Месяц"),))
    result = load_macro(tmp_path, config).set_index("period")
    assert result.shape[1] == 2
    assert sorted(result.loc["2023-02-01"].tolist()) == [10, 100]
    assert result.loc["2023-03-01"].isna().all()
    assert sorted(result.loc["2023-04-01"].tolist()) == [30, 300]


def test_weekly_aggregation(tmp_path: Path) -> None:
    data = source(["2023-01-08", "2023-01-15"], [10, 30])
    data["freq"] = "Неделя"
    data.to_csv(tmp_path / "weekly.csv", sep=";", index=False, encoding="utf-8")
    config = DataConfig(macro_sources=(MacroSourceConfig(name="weekly", pattern="weekly.csv", dimensions=(), frequency="Неделя"),))
    result = load_macro(tmp_path, config)
    assert result.iloc[-1, 1] == 20


def test_target_category_and_multiplier(tmp_path: Path) -> None:
    data = source(["2023-01-01"] * 3, [10, 4, 6])
    data["mo"] = "001"
    data["category_15"] = ["Все категории", "a", "b"]
    data["unit_mult"] = 3
    data.to_csv(tmp_path / "target.csv", sep=";", index=False, encoding="utf-8")
    result = load_target(tmp_path, TargetConfig(pattern="target.csv"))
    assert result.iloc[0].y == 10000
    assert result.iloc[0].mo == "001"
    summed = load_target(tmp_path, TargetConfig(pattern="target.csv", category_mode="sum", sum_categories=("a", "b")))
    assert summed.iloc[0].y == 10000


def test_news_calendar_lag() -> None:
    news = pd.DataFrame({"period": ["2023-01-01", "2023-03-01"], "news_sentiment": [0.2, 0.9], "news_shock_index": [1, 2]})
    result = merge_news_features(panel(), news)
    assert result.loc[result.period.eq("2023-02-01"), "news_sentiment"].eq(0.2).all()
    assert result.loc[result.period.eq("2023-03-01"), "news_sentiment"].isna().all()
    with pytest.raises(ValueError):
        merge_news_features(panel(), news, lag_months=0)


def test_ambiguous_vintages(tmp_path: Path) -> None:
    (tmp_path / "a.csv").touch()
    (tmp_path / "b.csv").touch()
    with pytest.raises(ValueError, match="Несколько"):
        discover_file(tmp_path, "*.csv")


def test_monthly_validation_adapter() -> None:
    features = generate_features(panel(), FeatureConfig())
    config = ValidationConfig(n_splits=3, horizon=1, fold_size=2, gap=0, min_train_periods=6, frequency="MS", feature_columns=("y_lag_1",))
    ready = to_validation_panel(features, config)
    assert ready.target.equals(ready.y)
    assert len(expanding_window_splits(ready, config)) == 3
