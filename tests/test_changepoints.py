"""Тесты причинного протокола детекции структурных сдвигов."""
import numpy as np
import pandas as pd
import pytest
import warnings

from src.changepoints import DetectionConfig, detect_changepoints


def test_constant_news_signal_has_undefined_correlations_without_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import evaluate_changepoints

    periods = pd.date_range("2024-01-01", periods=3, freq="MS")
    detected = pd.DataFrame({"period": periods, "mo": ["a"] * 3, "residual": [1.0, 2.0, 3.0]})
    news = pd.DataFrame({"period": periods, "news_shock_score": [0.0] * 3})
    monkeypatch.setattr(evaluate_changepoints, "_load_national_history", lambda config: None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate_changepoints._continuous_news_correlations(detected, news, DetectionConfig())

    assert result["correlation"].isna().all()
    assert result["spearman_correlation"].isna().all()
    assert result["n_comparable"].tolist() == [3]


def synthetic_series() -> pd.DataFrame:
    periods = pd.date_range("2020-01-01", periods=48, freq="MS")
    rng = np.random.default_rng(42)
    y = np.r_[rng.normal(100, 1, 24), rng.normal(150, 1, 24)]
    prediction = np.r_[y[:24], np.repeat(100.0, 24)]
    return pd.DataFrame({
        "period": periods,
        "y": y,
        "prediction": prediction,
        "news_shock_score": np.r_[np.zeros(22), 8.0, np.zeros(25)],
        "sentiment_index": np.r_[np.zeros(22), -0.8, np.zeros(25)],
    })


def test_detects_residual_and_news_supported_shock() -> None:
    result = detect_changepoints(synthetic_series(), DetectionConfig(window=12, min_history=12, pen=2))
    assert result["cusum_eligible"].iloc[12:].all()
    assert result["residual_news_shock"].sum() >= 1
    assert result["news_alert"].sum() >= 1
    assert set(result["residual_shock"].unique()).issubset({0, 1})


def test_scores_are_not_probabilities_and_dates_are_causal() -> None:
    frame = synthetic_series()
    result = detect_changepoints(frame, DetectionConfig(window=12, min_history=12))
    assert result.loc[:11, "cusum_eligible"].eq(False).all()
    assert result.loc[:11, "residual_eligible"].eq(False).all()
    assert result.loc[:11, "residual_news_eligible"].eq(False).all()
    assert result["cusum_score"].dropna().max() > 1
    assert not result["cusum_score"].dropna().between(0, 1).all()


def test_rejects_gaps_and_nonfinite() -> None:
    frame = synthetic_series().drop(index=3)
    with pytest.raises(ValueError, match="непрерывный"):
        detect_changepoints(frame, DetectionConfig())
    frame = synthetic_series()
    frame.loc[0, "y"] = np.inf
    with pytest.raises(ValueError, match="конечными"):
        detect_changepoints(frame, DetectionConfig())


def test_compare_shock_detectors_smoke() -> None:
    """Дымовой тест compare_shock_detectors: возвращает все ключи."""
    from src.changepoints import compare_shock_detectors

    rng = np.random.default_rng(42)
    periods = pd.date_range("2020-01-01", periods=36, freq="MS")
    frame = pd.DataFrame({
        "period": periods,
        "mo": "test_mo",
        "y": rng.normal(100, 5, 36),
        "prediction": rng.normal(100, 5, 36),
        "pelt_shock": np.zeros(36, dtype=bool),
        "binseg_shock": np.zeros(36, dtype=bool),
        "cusum_shock": np.zeros(36, dtype=bool),
        "residual_shock": np.zeros(36, dtype=bool),
        "residual_news_shock": np.zeros(36, dtype=bool),
        "pelt_eligible": np.full(36, True),
        "binseg_eligible": np.full(36, True),
        "cusum_eligible": np.full(36, True),
        "residual_eligible": np.full(36, True),
        "residual_news_eligible": np.full(36, True),
        "tda_wasserstein_dist": np.abs(rng.normal(0, 0.5, 36)),
        "news_alert": np.zeros(36, dtype=bool),
        "news_shock_score": np.zeros(36, dtype=float),
    })
    # Искусственные шоки
    frame.loc[20, ["pelt_shock", "tda_wasserstein_dist"]] = [True, 3.0]
    frame.loc[21, "news_alert"] = True
    frame.loc[30, "tda_wasserstein_dist"] = 4.0
    frame.loc[31, "news_alert"] = True

    result = compare_shock_detectors(frame, DetectionConfig(min_history=12, event_tolerance=2))
    assert "summary" in result
    assert "jaccard_matrix" in result
    assert "synergy_events" in result
    assert "lead_lag" in result
    assert len(result["summary"]) >= 4  # pelt, binseg, cusum, residual + возможно tda
    assert result["jaccard_matrix"].shape[0] >= 3


def test_compare_shock_detectors_empty() -> None:
    """compare_shock_detectors с данными без шоков."""
    from src.changepoints import compare_shock_detectors

    periods = pd.date_range("2020-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({
        "period": periods,
        "mo": "test_mo",
        "y": np.ones(24, dtype=float),
        "prediction": np.ones(24, dtype=float),
        "pelt_shock": np.zeros(24, dtype=bool),
        "binseg_shock": np.zeros(24, dtype=bool),
        "cusum_shock": np.zeros(24, dtype=bool),
        "residual_shock": np.zeros(24, dtype=bool),
        "residual_news_shock": np.zeros(24, dtype=bool),
        "pelt_eligible": np.full(24, True),
        "binseg_eligible": np.full(24, True),
        "cusum_eligible": np.full(24, True),
        "residual_eligible": np.full(24, True),
        "residual_news_eligible": np.full(24, True),
    })
    result = compare_shock_detectors(frame, DetectionConfig(min_history=12))
    assert result["summary"]["n_shocks"].sum() == 0
    assert result["synergy_events"].empty
