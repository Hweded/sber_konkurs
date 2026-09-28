"""Контракт причинного голосования и переключения ансамбля."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.changepoints import ConsensusShockDetector
from src.forecasting import RegimeAwareEnsemble, metric_table


def test_consensus_quorum_and_calendar_lag() -> None:
    months = pd.date_range("2023-01-01", periods=5, freq="MS")
    frame = pd.DataFrame({
        "period": months,
        "cusum_shock": [False, False, False, True, False],
        "tda_shock": [True, False, True, False, False],
        "pelt_shock": [False, False, True, False, False],
        "news_alert": [False, True, False, False, True],
    })
    voted = ConsensusShockDetector().predict(frame)
    assert voted.consensus_shock.tolist() == [False, True, True, True, False]
    short = frame.drop(index=0).reset_index(drop=True)
    assert not bool(ConsensusShockDetector().predict(short).consensus_shock.iloc[0])


def test_gating_uses_only_known_signals() -> None:
    frame = pd.DataFrame({
        "mo": ["a"] * 4,
        "period": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "pred_catboost": [100.0] * 4,
        "pred_chronos": [80.0] * 4,
        "cusum_shock": [False, True, False, False],
        "news_shock_score": [0.0, 0.0, 3.0, 0.0],
    })
    pred, alpha = RegimeAwareEnsemble().predict(frame)
    assert alpha.tolist() == [0.4, 0.4, 0.0, 0.4]
    np.testing.assert_allclose(pred, [92.0, 92.0, 100.0, 92.0])
    frame.loc[3, "cusum_shock"] = True
    changed, _ = RegimeAwareEnsemble().predict(frame)
    np.testing.assert_allclose(changed, pred)
    oof = frame.assign(target=95.0, fold=1, pred_regime_aware=pred)
    assert "RegimeAware (CatBoost + Chronos)" in metric_table(oof)["model"].tolist()
