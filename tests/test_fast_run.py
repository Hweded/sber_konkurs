from pathlib import Path

import numpy as np
import pandas as pd

from src.fast_run import recompute_ensemble, resolve_default_weights


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mo": ["a", "a"],
            "period": pd.to_datetime(["2024-01-01", "2024-07-01"]),
            "lead_months": [1, 6],
            "target": [100.0, 100.0],
            "pred_prophet": [90.0, 90.0],
            "pred_catboost": [110.0, 110.0],
            "pred_chronos": [100.0, 100.0],
        }
    )


def test_default_weights_use_validated_candidate_file() -> None:
    weights = resolve_default_weights()
    assert weights is not None
    assert weights[1] == (0.563, 0.437, 0.0)
    assert weights[6] == (0.85, 0.15, 0.0)


def test_recompute_ensemble_applies_horizon_specific_candidate_weights() -> None:
    result, _ = recompute_ensemble(_frame(), _frame().drop(columns="target"))
    assert np.isclose(result.loc[0, "pred_regime_aware"], 98.74)
    assert np.isclose(result.loc[1, "pred_regime_aware"], 93.0)
