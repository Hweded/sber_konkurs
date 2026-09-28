from __future__ import annotations

import numpy as np
import pytest

from src.ensemble import EnsembleConfig, OOFBlender
from src.forecasting import metric_table


def test_blender_fits_and_clips_predictions() -> None:
    blender = OOFBlender(EnsembleConfig(default_alpha=0.8))
    alpha = blender.fit_weights(
        [10.0, 20.0, 30.0],
        [11.0, 19.0, 31.0],
        [30.0, 1.0, 50.0],
    )
    assert 0.7 <= alpha <= 1.0
    predicted = blender.predict([10.0, -2.0], [np.nan, -10.0])
    np.testing.assert_allclose(predicted, [10.0, 0.0])


def test_blender_rejects_unimplemented_method() -> None:
    with pytest.raises(ValueError, match="ridge_stacking"):
        OOFBlender(EnsembleConfig(method="ridge_stacking"))


def test_metric_table_contains_ensemble_metrics() -> None:
    import pandas as pd

    frame = pd.DataFrame({
        "fold": [1, 1],
        "target": [10.0, 20.0],
        "pred_prophet": [8.0, 18.0],
        "pred_catboost": [9.0, 19.0],
        "pred_chronos": [11.0, 21.0],
        "pred_ensemble": [10.0, 20.0],
    })
    result = metric_table(frame)
    ensemble = result.loc[(result["fold"] == "pooled") & result["model"].eq("Ensemble (CatBoost + Chronos)")].iloc[0]
    assert ensemble["MAE"] == 0.0
    assert ensemble["RMSE"] == 0.0
    assert ensemble["WAPE"] == 0.0
    assert ensemble["delta_mae_vs_prophet_pct"] == 100.0
    assert ensemble["delta_mae_vs_catboost_pct"] == 100.0
