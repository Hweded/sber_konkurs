"""Long forecasts cannot consume actual observations after the origin."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.forecasting import recursive_catboost_path, _direct_validation_config
from src.validation import ValidationConfig


class LagIncrementModel:
    def predict(self, values):
        return values[:, 0] + 1.0


def test_recursive_path_is_invariant_to_future_targets():
    history = pd.DataFrame({"mo": ["a"] * 5, "period": pd.date_range("2024-01-01", periods=5, freq="MS"),
                            "y": [10., 20., 30., 999., 999.]})
    origin = pd.Timestamp("2024-04-01")
    periods = pd.date_range(origin, periods=2, freq="MS")
    imputer = SimpleImputer(keep_empty_features=True).fit([[10.]])
    baseline = recursive_catboost_path(LagIncrementModel(), imputer, history, origin, periods, ["y_lag_1"])
    history.loc[history.period.ge(origin), "y"] = -100000.
    changed = recursive_catboost_path(LagIncrementModel(), imputer, history, origin, periods, ["y_lag_1"])
    pd.testing.assert_frame_equal(baseline, changed)
    np.testing.assert_array_equal(baseline.pred_catboost, [31., 32.])


def test_direct_purge_excludes_unreleased_target_month():
    config = ValidationConfig(feature_columns=("x",))
    assert _direct_validation_config(config, 6).gap >= 6
