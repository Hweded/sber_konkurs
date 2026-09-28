"""Проверки frozen-origin оценки: никакой подмены горизонта фолда."""
from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting import horizon_metric_table


def test_one_step_does_not_claim_long_horizon() -> None:
    frame = pd.DataFrame({
        "fold": [1, 1], "mo": ["a", "a"],
        "period": pd.to_datetime(["2024-02-01", "2024-03-01"]),
        "target": [10.0, 20.0],
        "pred_prophet": [12.0, 22.0], "pred_catboost": [11.0, 21.0],
        "pred_chronos": [13.0, 23.0], "pred_ensemble": [11.0, 21.0],
    })
    table = horizon_metric_table(frame)
    pooled = table.loc[table["fold"].eq("pooled") & table["model"].eq("catboost")]
    assert pooled.loc[pooled["horizon"].eq(1) & pooled["scope"].eq("exact"), "MAE"].iloc[0] == 1.0
    assert pooled.loc[pooled["horizon"].eq(3), "status"].eq("not_evaluated").all()
    assert pooled.loc[pooled["horizon"].eq(12), "n"].eq(0).all()


def test_frozen_origin_exact_and_complete_cumulative() -> None:
    frame = pd.DataFrame({
        "fold": [1, 1, 1], "mo": ["a"] * 3,
        "origin": pd.to_datetime(["2024-01-01"] * 3),
        "period": pd.to_datetime(["2024-02-01", "2024-03-01", "2024-04-01"]),
        "target": [10.0, 20.0, 30.0], "pred_catboost": [9.0, 18.0, 27.0],
    })
    table = horizon_metric_table(frame, (1, 3, 6))
    values = table.loc[table["fold"].eq("pooled") & table["model"].eq("catboost")]
    assert values.loc[values["horizon"].eq(3) & values["scope"].eq("exact"), "MAE"].iloc[0] == 3.0
    assert values.loc[values["horizon"].eq(3) & values["scope"].eq("cumulative"), "MAE"].iloc[0] == 2.0
    assert values.loc[values["horizon"].eq(6) & values["scope"].eq("cumulative"), "status"].iloc[0] == "not_evaluated"


def test_reject_inconsistent_lead() -> None:
    frame = pd.DataFrame({
        "fold": [1], "mo": ["a"], "origin": pd.to_datetime(["2024-01-01"]),
        "period": pd.to_datetime(["2024-04-01"]), "lead_months": [1], "target": [2.0],
    })
    with pytest.raises(ValueError, match="origin"):
        horizon_metric_table(frame)
