"""Adapters for measured time-series foundation model experiments.

Chronos-Bolt is the configured foundation model. MOIRAI is intentionally not
listed until a pinned uni2ts checkpoint and adapter are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class FoundationForecast:
    values: np.ndarray
    model: str
    status: str
    note: str = ""


def foundation_benchmark_table(
    metrics: Any, horizons: Sequence[int] = (1, 3, 6, 12)
) -> dict[str, Any]:
    """Create an auditable Chronos benchmark payload from measured rows."""
    rows = []
    for horizon in horizons:
        measured = (
            metrics.loc[
                metrics["model"].eq("chronos")
                & metrics["horizon"].eq(horizon)
                & metrics["fold"].eq("pooled")
                & metrics["scope"].eq("exact")
            ]
            if hasattr(metrics, "loc")
            else []
        )
        record = {"horizon": horizon, "model": "chronos-bolt", "status": "not_evaluated"}
        if len(measured):
            row = measured.iloc[0]
            if str(row.get("status")) == "measured":
                record.update(
                    {
                        "status": "measured",
                        "MAE": float(row["MAE"]),
                        "R2": float(row["R2"]),
                        "WAPE": float(row["WAPE"]),
                    }
                )
        rows.append(record)
    return {"models": ["chronos-bolt"], "rows": rows}


def forecast_moirai(
    contexts: Sequence[Sequence[float]],
    prediction_length: int,
    *,
    device: str = "cpu",
) -> FoundationForecast:
    """Forecast with MOIRAI when available, otherwise return a labelled baseline.

    The adapter deliberately does not download weights.  This keeps the
    default competition run reproducible offline while providing a single
    integration point for a local ``uni2ts`` checkpoint.
    """
    if prediction_length < 1:
        raise ValueError("prediction_length должен быть положительным")
    if not contexts:
        return FoundationForecast(np.empty((0, prediction_length)), "moirai", "empty")
    try:
        import uni2ts  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        values = np.asarray(
            [
                np.repeat(float(np.asarray(context, dtype=float)[-1]), prediction_length)
                for context in contexts
            ],
            dtype=float,
        )
        return FoundationForecast(
            values,
            "moirai",
            "fallback_last_value",
            "uni2ts is not installed; no MOIRAI score was claimed",
        )
    # API uni2ts менялся; стабильный контракт задаётся wrapper-адаптером вызывающей стороны.
    raise RuntimeError(
        "uni2ts is installed but no configured MOIRAI checkpoint adapter is available; "
        "pass a local predictor before reporting measured scores"
    )
