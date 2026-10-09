"""Memory-safe helpers for retraining 2 094 municipal series on a small-RAM host.

The competition host is memory bound (14.7 GiB total, a large part taken by the
OS, browsers and containers).  A from-scratch run must therefore never hold two
copies of a panel, must stream per-batch results to disk, and must release
native/Stan/torch buffers explicitly.  Everything in this module is optional
infrastructure: it degrades to a no-op when ``psutil`` is unavailable.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:  # psutil is a hard dependency in requirements.txt but stay import-safe.
    import psutil
except ImportError:  # pragma: no cover - exercised only in minimal envs
    psutil = None  # type: ignore[assignment]

_MIB = 1024.0 * 1024.0

#: Columns that must keep full precision: the target drives every metric and the
#: submission gate, so downcasting it would silently move MAE.
PRECISION_COLUMNS = frozenset({"y", "target", "pred", "fold", "lead_months"})


def process_handle() -> Any | None:
    """Return the current-process psutil handle, or ``None`` when unavailable."""
    if psutil is None:
        return None
    try:
        return psutil.Process()
    except Exception:  # pragma: no cover - defensive
        return None


def rss_mib() -> float | None:
    """Resident set size of this process in MiB."""
    handle = process_handle()
    if handle is None:
        return None
    try:
        return float(handle.memory_info().rss) / _MIB
    except Exception:  # pragma: no cover - defensive
        return None


def available_mib() -> float | None:
    """System-wide available memory in MiB."""
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().available) / _MIB
    except Exception:  # pragma: no cover - defensive
        return None


def memory_percent() -> float | None:
    """System-wide memory utilisation in percent."""
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().percent)
    except Exception:  # pragma: no cover - defensive
        return None


def free() -> None:
    """Force a garbage-collection cycle.

    Kept as a function so call sites read as an explicit intent rather than a
    bare ``gc.collect()`` that a later cleanup pass might drop.
    """
    gc.collect()


def empty_torch_cache() -> None:
    """Release the CUDA caching allocator when a GPU is present (no-op on CPU)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a project dependency
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class MemoryMonitor:
    """Samples RSS so a run can report its real peak instead of an estimate."""

    label: str = "run"
    peak_rss_mib: float = 0.0
    peak_percent: float = 0.0
    samples: int = 0
    _baseline_rss_mib: float | None = field(default=None, repr=False)

    def sample(self) -> "MemoryMonitor":
        current = rss_mib()
        if current is not None:
            if self._baseline_rss_mib is None:
                self._baseline_rss_mib = current
            self.peak_rss_mib = max(self.peak_rss_mib, current)
            self.samples += 1
        percent = memory_percent()
        if percent is not None:
            self.peak_percent = max(self.peak_percent, percent)
        return self

    def log(self, tag: str = "") -> None:
        """Sample once and log the current footprint at INFO level."""
        self.sample()
        current = rss_mib()
        avail = available_mib()
        LOGGER.info(
            "[memory] %s%s rss=%s avail=%s peak=%s%%",
            self.label,
            f" {tag}" if tag else "",
            f"{current:.0f}MiB" if current is not None else "n/a",
            f"{avail:.0f}MiB" if avail is not None else "n/a",
            f"{self.peak_percent:.0f}" if self.samples else "n/a",
        )

    def describe(self) -> str:
        if not self.samples:
            return "psutil недоступен: пиковое потребление RAM не измерено"
        return (
            f"пиковое RSS процесса {self.peak_rss_mib:.0f} MiB; "
            f"пиковая загрузка памяти системы {self.peak_percent:.0f}%"
        )


def downcast_frame(
    frame: pd.DataFrame,
    *,
    protect: Sequence[str] = tuple(PRECISION_COLUMNS),
    categorical: Sequence[str] = ("mo",),
) -> pd.DataFrame:
    """Return a memory-lean copy: float32 features, small ints, categorical IDs.

    ``protect`` columns keep ``float64``.  ``mo`` becomes ``category`` because it
    is the only repeated string key in the panel and pandas stores it as a
    dictionary index, which is what every ``groupby(..., observed=True)`` call
    in the pipeline already expects.
    """
    protected = set(protect)
    result = frame.copy()
    for column in result.columns:
        if column in protected:
            continue
        dtype = result[column].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype):
            continue
        if pd.api.types.is_float_dtype(dtype):
            if dtype != np.float32:
                result[column] = result[column].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype):
            if dtype == np.int64:
                values = result[column]
                finite = values.dropna()
                if finite.empty:
                    result[column] = values.astype(np.int32)
                    continue
                low, high = int(finite.min()), int(finite.max())
                if -32_768 <= low and high <= 32_767:
                    result[column] = values.astype(np.int16)
                elif -2_147_483_648 <= low and high <= 2_147_483_647:
                    result[column] = values.astype(np.int32)
    for column in categorical:
        if column in result.columns and not isinstance(result[column].dtype, pd.CategoricalDtype):
            result[column] = result[column].astype("category")
    return result


def frame_footprint_mib(frame: pd.DataFrame) -> float:
    """Deep memory footprint of a frame in MiB (for before/after logging)."""
    try:
        return float(frame.memory_usage(index=True, deep=True).sum()) / _MIB
    except Exception:  # pragma: no cover - defensive
        return float("nan")


def spill_frame(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    """Persist an intermediate batch so it can leave RAM immediately."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_parquet(path, index=False)
    return path


__all__ = [
    "MemoryMonitor",
    "PRECISION_COLUMNS",
    "available_mib",
    "downcast_frame",
    "empty_torch_cache",
    "frame_footprint_mib",
    "free",
    "memory_percent",
    "rss_mib",
    "spill_frame",
]
