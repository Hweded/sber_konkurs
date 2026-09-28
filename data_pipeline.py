"""Leakage-safe preprocessing for municipal consumer-spending time series.

The public API intentionally works with a long dataframe containing one row per
``entity/category/date``.  ``fit`` learns only train-window statistics;
``transform`` regularises the supplied frame and imputes values causally.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import PchipInterpolator

try:  # statsmodels is an optional runtime dependency for the diagnostics.
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller
except Exception:  # pragma: no cover - exercised only in minimal installs.
    STL = acf = adfuller = None


@dataclass(frozen=True)
class ImputationConfig:
    """Parameters that control the causal imputation policy."""

    short_gap: int = 2
    medium_gap: int = 6
    seasonal_period: int | None = None
    critical_missing_fraction: float = 0.40
    periodic_run_fraction: float = 0.20
    min_history_for_seasonality: int = 4
    drop_critical: bool = False
    outlier_z_threshold: float = 8.0
    winsorize_outliers: bool = True
    structural_shock_dates: tuple[str, ...] = ()
    seasonal_decay: float = 0.92


@dataclass
class _SeriesStats:
    median: float = np.nan
    scale: float = np.nan
    seasonal: dict[int, float] = field(default_factory=dict)
    trend: float = 0.0
    count: int = 0


class TimeSeriesPreprocessor:
    """Regularise, audit and impute temporal panels without future leakage.

    Parameters
    ----------
    config:
        Optional mapping, path to YAML, or :class:`ImputationConfig`.
    date_column, value_column:
        Names of timestamp and target columns in the long dataframe.
    entity_columns:
        Columns identifying a municipality/category series.  Pass ``("id",)``
        for a single-series panel.  Missing entity columns raise a clear error.
    frequency:
        A pandas offset such as ``"MS"`` or ``"W-MON"``.
    hierarchy_columns:
        Optional parent/cluster columns used by external hierarchical fallback.
    """

    def __init__(
        self,
        config: ImputationConfig | Mapping[str, Any] | str | Path | None = None,
        *,
        date_column: str = "date",
        value_column: str = "value",
        entity_columns: Sequence[str] = ("municipality", "category"),
        frequency: str = "MS",
        freq: str | None = None,
        hierarchy_columns: Sequence[str] = (),
    ) -> None:
        self.date_column = date_column
        self.value_column = value_column
        self.entity_columns = tuple(entity_columns)
        self.frequency = freq or frequency
        self.hierarchy_columns = tuple(hierarchy_columns)
        raw_config: Mapping[str, Any] = {}
        if isinstance(config, Mapping):
            raw_config = config
        elif isinstance(config, (str, Path)):
            with Path(config).open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
            if isinstance(loaded, Mapping):
                raw_config = loaded
        if raw_config:
            self.date_column = str(raw_config.get("date_column", self.date_column))
            self.value_column = str(raw_config.get("value_column", self.value_column))
            if "entity_columns" in raw_config:
                self.entity_columns = tuple(raw_config["entity_columns"])
            if "hierarchy_columns" in raw_config:
                self.hierarchy_columns = tuple(raw_config["hierarchy_columns"])
            if freq is None and "frequency" in raw_config:
                self.frequency = str(raw_config["frequency"])
        self.config = self._read_config(config)
        if self.config.short_gap < 0 or self.config.medium_gap < self.config.short_gap:
            raise ValueError("medium_gap должен быть не меньше short_gap")
        if not 0 < self.config.critical_missing_fraction <= 1:
            raise ValueError("critical_missing_fraction должен быть в (0, 1]")
        self._stats: dict[tuple[Any, ...], _SeriesStats] = {}
        self._hierarchy_stats: dict[tuple[Any, ...], _SeriesStats] = {}
        self._global_stats = _SeriesStats()
        self._fit_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self._excluded_groups: set[tuple[Any, ...]] = set()
        self._quality_log = pd.DataFrame()
        self._fitted = False
        self.report_: dict[str, Any] = {}

    @staticmethod
    def _read_config(config: ImputationConfig | Mapping[str, Any] | str | Path | None) -> ImputationConfig:
        if config is None:
            return ImputationConfig()
        if isinstance(config, ImputationConfig):
            return config
        if isinstance(config, (str, Path)):
            with Path(config).open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream) or {}
        data = dict(config)
        # Accept the natural YAML nesting used by configs/data_config.yaml.
        nested = data.get("imputation", data)
        aliases = {"critical_missing_share": "critical_missing_fraction", "max_short_gap": "short_gap"}
        nested = {aliases.get(k, k): v for k, v in dict(nested).items()}
        quality = data.get("quality", {})
        if "critical_missing_fraction" not in nested and "critical_missing_fraction" in quality:
            nested["critical_missing_fraction"] = quality["critical_missing_fraction"]
        if str(quality.get("action", "")).lower() in {"exclude", "drop"}:
            nested["drop_critical"] = True
        known = {k: v for k, v in nested.items() if k in ImputationConfig.__dataclass_fields__}
        if "structural_shock_dates" in known:
            known["structural_shock_dates"] = tuple(str(x) for x in known["structural_shock_dates"])
        return ImputationConfig(**known)

    def _groups(self, frame: pd.DataFrame) -> tuple[str, ...]:
        missing = [c for c in self.entity_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Отсутствуют идентификаторы рядов: {missing}")
        return self.entity_columns

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame должен быть pandas.DataFrame")
        required = [self.date_column, self.value_column, *self.entity_columns]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
        data = frame.copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column], errors="coerce")
        if data[self.date_column].isna().any():
            raise ValueError("Дата содержит некорректные значения")
        data[self.value_column] = pd.to_numeric(data[self.value_column], errors="coerce")
        data = data.sort_values([*self.entity_columns, self.date_column], kind="stable")
        return data.reset_index(drop=True)

    def _align_dates(self, dates: pd.Series) -> pd.Series:
        if self.frequency.upper().startswith("W"):
            return dates - pd.to_timedelta(dates.dt.weekday, unit="D")
        if self.frequency.upper().startswith("M"):
            return dates.dt.to_period("M").dt.to_timestamp()
        return dates.dt.floor(self.frequency)

    def _season_key(self, date: pd.Timestamp) -> int:
        if self.frequency.upper().startswith("W"):
            return int(date.isocalendar().week)
        if self.frequency.upper().startswith("M"):
            return int(date.month)
        return int(date.dayofyear)

    def _period_length(self) -> int:
        if self.config.seasonal_period:
            return self.config.seasonal_period
        return 52 if self.frequency.upper().startswith("W") else 12

    @staticmethod
    def _key(value: Any) -> tuple[Any, ...]:
        return value if isinstance(value, tuple) else (value,)

    def _mark_technical_zeros(self, data: pd.DataFrame) -> pd.DataFrame:
        """Turn an all-category, one-period zero outage into a missing value."""
        data = data.copy()
        data["_technical_zero"] = False
        candidates = [c for c in self.entity_columns if c.lower() not in {"category", "cat", "metric"}]
        if not candidates:
            candidates = list(self.entity_columns)
        grouped = data.groupby([*candidates, self.date_column], dropna=False, sort=False)
        zeros = grouped[self.value_column].transform(lambda s: bool(s.notna().all() and s.eq(0).all()))
        tmp = data.assign(_zero_group=zeros)
        for _, idx in tmp.groupby(candidates, dropna=False, sort=False).groups.items():
            part = tmp.loc[idx].sort_values(self.date_column)
            z = part["_zero_group"].to_numpy(bool)
            nonzero = part[self.value_column].fillna(0).to_numpy() != 0
            # A one-sided neighbour can be a genuine series start/end.  Require
            # a return from zero on both sides before calling it a feed outage.
            neighbor = np.r_[False, nonzero[:-1]] & np.r_[nonzero[1:], False]
            chosen = part.index[z & neighbor & ~part[self.date_column].map(self._is_shock).to_numpy(bool)]
            data.loc[chosen, "_technical_zero"] = True
        data.loc[data["_technical_zero"], self.value_column] = np.nan
        return data

    def _regularize(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = self._prepare(frame)
        data[self.date_column] = self._align_dates(data[self.date_column])
        groups = list(data.groupby(list(self.entity_columns), dropna=False, sort=False))
        if not groups:
            return data
        result: list[pd.DataFrame] = []
        start = data[self.date_column].min()
        end = data[self.date_column].max()
        for key, part in groups:
            key_tuple = self._key(key)
            dates = pd.date_range(start, end, freq=self.frequency)
            base = pd.DataFrame({self.date_column: dates})
            for col, val in zip(self.entity_columns, key_tuple):
                base[col] = val
            for col in self.hierarchy_columns:
                if col in part.columns and col not in base:
                    non_null = part[col].dropna()
                    base[col] = non_null.iloc[0] if len(non_null) else np.nan
            values = part.groupby(self.date_column, sort=False)[self.value_column].agg(
                lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan
            )
            base[self.value_column] = base[self.date_column].map(values)
            if "_technical_zero" in part:
                tech = part.groupby(self.date_column, sort=False)["_technical_zero"].any()
                base["_technical_zero"] = base[self.date_column].map(tech).eq(True)
            result.append(base)
        return pd.concat(result, ignore_index=True).sort_values([*self.entity_columns, self.date_column], kind="stable")

    def audit_missingness(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return one auditable missingness row per municipality/category.

        ``classification`` distinguishes MCAR-like isolated gaps, leading or
        trailing structural tails, and periodic reporting failures.  The audit
        uses only the supplied frame and never imputes values.
        """
        data = self._prepare(frame)
        data[self.date_column] = self._align_dates(data[self.date_column])
        global_start = data[self.date_column].min()
        global_end = data[self.date_column].max()
        rows: list[dict[str, Any]] = []
        for key, part in data.groupby(list(self.entity_columns), dropna=False, sort=False):
            key_tuple = self._key(key)
            expected = pd.date_range(global_start, global_end, freq=self.frequency)
            observed = part.groupby(self.date_column)[self.value_column].agg(
                lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan
            ).reindex(expected)
            missing = observed.isna().to_numpy()
            runs: list[int] = []
            current = 0
            for flag in missing:
                if flag:
                    current += 1
                else:
                    if current:
                        runs.append(current)
                    current = 0
            if current:
                runs.append(current)
            lead = int(np.argmax(~missing)) if (~missing).any() else len(missing)
            trail = int(np.argmax((~missing)[::-1])) if (~missing).any() else len(missing)
            internal = int(missing.sum() - lead - trail)
            max_run = max(runs, default=0)
            fraction = float(missing.mean()) if len(missing) else 1.0
            if not (~missing).any():
                classification = "all_missing"
            elif fraction > self.config.critical_missing_fraction:
                classification = "critical"
            elif lead >= 1 and internal == 0:
                classification = "structural_leading"
            elif trail >= 1 and internal == 0:
                classification = "structural_trailing"
            else:
                missing_positions = np.flatnonzero(missing)
                periodic_spacing = (len(missing_positions) >= 3 and
                                    np.std(np.diff(missing_positions)) <= 1.0)
                periodic_runs = len(runs) >= 2 and max_run > 1
                elif_periodic = ((periodic_runs or periodic_spacing) and
                                 max_run / max(1, int(missing.sum())) >= self.config.periodic_run_fraction)
                if elif_periodic:
                    classification = "periodic"
                elif missing.any():
                    classification = "mcar"
                else:
                    classification = "complete"
            row = {col: val for col, val in zip(self.entity_columns, key_tuple)}
            row.update({"expected_periods": len(expected), "observed_periods": int((~missing).sum()),
                        "missing_periods": int(missing.sum()), "missing_fraction": fraction,
                        "leading_missing": lead, "trailing_missing": trail, "max_missing_run": max_run,
                        "classification": classification,
                        "quality_status": "exclude" if fraction > self.config.critical_missing_fraction else "keep"})
            rows.append(row)
        return pd.DataFrame(rows)

    def fit(self, frame: pd.DataFrame) -> "TimeSeriesPreprocessor":
        """Learn medians, seasonal indices and trends from a train window only."""
        data = self._mark_technical_zeros(self._prepare(frame))
        data[self.date_column] = self._align_dates(data[self.date_column])
        self._fit_bounds = (data[self.date_column].min(), data[self.date_column].max())
        self._quality_log = self.audit_missingness(data)
        self._excluded_groups = {
            tuple(row[c] for c in self.entity_columns)
            for _, row in self._quality_log[self._quality_log.quality_status.eq("exclude")].iterrows()
        }
        values = data[self.value_column].replace([np.inf, -np.inf], np.nan).dropna()
        self._global_stats = _SeriesStats(median=float(values.median()) if len(values) else np.nan,
                                          scale=float(values.std(ddof=0)) if len(values) > 1 else np.nan,
                                          count=int(len(values)))
        for key, part in data.groupby(list(self.entity_columns), dropna=False, sort=False):
            k = self._key(key)
            clean = part[self.value_column].replace([np.inf, -np.inf], np.nan)
            finite = clean.dropna()
            median = float(finite.median()) if len(finite) else self._global_stats.median
            scale = float(finite.std(ddof=0)) if len(finite) > 1 else np.nan
            seasonal: dict[int, float] = {}
            if len(finite) >= self.config.min_history_for_seasonality and np.isfinite(median) and median != 0:
                ratios = finite / median
                season_keys = part.loc[ratios.index, self.date_column].map(self._season_key)
                seasonal = {int(s): float(ratios[season_keys.eq(s)].median())
                             for s in season_keys.dropna().unique()
                             if np.isfinite(ratios[season_keys.eq(s)].median())}
            history = part[[self.date_column, self.value_column]].dropna().tail(8)
            trend = 0.0
            if len(history) > 1:
                trend = float(np.polyfit(np.arange(len(history)), history[self.value_column].to_numpy(float), 1)[0])
            self._stats[k] = _SeriesStats(median=median, scale=scale, seasonal=seasonal,
                                          trend=trend, count=int(len(finite)))
        # Parent/cluster statistics are a train-only spatial fallback.
        available_hierarchy = tuple(c for c in self.hierarchy_columns if c in data.columns)
        for key, part in (data.groupby(list(available_hierarchy), dropna=False, sort=False)
                          if available_hierarchy else []):
            finite = part[self.value_column].replace([np.inf, -np.inf], np.nan).dropna()
            median = float(finite.median()) if len(finite) else self._global_stats.median
            seasonal: dict[int, float] = {}
            if len(finite) >= self.config.min_history_for_seasonality and np.isfinite(median) and median != 0:
                ratios = finite / median
                season_keys = part.loc[ratios.index, self.date_column].map(self._season_key)
                seasonal = {int(s): float(ratios[season_keys.eq(s)].median())
                            for s in season_keys.dropna().unique()
                            if np.isfinite(ratios[season_keys.eq(s)].median())}
            self._hierarchy_stats[self._key(key)] = _SeriesStats(
                median=median, seasonal=seasonal, count=int(len(finite)),
                scale=float(finite.std(ddof=0)) if len(finite) > 1 else np.nan,
            )
        self._fitted = True
        return self

    def _is_shock(self, date: pd.Timestamp) -> bool:
        return date.strftime("%Y-%m-%d") in set(self.config.structural_shock_dates) or date.strftime("%Y-%m") in set(self.config.structural_shock_dates)

    def _causal_prediction(self, history: list[float], dates: list[pd.Timestamp], date: pd.Timestamp,
                           stats: _SeriesStats, gap_length: int) -> tuple[float, str]:
        if not history:
            return (stats.median, "hierarchical_seasonal" if np.isfinite(stats.median) else "unresolved")
        factor = stats.seasonal.get(self._season_key(date), 1.0)
        prev_factor = stats.seasonal.get(self._season_key(dates[-1]), 1.0)
        seasonal_last = history[-1] * factor / prev_factor if prev_factor else history[-1]
        if gap_length <= self.config.short_gap and len(history) < 2:
            return (float(max(seasonal_last, 0.0)), "seasonal_ffill")
        if gap_length <= self.config.short_gap and len(history) >= 2:
            n = min(5, len(history))
            x = np.arange(len(history) - n, len(history), dtype=float)
            y = np.asarray(history[-n:], dtype=float)
            try:
                pred = float(PchipInterpolator(x, y, extrapolate=True)(len(history)))
            except (ValueError, TypeError):
                pred = seasonal_last
            pred = 0.5 * pred + 0.5 * seasonal_last
            return (float(max(pred, 0.0)), "pchip_causal")
        # Causal STL forecast: decompose only pre-gap history, extrapolate a
        # robust local trend, and reapply the fitted seasonal factor.
        n = min(12, len(history))
        deseason = np.array([history[-n + i] / stats.seasonal.get(self._season_key(dates[-n + i]), 1.0)
                             for i in range(n)], dtype=float)
        if STL is not None and gap_length <= self.config.medium_gap and len(history) >= max(8, 2 * self._period_length()):
            try:
                period = min(self._period_length(), len(history) // 2)
                decomposition = STL(np.asarray(history, dtype=float), period=period, robust=True).fit()
                deseason = np.asarray(decomposition.trend[-n:], dtype=float)
            except (ValueError, np.linalg.LinAlgError):
                pass
        slope = float(np.polyfit(np.arange(n), deseason, 1)[0]) if n > 1 else stats.trend
        pred = (deseason[-1] + slope) * factor
        if not np.isfinite(pred):
            pred = stats.median
        return (float(max(pred, 0.0)), "stl_causal" if gap_length <= self.config.medium_gap else "hierarchical_seasonal")

    def _clean_outliers(self, values: pd.Series, dates: pd.Series, stats: _SeriesStats) -> tuple[pd.Series, pd.Series]:
        methods = pd.Series("observed", index=values.index, dtype="string")
        if not self.config.winsorize_outliers:
            return values, methods
        vals = values.copy()
        for i in range(len(vals)):
            if pd.isna(vals.iloc[i]) or self._is_shock(dates.iloc[i]) or i < 3:
                continue
            past = vals.iloc[max(0, i - 12):i].dropna()
            if len(past) < 3:
                continue
            med = float(past.median())
            mad = float(np.median(np.abs(past.to_numpy() - med)))
            scale = max(1.4826 * mad, 1e-12)
            if abs(float(vals.iloc[i]) - med) / scale > self.config.outlier_z_threshold:
                vals.iloc[i] = med
                methods.iloc[i] = "winsorized_outlier"
        return vals, methods

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Regularise and causally impute a frame using frozen ``fit`` state."""
        if not self._fitted:
            raise RuntimeError("TimeSeriesPreprocessor не обучен: вызовите fit()")
        data = self._mark_technical_zeros(self._prepare(frame))
        regular = self._regularize(data)
        output: list[pd.DataFrame] = []
        for key, part in regular.groupby(list(self.entity_columns), dropna=False, sort=False):
            k = self._key(key)
            if k in self._excluded_groups and self.config.drop_critical:
                continue
            part = part.sort_values(self.date_column, kind="stable").copy()
            stats = self._stats.get(k, self._global_stats)
            if self.hierarchy_columns and all(c in part.columns for c in self.hierarchy_columns):
                parent_key = tuple(part[c].iloc[0] for c in self.hierarchy_columns)
                parent_stats = self._hierarchy_stats.get(parent_key)
                if parent_stats is not None and parent_stats.count > stats.count:
                    stats = parent_stats
            original = part[self.value_column].copy()
            values, methods = self._clean_outliers(part[self.value_column], part[self.date_column], stats)
            history: list[float] = []
            history_dates: list[pd.Timestamp] = []
            for idx in range(len(values)):
                date = part[self.date_column].iloc[idx]
                if pd.notna(values.iloc[idx]):
                    history.append(float(values.iloc[idx])); history_dates.append(date)
                    continue
                run_end = idx
                while run_end < len(values) and pd.isna(values.iloc[run_end]):
                    run_end += 1
                gap_length = run_end - idx
                pred, method = self._causal_prediction(history, history_dates, date, stats, gap_length)
                values.iloc[idx] = pred
                methods.iloc[idx] = "technical_zero_interpolation" if bool(part.get("_technical_zero", pd.Series(False, index=part.index)).iloc[idx]) else method
                if np.isfinite(pred):
                    history.append(pred); history_dates.append(date)
            part[self.value_column] = values
            part["is_imputed"] = original.isna()
            part["imputation_method"] = methods.astype(str)
            part["is_structural_shock"] = part[self.date_column].map(self._is_shock)
            part["unresolved"] = part[self.value_column].isna()
            output.append(part)
        result = pd.concat(output, ignore_index=True) if output else regular.iloc[0:0].copy()
        self.report_ = {"rows": int(len(result)), "imputed": int(result.get("is_imputed", pd.Series(dtype=bool)).sum()),
                        "unresolved": int(result.get("unresolved", pd.Series(dtype=bool)).sum()),
                        "excluded_groups": len(self._excluded_groups)}
        return result.sort_values([*self.entity_columns, self.date_column], kind="stable").reset_index(drop=True)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    @property
    def quality_log_(self) -> pd.DataFrame:
        """Copy of the fit-window quality filter log."""
        return self._quality_log.copy()

    def save_quality_log(self, path: str | Path) -> None:
        """Persist the fit-window quality decisions as JSON for audit trails."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._quality_log.to_json(destination, orient="records", force_ascii=False, date_format="iso")

    def merge_exogenous(
        self,
        frame: pd.DataFrame,
        exogenous: pd.DataFrame,
        *,
        exogenous_date_column: str | None = None,
        lags: Sequence[int] = (1, 2),
        news_columns: Sequence[str] = (),
        news_decay: float = 0.85,
    ) -> pd.DataFrame:
        """Align external data causally and add lagged macro/news features.

        External observations are matched with ``merge_asof(direction='backward')``.
        News columns use recursive decay between events; missing news is never
        replaced by an artificial zero.  Lags are applied after alignment.
        """
        target = frame.copy()
        ext = exogenous.copy()
        e_date = exogenous_date_column or self.date_column
        if self.date_column not in target or e_date not in ext:
            raise ValueError("В обоих кадрах нужна колонка даты")
        target[self.date_column] = self._align_dates(pd.to_datetime(target[self.date_column]))
        ext[e_date] = self._align_dates(pd.to_datetime(ext[e_date]))
        release_column = next((c for c in ("released_at", "available_at", "published_at") if c in ext.columns), None)
        if release_column:
            ext[release_column] = pd.to_datetime(ext[release_column], errors="coerce")
            ext = ext.loc[ext[release_column].notna()].copy()
        right_on = release_column or e_date
        ext = ext.sort_values([*self.entity_columns, right_on] if all(c in ext for c in self.entity_columns) else [right_on])
        news = set(news_columns) or {c for c in ext.columns if str(c).lower().startswith("news") or "sentiment" in str(c).lower()}
        excluded_external = set(self.entity_columns) | {e_date}
        if release_column:
            excluded_external.add(release_column)
        ext_cols = [c for c in ext.columns if c not in excluded_external]
        by = [c for c in self.entity_columns if c in target.columns and c in ext.columns]
        merged = pd.merge_asof(target.sort_values(self.date_column), ext.sort_values(right_on),
                               left_on=self.date_column, right_on=right_on, by=by or None, direction="backward")
        if self.date_column not in merged.columns and f"{self.date_column}_x" in merged.columns:
            merged = merged.rename(columns={f"{self.date_column}_x": self.date_column})
            duplicate_date = f"{self.date_column}_y"
            if duplicate_date in merged.columns:
                merged = merged.drop(columns=duplicate_date)
        new_cols = [c for c in ext_cols if c in merged.columns]
        # Apply news decay on the target calendar.  This keeps the impulse
        # causal even when the source has no row for an otherwise observed
        # target period.
        if news:
            news = news.intersection(new_cols)
            target_groups = merged.groupby(by, dropna=False, sort=False) if by else [((), merged)]
            for group_key, target_part in target_groups:
                target_index = target_part.index
                source_part = ext
                if by:
                    key_tuple = self._key(group_key)
                    for column, value in zip(by, key_tuple):
                        source_part = source_part[source_part[column].eq(value)]
                source_part = source_part.sort_values(e_date)
                for col in news:
                    events = source_part[[e_date, col]].dropna(subset=[col]).groupby(e_date)[col].last()
                    state = np.nan
                    decayed: list[float] = []
                    for date in target_part[self.date_column]:
                        if date in events.index:
                            state = float(events.loc[date])
                        elif pd.notna(state):
                            state *= news_decay
                        decayed.append(state)
                    merged.loc[target_index, col] = decayed
        for lag in lags:
            if lag < 1:
                raise ValueError("Лаги экзогенных переменных должны быть положительными")
            for col in new_cols:
                name = f"{col}_lag_{lag}"
                if by:
                    merged[name] = merged.groupby(by, dropna=False, sort=False)[col].shift(lag)
                else:
                    merged[name] = merged[col].shift(lag)
        return merged.sort_index()

    def validate_imputation(self, frame: pd.DataFrame, *, mask_fraction: float = 0.05,
                            random_state: int = 42) -> dict[str, Any]:
        """Mask observed points, refit on the masked train panel and score recovery."""
        if not 0 < mask_fraction < 1:
            raise ValueError("mask_fraction должен быть между 0 и 1")
        original = self._prepare(frame)
        original[self.date_column] = self._align_dates(original[self.date_column])
        observed = original[self.value_column].notna()
        candidates = original.index[observed].to_numpy()
        if len(candidates) < 2:
            return {"masked_points": 0, "MAE": np.nan, "WAPE": np.nan, "status": "insufficient_observations"}
        n = max(1, int(round(len(candidates) * mask_fraction)))
        rng = np.random.default_rng(random_state)
        masked_idx = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        masked = original.copy(); truth = original.loc[masked_idx, self.value_column].astype(float)
        masked.loc[masked_idx, self.value_column] = np.nan
        # Validation must score the deliberately masked group even when the
        # production quality policy would exclude it from the output.
        probe_config = replace(self.config, drop_critical=False)
        probe = TimeSeriesPreprocessor(probe_config, date_column=self.date_column, value_column=self.value_column,
                                       entity_columns=self.entity_columns, frequency=self.frequency,
                                       hierarchy_columns=self.hierarchy_columns).fit(masked)
        restored = probe.transform(masked)
        key_cols = [*self.entity_columns, self.date_column]
        pred = restored.set_index(key_cols).loc[original.loc[masked_idx, key_cols].set_index(key_cols).index, self.value_column]
        errors = pred.to_numpy(float) - truth.to_numpy(float)
        actual = truth.to_numpy(float)
        variance_ratio = float(np.var(pred) / np.var(actual)) if np.var(actual) > 0 else np.nan
        acf_error = np.nan
        if acf is not None and len(actual) > 3:
            acf_error = float(np.nanmean(np.abs(acf(np.nan_to_num(pred), nlags=min(5, len(pred) - 1)) -
                                          acf(np.nan_to_num(actual), nlags=min(5, len(actual) - 1)))))
        adf_before = adf_after = np.nan
        if adfuller is not None and len(original[self.value_column].dropna()) >= 8:
            try:
                adf_before = float(adfuller(original[self.value_column].dropna(), autolag="AIC")[1])
                adf_after = float(adfuller(restored[self.value_column].dropna(), autolag="AIC")[1])
            except (ValueError, np.linalg.LinAlgError):
                pass
        return {"masked_points": int(len(actual)), "MAE": float(np.mean(np.abs(errors))),
                "WAPE": float(np.sum(np.abs(errors)) / max(np.sum(np.abs(actual)), 1e-12)),
                "variance_ratio": variance_ratio, "acf_mean_abs_error": acf_error,
                "ADF_pvalue_before": adf_before, "ADF_pvalue_after": adf_after, "status": "ok"}


def load_data_config(path: str | Path = "configs/data_config.yaml") -> dict[str, Any]:
    """Load the reproducible YAML configuration used by the pipeline."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
