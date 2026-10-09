"""Причинная детекция: сигнал датируется моментом обнаружения, не задним числом."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    predictions: str = "reports/predictions_oof.parquet"
    news: str = "data/processed/news_monthly_features.parquet"
    output: str = "reports/detected_shocks.parquet"
    figures: str = "reports/figures"
    period_column: str = "period"
    entity_column: str = "mo"
    actual_column: str = "y"
    prediction_column: str = "catboost_prediction"
    availability_column: str = "actual_available_at"
    window: int = Field(default=24, ge=4)
    min_history: int = Field(default=4, ge=3)
    model: Literal["rbf", "l1", "l2"] = "rbf"
    pen: float = Field(default=2.5, gt=0)
    min_size: int = Field(default=2, ge=2)
    threshold: float = Field(default=2.0, gt=0)
    drift: float = Field(default=0.5, ge=0)
    residual_k: float = Field(default=1.8, gt=0)
    residual_quantile: float = Field(default=0.90, gt=0.5, lt=1.0)
    tda_quantile: float = Field(default=0.85, gt=0.5, lt=1.0)
    macro_alert_share: float = Field(default=0.085, gt=0.0, le=1.0)
    national_history: str | None = (
        "datasets/potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij_ru_1764079373653.csv"
    )
    national_category_column: str = "category_15"
    national_category: str = "Все категории"
    news_k: float = Field(default=2.0, gt=0)
    negative_sentiment: float = Field(default=-0.3, ge=-1, le=1)
    news_lookback: int = Field(default=2, ge=0)
    event_tolerance: int = Field(default=3, ge=0)
    max_figures: int = Field(default=5, ge=0)
    event_catalog: Literal["legacy", "extended"] = "extended"


def validate_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Нормализованный вход: period, y, prediction, optional news columns."""
    if not {"period", "y", "prediction"}.issubset(frame) or frame.empty:
        raise ValueError("Ожидается непустой ряд period/y/prediction")
    data = frame.copy().sort_values("period").reset_index(drop=True)
    data["period"] = (
        pd.to_datetime(data["period"], errors="raise", utc=True)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if data.period.isna().any() or data.period.duplicated().any():
        raise ValueError("Пропуски или дубликаты дат")
    expected = pd.date_range(data.period.iloc[0], data.period.iloc[-1], freq="MS")
    if not pd.DatetimeIndex(data.period).equals(expected):
        raise ValueError("Нужен непрерывный месячный ряд с датами начала месяца")
    if not np.isfinite(data[["y", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("Цель и OOF-прогноз должны быть конечными")
    return data


def _scale(values: np.ndarray) -> float:
    return max(float(np.std(values, ddof=1)), 1e-8)


def detect_changepoints(frame: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Считает причинные алерты, скоры и маски доступности детекторов."""
    data = validate_series(frame)
    try:
        import ruptures as rpt
    except ImportError as exc:
        raise RuntimeError(
            "Для PELT/BinSeg установите ruptures; подмена алгоритмов запрещена"
        ) from exc
    n = len(data)
    result = data.copy()
    residual = data.y.to_numpy(dtype=float) - data.prediction.to_numpy(dtype=float)
    levels = data.y.to_numpy(dtype=float)
    methods = ("pelt", "binseg", "cusum", "residual", "residual_news")
    for method in methods:
        result[f"{method}_shock"] = 0
        result[f"{method}_score"] = np.nan
        result[f"{method}_eligible"] = False
    for method in ("pelt", "binseg"):
        result[f"{method}_break_period"] = pd.NaT
    news = data.get("news_shock_score", pd.Series(np.nan, index=data.index))
    past = news.shift(1).rolling(config.window, min_periods=config.min_history)
    news_z = (news - past.mean()) / past.std().clip(lower=1e-8)
    sentiment = data.get("sentiment_index", pd.Series(np.nan, index=data.index))
    result["news_alert"] = (
        news_z.ge(config.news_k) | sentiment.le(config.negative_sentiment)
    ).astype(int)
    result["news_eligible"] = news_z.notna() | sentiment.notna()
    result["news_score"] = news_z
    result["residual"] = residual
    seen: dict[str, set[pd.Timestamp]] = {"pelt": set(), "binseg": set()}
    positive = negative = 0.0
    if n <= config.min_history:
        return result
    center = float(np.mean(levels[: config.min_history]))
    scale = _scale(levels[: config.min_history])
    absolute_residual = np.abs(residual)
    for i in range(config.min_history, n):
        start = max(0, i - config.window + 1)
        history = levels[max(0, i - config.window) : i]
        z = (levels[i] - center) / scale
        positive = max(0.0, positive + z - config.drift)
        negative = max(0.0, negative - z - config.drift)
        score = max(positive, negative) / config.threshold
        result.loc[i, ["cusum_score", "cusum_eligible", "cusum_shock"]] = [
            score,
            True,
            int(score >= 1),
        ]
        if score >= 1:
            positive = negative = 0.0
        residual_history = absolute_residual[max(0, i - config.window) : i]
        residual_mean = float(np.mean(residual_history))
        residual_sigma = _scale(residual_history)
        sigma_barrier = residual_mean + config.residual_k * residual_sigma
        quantile_barrier = float(np.quantile(residual_history, config.residual_quantile))
        barrier = min(sigma_barrier, quantile_barrier)
        barrier = max(barrier, 1e-8)
        ratio = absolute_residual[i] / barrier
        result.loc[i, ["residual_score", "residual_eligible", "residual_shock"]] = [
            ratio,
            True,
            int(ratio >= 1),
        ]
        recent = result.iloc[max(0, i - config.news_lookback) : i + 1]
        known = bool(recent.news_eligible.any())
        supported = bool(recent.news_alert.any())
        result.loc[i, ["residual_news_score", "residual_news_eligible", "residual_news_shock"]] = [
            ratio if known else np.nan,
            known,
            int(ratio > 1 and supported),
        ]
        segment = (levels[start : i + 1] - np.mean(history)) / _scale(history)
        if len(segment) < 2 * config.min_size:
            continue
        breaks_by_method: dict[str, list[int]] = {}
        # Голый BIC на коротком OOF зануляет PELT, поэтому штраф зажат сверху.
        adaptive_penalty = min(config.pen, max(0.5, 0.35 * float(np.log(len(segment)))))
        for name, algorithm in (("pelt", rpt.Pelt), ("binseg", rpt.Binseg)):
            fitted = algorithm(model=config.model, min_size=config.min_size, jump=1).fit(
                segment.reshape(-1, 1)
            )
            breaks_by_method[name] = fitted.predict(pen=adaptive_penalty)[:-1]
        for name, breaks in breaks_by_method.items():
            result.loc[i, f"{name}_eligible"] = True
            result.loc[i, f"{name}_score"] = 0.0
            for boundary in breaks:
                date = pd.Timestamp(data.period.iloc[start + boundary])
                if date in seen[name] or len(segment) - boundary > config.min_size:
                    continue
                seen[name].add(date)
                contrast = abs(float(segment[:boundary].mean() - segment[boundary:].mean()))
                result.loc[i, [f"{name}_shock", f"{name}_score", f"{name}_break_period"]] = [
                    1,
                    contrast,
                    date,
                ]
    return result


class ConsensusShockDetector:
    """Кворум CUSUM или TDA+PELT или TDA прошлого месяца+новости."""

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"period", "cusum_shock", "tda_shock", "pelt_shock", "news_alert"}
        if not required.issubset(frame):
            raise ValueError(f"Нет входов консенсуса: {sorted(required - set(frame))}")
        data = frame.sort_values("period", kind="stable").copy()
        if data["period"].duplicated().any():
            raise ValueError("Консенсус ожидает один агрегированный сигнал на месяц")
        dates = pd.to_datetime(data["period"], errors="raise")
        tda = data["tda_shock"].fillna(False).astype(bool)
        previous = (
            pd.Series(tda.to_numpy(), index=dates)
            .reindex(
                dates - pd.offsets.MonthBegin(1),
                fill_value=False,
            )
            .to_numpy(dtype=bool)
        )
        news = data["news_alert"].fillna(False).astype(bool)
        cusum = data["cusum_shock"].fillna(False).astype(bool)
        pelt = data["pelt_shock"].fillna(False).astype(bool)
        data["consensus_shock"] = cusum | (tda & pelt) | (previous & news)
        return data.reindex(frame.index)


def aggregate_macro_alerts(frame: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Агрегирует локальные сигналы в события страны по доле МО или national-ряду."""
    if not {"period", config.entity_column}.issubset(frame.columns):
        raise ValueError("Для макроагрегации требуются period и колонка территории")
    data = frame.copy()
    data["period"] = (
        pd.to_datetime(data["period"], errors="raise").dt.to_period("M").dt.to_timestamp()
    )
    methods = tuple(
        column.removesuffix("_shock")
        for column in data.columns
        if column.endswith("_shock") and not column.endswith("_macro_shock")
    )
    national_mask = data[config.entity_column].astype(str).eq("__national__")
    for method in methods:
        shock_column = f"{method}_shock"
        eligible_column = f"{method}_eligible"
        macro_column = f"{method}_macro_shock"
        local = data.loc[
            ~national_mask,
            ["period", shock_column] + ([eligible_column] if eligible_column in data else []),
        ].copy()
        if eligible_column in local:
            eligible = local[eligible_column].eq(True)
            local["eligible_count"] = eligible.astype(int)
            local["alert_count"] = (local[shock_column].eq(True) & eligible).astype(int)
        else:
            local["eligible_count"] = 1
            local["alert_count"] = local[shock_column].fillna(False).astype(bool).astype(int)
        monthly = local.groupby("period", observed=True)[["alert_count", "eligible_count"]].sum()
        monthly_share = (
            monthly["alert_count"].div(monthly["eligible_count"].replace(0, np.nan)).fillna(0.0)
        )
        national_periods = set(data.loc[national_mask & data[shock_column].eq(True), "period"])
        candidates = sorted(
            set(monthly_share.index[monthly_share.ge(config.macro_alert_share)]) | national_periods
        )
        # Режем длинное CUSUM-плато, иначе один эпизод съедает следующее решение ЦБ.
        macro_periods: set[pd.Timestamp] = set()
        clusters: list[list[pd.Timestamp]] = []
        for candidate in candidates:
            timestamp = pd.Timestamp(candidate)
            continues = bool(
                clusters
                and (timestamp.to_period("M") - clusters[-1][-1].to_period("M")).n <= 1
                and len(clusters[-1]) < 2
            )
            if continues:
                clusters[-1].append(timestamp)
            else:
                clusters.append([timestamp])
        for cluster in clusters:
            best = max(
                cluster,
                key=lambda period: (
                    float(monthly_share.get(period, 0.0)) + float(period in national_periods),
                    -period.value,
                ),
            )
            macro_periods.add(best)
        data[macro_column] = data["period"].isin(macro_periods)
    return data


def compare_shock_detectors(
    frame: pd.DataFrame,
    config: DetectionConfig,
) -> dict[str, Any]:
    """Сравнивает детекторы и собирает TDA-новостные совпадения."""
    data = frame.copy()
    if "period" not in data or "mo" not in data:
        raise ValueError("Требуются колонки period и mo")

    data["period"] = pd.to_datetime(data["period"], errors="raise")
    methods: list[str] = [
        col.replace("_shock", "")
        for col in data.columns
        if col.endswith("_shock") and not col.startswith("tda_news_synergy")
    ]
    if "tda" in methods:
        methods.remove("tda")
        methods.append("tda")

    if "tda_wasserstein_dist" in data.columns:
        data["tda_shock"] = False
        data["tda_eligible"] = False
        for _, group in data.groupby("mo", sort=False, observed=True):
            ordered = group.sort_values("period", kind="stable")
            wass = ordered["tda_wasserstein_dist"].to_numpy(dtype=float)
            idx = ordered.index
            for i in range(config.min_history, len(wass)):
                history = wass[max(0, i - config.window) : i]
                finite = np.isfinite(history)
                if finite.sum() < 3 or not np.isfinite(wass[i]):
                    continue
                threshold = float(np.quantile(history[finite], config.tda_quantile))
                data.loc[idx[i], "tda_eligible"] = True
                data.loc[idx[i], "tda_shock"] = bool(wass[i] >= threshold)
        if "tda" not in methods:
            methods.append("tda")

    if "tda_shock" in data.columns and any(
        c in data.columns for c in ("news_shock_score", "news_alert")
    ):
        data["tda_news_synergy_shock"] = False
        data["tda_news_synergy_eligible"] = False
        tolerance = config.event_tolerance
        has_news = data.get("pelt_shock", pd.Series(False, index=data.index)) | data.get(
            "residual_news_shock", pd.Series(False, index=data.index)
        )
        if "news_alert" in data:
            has_news = has_news | data["news_alert"].astype(bool)

        for mo, group in data.groupby("mo", sort=False, observed=True):
            idx = group.index
            tda_shocks = group["tda_shock"].to_numpy(dtype=bool)
            news_shocks = has_news.loc[idx].to_numpy(dtype=bool)
            for i in range(len(tda_shocks)):
                if not tda_shocks[i]:
                    continue
                window_start = max(0, i - tolerance)
                window_end = min(len(news_shocks), i + tolerance + 1)
                if news_shocks[window_start:window_end].any():
                    data.loc[idx[i], "tda_news_synergy_eligible"] = True
                    data.loc[idx[i], "tda_news_synergy_shock"] = True
        methods.append("tda_news_synergy")

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        shock_col = f"{method}_shock"
        eligible_col = f"{method}_eligible"
        if shock_col not in data.columns:
            continue
        shocks = data[shock_col].to_numpy(dtype=bool)
        n_shocks = int(shocks.sum())
        n_eligible = (
            int(data[eligible_col].to_numpy(dtype=bool).sum())
            if eligible_col in data.columns
            else len(data)
        )
        summary_rows.append(
            {
                "method": method,
                "n_shocks": n_shocks,
                "shock_rate": n_shocks / max(n_eligible, 1),
                "n_eligible": n_eligible,
                "mean_gap_months": _mean_shock_gap(data, shock_col),
                "n_entities": int(data.loc[shocks, "mo"].nunique()) if n_shocks else 0,
            }
        )

    summary = pd.DataFrame(summary_rows)

    jaccard_data: dict[str, dict[str, float]] = {}
    for m1 in methods:
        col1 = f"{m1}_shock"
        if col1 not in data.columns:
            continue
        jaccard_data[m1] = {}
        s1 = set(data.loc[data[col1].astype(bool), "period"])
        for m2 in methods:
            col2 = f"{m2}_shock"
            if col2 not in data.columns:
                jaccard_data[m1][m2] = 0.0
                continue
            s2 = set(data.loc[data[col2].astype(bool), "period"])
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard_data[m1][m2] = intersection / max(union, 1)
    jaccard_matrix = pd.DataFrame(jaccard_data)

    synergy_col = "tda_news_synergy_shock"
    synergy_events = (
        data.loc[data.get(synergy_col, pd.Series(False, index=data.index)).astype(bool)]
        if synergy_col in data.columns
        else pd.DataFrame()
    )

    lead_lag: dict[str, Any] = {}
    if (
        "tda_shock" in data.columns
        and "news_alert" in data.columns
        and data["tda_shock"].any()
        and data["news_alert"].any()
    ):
        tda_dates = data.loc[data["tda_shock"].astype(bool), "period"]
        news_dates = data.loc[data["news_alert"].astype(bool), "period"]
        diffs: list[int] = []
        for td in tda_dates:
            nearest_news = news_dates.to_numpy()
            if len(nearest_news) == 0:
                continue
            delta = (td - pd.Timestamp(nearest_news.min())).days
            diffs.append(int(delta))
        if diffs:
            lead_lag = {
                "mean_diff_days": float(np.mean(diffs)),
                "median_diff_days": float(np.median(diffs)),
                "n_pairs": len(diffs),
                "interpretation": (
                    "TDA опережает новости"
                    if np.mean(diffs) < -config.event_tolerance
                    else "Новости опережают TDA"
                    if np.mean(diffs) > config.event_tolerance
                    else "Одновременно"
                ),
            }

    LOGGER.info(
        "Сравнение детекторов: %d методов, %d шоков всего",
        len(methods),
        int(
            data[[f"{m}_shock" for m in methods if f"{m}_shock" in data.columns]].any(axis=1).sum()
        ),
    )

    return {
        "data": data,
        "summary": summary,
        "jaccard_matrix": jaccard_matrix,
        "synergy_events": synergy_events,
        "lead_lag": lead_lag,
    }


def _mean_shock_gap(data: pd.DataFrame, shock_col: str) -> float:
    """Средний интервал между шоками (в месяцах)."""
    shocks = data.loc[data[shock_col].astype(bool), "period"]
    if len(shocks) < 2:
        return float("nan")
    gaps = shocks.diff().dropna()
    return float(gaps.dt.days.mean() / 30.4375)
