"""Экономическая валидация детекторов структурных изменений."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.changepoints import (
    ConsensusShockDetector,
    DetectionConfig,
    aggregate_macro_alerts,
    compare_shock_detectors,
    detect_changepoints,
)
from src.tda_engine import TDAConfig, TopologicalAnalyzer
from src.visualize import plot_case_study

LOGGER = logging.getLogger(__name__)
METHODS = ("pelt", "binseg", "cusum", "residual", "residual_news")
VALIDATION_METHODS = ("tda", "pelt", "cusum", "residual", "consensus", "residual_news")
REFERENCE_EVENTS: dict[pd.Timestamp, str] = {
    pd.Timestamp("2023-08-01"): "Внеплановое заседание Банка России: ставка 8,5% → 12,0%",
    pd.Timestamp("2023-12-01"): "Ставка 16,0% и предновогодний фискально-потребительский всплеск",
    pd.Timestamp("2024-07-01"): "Завершение безадресной льготной ипотеки и ставка 18,0%",
    pd.Timestamp("2024-10-01"): "Рекордное повышение ключевой ставки до 21,0%",
}
EXTENDED_REFERENCE_EVENTS: dict[pd.Timestamp, str] = {
    pd.Timestamp("2022-03-01"): "Ажиотажный спрос и перестройка потребительских потоков, март 2022",
    pd.Timestamp("2022-10-01"): "Осенняя перестройка логистических цепочек, октябрь 2022",
    **REFERENCE_EVENTS,
}


def benchmark_offline_algorithms(
    frame: pd.DataFrame,
    config: DetectionConfig,
    *,
    event_catalog: str = "extended",
) -> pd.DataFrame:
    """Compare PELT/Binary Segmentation costs and online CUSUM on one pooled series.

    Pooling is performed by calendar month before detection, which gives a
    stable macro signal and keeps the benchmark affordable for thousands of
    municipal series.  The returned rows include precision, recall, F1 and
    non-negative detection delay for every cost function.
    """
    actual = "target" if "target" in frame.columns else "y"
    prediction = "pred_catboost" if "pred_catboost" in frame.columns else "prediction"
    if not {"period", actual, prediction}.issubset(frame.columns):
        raise ValueError("Для benchmark нужны period, target/y и pred_catboost/prediction")
    pooled = frame.assign(period=pd.to_datetime(frame["period"], errors="raise"))
    pooled = pooled.groupby("period", observed=True)[[actual, prediction]].median().reset_index()
    pooled = pooled.rename(columns={actual: "y", prediction: "prediction"})
    # Это retrospective-бенчмарк; основной детектор остаётся причинным.
    # Все фиксированные penalty выводятся без выбора по event F1.
    import ruptures as rpt

    signal = pooled["y"].to_numpy(dtype=float)
    signal = ((signal - np.nanmean(signal)) / max(float(np.nanstd(signal)), 1e-8)).reshape(-1, 1)
    rows: list[dict[str, Any]] = []
    for algorithm in ("l1", "l2", "rbf"):
        for penalty in (0.5, 1.0, 2.5):
            for method, estimator in (("pelt", rpt.Pelt), ("binseg", rpt.Binseg)):
                breakpoints = (
                    estimator(model=algorithm, min_size=config.min_size, jump=1)
                    .fit(signal)
                    .predict(pen=penalty)[:-1]
                )
                detected = pooled[["period"]].copy()
                detected[f"{method}_shock"] = 0
                detected[f"{method}_eligible"] = True
                for boundary in breakpoints:
                    if 0 < boundary < len(detected):
                        detected.loc[boundary, f"{method}_shock"] = 1
                metrics = validate_against_reference_events(
                    detected,
                    event_catalog=event_catalog,
                    methods=(method,),
                    restrict_to_observed=True,
                )
                for row in metrics.to_dict(orient="records"):
                    row["cost_model"] = algorithm
                    row["penalty"] = penalty
                    row["fit_scope"] = "retrospective_pooled_series"
                    rows.append(row)
    return pd.DataFrame(rows)


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    a, b = left.eq(1), right.eq(1)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def _absolute_signal(values: pd.Series) -> pd.Series:
    """Приводит двусторонний новостной шок к неотрицательной интенсивности."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.abs()


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    """Возвращает конечную корреляцию Пирсона для попарно доступных месяцев."""
    paired = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return 0.0
    value = float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="pearson"))
    return value if np.isfinite(value) else 0.0


def _load_news_features(config: DetectionConfig) -> pd.DataFrame | None:
    """Загружает основной или JSON-кэш новостей и унифицирует его схему."""
    configured = Path(config.news)
    candidates = (configured, configured.with_name("news_features.parquet"))
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        LOGGER.warning("Новостные признаки не найдены: %s", [str(path) for path in candidates])
        return None
    news = pd.read_parquet(source)
    aliases = {
        "telegram_sentiment": "sentiment_index",
        "telegram_volume": "news_volume",
        "telegram_shock_score": "news_shock_score",
    }
    news = news.rename(
        columns={old: new for old, new in aliases.items() if new not in news.columns}
    )
    required = {"period", "news_shock_score"}
    if not required.issubset(news.columns):
        raise ValueError(
            f"Новостной кэш {source} не содержит {sorted(required.difference(news.columns))}"
        )
    columns = ["period", "news_shock_score"]
    columns.extend(
        column for column in ("sentiment_index", "news_volume") if column in news.columns
    )
    result = news.loc[:, columns].copy()
    result["period"] = (
        pd.to_datetime(result["period"], errors="raise").dt.to_period("M").dt.to_timestamp()
    )
    if result["period"].duplicated().any():
        raise ValueError(f"Новостной кэш {source} содержит дубликаты месяцев")
    LOGGER.info("Новостные признаки загружены: %s, месяцев=%d", source, len(result))
    return result


def _monthly_news_validation(
    detected: pd.DataFrame,
    config: DetectionConfig,
) -> dict[str, float | int]:
    """Сопоставляет новостной индекс t-1 с алертами и невязками месяца t."""
    required = {
        "period",
        config.entity_column,
        "news_shock_score",
        "residual",
        "pelt_shock",
        "cusum_shock",
        "residual_shock",
    }
    if not required.issubset(detected.columns):
        return {
            "jaccard": 0.0,
            "correlation": 0.0,
            "correlation_alert_share": 0.0,
            "correlation_residual_magnitude": 0.0,
            "n_comparable": 0,
            "mean_news_lead_months": 0.0,
            "news_validated_events": 0,
        }
    local = detected.loc[~detected[config.entity_column].astype(str).eq("__national__")].copy()
    local["period"] = (
        pd.to_datetime(local["period"], errors="raise").dt.to_period("M").dt.to_timestamp()
    )
    local["absolute_residual"] = pd.to_numeric(local["residual"], errors="coerce").abs()
    shares = (
        local.groupby("period", observed=True)[["pelt_shock", "cusum_shock", "residual_shock"]]
        .mean()
        .max(axis=1)
        .rename("alert_share")
    )
    monthly = (
        local.groupby("period", observed=True)
        .agg(
            residual_magnitude=("absolute_residual", "mean"),
            news_shock_score=("news_shock_score", "median"),
        )
        .join(shares)
        .sort_index()
    )
    # Shift(1) не даёт новостям месяца разлома выдать себя за ранний сигнал.
    monthly["news_lead_score"] = _absolute_signal(monthly["news_shock_score"]).shift(1)
    comparable = monthly.dropna(subset=["news_lead_score", "alert_share", "residual_magnitude"])
    if comparable.empty:
        return {
            "jaccard": 0.0,
            "correlation": 0.0,
            "correlation_alert_share": 0.0,
            "correlation_residual_magnitude": 0.0,
            "n_comparable": 0,
            "mean_news_lead_months": 0.0,
            "news_validated_events": 0,
        }
    news_threshold = float(comparable["news_lead_score"].quantile(0.75))
    news_peaks = comparable["news_lead_score"].ge(news_threshold)
    local_breaks = comparable["alert_share"].ge(config.macro_alert_share)
    union = int((news_peaks | local_breaks).sum())
    validated = int((news_peaks & local_breaks).sum())
    correlation_alerts = _safe_correlation(comparable["news_lead_score"], comparable["alert_share"])
    correlation_residuals = _safe_correlation(
        comparable["news_lead_score"], comparable["residual_magnitude"]
    )
    return {
        "jaccard": float(validated / union) if union else 0.0,
        "correlation": correlation_residuals,
        "correlation_alert_share": correlation_alerts,
        "correlation_residual_magnitude": correlation_residuals,
        "n_comparable": len(comparable),
        "mean_news_lead_months": 1.0 if validated else 0.0,
        "news_validated_events": validated,
    }


def _lag_to_news(frame: pd.DataFrame, method: str, config: DetectionConfig) -> list[int]:
    dates = frame.loc[frame[f"{method}_shock"].eq(1), "period"]
    news = frame.loc[frame.news_alert.eq(1), "period"].tolist() if "news_alert" in frame else []
    lags: list[int] = []
    for date in dates:
        prior = [
            int((date.to_period("M") - value.to_period("M")).n)
            for value in news
            if value <= date and value >= date - pd.offsets.MonthBegin(config.event_tolerance)
        ]
        if prior:
            lags.append(min(prior))
    return lags


def _detected_dates(detected: pd.DataFrame, method: str) -> pd.DatetimeIndex:
    aliases: dict[str, tuple[str, ...]] = {
        "residual": ("residual_macro_shock", "residual_shock", "residuals_shock"),
        "tda": ("tda_macro_shock", "tda_shock"),
    }
    candidates = aliases.get(method, (f"{method}_macro_shock", f"{method}_shock"))
    column = next((name for name in candidates if name in detected.columns), None)
    if column is None:
        return pd.DatetimeIndex([])
    mask = detected[column].fillna(False).astype(bool)
    periods = pd.to_datetime(
        pd.Series(detected.loc[mask, "period"], copy=True), errors="coerce"
    ).dropna()
    month_starts = periods.dt.to_period("M").dt.to_timestamp()
    return pd.DatetimeIndex(month_starts.unique()).sort_values()


def validate_against_reference_events(
    detected: pd.DataFrame,
    *,
    tolerance_days: int = 31,
    methods: tuple[str, ...] = VALIDATION_METHODS,
    event_catalog: str = "legacy",
    restrict_to_observed: bool = False,
) -> pd.DataFrame:
    """Сопоставляет месяцы сигналов и событий взаимно-однозначно в окне ±1 месяц."""
    if tolerance_days < 0:
        raise ValueError("tolerance_days должен быть неотрицательным")
    if "period" not in detected.columns:
        raise ValueError("Для валидации требуется колонка period")
    catalogs = {"legacy": REFERENCE_EVENTS, "extended": EXTENDED_REFERENCE_EVENTS}
    if event_catalog not in catalogs:
        raise ValueError(f"Неизвестный каталог событий: {event_catalog}")
    events = pd.DatetimeIndex(
        pd.Series(list(catalogs[event_catalog])).dt.to_period("M").dt.to_timestamp()
    )
    catalog_count = len(events)
    if restrict_to_observed and not detected.empty:
        observed = pd.to_datetime(detected["period"], utc=True).dt.tz_localize(None)
        events = events[(events >= observed.min()) & (events <= observed.max())]
    rows: list[dict[str, Any]] = []
    for method in methods:
        predictions = _detected_dates(detected, method)
        candidates: list[tuple[int, int, int]] = []
        for prediction_index, prediction in enumerate(predictions):
            for event_index, event in enumerate(events):
                delta = int((prediction - event).days)
                if abs(delta) <= tolerance_days:
                    candidates.append((abs(delta), prediction_index, event_index))
        used_predictions: set[int] = set()
        used_events: set[int] = set()
        leads: list[int] = []
        for _, prediction_index, event_index in sorted(candidates):
            if prediction_index in used_predictions or event_index in used_events:
                continue
            used_predictions.add(prediction_index)
            used_events.add(event_index)
            leads.append(int((predictions[prediction_index] - events[event_index]).days))
        true_positives = len(used_events)
        precision = true_positives / len(predictions) if len(predictions) else 0.0
        recall = true_positives / len(events) if len(events) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "method": method.upper() if method in {"tda", "cusum"} else method.capitalize(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_lead_time_days": float(np.mean(leads)) if leads else np.nan,
                "coverage_pct": recall * 100.0,
                "true_positives": true_positives,
                "reference_events": len(events),
                "detected_events": len(predictions),
                "mean_detection_delay_days": float(np.mean([max(value, 0) for value in leads]))
                if leads
                else np.nan,
                "detection_delay_months": float(
                    np.mean([max(value, 0) for value in leads]) / 30.4375
                )
                if leads
                else np.nan,
                "event_catalog": event_catalog,
                "catalog_events": catalog_count,
                "out_of_range_events": catalog_count - len(events),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "precision",
            "recall",
            "f1",
            "mean_lead_time_days",
            "coverage_pct",
            "true_positives",
            "reference_events",
            "detected_events",
            "mean_detection_delay_days",
            "detection_delay_months",
            "event_catalog",
            "catalog_events",
            "out_of_range_events",
        ],
    )


def export_changepoint_validation(
    detected: pd.DataFrame,
    output: Path = Path("reports/artifacts/changepoint_validation.csv"),
    *,
    event_catalog: str = "legacy",
) -> pd.DataFrame:
    validation = validate_against_reference_events(
        detected, event_catalog=event_catalog, restrict_to_observed=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    LOGGER.info("Экономическая валидация сохранена: %s", output)
    return validation


def _split_contiguous_monthly_runs(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    ordered = frame.copy()
    ordered["period"] = pd.to_datetime(ordered["period"], errors="raise")
    ordered = ordered.sort_values("period", kind="stable")
    if ordered["period"].duplicated().any():
        raise ValueError("OOF-ряд содержит дубликаты периода внутри одного МО")
    month_numbers = ordered["period"].dt.year.mul(12).add(ordered["period"].dt.month)
    run_ids = month_numbers.diff().ne(1).cumsum()
    return [group.reset_index(drop=True) for _, group in ordered.groupby(run_ids, sort=False)]


def _resolve_oof_columns(frame: pd.DataFrame, config: DetectionConfig) -> tuple[str, str]:
    actual_candidates = tuple(dict.fromkeys((config.actual_column, "target", "y")))
    aliases = {
        "catboost_prediction": ("pred_catboost",),
        "ensemble_prediction": ("pred_ensemble",),
        "prophet_prediction": ("pred_prophet",),
        "chronos_prediction": ("pred_chronos",),
    }
    prediction_candidates = tuple(
        dict.fromkeys((config.prediction_column, *aliases.get(config.prediction_column, ())))
    )
    actual = next((column for column in actual_candidates if column in frame), None)
    prediction = next((column for column in prediction_candidates if column in frame), None)
    if actual is None or prediction is None:
        raise ValueError(
            f"OOF не содержит цель/прогноз: actual={actual_candidates}, prediction={prediction_candidates}"
        )
    return actual, prediction


def _load_national_history(config: DetectionConfig) -> pd.DataFrame | None:
    """Строит полный общероссийский ряд; OOF-прогнозы не растягиваются в прошлое."""
    if config.national_history is None:
        return None
    source = Path(config.national_history)
    if not source.exists():
        LOGGER.warning("История для общероссийского детектора не найдена: %s", source)
        return None
    required = {"period", "value", config.national_category_column}
    chunks = pd.read_csv(source, sep=";", encoding="utf-8-sig", chunksize=100_000)
    selected_parts: list[pd.DataFrame] = []
    available: set[str] = set()
    for chunk in chunks:
        if not required.issubset(chunk.columns):
            raise ValueError(
                f"National history не содержит {sorted(required.difference(chunk.columns))}"
            )
        available.update(chunk[config.national_category_column].dropna().astype(str).unique())
        part = chunk.loc[
            chunk[config.national_category_column].eq(config.national_category), ["period", "value"]
        ]
        if not part.empty:
            selected_parts.append(part)
    if not selected_parts:
        raise ValueError(
            f"Категория {config.national_category!r} не найдена; доступны {sorted(available)}"
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["period"] = (
        pd.to_datetime(selected["period"], errors="raise").dt.to_period("M").dt.to_timestamp()
    )
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    national = pd.DataFrame(selected.groupby("period", observed=True)["value"].mean()).reset_index()
    national = (
        national.rename(columns={"value": "y"})
        .sort_values("period", kind="stable")
        .reset_index(drop=True)
    )
    national["prediction"] = (
        national["y"].shift(1).rolling(3, min_periods=1).mean().fillna(national["y"])
    )
    national[config.entity_column] = "__national__"
    return national


def _append_national_detection(output: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    national = _load_national_history(config)
    if national is None or len(national) <= config.min_history:
        return output
    detected = detect_changepoints(national, config)
    try:
        tda = TopologicalAnalyzer(
            TDAConfig(window_size=5, embedding_dimension=3, time_delay=1)
        ).fit_transform(
            national.set_index("period")["y"],
        )
        detected = detected.merge(tda, on="period", how="left", validate="one_to_one")
    except (ImportError, ValueError, RuntimeError):
        LOGGER.warning("TDA общероссийского ряда недоступен", exc_info=True)
    detected[config.entity_column] = "__national__"
    detected["detected_at"] = detected["period"]
    return pd.concat([output, detected], ignore_index=True, sort=False)


def _append_territorial_residual_evidence(
    detected: pd.DataFrame, config: DetectionConfig
) -> pd.DataFrame:
    """Добавляет в fallback причинные residual-алерты всех МО для макроагрегации."""
    if config.national_history is None:
        return detected
    source = Path(config.national_history)
    if not source.exists():
        return detected
    raw = pd.read_csv(source, sep=";", encoding="utf-8-sig")
    required = {"period", "mo", "value", config.national_category_column}
    if not required.issubset(raw.columns):
        return detected
    local = raw.loc[
        raw[config.national_category_column].eq(config.national_category), ["period", "mo", "value"]
    ].copy()
    local["period"] = (
        pd.to_datetime(local["period"], errors="raise").dt.to_period("M").dt.to_timestamp()
    )
    local["y"] = pd.to_numeric(local.pop("value"), errors="raise")
    local = local.sort_values(["mo", "period"], kind="stable")
    local["prediction"] = local.groupby("mo", observed=True)["y"].shift(1)
    local["residual"] = (local["y"] - local["prediction"]).abs()
    local["residual_threshold"] = local.groupby("mo", observed=True)["residual"].transform(
        lambda values: (
            values.shift(1)
            .rolling(config.window, min_periods=config.min_history)
            .quantile(config.residual_quantile)
        ),
    )
    local["residual_eligible"] = local["residual_threshold"].notna()
    local["residual_shock"] = (
        local["residual_eligible"] & local["residual"].ge(local["residual_threshold"])
    ).astype(int)
    local["residual_score"] = local["residual"].div(local["residual_threshold"].clip(lower=1e-8))
    local["detected_at"] = local["period"] + pd.offsets.MonthBegin(1)
    methods = ("pelt", "binseg", "cusum", "residual_news")
    for method in methods:
        local[f"{method}_shock"] = 0
        local[f"{method}_eligible"] = False
        local[f"{method}_score"] = np.nan
    local["news_alert"] = 0
    local["news_eligible"] = False
    local["news_score"] = np.nan
    local["pelt_break_period"] = pd.NaT
    local["binseg_break_period"] = pd.NaT
    return pd.concat([detected, local], ignore_index=True, sort=False)


def _continuous_news_correlations(
    detected: pd.DataFrame,
    news: pd.DataFrame | None,
    config: DetectionConfig,
) -> pd.DataFrame:
    """Попарные корреляции по календарным месяцам, без заполнения отсутствующего OOF."""
    columns = ["method", "other", "correlation", "spearman_correlation", "n_comparable"]
    if news is None:
        return pd.DataFrame(columns=columns)
    monthly = news.set_index("period")["news_shock_score"].sort_index().abs()
    # Кэш уже лагирован по дате публикации; повторный shift(1) создал бы t-2.
    monthly.index = pd.to_datetime(monthly.index).to_period("M").to_timestamp()
    local = detected.loc[~detected[config.entity_column].astype(str).eq("__national__")]
    residual = local.groupby("period", observed=True)["residual"].apply(
        lambda values: pd.to_numeric(values, errors="coerce").abs().mean()
    )
    national = _load_national_history(config)
    targets: dict[str, pd.Series] = {"mean_abs_catboost_residual_oof": residual}
    if national is not None:
        values = national.set_index("period")["y"].sort_index()
        targets["national_spending_growth_mom"] = values.pct_change(fill_method=None)
    rows: list[dict[str, Any]] = []
    for name, series in targets.items():
        paired = (
            pd.concat([monthly.rename("news"), series.rename("outcome")], axis=1)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )
        variable_pair = (
            len(paired) > 1 and paired.news.nunique() > 1 and paired.outcome.nunique() > 1
        )
        rows.append(
            {
                "method": "news_lead_t-1",
                "other": name,
                "correlation": float(paired.news.corr(paired.outcome)) if variable_pair else np.nan,
                "spearman_correlation": float(paired.news.corr(paired.outcome, method="spearman"))
                if variable_pair
                else np.nan,
                "n_comparable": len(paired),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _add_consensus(detected: pd.DataFrame) -> pd.DataFrame:
    """Голосование по макросигналам на датах фактического обнаружения."""
    months = detected["period"].drop_duplicates().sort_values()
    monthly = pd.DataFrame({"period": months})
    for method in ("cusum", "pelt", "tda"):
        key = f"{method}_macro_shock"
        active = (
            detected.loc[detected[key].fillna(False).astype(bool), "period"]
            if key in detected
            else pd.Series(dtype="datetime64[ns]")
        )
        monthly[f"{method}_shock"] = monthly.period.isin(active)
    # news_alert относится к месяцу, в который лагированный индекс уже доступен.
    news_months = detected.loc[detected["news_alert"].fillna(False).astype(bool), "period"]
    monthly["news_alert"] = monthly.period.isin(news_months)
    voted = ConsensusShockDetector().predict(monthly)
    result = detected.copy()
    result["consensus_macro_shock"] = result.period.isin(
        voted.loc[voted.consensus_shock, "period"],
    )
    return result


def evaluate(
    frame: pd.DataFrame, config: DetectionConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {config.entity_column, config.period_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"OOF не содержит колонки: {sorted(required.difference(frame.columns))}")
    actual_column, prediction_column = _resolve_oof_columns(frame, config)
    records: list[pd.DataFrame] = []
    split_entities = 0
    for entity, group in frame.groupby(config.entity_column, sort=True, observed=True):
        current = group.rename(
            columns={
                prediction_column: "prediction",
                config.period_column: "period",
                actual_column: "y",
            }
        )
        runs = _split_contiguous_monthly_runs(current)
        split_entities += int(len(runs) > 1)
        for run in runs:
            detected = detect_changepoints(run, config)
            detected["detected_at"] = (
                pd.to_datetime(detected[config.availability_column])
                if config.availability_column in detected
                else detected.period + pd.offsets.MonthBegin(1)
            )
            detected[config.entity_column] = entity
            records.append(detected)
    if not records:
        raise ValueError("Нет временных рядов для анализа")
    output = pd.concat(records, ignore_index=True)
    national_mask = output[config.entity_column].astype(str).eq("__national__")
    if national_mask.any() and "tda_wasserstein_dist" not in output.columns:
        try:
            national_group = output.loc[national_mask].sort_values("period", kind="stable")
            tda = TopologicalAnalyzer(
                TDAConfig(window_size=5, embedding_dimension=3, time_delay=1)
            ).fit_transform(
                national_group.set_index("period")["y"],
            )
            tda_by_period = tda.set_index("period")
            for column in ("tda_entropy", "tda_wasserstein_dist"):
                output.loc[national_mask, column] = output.loc[national_mask, "period"].map(
                    tda_by_period[column]
                )
        except (ImportError, ValueError, RuntimeError):
            LOGGER.warning("TDA OOF national-ряда недоступен", exc_info=True)
    output = _append_national_detection(output, config)
    comparison = compare_shock_detectors(output, config)
    compared = comparison.get("data")
    if isinstance(compared, pd.DataFrame):
        output = compared
    output = aggregate_macro_alerts(output, config)
    output = _add_consensus(output)
    news_validation = _monthly_news_validation(output, config)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for other in METHODS:
            if method < other:
                eligible = output[method + "_eligible"] & output[other + "_eligible"]
                left, right = (
                    output.loc[eligible, method + "_shock"],
                    output.loc[eligible, other + "_shock"],
                )
                rows.append(
                    {
                        "method": method,
                        "other": other,
                        "jaccard": _jaccard(left, right) if len(left) else np.nan,
                        "correlation": float(left.corr(right))
                        if left.nunique() > 1 and right.nunique() > 1
                        else np.nan,
                        "n_comparable": len(left),
                    }
                )
        lags: list[int] = []
        for _, group in output.groupby(config.entity_column, observed=True):
            lags.extend(_lag_to_news(group, method, config))
        summary_row: dict[str, Any] = {
            "method": method,
            "mean_news_lead_months": float(np.mean(lags)) if lags else 0.0,
            "news_validated_events": len(lags),
            "alerts": int(output[method + "_shock"].sum()),
            "eligible": int(output[method + "_eligible"].sum()),
        }
        if method == "residual_news":
            summary_row.update(news_validation)
            summary_row["other"] = "news_lead_t-1"
        rows.append(summary_row)
    summary: dict[str, Any] = {
        "entities": int(output[config.entity_column].nunique()),
        "rows": len(output),
        "methods": list(METHODS),
        "split_entities": split_entities,
        "macro_alert_share": config.macro_alert_share,
        "interpretation": "Метрики согласованности описательные; экономическая проверка использует фиксированные события ДКП.",
    }
    return output, pd.DataFrame(rows), summary


def run_benchmark(config: DetectionConfig) -> dict[str, Any]:
    source = Path(config.predictions)
    original_config = config
    if source.exists():
        frame = pd.read_parquet(source)
        if "lead_months" in frame:
            frame = frame.loc[frame["lead_months"].eq(1)].copy()
    else:
        # Без OOF считаем честный rolling baseline, модели заново не обучаем.
        national = _load_national_history(config)
        if national is None:
            raise FileNotFoundError(f"Нет OOF-файла и national history: {source}")
        frame = national.rename(columns={"y": "target", "prediction": "pred_catboost"})
        frame[config.availability_column] = frame["period"] + pd.offsets.MonthBegin(1)
        config = config.model_copy(update={"national_history": None})
        LOGGER.warning("OOF-файл %s отсутствует: детекция выполняется на national history", source)
    if config.period_column != "period":
        frame = frame.rename(columns={config.period_column: "period"})
        config = config.model_copy(update={"period_column": "period"})
    news = _load_news_features(config)
    if news is not None:
        frame["period"] = pd.to_datetime(frame["period"])
        frame = frame.merge(news, on="period", how="left", validate="many_to_one")
    detected, metrics, summary = evaluate(frame, config)
    summary["event_catalog"] = config.event_catalog
    summary["reference_events"] = len(
        EXTENDED_REFERENCE_EVENTS if config.event_catalog == "extended" else REFERENCE_EVENTS
    )
    summary["detection_metrics"] = ["precision", "recall", "f1", "detection_delay_months"]
    if not source.exists():
        detected = _append_territorial_residual_evidence(detected, original_config)
        detected = aggregate_macro_alerts(detected, original_config)
        config = original_config
        residual_row = metrics["method"].eq("residual") & metrics["other"].isna()
        metrics.loc[residual_row, "alerts"] = int(detected["residual_shock"].sum())
        metrics.loc[residual_row, "eligible"] = int(detected["residual_eligible"].sum())
    detected = _add_consensus(detected)
    metrics = pd.concat(
        [metrics, _continuous_news_correlations(detected, news, config)],
        ignore_index=True,
    )
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detected.to_parquet(output, index=False)
    metrics.to_csv(output.with_suffix(".metrics.csv"), index=False, encoding="utf-8")
    try:
        national = _load_national_history(config)
        benchmark_frame = national if national is not None else frame
        algorithm_metrics = benchmark_offline_algorithms(
            benchmark_frame, config, event_catalog=config.event_catalog
        )
        algorithm_path = output.parent / "artifacts" / "changepoint_algorithm_benchmark.csv"
        algorithm_path.parent.mkdir(parents=True, exist_ok=True)
        algorithm_metrics.to_csv(algorithm_path, index=False, encoding="utf-8")
    except (ValueError, RuntimeError) as error:
        LOGGER.warning("Сравнение cost-функций CPD пропущено: %s", error)
    validation_path = output.parent / "artifacts" / "changepoint_validation.csv"
    export_changepoint_validation(detected, validation_path, event_catalog=config.event_catalog)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for number, (entity, group) in enumerate(detected.groupby(config.entity_column, observed=True)):
        if number >= config.max_figures:
            break
        plot_case_study(group, str(entity), Path(config.figures))
    return summary


def main() -> int:
    run_benchmark(DetectionConfig())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
