"""Причинные месячные признаки; никаких глобальных обучаемых преобразований."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_config import DataConfig, FeatureConfig
from src.data_loader import load_data, month_start
from src.validation import ValidationConfig
from src.nlp_features import (
    JSON_TO_NEWS_COLUMNS,
    NEWS_COLUMNS,
    TELEGRAM_NEWS_COLUMNS,
    build_news_features,
    build_telegram_news_features,
    news_coverage,
)
from src.rosstat_loader import merge_rosstat
from src.tda_engine import TDAConfig, build_tda_features

LOGGER = logging.getLogger(__name__)


def regularize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Вставляет пропущенные месяцы внутри каждого МО, не заполняя значения."""
    if not {"period", "mo", "y"}.issubset(frame) or frame.empty:
        raise ValueError("Нужна непустая панель [period, mo, y]")
    data = frame.copy()
    data["period"] = month_start(data["period"])
    if data["mo"].isna().any() or data.duplicated(["mo", "period"]).any():
        raise ValueError("Пустые МО или повторяющиеся месяцы")
    data["y"] = pd.to_numeric(data["y"], errors="raise")
    if np.isinf(data["y"].to_numpy(dtype=float)).any():
        raise ValueError("Бесконечный таргет")
    parts: list[pd.DataFrame] = []
    for mo, group in data.groupby("mo", sort=True, observed=True):
        calendar = pd.date_range(group["period"].min(), group["period"].max(), freq="MS")
        part = group.set_index("period").reindex(calendar)
        part.index.name = "period"
        part["mo"] = mo
        parts.append(part.reset_index())
    result = pd.concat(parts, ignore_index=True)
    LOGGER.info("Календаризация: %d -> %d строк, %d МО", len(frame), len(result), result["mo"].nunique())
    return result


def generate_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Строит причинные лаги, темпы роста и rolling-признаки."""
    data = regularize_panel(frame)
    month = data["period"].dt.month
    data["month"] = month
    data["quarter"] = data["period"].dt.quarter
    data["sin_month"] = np.sin(2.0 * np.pi * (month - 1) / 12)
    data["cos_month"] = np.cos(2.0 * np.pi * (month - 1) / 12)
    grouped = data.groupby("mo", sort=False, observed=True)["y"]
    lags = {lag: grouped.shift(lag) for lag in sorted(set(config.lags) | {1, 2, 13})}
    for lag in config.lags:
        data[f"y_lag_{lag}"] = lags[lag]
    for name, denominator in (("growth_mom", 2), ("growth_yoy", 13)):
        data[name] = lags[1].div(lags[denominator].where(lags[denominator].ne(0))).sub(1).replace([np.inf, -np.inf], np.nan)
    shifted = lags[1].groupby(data["mo"], sort=False, observed=True)
    for window in config.rolling_windows:
        rolling = shifted.rolling(window=window, min_periods=window)
        for statistic in config.rolling_statistics:
            values = rolling.agg(statistic).reset_index(level=0, drop=True)
            data[f"y_rolling_{statistic}_{window}"] = values.reindex(data.index)
    # NOTE: текущий месяц не участвует в оценке локальной волатильности.
    expanding = grouped.expanding(min_periods=2)
    past_mean = expanding.mean().reset_index(level=0, drop=True).groupby(data["mo"], observed=True).shift(1)
    past_std = expanding.std().reset_index(level=0, drop=True).groupby(data["mo"], observed=True).shift(1)
    data["spending_volatility"] = past_std.div(past_mean.abs().where(past_mean.ne(0)))
    age = data.groupby("mo", sort=False, observed=True).cumcount()
    result = data.loc[age.ge(config.warmup_months) & data["y"].notna()].copy()
    if result.empty:
        raise ValueError("После warmup и удаления отсутствующих целей не осталось строк")
    result = result.sort_values(["period", "mo"]).reset_index(drop=True)
    LOGGER.info("Признаки: %d строк, %d МО, %d NaN; удалено %d строк", len(result), result["mo"].nunique(), result.isna().sum().sum(), len(data) - len(result))
    return result


def merge_news_features(frame: pd.DataFrame, news: pd.DataFrame, *, lag_months: int = 1) -> pd.DataFrame:
    """Присоединяет нелагированные NLP-агрегаты со сдвигом по календарю."""
    columns = ["news_sentiment", "news_shock_index"]
    if lag_months < 1 or not {"period", *columns}.issubset(news):
        raise ValueError("Нужны period, news_sentiment, news_shock_index и положительный лаг")
    if set(columns).intersection(frame.columns):
        raise ValueError("Новостные признаки уже присутствуют")
    prepared = news[["period", *columns]].copy()
    prepared["period"] = month_start(prepared["period"])
    if prepared["period"].duplicated().any():
        raise ValueError("Новости должны быть предварительно агрегированы по месяцу")
    for column in columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    if np.isinf(prepared[columns].to_numpy(dtype=float)).any():
        raise ValueError("Бесконечные NLP-признаки")
    # NOTE: переносим календарную метку, чтобы не перескочить через пустой месяц.
    prepared["period"] += pd.offsets.MonthBegin(lag_months)
    result = frame.copy()
    result["period"] = month_start(result["period"])
    return result.merge(prepared, on="period", how="left", validate="many_to_one")


def build_dataset(
    directory: Path,
    data_config: DataConfig,
    feature_config: FeatureConfig,
    *,
    news: pd.DataFrame | None = None,
    tda_config: TDAConfig | None = None,
    skip_news: bool = False,
    skip_tda: bool = False,
    skip_telegram: bool = False,
) -> pd.DataFrame:
    """Собирает причинную месячную матрицу без глобальной импутации."""
    loaded = load_data(directory, data_config)
    result = generate_features(loaded, feature_config)
    root = Path(__file__).resolve().parents[1]

    if not skip_news and data_config.nlp.enabled:
        if news is not None:
            result = merge_news_features(result, news, lag_months=feature_config.news_lag_months)
        else:
            nlp = data_config.nlp.model_copy(update={
                "path": root / data_config.nlp.path,
                "cache": root / data_config.nlp.cache,
            })
            first_target = pd.Timestamp(result["period"].min()).to_period("M").to_timestamp()
            json_features = pd.DataFrame(columns=["period", *TELEGRAM_NEWS_COLUMNS])
            if not skip_telegram:
                json_features = build_telegram_news_features(
                    nlp,
                    project_root=root,
                    data_directory=directory,
                )
            json_covers = (
                not json_features.empty
                and pd.Timestamp(json_features["period"].max()) >= first_target
            )
            if json_covers:
                if json_features["period"].duplicated().any():
                    raise ValueError("Дубликаты периодов в JSON NLP-признаках")
                compatible = json_features.rename(columns=JSON_TO_NEWS_COLUMNS)
                result = result.merge(
                    compatible[["period", *NEWS_COLUMNS]],
                    on="period",
                    how="left",
                    validate="many_to_one",
                )
                # NOTE: старые имена нужны сохранённым матрицам и тестам схемы.
                result = result.merge(
                    json_features,
                    on="period",
                    how="left",
                    validate="many_to_one",
                )
                LOGGER.info("JSON NLP: актуальный источник выбран, присоединено %d признаков", len(NEWS_COLUMNS))
            else:
                coverage = news_coverage(nlp) if nlp.path.exists() else None
                lag = pd.offsets.MonthBegin(nlp.lag_months)
                csv_covers = (
                    bool(coverage)
                    and pd.Timestamp(coverage[1]).to_period("M").to_timestamp() + lag >= first_target
                )
                if csv_covers:
                    monthly = build_news_features(nlp)
                    result = result.merge(monthly, on="period", how="left", validate="many_to_one")
                else:
                    json_range = (
                        (json_features["period"].min(), json_features["period"].max())
                        if not json_features.empty else (None, None)
                    )
                    LOGGER.warning(
                        "Актуальные новости не покрывают target=%s; JSON=%s—%s, legacy CSV=%s—%s, "
                        "минимально необходим месяц публикации=%s (lag=%d)",
                        first_target.date(), json_range[0], json_range[1],
                        coverage[0].date() if coverage else None,
                        coverage[1].date() if coverage else None,
                        (first_target - pd.offsets.MonthBegin(nlp.lag_months)).date(),
                        nlp.lag_months,
                    )

    if not skip_tda and tda_config is not None and tda_config.enabled:
        # NOTE: считаем TDA из текущей панели, иначе первый запуск зависит от stale-кэша.
        tda_features = build_tda_features(result[["mo", "period", "y"]], tda_config)
        if not tda_features.empty:
            result = result.merge(
                tda_features,
                on=["mo", "period"],
                how="left",
                validate="many_to_one",
            )
            LOGGER.info(
                "TDA: присоединено %d колонок",
                len({"tda_entropy", "tda_wasserstein_dist"} & set(result.columns)),
            )

    for source in data_config.rosstat_sources:
        result = merge_rosstat(result, source)

    news_shock_columns = [
        column for column in ("news_shock_score", "telegram_shock_score")
        if column in result
    ]
    for column in news_shock_columns:
        result[f"local_{column}"] = result[column] * result["spending_volatility"]

    LOGGER.info(
        "Финальная матрица: %s, МО=%d, пропуски=%d",
        result.shape,
        result["mo"].nunique(),
        result.isna().sum().sum(),
    )
    first = ["period", "mo", "y"]
    return result[[*first, *(column for column in result.columns if column not in first)]]


def to_validation_panel(frame: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    """Переводит месячную матрицу в контракт rolling-origin CV."""
    if config.horizon != 1 or config.frequency != "MS":
        raise ValueError("Месячная матрица t-1 поддерживает только horizon=1, frequency=MS")
    result = frame.copy()
    result[config.time_column] = pd.to_datetime(result["period"], utc=True)
    result[config.target_end_column] = result[config.time_column] + pd.offsets.MonthBegin(1)
    result[config.availability_column] = result[config.time_column] - pd.Timedelta(nanoseconds=1)
    result[config.entity_column] = result["mo"]
    result[config.target_column] = result["y"]
    return result
