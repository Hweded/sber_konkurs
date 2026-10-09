"""Типизированные настройки чтения источников и причинных признаков."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ETLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetConfig(ETLConfig):
    pattern: str = "potrebitelskie-beznalicnye-rashody-na-urovne*.csv"
    category_column: str = "category_15"
    category_mode: Literal["select", "sum"] = "select"
    category: str = "Все категории"
    sum_categories: tuple[str, ...] = ()
    unit: str = "руб."

    duplicate_policy: Literal["error", "exclude_ambiguous_mo"] = "error"

    @model_validator(mode="after")
    def check_sum(self) -> TargetConfig:
        if self.category_mode == "sum":
            if not self.sum_categories or len(set(self.sum_categories)) != len(self.sum_categories):
                raise ValueError("Для sum нужен явный список уникальных непересекающихся категорий")
            if self.category in self.sum_categories:
                raise ValueError("Итоговую категорию нельзя складывать с компонентами")
        return self


class MacroSourceConfig(ETLConfig):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pattern: str
    dimensions: tuple[str, ...]
    frequency: Literal["Месяц", "Неделя"]
    aggregation: Literal["mean", "sum", "last"] = "mean"
    required: bool = True


class RosstatSourceConfig(ETLConfig):
    path: Path
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    required: bool = False
    format: Literal["auto", "csv", "xls", "xlsx", "json", "zip"] = "auto"
    read_timeout_seconds: PositiveInt = 60
    max_file_size_mb: PositiveInt = 512
    max_rows: PositiveInt = 2_000_000
    sheet_name: str | int | None = None
    header_row: int | None = None
    separator: str = ","
    encoding: str = "utf-8-sig"
    entity_column: str = "mo"
    period_column: str = "period"
    release_column: str = "released_at"
    value_columns: tuple[str, ...] = ()
    region_mapping: Path | None = None
    autodiscover: bool = False
    required_fields: tuple[str, ...] = ()


class NLPConfig(ETLConfig):
    enabled: bool = True
    path: Path = Path("datasets/lenta-ru-news.csv")
    cache: Path = Path("data/processed/news_monthly_features.parquet")
    chunksize: PositiveInt = 50_000
    batch_size: PositiveInt = 256
    max_length: PositiveInt = 128
    topics: tuple[str, ...] = ("Экономика",)
    tags: tuple[str, ...] = ()
    model_id: str = "cointegrated/rubert-tiny-sentiment-balanced"
    revision: str = "main"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    mixed_precision: bool = False
    fallback_to_cpu_on_oom: bool = True
    lag_months: PositiveInt = 1
    shock_window: PositiveInt = 3
    min_date: str = "1999-01-01"
    # Без first_seen_at остаётся считать дату архива датой доступности.
    publication_date_is_availability: bool = True
    telegram_input: Path | None = None
    news_json_paths: tuple[Path, ...] = ()
    news_auto_discover: bool = True
    telegram_keywords: tuple[str, ...] = ()
    telegram_min_length: PositiveInt = 40
    news_timezone: str = "Europe/Moscow"
    news_sentiment_enabled: bool = True
    news_strip_html: bool = True
    news_remove_urls: bool = True
    news_deduplicate: bool = True
    news_max_text_length: PositiveInt = 10_000


class DataConfig(ETLConfig):
    directory: Path = Path("datasets")
    separator: Literal[";", ",", "\t"] = ";"
    target: TargetConfig = Field(default_factory=TargetConfig)
    macro_sources: tuple[MacroSourceConfig, ...] = ()
    macro_lag_months: PositiveInt = 1
    rosstat_sources: tuple[RosstatSourceConfig, ...] = ()
    nlp: NLPConfig = Field(default_factory=lambda: NLPConfig(enabled=False))

    @model_validator(mode="after")
    def check_names(self) -> DataConfig:
        names = [source.name for source in self.macro_sources]
        if len(names) != len(set(names)):
            raise ValueError("Имена макроисточников должны быть уникальны")
        return self


class FeatureConfig(ETLConfig):
    lags: tuple[PositiveInt, ...] = (1, 2, 3, 12)
    rolling_windows: tuple[PositiveInt, ...] = (3, 12)
    rolling_statistics: tuple[Literal["mean", "std", "min", "max"], ...] = (
        "mean",
        "std",
        "min",
        "max",
    )
    warmup_months: PositiveInt = 12
    news_lag_months: PositiveInt = 1

    @model_validator(mode="after")
    def check_windows(self) -> FeatureConfig:
        for values in (self.lags, self.rolling_windows, self.rolling_statistics):
            if not values or len(set(values)) != len(values):
                raise ValueError("Списки признаков должны быть непустыми и уникальными")
        # Long lags and rolling windows are allowed to start as NaN.  Fold
        # local imputers handle those early values, while retaining the first
        # available calendar months is necessary for h=6/12 evaluation.
        if self.warmup_months < 1:
            raise ValueError("warmup_months должен быть положительным")
        return self
