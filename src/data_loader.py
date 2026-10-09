"""Чтение UTF-8 выгрузок и склейка календарно лагированных макроданных."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_config import DataConfig, MacroSourceConfig, TargetConfig

LOGGER = logging.getLogger(__name__)


def discover_file(directory: Path, pattern: str, *, required: bool = True) -> Path | None:
    """Находит единственную выгрузку; неоднозначные винтажи не выбирает молча."""
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(files) > 1:
        raise ValueError(f"Несколько выгрузок для {pattern}: {[p.name for p in files]}")
    if not files:
        if required:
            raise FileNotFoundError(f"Не найден {directory / pattern}")
        LOGGER.warning("Не найден необязательный источник %s", pattern)
        return None
    return files[0]


def month_start(values: pd.Series) -> pd.Series:
    """Нормализует даты в timezone-naive datetime64[ns], начало месяца."""
    dates = pd.to_datetime(values, errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Нужны непустые календарные даты без timezone")
    return dates.dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")


def _read_csv_streaming(path: Path, separator: str) -> pd.DataFrame:
    """Читает CSV построчно, не раздувая промежуточный буфер pandas."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=separator)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Пустой CSV: {path.name}") from exc
        if not header or any(not isinstance(name, str) or not name.strip() for name in header):
            raise ValueError(f"Некорректный заголовок CSV: {path.name}")
        header = [name.strip() for name in header]
        if len(set(header)) != len(header):
            raise ValueError(f"Повторяющиеся колонки в CSV: {path.name}")
        rows: list[list[str]] = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"{path.name}: строка содержит {len(row)} полей при {len(header)} колонках",
                )
            rows.append(row)
    return pd.DataFrame.from_records(rows, columns=header, coerce_float=False)


def read_source(path: Path, separator: str = ";") -> pd.DataFrame:
    """Читает UTF-8 выгрузку и применяет множитель единиц."""
    try:
        frame = _read_csv_streaming(path, separator)
        required = {"period", "value", "obs_status", "freq", "unit_mult"}
        if not required.issubset(frame):
            raise ValueError(
                f"{path.name}: нет колонок {sorted(required.difference(frame.columns))}"
            )
        before = len(frame)
        frame = frame.loc[frame["obs_status"].eq("A")].copy()
        frame["period"] = pd.to_datetime(frame["period"], format="%Y-%m-%d", errors="raise").astype(
            "datetime64[ns]"
        )
        if frame["period"].isna().any() or frame.empty:
            raise ValueError(f"{path.name}: нет валидных дат/строк")
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            frame[column] = frame[column].astype("string").str.strip()
        unit_columns = [name for name in ("unit_measure", "unit_meas") if name in frame]
        if len(unit_columns) != 1:
            raise ValueError("Ожидается ровно одна колонка единицы измерения")
        frame = frame.rename(columns={unit_columns[0]: "unit"})
        multiplier = pd.to_numeric(frame["unit_mult"], errors="raise")
        if multiplier.isna().any() or not multiplier.between(-12, 12).all():
            raise ValueError("Некорректный unit_mult")
        frame["value"] = pd.to_numeric(frame["value"], errors="raise") * np.power(10.0, multiplier)
        if np.isinf(frame["value"].to_numpy(dtype=float)).any():
            raise ValueError("Бесконечное значение источника")
        LOGGER.info(
            "%s: прочитано %d, статус A %d, пропуски value %d",
            path.name,
            before,
            len(frame),
            frame["value"].isna().sum(),
        )
        return frame
    except Exception:
        LOGGER.exception("Ошибка чтения %s", path)
        raise


def load_target(directory: Path, config: TargetConfig, separator: str = ";") -> pd.DataFrame:
    """Возвращает уникальные (period, mo, y), не складывая итог с категориями."""
    path = discover_file(directory, config.pattern)
    if path is None:
        raise FileNotFoundError(config.pattern)
    frame = read_source(path, separator)
    required = {"mo", config.category_column}
    if not required.issubset(frame):
        raise ValueError(f"Нет измерений таргета: {sorted(required.difference(frame.columns))}")
    if not frame["freq"].eq("Месяц").all() or not frame["unit"].eq(config.unit).all():
        raise ValueError("Неожиданная частота или единицы таргета")
    frame["period"] = month_start(frame["period"])
    if frame[["mo", config.category_column]].isna().any().any():
        raise ValueError("Пропущена территория или категория")
    categories = (config.category,) if config.category_mode == "select" else config.sum_categories
    missing = set(categories).difference(frame[config.category_column].unique())
    if missing:
        raise ValueError(f"Нет целевых категорий: {sorted(missing)}")
    frame = frame.loc[frame[config.category_column].isin(categories)].copy()
    keys = ["period", "mo"]
    duplicates = frame.duplicated([*keys, config.category_column], keep=False)
    if duplicates.any():
        if config.duplicate_policy == "error":
            raise ValueError("Дубликаты целевых наблюдений: требуется явная политика винтажей")
        # FIXME: пока нет ОКАТО, разводим 49 дублей через source и порядок в выгрузке.
        if "source" not in frame or frame.loc[duplicates, "source"].isna().any():
            raise ValueError(
                "Неоднозначные МО нельзя разрешить без стабильного source/кода региона"
            )
        ambiguous = set(frame.loc[duplicates, "mo"].astype(str))
        frame["mo_original"] = frame["mo"].astype(str)
        frame["mo_source"] = frame["mo_original"] + " [" + frame["source"].astype(str) + "]"
        slot = frame.groupby(
            ["period", "mo_source", config.category_column], sort=False, observed=True
        ).cumcount()
        frame["mo"] = frame["mo_source"]
        frame.loc[slot.gt(0), "mo"] = (
            frame.loc[slot.gt(0), "mo_source"] + "#" + (slot[slot.gt(0)] + 1).astype(str)
        )
        if frame.duplicated([*keys, config.category_column]).any():
            raise ValueError("Не удалось сформировать уникальный слот неоднозначного наблюдения")
        LOGGER.warning(
            "Разрешено %d неоднозначных названий МО: source + порядковый слот; "
            "это не восстановление регионального кода",
            len(ambiguous),
        )
    if config.category_mode == "sum":
        result = (
            frame.groupby(keys, observed=True)["value"]
            .sum(min_count=len(categories))
            .rename("y")
            .reset_index()
        )
    else:
        result = frame[[*keys, "value"]].rename(columns={"value": "y"})
    LOGGER.info(
        "Таргет: %d строк, %d МО, %s — %s",
        len(result),
        result["mo"].nunique(),
        result["period"].min(),
        result["period"].max(),
    )
    return result.sort_values(["mo", "period"]).reset_index(drop=True)


def _macro_table(frame: pd.DataFrame, source: MacroSourceConfig) -> pd.DataFrame:
    dimensions = list(dict.fromkeys([*source.dimensions, "unit"]))
    if not set(dimensions).issubset(frame) or frame[dimensions].isna().any().any():
        raise ValueError(f"{source.name}: отсутствуют измерения")
    if not frame["freq"].eq(source.frequency).all():
        raise ValueError(f"{source.name}: неверная частота")
    known = {
        "period",
        "value",
        "obs_status",
        "source",
        "freq",
        "decimals",
        "unit_mult",
        *dimensions,
    }
    if set(frame.columns).difference(known):
        raise ValueError(
            f"{source.name}: неописанные измерения {set(frame.columns).difference(known)}"
        )
    if frame.duplicated(["period", *dimensions]).any():
        raise ValueError(f"{source.name}: дубликаты измерений и дат")
    frame = frame.sort_values("period").copy()
    frame["period"] = month_start(frame["period"])
    if source.frequency == "Месяц" and frame.duplicated(["period", *dimensions]).any():
        raise ValueError(f"{source.name}: несколько наблюдений одного месяца")
    # JSON-ключ длинный, зато разные срезы не схлопнутся после slugify.
    frame["indicator"] = frame[dimensions].apply(
        lambda row: (
            source.name + "::" + json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
        ),
        axis=1,
    )
    grouped = frame.groupby(["period", "indicator"], observed=True)["value"]
    if source.aggregation == "sum":
        values = grouped.sum(min_count=1)
    elif source.aggregation == "last":
        values = grouped.last()
    else:
        values = grouped.mean()
    return values.unstack("indicator").sort_index()


def load_macro(directory: Path, config: DataConfig) -> pd.DataFrame:
    """Собирает макропризнаки и сдвигает их по полной месячной сетке."""
    tables: list[pd.DataFrame] = []
    for source in config.macro_sources:
        path = discover_file(directory, source.pattern, required=source.required)
        if path is not None:
            tables.append(_macro_table(read_source(path, config.separator), source))
    if not tables:
        return pd.DataFrame({"period": pd.Series(dtype="datetime64[ns]")})
    combined = pd.concat(tables, axis=1).sort_index()
    calendar = pd.date_range(
        combined.index.min(),
        combined.index.max() + pd.offsets.MonthBegin(config.macro_lag_months),
        freq="MS",
    )
    result = combined.reindex(calendar).shift(config.macro_lag_months)
    result = result.add_prefix("macro_").add_suffix(f"_lag_{config.macro_lag_months}")
    result.index.name = "period"
    LOGGER.info(
        "Макро: %d месяцев, %d признаков, %d пропусков",
        len(result),
        len(result.columns),
        result.isna().sum().sum(),
    )
    return result.reset_index()


def load_data(directory: Path, config: DataConfig) -> pd.DataFrame:
    """Склеивает таргет и уже лагированные общероссийские признаки m:1."""
    target = load_target(directory, config.target, config.separator)
    macro = load_macro(directory, config)
    return target.merge(macro, on="period", how="left", validate="many_to_one")
