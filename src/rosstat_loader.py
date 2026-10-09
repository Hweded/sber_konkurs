"""Универсальная point-in-time загрузка и content-based discovery Росстата."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_config import RosstatSourceConfig

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
_DATE_HINTS = ("period", "date", "year", "год", "период", "дата", "month", "месяц")
_ENTITY_HINTS = ("mo", "municip", "region", "territ", "муниц", "район", "регион", "террит")
_RELEASE_HINTS = ("release", "published", "updated", "выпуск", "публикац", "дата размещ")


def _norm_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def discover_rosstat_files(directory: Path = ROOT / "datasets") -> list[Path]:
    root = directory if directory.is_absolute() else ROOT / directory
    tokens = (
        "rosstat",
        "trud",
        "zpl",
        "rabot",
        "mediana",
        "balans",
        "raspr",
        "travm",
        "lik",
        "vib",
    )
    candidates = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.casefold() in {".csv", ".xls", ".xlsx", ".json", ".zip"}
        and any(token in _norm_name(path.name) for token in tokens)
    ]
    LOGGER.info("Росстат discovery: найдено %d потенциальных файлов в %s", len(candidates), root)
    return candidates


def _decode_bytes(raw: bytes, preferred: str = "utf-8-sig") -> str:
    for encoding in tuple(
        dict.fromkeys((preferred, "utf-8-sig", "utf-8", "cp1251", "windows-1251"))
    ):
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_csv_bytes(raw: bytes, separator: str | None, encoding: str) -> pd.DataFrame:
    text = _decode_bytes(raw, encoding)
    if separator:
        delimiter = separator
    else:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=";,\t|").delimiter
        except csv.Error:
            delimiter = ";" if text.count(";") >= text.count(",") else ","
    return pd.read_csv(
        io.StringIO(text), sep=delimiter, dtype="string", engine="python", on_bad_lines="warn"
    )


def _unique_columns(values: list[Any]) -> list[str]:
    result: list[str] = []
    counts: dict[str, int] = {}
    for value in values:
        name = re.sub(r"\s+", " ", str(value).replace("\n", " ").strip())
        if not name or name.casefold().startswith("unnamed") or name == "nan":
            name = f"column_{len(result) + 1}"
        counts[name] = counts.get(name, 0) + 1
        result.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return result


def _detect_header_row(raw: pd.DataFrame) -> int:
    hints = _DATE_HINTS + _ENTITY_HINTS + _RELEASE_HINTS
    best_row, best_score = 0, float("-inf")
    for row_number in range(min(len(raw), 40)):
        values = [
            _norm_name(value)
            for value in raw.iloc[row_number].tolist()
            if pd.notna(value) and str(value).strip()
        ]
        if len(values) < 2:
            continue
        score = len(values) + 5 * sum(any(hint in value for hint in hints) for value in values)
        score += 2 * sum(
            bool(re.fullmatch(r"(?:19|20)\d{2}(?:[./-]\d{1,2})?", value)) for value in values
        )
        if score > best_score:
            best_row, best_score = row_number, score
    return best_row


def _read_excel_tables(
    raw: bytes, path: Path, config: RosstatSourceConfig
) -> dict[str, pd.DataFrame]:
    engine = "xlrd" if path.suffix.casefold() == ".xls" else "openpyxl"
    book = pd.ExcelFile(io.BytesIO(raw), engine=engine)
    sheets: list[str | int] = (
        [config.sheet_name] if config.sheet_name is not None else list(book.sheet_names)
    )
    result: dict[str, pd.DataFrame] = {}
    for sheet in sheets:
        if isinstance(sheet, str) and sheet not in book.sheet_names:
            raise ValueError(f"Лист {sheet!r} отсутствует в {path.name}")
        if config.header_row is not None:
            table = pd.read_excel(book, sheet_name=sheet, header=config.header_row)  # type: ignore[call-overload]
        else:
            raw_table = pd.read_excel(book, sheet_name=sheet, header=None, dtype=object)  # type: ignore[call-overload]
            header = _detect_header_row(raw_table)
            table = raw_table.iloc[header + 1 :].copy()
            table.columns = _unique_columns(raw_table.iloc[header].tolist())
            table = table.dropna(how="all").reset_index(drop=True)
        table.columns = _unique_columns(list(table.columns))
        result[str(sheet)] = table
    return result


def _read_json_bytes(raw: bytes) -> pd.DataFrame:
    data: Any = json.loads(_decode_bytes(raw))
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        for key in ("data", "items", "rows", "results"):
            if isinstance(data.get(key), list):
                return pd.DataFrame(data[key])
        return pd.DataFrame([data])
    raise ValueError("JSON Росстата не содержит таблицу")


def read_rosstat_tables(path: Path, config: RosstatSourceConfig) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    if size_mb > config.max_file_size_mb:
        raise ValueError(f"Файл {path.name} слишком большой: {size_mb:.1f} MB")
    if path.suffix.casefold() == ".zip":
        result: dict[str, pd.DataFrame] = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                inner = Path(member.filename)
                if member.is_dir() or inner.suffix.casefold() not in {
                    ".csv",
                    ".json",
                    ".xls",
                    ".xlsx",
                }:
                    continue
                with archive.open(member) as stream:
                    payload = stream.read()
                result.update(
                    {
                        f"{member.filename}:{key}": value
                        for key, value in read_rosstat_bytes(payload, inner, config).items()
                    }
                )
        return result
    return read_rosstat_bytes(path.read_bytes(), path, config)


def read_rosstat_bytes(
    raw: bytes, path: Path, config: RosstatSourceConfig
) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.casefold()
    if config.format == "csv" or (config.format == "auto" and suffix == ".csv"):
        return {"data": _read_csv_bytes(raw, config.separator or None, config.encoding)}
    if config.format == "json" or (config.format == "auto" and suffix == ".json"):
        return {"data": _read_json_bytes(raw)}
    if suffix in {".xls", ".xlsx"} or config.format in {"xls", "xlsx"}:
        return _read_excel_tables(raw, path, config)
    raise ValueError(f"Неподдерживаемый формат Росстата: {path}")


def _find_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for column in columns:
        if any(hint in _norm_name(column) for hint in hints):
            return column
    return None


def parse_rosstat_period(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.Timestamp("NaT")
    text = str(value).strip().casefold().replace("ё", "е")
    months = {
        "январь": 1,
        "янв": 1,
        "февраль": 2,
        "фев": 2,
        "март": 3,
        "апрель": 4,
        "апр": 4,
        "май": 5,
        "июнь": 6,
        "июль": 7,
        "август": 8,
        "сентябрь": 9,
        "октябрь": 10,
        "ноябрь": 11,
        "декабрь": 12,
    }
    for name, month in months.items():
        match = re.search(rf"{name}[^0-9]*(20\d{{2}})", text)
        if match:
            return pd.Timestamp(year=int(match.group(1)), month=month, day=1)
    match = re.search(r"(20\d{2})\s*(?:m|/|-|\.)\s*(0?[1-9]|1[0-2])", text) or re.search(
        r"(0?[1-9]|1[0-2])\s*[./-]\s*(20\d{2})", text
    )
    if match:
        year, month = (
            (int(match.group(1)), int(match.group(2)))
            if match.group(1).startswith("20")
            else (int(match.group(2)), int(match.group(1)))
        )
        return pd.Timestamp(year=year, month=month, day=1)
    if re.fullmatch(r"20\d{2}\s*(?:год|г)?", text):
        return pd.Timestamp(year=int(text[:4]), month=1, day=1)
    parsed_text: str = str(
        pd.to_datetime(str(value), errors="coerce", format="mixed", dayfirst=True)
    )
    return (
        pd.Timestamp("NaT")
        if parsed_text == "NaT"
        else pd.Timestamp(parsed_text).to_period("M").to_timestamp()
    )


def _clean_numeric(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned = cleaned.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    cleaned = cleaned.mask(cleaned.isin(["", "-", ".", "..", "…"]))
    return pd.to_numeric(cleaned, errors="coerce")


def infer_rosstat_schema(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    date_column = _find_column(list(data.columns), _DATE_HINTS)
    entity_column = _find_column(list(data.columns), _ENTITY_HINTS)
    release_column = _find_column(list(data.columns), _RELEASE_HINTS)
    parsed_dates = {
        column: int(pd.to_datetime(data[column], errors="coerce", format="mixed").notna().sum())
        for column in data.columns
    }
    if date_column is None and parsed_dates:
        date_column = max(parsed_dates, key=lambda column: parsed_dates[column])
    numeric = [
        column
        for column in data.columns
        if int(_clean_numeric(data[column]).notna().sum()) >= max(1, int(len(data) * 0.5))
    ]
    return {
        "period_column": date_column,
        "entity_column": entity_column,
        "release_column": release_column,
        "value_columns": tuple(
            column
            for column in numeric
            if column not in {date_column, entity_column, release_column}
        ),
    }


def normalize_rosstat_table(frame: pd.DataFrame, config: RosstatSourceConfig) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    inferred = infer_rosstat_schema(data)
    period_column = (
        config.period_column if config.period_column in data else inferred["period_column"]
    )
    entity_column = (
        config.entity_column if config.entity_column in data else inferred["entity_column"]
    )
    release_column = (
        config.release_column if config.release_column in data else inferred["release_column"]
    )
    values = config.value_columns or tuple(inferred["value_columns"])
    if period_column is None or not values:
        raise ValueError(f"{config.name}: не удалось определить дату и числовые показатели")
    if entity_column is None:
        data["__entity"] = "aggregate"
        entity_column = "__entity"
    if release_column is None:
        raise ValueError(f"{config.name}: отсутствует дата доступности/release")
    required = {
        str(period_column),
        str(entity_column),
        str(release_column),
        *(str(value) for value in values),
    }
    if not required.issubset(data.columns):
        raise ValueError(
            f"{config.name}: отсутствуют колонки {sorted(required.difference(data.columns))}"
        )
    result = data[[period_column, entity_column, release_column, *values]].rename(
        columns={period_column: "_reference", entity_column: "_entity", release_column: "_release"}
    )
    result["_reference"] = result["_reference"].map(parse_rosstat_period)
    result["_release"] = pd.to_datetime(result["_release"], utc=True, errors="coerce")
    result = result.dropna(subset=["_reference", "_entity", "_release"])
    for column in values:
        result[f"rosstat_{config.name}_{column}"] = _clean_numeric(result[column]).astype(float)
    names = [f"rosstat_{config.name}_{column}" for column in values]
    return result[["_reference", "_entity", "_release", *names]]


class RosstatPipeline:
    def __init__(self, directory: Path = ROOT / "datasets") -> None:
        self.directory = directory
        self.diagnostics: list[dict[str, Any]] = []

    def discover(self) -> list[Path]:
        return discover_rosstat_files(self.directory)

    def load(self, config: RosstatSourceConfig) -> pd.DataFrame:
        paths = (
            [config.path]
            if config.path.exists()
            else self.discover()
            if config.autodiscover
            else []
        )
        parts: list[pd.DataFrame] = []
        for path in paths:
            try:
                for sheet, table in read_rosstat_tables(path, config).items():
                    if len(table) > config.max_rows:
                        raise ValueError(f"Лист {sheet} содержит слишком много строк: {len(table)}")
                    normalized = normalize_rosstat_table(table, config)
                    normalized["source_file"] = str(path)
                    parts.append(normalized)
                    self.diagnostics.append(
                        {"path": str(path), "sheet": sheet, "rows": len(normalized), "status": "ok"}
                    )
            except Exception as error:
                LOGGER.warning("РосстатPipeline: пропуск %s: %s", path, error)
                self.diagnostics.append(
                    {"path": str(path), "status": "error", "reason": str(error)}
                )
        if not parts:
            return pd.DataFrame(columns=["period", "region", "mo", "indicator", "value"])
        return pd.concat(parts, ignore_index=True)


def _build_causal_proxy(frame: pd.DataFrame, config: RosstatSourceConfig) -> pd.DataFrame:
    result = frame.copy()
    ordered = result.sort_values(["mo", "period"], kind="stable")
    lagged = (
        pd.to_numeric(ordered["y"], errors="coerce").groupby(ordered["mo"], observed=True).shift(1)
    )
    prefix = f"rosstat_{config.name}_proxy"
    ordered[prefix + "_spending_level"] = (
        lagged.groupby(ordered["mo"], observed=True)
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=float)
    )
    ordered[prefix + "_spending_growth"] = (
        lagged.groupby(ordered["mo"], observed=True)
        .pct_change(fill_method=None)
        .to_numpy(dtype=float)
    )
    ordered = ordered.sort_index()
    result[prefix + "_spending_level"] = ordered[prefix + "_spending_level"].to_numpy()
    result[prefix + "_spending_growth"] = ordered[prefix + "_spending_growth"].to_numpy()
    return result


def merge_rosstat(frame: pd.DataFrame, config: RosstatSourceConfig) -> pd.DataFrame:
    path = config.path if config.path.is_absolute() else ROOT / config.path
    if not path.exists():
        if config.required:
            raise FileNotFoundError(path)
        return _build_causal_proxy(frame, config)
    try:
        data = pd.concat(
            [
                normalize_rosstat_table(table, config)
                for table in read_rosstat_tables(path, config).values()
            ],
            ignore_index=True,
        )
        if data.duplicated(["_reference", "_entity", "_release"]).any():
            raise ValueError(f"{config.name}: дубликаты винтажей")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        if config.required:
            raise ValueError(f"Росстат {path.name}: {error}") from error
        return _build_causal_proxy(frame, config)
    left = frame.copy()
    left["_row"] = np.arange(len(left))
    left["_entity"] = left["mo"].astype("string")
    left["_reference"] = left["period"] - pd.offsets.MonthBegin(1)
    left["_cutoff"] = pd.to_datetime(left["period"], utc=True)
    merged = pd.merge_asof(
        left.sort_values("_cutoff"),
        data.sort_values("_release"),
        left_on="_cutoff",
        right_on="_release",
        by=["_entity", "_reference"],
        direction="backward",
        allow_exact_matches=False,
    )
    return (
        merged.sort_values("_row")
        .drop(columns=["_row", "_entity", "_reference", "_cutoff", "_release"])
        .reset_index(drop=True)
    )
