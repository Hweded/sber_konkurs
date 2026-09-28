"""Канонизация географии муниципальных образований и региональные join-ы."""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

_REGION_ALIASES: dict[str, str] = {
    "чувашская республика чувашия": "чувашия",
    "чувашская республика — чувашия": "чувашия",
    "чувашская республика - чувашия": "чувашия",
    "г санкт петербург": "санкт-петербург",
    "город санкт петербург": "санкт-петербург",
    "москва город": "москва",
}


def _clean_geo_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.replace("\ufeff", "").replace("ё", "е").casefold()
    text = re.sub(r"[\[\]()*†‡]+", " ", text)
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip(" .,;:'\"-")
    return text


def canonical_region(value: Any) -> str:
    """Возвращает консервативный ключ субъекта РФ."""
    text = _clean_geo_text(value)
    if text in _REGION_ALIASES:
        return _REGION_ALIASES[text]
    text = re.sub(r"^г\.?\s+", "", text)
    text = re.sub(r"\b(область|край|республика|автономная область|автономный округ|ао)\b", "", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def canonical_municipality(value: Any, region: Any = None) -> str:
    """Канонизирует МО, сохраняя его тип и не удаляя значимые слова."""
    text = _clean_geo_text(value)
    text = re.sub(r"^г\.?\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    region_key = canonical_region(region) if region is not None else ""
    return f"{region_key}::{text}" if region_key else text


@dataclass(frozen=True)
class GeoResolution:
    """Решение географического сопоставления с объяснимым статусом."""

    key: str
    region: str
    status: str
    confidence: float
    reason: str
    source: str


def resolve_municipality(
    name: Any,
    *,
    region: Any = None,
    code: Any = None,
    crosswalk: pd.DataFrame | None = None,
) -> GeoResolution:
    """Разрешает МО через код/crosswalk/составной регион без ложного выбора."""
    region_key = canonical_region(region)
    name_key = _clean_geo_text(name)
    if crosswalk is not None and not crosswalk.empty:
        table = crosswalk.copy()
        table["_name"] = table["mo"].map(_clean_geo_text)
        table["_region"] = table.get("region", "").map(canonical_region) if "region" in table else ""
        if code is not None and "code" in table:
            matched = table.loc[table["code"].astype(str).eq(str(code))]
            if len(matched) == 1:
                row = matched.iloc[0]
                return GeoResolution(str(row.get("canonical_id", row["_name"])), str(row["_region"]), "exact", 1.0, "code", "crosswalk")
        matched = table.loc[table["_name"].eq(name_key) & (table["_region"].eq(region_key) if region_key else True)]
        if len(matched) == 1:
            row = matched.iloc[0]
            return GeoResolution(str(row.get("canonical_id", row["_name"])), str(row["_region"]), "crosswalk", .95, "name+region", "crosswalk")
        if len(matched) > 1:
            LOGGER.warning("Неоднозначное МО %s/%s: %d кандидатов", name_key, region_key, len(matched))
    key = canonical_municipality(name, region)
    return GeoResolution(key, region_key, "ambiguous" if not region_key else "inferred", .2 if not region_key else .55, "surrogate key", "fallback")


def merge_regional_to_municipal(df_target: pd.DataFrame, df_macro_regions: pd.DataFrame) -> pd.DataFrame:
    """Left join регионального показателя на МО и добавляет coverage-диагностику."""
    required_target = {"period", "mo"}
    required_macro = {"period", "region"}
    if not required_target.issubset(df_target) or not required_macro.issubset(df_macro_regions):
        raise ValueError("Нужны period, mo у target и period, region у macro")
    left = df_target.copy()
    right = df_macro_regions.copy()
    left["period"] = pd.to_datetime(left["period"]).dt.to_period("M").dt.to_timestamp()
    right["period"] = pd.to_datetime(right["period"]).dt.to_period("M").dt.to_timestamp()
    left["_region_key"] = left.get("region", left["mo"]).map(canonical_region)
    right["_region_key"] = right["region"].map(canonical_region)
    if right.duplicated(["_region_key", "period"]).any():
        LOGGER.warning("Regional join: duplicate region-period keys; first row retained")
        right = right.drop_duplicates(["_region_key", "period"], keep="last")
    payload = [column for column in right.columns if column not in {"period", "region", "_region_key"}]
    result = left.merge(right[["period", "_region_key", *payload]], on=["period", "_region_key"], how="left", validate="many_to_one")
    result["regional_match"] = result[payload].notna().any(axis=1) if payload else False
    result = result.drop(columns="_region_key")
    LOGGER.info("Regional join: %d/%d rows matched", int(result["regional_match"].sum()), len(result))
    return result
