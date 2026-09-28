"""География, fold-local preprocessing и канонический период Росстата."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.geo_resolver import (
    canonical_municipality,
    canonical_region,
    merge_regional_to_municipal,
    resolve_municipality,
)
from src.preprocessing import ExternalFeaturePreprocessor, add_normalized_features, save_feature_store
from src.rosstat_loader import parse_rosstat_period


def test_geo_canonicalization_and_ambiguity() -> None:
    assert canonical_region("Чувашская Республика — Чувашия") == "чувашия"
    assert canonical_region("г. Санкт-Петербург") == "санкт-петербург"
    assert canonical_municipality("Советский муниципальный район", "Чувашия").startswith("чувашия::")
    resolution = resolve_municipality("Советский муниципальный район")
    assert resolution.status == "ambiguous"
    assert resolution.key


def test_regional_merge_keeps_all_target_rows() -> None:
    target = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "mo": ["a", "b"],
            "region": ["Чувашская Республика — Чувашия", "г. Санкт-Петербург"],
        }
    )
    macro = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-01-01"]),
            "region": ["чувашия"],
            "wage": [100.0],
        }
    )
    result = merge_regional_to_municipal(target, macro)
    assert len(result) == 2
    assert result["regional_match"].tolist() == [True, False]


def test_period_parser_russian_and_emiss_formats() -> None:
    assert parse_rosstat_period("Январь 2023 г.") == pd.Timestamp("2023-01-01")
    assert parse_rosstat_period("2023M02") == pd.Timestamp("2023-02-01")
    assert parse_rosstat_period("03.2023") == pd.Timestamp("2023-03-01")
    assert parse_rosstat_period("2023 год") == pd.Timestamp("2023-01-01")
    assert pd.isna(parse_rosstat_period("не дата"))


def test_preprocessor_uses_only_train_statistics() -> None:
    train = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "mo": ["a", "a"],
            "federal_district": ["x", "x"],
            "wage": [10.0, 20.0],
        }
    )
    test = pd.DataFrame(
        {
            "period": pd.to_datetime(["2024-03-01"]),
            "mo": ["b"],
            "federal_district": ["x"],
            "wage": [np.nan],
        }
    )
    processor = ExternalFeaturePreprocessor(["wage"]).fit(train)
    result = processor.transform(test)
    assert result["wage"].iloc[0] == 15.0
    assert result["wage_imputation_source"].iloc[0] == "federal_district"


def test_normalized_features_do_not_cross_entities() -> None:
    periods = pd.date_range("2023-01-01", periods=13, freq="MS")
    frame = pd.concat(
        [
            pd.DataFrame({"period": periods, "mo": mo, "y": 100.0, "wage": np.arange(1, 14) * factor, "employment": 10.0})
            for mo, factor in (("a", 1.0), ("b", 100.0))
        ],
        ignore_index=True,
    )
    result = add_normalized_features(frame)
    assert result.loc[result["mo"].eq("a"), "wage_growth_yoy"].iloc[-1] == 12.0
    assert result.loc[result["mo"].eq("b"), "wage_growth_yoy"].iloc[-1] == 12.0


def test_atomic_feature_store(tmp_path: Path) -> None:
    path = tmp_path / "processed" / "features.parquet"
    frame = pd.DataFrame({"period": pd.to_datetime(["2024-01-01"]), "value": [1.0]})
    save_feature_store(frame, path, sources=("source.csv",), metadata={"point_in_time": True})
    assert path.exists()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 1
    assert manifest["metadata"]["point_in_time"] is True
    assert not path.with_suffix(".tmp.parquet").exists()
