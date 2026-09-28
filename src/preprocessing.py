"""Fold-local подготовка внешних социально-экономических признаков."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

LOGGER = logging.getLogger(__name__)


class ExternalFeaturePreprocessor:
    """Иерархическая импутация и масштабирование без переобучения на test."""

    def __init__(self, feature_columns: Sequence[str], *, scaler: str | None = None) -> None:
        self.feature_columns = tuple(feature_columns)
        self.scaler_name = scaler
        self._district_medians: dict[str, dict[str, float]] = {}
        self._global_medians: dict[str, float] = {}
        self._scaler: RobustScaler | StandardScaler | None = None
        self.fitted = False
        self.report_: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> ExternalFeaturePreprocessor:
        """Обучает fallback-статистики только на переданном train fold."""
        data = frame.copy()
        for column in self.feature_columns:
            values = pd.to_numeric(data[column], errors="coerce")
            finite = values[np.isfinite(values)]
            self._global_medians[column] = float(finite.median()) if len(finite) else np.nan
            if "federal_district" in data:
                grouped = data.assign(_value=values).groupby("federal_district", observed=True)["_value"].median()
                self._district_medians[column] = {str(k): float(v) for k, v in grouped.dropna().items()}
        matrix = data.loc[:, list(self.feature_columns)].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.fillna(pd.Series(self._global_medians))
        if self.scaler_name == "robust":
            self._scaler = RobustScaler().fit(matrix)
        elif self.scaler_name == "standard":
            self._scaler = StandardScaler().fit(matrix)
        elif self.scaler_name is not None:
            raise ValueError(f"Неизвестный scaler: {self.scaler_name}")
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Применяет сохранённые train-статистики и причинный local ffill."""
        if not self.fitted:
            raise RuntimeError("ExternalFeaturePreprocessor не обучен")
        data = frame.copy()
        if "period" in data:
            data["period"] = pd.to_datetime(data["period"], errors="raise")
        sort_columns = [column for column in ("mo", "period") if column in data]
        if sort_columns:
            data = data.sort_values(sort_columns, kind="stable")
        counts: dict[str, int] = {}
        for column in self.feature_columns:
            values = pd.to_numeric(data[column], errors="coerce")
            source = pd.Series("observed", index=data.index, dtype="string")
            if "region_value" in data:
                mask = values.isna() & pd.to_numeric(data["region_value"], errors="coerce").notna()
                values.loc[mask] = pd.to_numeric(data.loc[mask, "region_value"], errors="coerce")
                source.loc[mask] = "region"
            if "federal_district" in data:
                medians = data["federal_district"].astype(str).map(self._district_medians.get(column, {}))
                mask = values.isna() & medians.notna()
                values.loc[mask] = medians.loc[mask]
                source.loc[mask] = "federal_district"
            if "mo" in data:
                filled = values.groupby(data["mo"], observed=True).ffill(limit=2)
                mask = values.isna() & filled.notna()
                values.loc[mask] = filled.loc[mask]
                source.loc[mask] = "local_ffill"
            source.loc[values.isna()] = "unresolved"
            data[column] = values
            data[f"{column}_imputation_source"] = source
            for key, count in source.value_counts(dropna=False).items():
                counts[f"{column}:{key}"] = int(count)
        if self._scaler is not None:
            matrix = data.loc[:, list(self.feature_columns)].fillna(pd.Series(self._global_medians))
            data.loc[:, list(self.feature_columns)] = self._scaler.transform(matrix)
        self.report_ = {"rows": len(data), "counts": counts}
        return data.sort_index()

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def add_normalized_features(
    frame: pd.DataFrame,
    *,
    spending_column: str = "y",
    wage_column: str = "wage",
    employment_column: str = "employment",
) -> pd.DataFrame:
    """Добавляет ratios и причинный годовой рост зарплаты по территории."""
    data = frame.copy().sort_values(["mo", "period"], kind="stable")
    spending = pd.to_numeric(data[spending_column], errors="coerce")
    wage = pd.to_numeric(data[wage_column], errors="coerce")
    employment = pd.to_numeric(data[employment_column], errors="coerce")
    data["wage_to_spending_ratio"] = spending.div(wage.where(wage.gt(0)))
    data["spending_per_employed"] = spending.div(employment.where(employment.gt(0)))
    data["wage_growth_yoy"] = wage.groupby(data["mo"], observed=True).pct_change(12, fill_method=None)
    return data


def save_feature_store(
    frame: pd.DataFrame,
    path: Path,
    *,
    sources: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
) -> None:
    """Атомарно сохраняет Parquet и JSON-метаданные схемы."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
    except ImportError as error:
        raise RuntimeError("Для Parquet установите pyarrow или fastparquet") from error
    temporary.replace(path)
    payload = {
        "schema_version": 1,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "sources": list(sources),
        "rows": len(frame),
        "missing_values": int(frame.isna().sum().sum()),
        "metadata": metadata or {},
    }
    manifest = path.with_suffix(".manifest.json")
    temp_manifest = manifest.with_suffix(".tmp.json")
    temp_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_manifest.replace(manifest)
