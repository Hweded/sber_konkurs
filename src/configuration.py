"""Загрузка YAML и проверка связности настроек."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.validation import ValidationConfig
from src.data_config import DataConfig, FeatureConfig
from src.changepoints import DetectionConfig
from src.tda_engine import TDAConfig
from src.ensemble import EnsembleConfig


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(StrictConfig):
    supervised: Path
    artifacts: Path


class ReproducibilityConfig(StrictConfig):
    seed: int = Field(ge=0, le=4294967295)
    deterministic_torch: bool
    threads: int = Field(gt=0)


class DeviceConfig(StrictConfig):
    """Централизованные параметры ускорения Torch-моделей."""

    mode: Literal["auto", "cuda", "cpu"] = "auto"
    fallback_to_cpu_on_oom: bool = True
    batch_size: int = Field(default=128, gt=0)


class ChronosConfig(StrictConfig):
    model_id: str
    revision: str
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "bfloat16"]
    context_length: int = Field(gt=0, le=2048)
    prediction_length: int = Field(gt=0, le=64)
    point_forecast: Literal["median"]
    zero_shot: Literal[True]


class ModelsConfig(StrictConfig):
    prophet: dict[str, Any]
    catboost: dict[str, Any]
    chronos: ChronosConfig
    foundation_models: tuple[str, ...] = ("chronos-bolt",)


class OptimizationConfig(StrictConfig):
    enabled: bool
    sampler: Literal["TPESampler"]
    n_trials: int = Field(gt=0)
    inner_n_splits: int = Field(ge=2)
    objective: Literal["MAE"]


class AppConfig(StrictConfig):
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    paths: PathsConfig
    validation: ValidationConfig
    reproducibility: ReproducibilityConfig
    models: ModelsConfig
    optimization: OptimizationConfig
    data: DataConfig
    features: FeatureConfig

    changepoint_detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tda: TDAConfig = Field(default_factory=TDAConfig)

    @model_validator(mode="after")
    def check_horizon(self) -> AppConfig:
        if self.models.chronos.prediction_length != self.validation.horizon:
            raise ValueError("Горизонты Chronos и валидации должны совпадать")
        return self


def load_config(path: Path) -> AppConfig:
    with path.open(encoding="utf-8") as stream:
        return AppConfig.model_validate(yaml.safe_load(stream))
