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
    submission_roster: Path = Path("configs/submission_roster.csv")


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


class MemoryConfig(StrictConfig):
    """Политика памяти для полного переобучения на хосте с 14.7 GiB RAM.

    Значения по умолчанию воспроизводят прежнее поведение.  Флаг
    ``--retrain-from-scratch`` переключает ``safe_mode`` и включает
    даункастинг, мини-батчи Prophet и очистку буферов.
    """

    safe_mode: bool = False
    downcast_features: bool = False
    prophet_batch_size: int = Field(default=128, gt=0)
    prophet_spill_dir: Path = Path("tmp/prophet_batches")
    chronos_batch_size: int | None = Field(default=None, gt=0)
    catboost_thread_count: int | None = Field(default=None, gt=0)
    catboost_border_count: int = Field(default=254, ge=2)
    gc_every_batch: bool = True


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
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @model_validator(mode="after")
    def check_horizon(self) -> AppConfig:
        if self.models.chronos.prediction_length != self.validation.horizon:
            raise ValueError("Горизонты Chronos и валидации должны совпадать")
        return self


def load_config(path: Path) -> AppConfig:
    with path.open(encoding="utf-8") as stream:
        return AppConfig.model_validate(yaml.safe_load(stream))
