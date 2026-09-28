"""Причинный OOF-блендинг CatBoost и Chronos."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error


class EnsembleConfig(BaseModel):
    """Строгая конфигурация weighted-average ансамбля."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    method: Literal["weighted_average", "ridge_stacking"] = "weighted_average"
    metric_to_optimize: Literal["MAE"] = "MAE"
    alpha_bounds: tuple[float, float] = (0.5, 1.0)
    default_alpha: float = Field(default=0.82, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_bounds(self) -> EnsembleConfig:
        low, high = self.alpha_bounds
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("alpha_bounds должны удовлетворять 0 <= low < high <= 1")
        if not low <= self.default_alpha <= high:
            raise ValueError("default_alpha должен находиться внутри alpha_bounds")
        return self


class OOFBlender:
    """Подбирает вес только на переданной прошлой OOF-выборке."""

    def __init__(self, config: EnsembleConfig) -> None:
        if config.method != "weighted_average":
            raise ValueError("ridge_stacking пока не реализован")
        self.config = config
        self.alpha_: float = config.default_alpha

    @staticmethod
    def _finite(values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64).reshape(-1)

    def fit_weights(
        self,
        y_val: ArrayLike,
        preds_catboost_val: ArrayLike,
        preds_chronos_val: ArrayLike,
    ) -> float:
        actual = self._finite(y_val)
        catboost = self._finite(preds_catboost_val)
        chronos = self._finite(preds_chronos_val)
        if not (actual.shape == catboost.shape == chronos.shape) or actual.size == 0:
            raise ValueError("Для подбора alpha нужны непустые массивы одинаковой длины")
        valid = np.isfinite(actual) & np.isfinite(catboost) & np.isfinite(chronos)
        if not valid.any():
            self.alpha_ = self.config.default_alpha
            return self.alpha_
        actual, catboost, chronos = actual[valid], catboost[valid], chronos[valid]
        mae_catboost = float(mean_absolute_error(actual, catboost))
        mae_chronos = float(mean_absolute_error(actual, chronos))
        low, high = self.config.alpha_bounds
        if mae_chronos > mae_catboost * 1.5:
            low = max(low, 0.70)

        def objective(alpha: float) -> float:
            return float(mean_absolute_error(actual, alpha * catboost + (1.0 - alpha) * chronos))

        result = minimize_scalar(objective, bounds=(low, high), method="bounded")
        self.alpha_ = float(np.clip(result.x if result.success else self.config.default_alpha, low, high))
        return self.alpha_

    def predict(self, preds_catboost: ArrayLike, preds_chronos: ArrayLike) -> NDArray[np.float64]:
        catboost = self._finite(preds_catboost)
        chronos = self._finite(preds_chronos)
        if catboost.shape != chronos.shape:
            raise ValueError("Прогнозы CatBoost и Chronos должны иметь одинаковую форму")
        blended = self.alpha_ * catboost + (1.0 - self.alpha_) * chronos
        invalid = ~np.isfinite(blended)
        blended[invalid] = catboost[invalid]
        if not np.isfinite(blended).all():
            raise ValueError("CatBoost fallback содержит NaN/Inf")
        return np.clip(blended, a_min=0.0, a_max=None)
