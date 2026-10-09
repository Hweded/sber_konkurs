"""Причинные блендеры для CatBoost, Chronos-Bolt и Prophet."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error

LOGGER = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    """Строгая конфигурация weighted-average ансамбля."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    method: Literal["weighted_average", "ridge_stacking"] = "weighted_average"
    metric_to_optimize: Literal["MAE"] = "MAE"
    alpha_bounds: tuple[float, float] = (0.5, 1.0)
    default_alpha: float = Field(default=0.82, ge=0.0, le=1.0)
    # Заводская смесь трёх моделей до OOF-калибровки.
    prophet_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    catboost_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    chronos_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    long_horizon_catboost_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    # OOF-веса действуют и в OOF, и в frozen-origin submission.
    weights_path: Path | None = None

    @model_validator(mode="after")
    def check_bounds(self) -> EnsembleConfig:
        low, high = self.alpha_bounds
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("alpha_bounds должны удовлетворять 0 <= low < high <= 1")
        if not low <= self.default_alpha <= high:
            raise ValueError("default_alpha должен находиться внутри alpha_bounds")
        if self.prophet_weight + self.catboost_weight + self.chronos_weight <= 0:
            raise ValueError("Сумма базовых весов ансамбля должна быть положительной")
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
        self.alpha_ = float(
            np.clip(result.x if result.success else self.config.default_alpha, low, high)
        )
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


class RegimeAwareBlender:
    """Трёхкомпонентный MAE-блендер с отдельными весами по горизонту.

    Веса калибруются только на уже закрытой OOF-панели.  Для длинных
    горизонтов используется CatBoost-heavy режим, потому что у Prophet и
    zero-shot Chronos на этих горизонтах быстро растёт ошибка тренда.
    """

    DEFAULT_WEIGHTS: dict[int, tuple[float, float, float]] = {
        # Prophet, CatBoost без news и Chronos на измеренной OOF-панели.
        1: (0.5741, 0.4258, 0.0001),
        3: (0.6876, 0.1934, 0.1190),
        6: (0.8220, 0.1779, 0.0001),
        12: (0.0, 1.0, 0.0),
    }

    def __init__(self, weights: dict[int, Sequence[float]] | None = None) -> None:
        source = weights or self.DEFAULT_WEIGHTS
        self.weights = {int(horizon): self._normalise(values) for horizon, values in source.items()}

    @staticmethod
    def _finite(values: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64).reshape(-1)

    @staticmethod
    def _normalise(values: Sequence[float]) -> tuple[float, float, float]:
        if len(values) != 3:
            raise ValueError("Нужны три веса: Prophet, CatBoost, Chronos")
        values = np.maximum(np.asarray(values, dtype=float), 0.0)
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Сумма весов ансамбля должна быть положительной")
        result = values / total
        return tuple(float(value) for value in result)

    def fit(self, frame: Any, *, horizons: Sequence[int] = (1, 3, 6, 12)) -> "RegimeAwareBlender":
        """Подбирает неотрицательные веса по MAE на закрытой OOF-панели."""
        from scipy.optimize import minimize

        required = {"target", "pred_prophet", "pred_catboost", "pred_chronos"}
        if not required.issubset(frame):
            raise ValueError(f"OOF не содержит {sorted(required - set(frame))}")
        groups = (
            frame.groupby("lead_months", sort=False) if "lead_months" in frame else [(1, frame)]
        )
        for horizon, group in groups:
            horizon = int(horizon)
            if horizon not in horizons:
                continue
            valid = group[list(required)].apply(np.isfinite).all(axis=1)
            group = group.loc[valid]
            if group.empty:
                continue
            y = group["target"].to_numpy(dtype=float)
            x = group[["pred_prophet", "pred_catboost", "pred_chronos"]].to_numpy(dtype=float)
            result = minimize(
                lambda weights: float(np.abs(y - x @ weights).mean()),
                np.asarray(self.weights.get(horizon, (0.45, 0.35, 0.20))),
                bounds=[(0.0, 1.0)] * 3,
                constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
                method="SLSQP",
            )
            if result.success:
                self.weights[horizon] = self._normalise(result.x)
        return self

    def weights_for(self, horizon: int, *, alert: bool = False) -> tuple[float, float, float]:
        horizon = int(horizon)
        if horizon in self.weights:
            prophet, catboost, chronos = self.weights[horizon]
        elif horizon >= 6:
            prophet, catboost, chronos = (0.20, 0.75, 0.05)
        else:
            prophet, catboost, chronos = (0.45, 0.35, 0.20)
        if alert:
            # При шоке CatBoost получает основной вес, Prophet/Chronos стабилизируют прогноз.
            catboost = max(catboost, 0.80 if horizon >= 6 else 0.70)
            remainder = max(0.0, 1.0 - catboost)
            prophet, chronos = remainder * 0.75, remainder * 0.25
        return self._normalise((prophet, catboost, chronos))

    def predict(
        self,
        prophet: ArrayLike,
        catboost: ArrayLike,
        chronos: ArrayLike,
        *,
        horizon: int = 1,
        alerts: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        arrays = [self._finite(values) for values in (prophet, catboost, chronos)]
        if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
            raise ValueError("Прогнозы ансамбля должны иметь одинаковую форму")
        alert_values = (
            np.zeros(arrays[0].shape, dtype=bool)
            if alerts is None
            else np.asarray(alerts, dtype=bool).reshape(-1)
        )
        if alert_values.shape != arrays[0].shape:
            raise ValueError("alerts должен совпадать по форме с прогнозами")
        matrix = np.column_stack(arrays)
        calm = np.asarray(self.weights_for(horizon), dtype=float)
        shock = np.asarray(self.weights_for(horizon, alert=True), dtype=float)
        output = matrix @ calm
        if alert_values.any():
            output[alert_values] = matrix[alert_values] @ shock
        if not np.isfinite(output).all():
            raise ValueError("Ансамбль получил неконечный прогноз")
        return np.clip(output, a_min=1e-6, a_max=None)


def load_blend_weights(
    path: str | Path | None,
    *,
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> dict[int, Sequence[float]] | None:
    """Load measured ``{horizon: [prophet, catboost, chronos]}`` OOF weights.

    Returns ``None`` when no path is configured or the file is absent, which
    keeps the factory :data:`RegimeAwareBlender.DEFAULT_WEIGHTS` in force.
    """
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        LOGGER.warning("Файл калиброванных весов %s не найден: используем factory-веса", resolved)
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved}: ожидался объект {{horizon: [w1, w2, w3]}}")
    weights: dict[int, Sequence[float]] = {}
    for key, values in payload.items():
        if key in {"source", "protocol", "timestamp", "metric"}:
            continue
        try:
            horizon = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{resolved}: горизонт {key!r} не является целым") from error
        if horizon not in horizons:
            continue
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{resolved}: для h={horizon} нужны три веса")
        weights[horizon] = RegimeAwareBlender._normalise(values)
    if not weights:
        LOGGER.warning("Файл весов %s не содержит горизонтов %s", resolved, list(horizons))
        return None
    LOGGER.info(
        "Blend weights из %s: %s",
        resolved,
        ", ".join(f"h={h}:{tuple(round(v, 4) for v in w)}" for h, w in sorted(weights.items())),
    )
    return weights


def blender_from_config(config: EnsembleConfig) -> "RegimeAwareBlender":
    """Build the production blender, preferring the calibrated weight file.

    Horizons missing from the file keep the factory-calibrated
    :data:`RegimeAwareBlender.DEFAULT_WEIGHTS` instead of dropping to the crude
    ``horizon >= 6`` fallback branch.
    """
    weights = load_blend_weights(config.weights_path)
    if not weights:
        return RegimeAwareBlender()
    merged = dict(RegimeAwareBlender.DEFAULT_WEIGHTS)
    merged.update(weights)
    return RegimeAwareBlender(merged)


# Имя публичного блендера для API поставки.
RegimeAwareEnsemble = RegimeAwareBlender
