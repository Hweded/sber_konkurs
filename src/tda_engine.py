"""TDA-признаки месячных расходов на причинных скользящих окнах."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=UserWarning, module="ripser")

TDA_COLUMNS = ("tda_entropy", "tda_wasserstein_dist", "tda_shock")


class TDAConfig(BaseModel):
    """Параметры топологического анализа."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    window_size: PositiveInt = Field(default=5, ge=3, le=24)
    embedding_dimension: PositiveInt = Field(default=3, ge=2, le=10)
    time_delay: PositiveInt = Field(default=1, ge=1, le=5)
    homology_dimensions: tuple[int, ...] = (0, 1)
    metric: str = "euclidean"
    cache_path: Path = Field(default=Path("data/processed/tda_features.parquet"))
    shock_threshold_std: float = Field(default=2.5, gt=0)


class TopologicalAnalyzer:
    """Считает TDA-признаки одного ряда без подглядывания в текущий месяц."""

    def __init__(self, config: TDAConfig) -> None:
        self.config = config
        self._rips: Any = None

    def _get_rips(self) -> Any:
        if self._rips is None:
            from ripser import Rips

            self._rips = Rips(maxdim=max(self.config.homology_dimensions))
        return self._rips

    @staticmethod
    def _takens_embedding(
        series: NDArray[np.float64],
        dimension: int,
        delay: int,
    ) -> NDArray[np.float64]:
        """Преобразует одномерный ряд во вложение Такенса."""
        n = len(series)
        required = (dimension - 1) * delay + 1
        if n < required:
            raise ValueError(
                f"Ряд длиной {n} слишком короток для d={dimension}, τ={delay} "
                f"(нужно минимум {required})"
            )
        indices = np.arange(n - required + 1)[:, None] + np.arange(dimension)[None, :] * delay
        return series[np.flip(indices, axis=1)]

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        """Возвращает лагированные энтропию и расстояние Вассерштейна."""
        from sklearn.preprocessing import StandardScaler

        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("Индекс должен быть DatetimeIndex")
        values = series.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Ряд содержит неконечные значения")
        if len(values) < self.config.window_size:
            raise ValueError(f"Ряд длиной {len(values)} короче окна {self.config.window_size}")

        ws = self.config.window_size
        d = self.config.embedding_dimension
        tau = self.config.time_delay
        scaler = StandardScaler()
        rips = self._get_rips()
        hom_dims = list(self.config.homology_dimensions)

        entropy_values: list[float | None] = []
        wasserstein_values: list[float | None] = []
        previous_diagram: NDArray[np.float64] | None = None

        for t in range(ws - 1, len(values)):
            window = values[t - ws + 1 : t + 1].copy()
            window = (window - window.mean()) / (window.std(ddof=1) + 1e-10)

            # На коротких окнах H₁ часто пуст, поэтому откатываемся к конечным H₀.
            topology_diagram: NDArray[np.float64] = np.empty((0, 2), dtype=np.float64)
            try:
                embedded = self._takens_embedding(window, d, tau)
                embedded = np.asarray(scaler.fit_transform(embedded), dtype=np.float64)
                # Квадратный point cloud не является матрицей расстояний.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module=r"ripser(\\..*)?",
                        message=r"The input matrix is square.*",
                    )
                    diagrams = rips.fit_transform(embedded, distance_matrix=False)

                h1 = (
                    np.asarray(diagrams[1], dtype=np.float64)
                    if 1 in hom_dims and len(diagrams) > 1
                    else np.empty((0, 2), dtype=np.float64)
                )
                selected = h1 if len(h1) > 0 else np.asarray(diagrams[0], dtype=np.float64)
                topology_diagram = selected[np.isfinite(selected).all(axis=1), :2]
            except Exception:
                LOGGER.debug(
                    "TDA: ошибка в окне t=%s, MO=%s",
                    series.index[t],
                    getattr(series, "name", "?"),
                    exc_info=True,
                )

            if len(topology_diagram) > 0:
                from persim.persistent_entropy import persistent_entropy

                entropy_result = persistent_entropy([topology_diagram])
                entropy = float(np.asarray(entropy_result, dtype=np.float64).reshape(-1)[0])
                entropy_values.append(entropy if np.isfinite(entropy) else None)
            else:
                entropy_values.append(None)

            if previous_diagram is not None and len(topology_diagram) > 0:
                from persim import wasserstein

                distance = float(wasserstein(topology_diagram, previous_diagram))
                wasserstein_values.append(distance if np.isfinite(distance) else None)
            else:
                wasserstein_values.append(None)

            if len(topology_diagram) > 0:
                previous_diagram = topology_diagram

        result_index = series.index[ws - 1 :]
        result = pd.DataFrame(index=result_index)
        result["tda_entropy"] = np.asarray(entropy_values, dtype=np.float64)
        result["tda_wasserstein_dist"] = np.asarray(wasserstein_values, dtype=np.float64)

        # Признак окна, закрывшегося в t, становится доступен только в t+1.
        result = result.shift(1)
        result.index.name = "period"
        result = result.reset_index()
        return result

    def detect_shocks(
        self,
        wasserstein_series: pd.Series,
        threshold: float | None = None,
    ) -> pd.Series:
        """Ставит онлайн-алерт по порогу, оценённому только на прошлом."""
        n_std = threshold if threshold is not None else self.config.shock_threshold_std
        values = wasserstein_series.to_numpy(dtype=np.float64)
        result = pd.Series(False, index=wasserstein_series.index, name="tda_shock")
        for i in range(3, len(values)):
            history = values[:i]
            if np.isfinite(history).sum() < 3:
                continue
            mu = float(np.nanmean(history))
            sigma = float(np.nanstd(history, ddof=1))
            if sigma < 1e-10:
                continue
            result.iloc[i] = bool(values[i] > mu + n_std * sigma)
        return result


def build_tda_features(
    source: Path | pd.DataFrame,
    config: TDAConfig,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Считает TDA-признаки всех МО и атомарно обновляет кэш."""
    cache = config.cache_path if config.cache_path.is_absolute() else ROOT / config.cache_path
    manifest_path = cache.with_suffix(".manifest.json")

    if not force and cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_config = manifest.get("config")
        if cached_config == config.model_dump(mode="json"):
            cached = pd.read_parquet(cache)
            if set(TDA_COLUMNS).issubset(cached.columns):
                LOGGER.info("TDA: загружен кэш %s, %d строк", cache, len(cached))
                return cached

    if isinstance(source, pd.DataFrame):
        data = source.copy()
    else:
        if not source.exists():
            raise FileNotFoundError(f"Панель для TDA не найдена: {source}")
        data = pd.read_parquet(source)
    if "mo" not in data or "period" not in data or "y" not in data:
        raise ValueError("Панель TDA должна содержать mo, period, y")
    data["period"] = pd.to_datetime(data["period"], errors="raise")

    # TDA необязательна: без ripser/persim ETL продолжает работать.
    try:
        import ripser  # noqa: F401
        import persim  # noqa: F401
    except ImportError as error:
        LOGGER.warning("TDA пропущен: необязательная зависимость недоступна (%s)", error)
        return pd.DataFrame(
            columns=["period", "mo", "tda_entropy", "tda_wasserstein_dist", "tda_shock"]
        )

    analyzer = TopologicalAnalyzer(config)
    parts: list[pd.DataFrame] = []
    mo_count = data["mo"].nunique()
    LOGGER.info("TDA: начало обработки %d МО", mo_count)

    for idx, (mo, group) in enumerate(data.groupby("mo", sort=True, observed=True)):
        ts = group.set_index("period")["y"].sort_index()
        if len(ts) < config.window_size:
            LOGGER.debug(
                "TDA: МО %s пропущен (длина %d < окна %d)", mo, len(ts), config.window_size
            )
            continue
        try:
            features = analyzer.fit_transform(ts)
            features["mo"] = str(mo)
            parts.append(features)
        except Exception:
            LOGGER.warning("TDA: ошибка для МО %s", mo, exc_info=True)
        if (idx + 1) % 200 == 0:
            LOGGER.info("TDA: обработано %d/%d МО", idx + 1, mo_count)

    if not parts:
        raise ValueError("TDA: ни один МО не прошёл обработку")

    result = pd.concat(parts, ignore_index=True)
    result = result.rename(columns={"index": "period"}) if "index" in result else result

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(cache)
    manifest_path.write_text(
        json.dumps(
            {"config": config.model_dump(mode="json"), "n_mo": result["mo"].nunique()},
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("TDA: кэш сохранён %s, %d строк, %d МО", cache, len(result), result["mo"].nunique())
    return result
