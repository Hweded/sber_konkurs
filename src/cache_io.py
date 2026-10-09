"""Загрузка проверенных frozen-origin кэшей для режима ``--from-cache``."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CACHE_ERROR = (
    "Кэш базовых прогнозов не найден в reports/artifacts/. "
    "Сначала выполните базовый расчет или предоставьте frozen-origin артефакты."
)

#: канонические имена, которых ждёт ТЗ
CANONICAL_OOF = ("oof_predictions.parquet", "oof_predictions.csv")
CANONICAL_TEST = ("test_predictions.parquet", "test_predictions.csv")
#: фактические имена в этом репозитории
REPO_OOF = (
    "reports/predictions_oof.parquet",
    "reports/artifacts/predictions_oof.parquet",
    "reports/artifacts/oof_predictions.parquet",
)
REPO_TEST = (
    "reports/predictions_submission.parquet",
    "reports/submission_predictions.parquet",
    "reports/artifacts/test_predictions.parquet",
)
#: раздельные кэши по моделям
PER_MODEL = {
    "pred_prophet": ("prophet_preds.parquet", "prophet_preds.csv"),
    "pred_catboost": ("catboost_preds.parquet", "catboost_preds.csv"),
    "pred_chronos": ("chronos_preds.parquet", "chronos_preds.csv"),
}
KEY = ["mo", "period"]


def _candidate_cache_roots(cache_dir: Path) -> tuple[Path, ...]:
    """Return only existing cache roots, keeping the caller's directory first."""
    roots = [cache_dir]
    if cache_dir.resolve() == (ROOT / "reports" / "artifacts").resolve():
        roots.extend([ROOT / "reports", ROOT / "reports" / "artifacts"])
    return tuple(dict.fromkeys(path.resolve() for path in roots if path.exists()))


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["period"] = pd.to_datetime(frame["period"], errors="raise").dt.tz_localize(None)
    if "lead_months" not in frame and "horizon" in frame:
        frame = frame.rename(columns={"horizon": "lead_months"})
    return frame


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_paths(
    cache_dir: Path, names: tuple[str, ...], repo_names: tuple[str, ...]
) -> list[Path]:
    """Ищи канонические и репозиторные пути, не смешивая OOF и test."""
    return [cache_dir / name for name in names] + [ROOT / name for name in repo_names]


def _load_per_model(cache_dir: Path) -> pd.DataFrame | None:
    """Склеить раздельные кэши моделей в один файл, если единого нет."""
    found: dict[str, Path] = {}
    for column, names in PER_MODEL.items():
        path = _first_existing([cache_dir / name for name in names])
        if path is not None:
            found[column] = path
    if len(found) != len(PER_MODEL):
        return None
    merged: pd.DataFrame | None = None
    for column, path in found.items():
        frame = _read(path)
        if not set(KEY).issubset(frame.columns):
            raise ValueError(f"{path.name}: нет ключей {KEY}")
        columns = KEY + [column]
        # Оставляем одну target: дублирование создаст target_x/target_y.
        if "target" in frame.columns and merged is None:
            columns.append("target")
        columns += ["lead_months"] if "lead_months" in frame.columns else []
        columns += ["fold"] if "fold" in frame.columns else []
        frame = frame[list(dict.fromkeys(columns))]
        merged = frame if merged is None else merged.merge(frame, on=KEY, how="outer")
    assert merged is not None
    LOGGER.warning("Кэш собран из раздельных моделей: %s", sorted(found))
    return merged


def _require_columns(frame: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: в кэше отсутствуют колонки {missing}")


def load_cached_predictions(
    cache_dir: str | Path = "reports/artifacts/",
    *,
    fast_mode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Вернуть ``(oof, test)`` из кэша. Бросает ``FileNotFoundError`` с понятным текстом."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    # Репозиторные пути допустимы только для стандартного каталога.
    default_dir = (ROOT / "reports" / "artifacts").resolve()
    repo_oof = REPO_OOF if cache_dir.resolve() == default_dir else ()
    repo_test = REPO_TEST if cache_dir.resolve() == default_dir else ()
    roots = _candidate_cache_roots(cache_dir)
    oof_path = _first_existing(
        [
            path
            for root in roots
            for path in _candidate_paths(
                root, CANONICAL_OOF, repo_oof if root == cache_dir.resolve() else REPO_OOF
            )
        ]
    )
    test_path = _first_existing(
        [
            path
            for root in roots
            for path in _candidate_paths(
                root, CANONICAL_TEST, repo_test if root == cache_dir.resolve() else REPO_TEST
            )
        ]
    )
    if oof_path is None or test_path is None:
        merged = _load_per_model(cache_dir)
        if merged is not None:
            frame = merged
            if "target" not in frame:
                raise ValueError(
                    "Раздельные кэши моделей должны содержать target для OOF "
                    "и строки без target для test."
                )
            oof = frame.loc[frame["target"].notna()].copy()
            test = frame.loc[frame["target"].isna()].copy()
            if oof.empty or test.empty:
                raise ValueError("Раздельные кэши моделей должны содержать и OOF, и test строки.")
            _require_columns(
                oof, KEY + ["target", "pred_catboost", "pred_prophet", "pred_chronos"], "OOF-кэш"
            )
            _require_columns(
                test, KEY + ["pred_catboost", "pred_prophet", "pred_chronos"], "test-кэш"
            )
            LOGGER.info(
                "Кэш собран из раздельных моделей: OOF=%d строк, test=%d строк", len(oof), len(test)
            )
            return oof, test
    if oof_path is None or test_path is None:
        raise FileNotFoundError(CACHE_ERROR)

    oof = _read(oof_path)
    test = _read(test_path)
    _require_columns(
        oof,
        KEY + ["target", "pred_catboost", "pred_prophet", "pred_chronos"],
        f"OOF-кэш {oof_path.name}",
    )
    _require_columns(
        test, KEY + ["pred_catboost", "pred_prophet", "pred_chronos"], f"test-кэш {test_path.name}"
    )
    LOGGER.info(
        "Кэш загружен: OOF=%s (%d строк), test=%s (%d строк)",
        oof_path.name,
        len(oof),
        test_path.name,
        len(test),
    )
    return oof, test


def cached_paths(cache_dir: str | Path = "reports/artifacts/") -> dict[str, str | None]:
    """Что именно нашлось — для протокола и диагностики."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    default_dir = (ROOT / "reports" / "artifacts").resolve()
    repo_oof = REPO_OOF if cache_dir.resolve() == default_dir else ()
    repo_test = REPO_TEST if cache_dir.resolve() == default_dir else ()
    oof = _first_existing(_candidate_paths(cache_dir, CANONICAL_OOF, repo_oof))
    test = _first_existing(_candidate_paths(cache_dir, CANONICAL_TEST, repo_test))

    def rel(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    return {"oof": rel(oof), "test": rel(test), "cache_dir": rel(cache_dir)}
