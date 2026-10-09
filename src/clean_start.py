"""Удаление производных результатов перед воспроизводимым запуском."""

from __future__ import annotations

import shutil
import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from src.configuration import AppConfig


@contextmanager
def pipeline_lock(root: Path) -> Iterator[None]:
    """Запрети параллельную запись и очистку артефактов одного проекта."""
    path = root.resolve() / ".pipeline.lock"
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "Пайплайн уже запущен в этом проекте. Дождитесь завершения или "
                "остановите предыдущий прогон; --clean-start сейчас запрещён."
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_target(path: Path, root: Path) -> bool:
    """Запрещает удалять корень проекта, исходные данные и конфигурацию."""
    resolved = path.resolve()
    if resolved == root.resolve():
        return False
    protected = (root / "datasets", root / "configs", root / "src")
    return all(resolved != item.resolve() for item in protected)


def _files_to_remove(config: AppConfig, root: Path) -> tuple[Path, ...]:
    """Возвращает явный список производных файлов и каталогов для удаления."""
    artifacts = Path(config.paths.artifacts)
    figures = Path(config.changepoint_detection.figures)
    candidates = [
        Path(config.paths.supervised),
        Path(config.data.nlp.cache),
        Path(config.data.nlp.cache).with_suffix(".manifest.json"),
        Path("data/processed/news_features.parquet"),
        Path("data/processed/news_features.manifest.json"),
        Path(config.tda.cache_path),
        Path(config.tda.cache_path).with_suffix(".manifest.json"),
        Path(config.changepoint_detection.predictions),
        Path(config.changepoint_detection.output),
        Path("reports/detected_shocks.metrics.csv"),
        Path("reports/detected_shocks.summary.json"),
        Path("reports/predictions_submission.parquet"),
        Path("reports/submission.csv"),
        Path("submission.csv"),
        Path("metrics.json"),
        Path("presentation.pdf"),
        Path("deliverables/presentation.pdf"),
        Path("deliverables/presentation.pptx"),
        artifacts,
        figures,
    ]
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else root / candidate
        if not _under_root(path, root) or not _safe_target(path, root):
            raise ValueError(f"clean-start path выходит за пределы проекта: {path}")
        if path not in resolved:
            resolved.append(path)
    return tuple(resolved)


def clean_generated_outputs(config: AppConfig, *, project_root: Path | None = None) -> list[Path]:
    """Удаляет только разрешённые артефакты; исходные данные и roster не трогает."""
    root = (project_root or Path.cwd()).resolve()
    removed: list[Path] = []
    for path in _files_to_remove(config, root):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed


__all__ = ["clean_generated_outputs"]
