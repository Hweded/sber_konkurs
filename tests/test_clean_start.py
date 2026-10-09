"""Контракт безопасной очистки производных результатов."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


from src.clean_start import clean_generated_outputs
from src.configuration import load_config


def test_clean_start_removes_outputs_and_preserves_sources(tmp_path: Path) -> None:
    base = load_config(Path("configs/config.yaml"))
    config = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "supervised": tmp_path / "data" / "processed" / "supervised.parquet",
                    "artifacts": tmp_path / "reports" / "artifacts",
                }
            ),
            "changepoint_detection": base.changepoint_detection.model_copy(
                update={
                    "predictions": str(tmp_path / "reports" / "predictions_oof.parquet"),
                    "output": str(tmp_path / "reports" / "detected_shocks.parquet"),
                    "figures": str(tmp_path / "reports" / "figures"),
                }
            ),
            "tda": base.tda.model_copy(
                update={
                    "cache_path": tmp_path / "data" / "processed" / "tda_features.parquet",
                }
            ),
            "data": base.data.model_copy(
                update={
                    "nlp": base.data.nlp.model_copy(
                        update={
                            "cache": tmp_path
                            / "data"
                            / "processed"
                            / "news_monthly_features.parquet",
                        }
                    ),
                }
            ),
        }
    )
    generated = [
        tmp_path / "data" / "processed" / "supervised.parquet",
        tmp_path / "data" / "processed" / "news_features.parquet",
        tmp_path / "reports" / "artifacts" / "metrics.json",
        tmp_path / "reports" / "figures" / "plot.png",
        tmp_path / "submission.csv",
    ]
    for path in generated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated", encoding="utf-8")
    source = tmp_path / "datasets" / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")

    removed = clean_generated_outputs(config, project_root=tmp_path)

    assert all(not path.exists() for path in generated)
    assert any(path.name == "artifacts" for path in removed)
    assert any(path.name == "figures" for path in removed)
    assert source.exists()


def test_clean_start_rejects_paths_outside_project(tmp_path: Path) -> None:
    base = load_config(Path("configs/config.yaml"))
    config = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "supervised": Path("C:/outside/supervised.parquet"),
                    "artifacts": tmp_path / "artifacts",
                }
            ),
        }
    )

    try:
        clean_generated_outputs(config, project_root=tmp_path)
    except ValueError as error:
        assert "за пределы проекта" in str(error)
    else:
        raise AssertionError("Ожидалась блокировка пути вне проекта")


def test_clean_start_rejects_project_root_as_artifacts(tmp_path: Path) -> None:
    base = load_config(Path("configs/config.yaml"))
    config = base.model_copy(
        update={
            "paths": base.paths.model_copy(update={"artifacts": tmp_path}),
        }
    )

    try:
        clean_generated_outputs(config, project_root=tmp_path)
    except ValueError as error:
        assert "clean-start path" in str(error)
    else:
        raise AssertionError("Нельзя разрешать очистку корня проекта")


def test_pipeline_lock_rejects_second_process_before_cleanup(tmp_path: Path) -> None:
    from src.clean_start import pipeline_lock

    artifact = tmp_path / "keep.txt"
    artifact.write_text("keep", encoding="utf-8")
    script = (
        "from pathlib import Path; from src.clean_start import pipeline_lock; "
        "root=Path(__import__('sys').argv[1]); "
        "\nwith pipeline_lock(root): (root/'keep.txt').unlink()"
    )
    with pipeline_lock(tmp_path):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
        assert child.returncode != 0
        assert "Пайплайн уже запущен" in child.stderr
        assert artifact.read_text(encoding="utf-8") == "keep"
    with pipeline_lock(tmp_path):
        assert artifact.exists()


def test_main_rejects_clean_start_while_project_is_locked() -> None:
    from src.clean_start import pipeline_lock

    root = Path.cwd()
    config = root / "configs/config.yaml"
    before = config.read_bytes()
    with pipeline_lock(root):
        child = subprocess.run(
            [sys.executable, "main.py", "--all", "--clean-start"],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
        assert child.returncode == 1
        assert "Пайплайн уже запущен" in child.stderr
        assert "START prepare_data" not in child.stderr
        assert config.read_bytes() == before
