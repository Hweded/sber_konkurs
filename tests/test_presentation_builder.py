"""Проверки необязательного конкурсного файла при создании PDF."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.presentation_builder import build_presentation


def _skip_image(figure: Figure, path: Path, bounds: tuple[float, float, float, float]) -> None:
    """Изолирует сборку слайдов от наличия PNG в тестовой директории."""
    return None


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    figures = tmp_path / "figures"
    artifacts.mkdir()
    figures.mkdir()
    pd.DataFrame({"fold": ["pooled"], "model": ["catboost"], "n": [10],
                  "MAE": [1.0], "R2": [0.5]}).to_csv(artifacts / "forecast_metrics.csv", index=False)
    pd.DataFrame({"fold": ["pooled"], "scope": ["exact"], "model": ["catboost"],
                  "horizon": [1], "n": [10], "status": ["измерено"]}).to_csv(
                      artifacts / "forecast_metrics_by_horizon.csv", index=False)
    pd.DataFrame({"method": ["CUSUM"], "precision": [1.0], "recall": [1.0],
                  "f1": [1.0], "mean_lead_time_days": [0]}).to_csv(
                      artifacts / "changepoint_validation.csv", index=False)
    return artifacts, figures


def test_presentation_without_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts, figures = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.presentation_builder._image", _skip_image)
    result = build_presentation(artifacts, figures, tmp_path / "presentation.pdf")
    assert result.is_file()
    assert result.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "submission.csv").exists()


def test_presentation_rejects_invalid_existing_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts, figures = _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"mo": ["a"], "period": ["2024-07-01"], "pred": [1.0]}).to_csv(
        tmp_path / "submission.csv", index=False)
    with pytest.raises(ValueError, match="Конкурсный сабмит"):
        build_presentation(artifacts, figures, tmp_path / "presentation.pdf")
