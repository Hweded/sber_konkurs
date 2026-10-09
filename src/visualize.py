"""Презентационные графики месячного ряда и причинных сигналов, 300 DPI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = ("pelt", "binseg", "cusum", "residual", "residual_news")


def plot_case_study(frame: pd.DataFrame, entity: str, directory: Path) -> Path:
    """Рисует сигналы на дате фактической доступности, если она предоставлена."""
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    try:
        axes[0].plot(frame.period, frame.y, label="Фактические расходы", color="black")
        axes[0].plot(frame.period, frame.prediction, label="CatBoost OOF", color="tab:blue")
        colors = ("tab:red", "tab:purple", "tab:orange", "tab:green", "tab:brown")
        for method, color in zip(METHODS, colors, strict=True):
            dates = frame.loc[
                frame[f"{method}_shock"].eq(1),
                "detected_at" if "detected_at" in frame else "period",
            ]
            for number, date in enumerate(dates):
                axes[0].axvline(
                    date,
                    color=color,
                    alpha=0.55,
                    linestyle="--",
                    label=method if number == 0 else None,
                )
        axes[0].set_title(f"{entity}: расходы, прогноз и даты обнаружения")
        axes[0].set_ylabel("Расходы, исходные единицы")
        axes[0].legend(loc="best", ncol=3)
        if "sentiment_index" in frame:
            axes[1].plot(
                frame.period,
                frame.sentiment_index,
                color="tab:blue",
                label="Лагированный сентимент",
            )
        axes[1].set_ylabel("P(positive) − P(negative)")
        secondary = axes[1].twinx()
        if "news_shock_score" in frame:
            secondary.plot(
                frame.period,
                frame.news_shock_score,
                color="tab:red",
                label="Лагированный новостной ажиотаж",
            )
        secondary.set_ylabel("Объём минус прошлое среднее")
        handles, labels = axes[1].get_legend_handles_labels()
        other, names = secondary.get_legend_handles_labels()
        if handles or other:
            axes[1].legend(handles + other, labels + names, loc="best")
        axes[1].set_xlabel("Месяц; отсутствие новостей не означает нейтральность")
        for axis in axes:
            axis.grid(alpha=0.2)
        name = hashlib.sha256(entity.encode()).hexdigest()[:12]
        path = directory / f"case_study_{name}.png"
        figure.savefig(path, dpi=150)
        return path
    finally:
        plt.close(figure)
