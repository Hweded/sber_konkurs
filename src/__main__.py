"""CLI: сборка месячной матрицы, проверка фолдов и табличная CV."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.configuration import load_config
from src.validation import ValidationConfig, cross_validate, expanding_window_splits
from src.features import build_dataset, to_validation_panel

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-config", "build-dataset", "validate", "cv-catboost"))
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(resolve_path(args.config))
        if args.command == "check-config":
            print(config.model_dump_json(indent=2))
            return 0
        seed = config.reproducibility.seed
        random.seed(seed)
        np.random.seed(seed)
        os.environ["OMP_NUM_THREADS"] = str(config.reproducibility.threads)
        if args.command == "build-dataset":
            frame = build_dataset(resolve_path(config.data.directory), config.data, config.features)
            output = resolve_path(config.paths.supervised)
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output, index=False)
            LOGGER.info("Матрица %s сохранена: %s", frame.shape, output)
            return 0
        frame = pd.read_parquet(resolve_path(config.paths.supervised))
        extra = sorted(column for column in frame.columns if column.startswith(("macro_", "rosstat_")) or column in ("news_sentiment", "news_shock_index", "sentiment_index", "news_volume", "news_shock_score"))
        settings = config.validation.model_dump()
        settings["feature_columns"] = tuple(dict.fromkeys([*config.validation.feature_columns, *extra]))
        validation = ValidationConfig.model_validate(settings)
        frame = to_validation_panel(frame, validation)
        if args.command == "validate":
            folds = expanding_window_splits(frame, validation)
            print(json.dumps([fold.model_dump(exclude={"train_positions", "test_positions"}) | {"n_train": len(fold.train_positions), "n_test": len(fold.test_positions)} for fold in folds], indent=2))
            return 0
        if config.optimization.enabled:
            raise ValueError("Optuna не реализован на шаге 1; отключите optimization.enabled")
        from catboost import CatBoostRegressor

        parameters = dict(config.models.catboost)
        parameters.update(random_seed=seed, thread_count=config.reproducibility.threads)
        report = cross_validate(frame, validation, CatBoostRegressor(**parameters))
        destination = resolve_path(config.paths.artifacts)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "catboost_cv.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        resolved = config.model_copy(update={"validation": validation})
        (destination / "config.resolved.json").write_text(resolved.model_dump_json(indent=2), encoding="utf-8")
        LOGGER.info("Отчёт записан в %s", destination)
        return 0
    except Exception:
        LOGGER.exception("Команда завершилась ошибкой")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
