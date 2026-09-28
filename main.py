"""Единый последовательный runner; запускать из корня репозитория."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

LOGGER = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--all", action="store_true")
    available = ("prepare_data", "train_forecast", "detect_shocks", "generate_report", "submission", "build_presentation")
    for stage in available:
        parser.add_argument("--" + stage.replace("_", "-"), action="store_true")
    args = parser.parse_args()
    stages = [stage for stage in available if args.all or getattr(args, stage)]
    if not stages:
        parser.error("Выберите --all или хотя бы один этап")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        # NOTE: hash seed читается при старте Python, поэтому один раз перезапускаемся.
        import yaml
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        repro = raw["reproducibility"]
        seed = int(repro["seed"])
        if os.environ.get("PYTHONHASHSEED") != str(seed):
            environment = dict(os.environ, PYTHONHASHSEED=str(seed))
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=environment, check=False).returncode
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(repro["threads"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import pandas as pd
        from src.configuration import load_config
        config = load_config(args.config)
        random.seed(seed)
        np.random.seed(seed)
        if ("prepare_data" in stages and config.data.nlp.enabled) or "train_forecast" in stages:
            import torch
            torch.manual_seed(seed)
            torch.set_num_threads(config.reproducibility.threads)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(config.reproducibility.deterministic_torch)
            torch.backends.cudnn.benchmark = False
        from src.device import resolve_device
        device_info = resolve_device(config.device.mode)
        LOGGER.info("Выбрано устройство pipeline: %s", device_info.device)
        config.paths.artifacts.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(config.paths.artifacts / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        (config.paths.artifacts / "config.resolved.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
        timings: dict[str, float] = {}
        for stage in stages:
            started = perf_counter()
            LOGGER.info("START %s", stage)
            if stage == "prepare_data":
                from src.features import build_dataset
                frame = build_dataset(
                    config.data.directory,
                    config.data,
                    config.features,
                    tda_config=config.tda,
                )
                config.paths.supervised.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(config.paths.supervised, index=False)
            elif stage == "train_forecast":
                from src.forecasting import forecast_multi_horizon, metric_table, save_horizon_metrics
                oof = forecast_multi_horizon(config, config.validation.horizons)
                metrics = metric_table(oof.loc[oof["lead_months"].eq(1)])
                horizons = save_horizon_metrics(
                    oof, config.paths.artifacts, Path(config.changepoint_detection.figures),
                    config.validation.horizons,
                )
                pooled = horizons.loc[horizons["fold"].eq("pooled") & horizons["scope"].eq("exact")].copy()
                pooled["fold"] = pooled["horizon"].map(lambda h: f"h={h}")
                metrics = pd.concat([metrics, pooled], ignore_index=True)
                metrics.to_csv(config.paths.artifacts / "forecast_metrics.csv", index=False)
            elif stage == "detect_shocks":
                from src.evaluate_changepoints import run_benchmark
                run_benchmark(config.changepoint_detection)
            elif stage == "generate_report":
                from src.reporting import generate_report
                generate_report(config)
            elif stage == "build_presentation":
                from src.presentation_builder import build_presentation
                build_presentation(config.paths.artifacts, Path(config.changepoint_detection.figures))
            elif stage == "submission":
                from src.forecasting import forecast_submission
                from src.submission import build_submission
                predictions_path = forecast_submission(config)
                build_submission(predictions_path, prediction_column="pred_prophet")
            timings[stage] = perf_counter() - started
            LOGGER.info("DONE %s %.3fs", stage, timings[stage])
            (config.paths.artifacts / "run.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "seed": seed, "completed_stages_seconds": timings}, indent=2), encoding="utf-8")
        return 0
    except Exception:
        LOGGER.exception("Пайплайн остановлен: последующие этапы не запускались")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
