"""Единый последовательный runner; запускать из корня репозитория."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from contextlib import ExitStack
from time import perf_counter

LOGGER = logging.getLogger(__name__)


def _print_retrain_summary(config: object, timings: dict[str, float], monitor: object) -> None:
    """Печатает итог переобучения: время, RAM, OOF и MD5."""
    artifacts = Path(getattr(config, "paths").artifacts)
    lines = ["", "=" * 78, "ПОЛНОЕ ПЕРЕОБУЧЕНИЕ С НУЛЯ — ИТОГОВЫЙ СТАТУС", "=" * 78]
    total = sum(timings.values())
    lines.append(f"1) Общее время расчета : {total:.1f} c ({total / 60.0:.1f} мин)")
    for stage, seconds in timings.items():
        lines.append(f"     - {stage:<18} {seconds:8.1f} c")
    lines.append(f"2) Память              : {monitor.describe()}")

    horizon_path = artifacts / "forecast_metrics_by_horizon.csv"
    lines.append("3) OOF MAE по горизонтам (pooled, exact) против Prophet:")
    if horizon_path.exists():
        import pandas as pd

        table = pd.read_csv(horizon_path)
        pooled = table.loc[
            table["fold"].astype(str).eq("pooled")
            & table["scope"].eq("exact")
            & table["status"].eq("measured")
        ]
        for horizon in sorted(pooled["horizon"].unique()):
            row = pooled.loc[pooled["horizon"].eq(horizon)].set_index("model")["MAE"]
            prophet = float(row["prophet"]) if "prophet" in row else float("nan")
            regime = float(row["regime_aware"]) if "regime_aware" in row else float("nan")
            delta = (prophet - regime) / prophet * 100.0 if prophet else float("nan")
            lines.append(
                f"     h={int(horizon):<2} Prophet {prophet:10.2f} | "
                f"RegimeAware {regime:10.2f} | Δ {delta:+6.2f}% / {prophet - regime:+.2f} руб."
            )
    else:
        lines.append("     н/д: forecast_metrics_by_horizon.csv не найден")

    submission_path = Path("submission.csv")
    if submission_path.exists():
        digest = hashlib.md5(submission_path.read_bytes()).hexdigest()
        audited = Path("reports/submission.csv")
        same = audited.is_file() and audited.read_bytes() == submission_path.read_bytes()
        lines.append(
            f"4) MD5 сабмита         : {digest} (reports/submission.csv идентичен: {same})"
        )
        import numpy as np
        import pandas as pd

        delivered = pd.read_csv(submission_path)
        lines.append(
            f"   Строк={len(delivered)}, МО={delivered['mo'].nunique()}, "
            f"NaN={delivered.isna().sum().sum()}, "
            f"finite={np.isfinite(delivered['pred']).all()}, "
            f"positive={delivered['pred'].gt(0).all()}"
        )
    else:
        lines.append("4) MD5 сабмита         : н/д — submission.csv отсутствует")
    lines.extend(["=" * 78, ""])
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--reuse-submission-base",
        action="store_true",
        help="явно использовать проверенные CatBoost/Prophet прогнозы из существующего frozen-origin parquet",
    )
    parser.add_argument(
        "--clean-start",
        action="store_true",
        help="удалить производные кэши и результаты перед выбранными этапами",
    )
    parser.add_argument(
        "--from-cache",
        "--reuse-cached",
        dest="from_cache",
        action="store_true",
        help="не обучать базовые модели: пересобрать ансамбль/метрики/сабмит из кэша frozen-origin прогнозов",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("reports/artifacts/"),
        help="каталог с сохранёнными OOF/Test предиктами базовых моделей",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="стратифицированная подвыборка 150 МО; работает только вместе с --from-cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="посчитать и показать результат, не записывая боевые артефакты",
    )
    parser.add_argument(
        "--weights-json",
        type=Path,
        default=None,
        help='веса кандидата: {"1": [prophet, catboost, chronos], ...}',
    )
    parser.add_argument(
        "--tag", default=None, help="метка эксперимента для reports/fast_run/experiments/"
    )
    parser.add_argument(
        "--fast-submission",
        action="store_true",
        help="в быстром режиме дополнительно собрать reports/fast_run/submission_150.csv",
    )
    parser.add_argument(
        "--retrain-from-scratch",
        action="store_true",
        help=(
            "полное детерминированное переобучение с нуля: clean-start + все этапы "
            "в memory-safe режиме (стриминг, даункастинг, мини-батчи Prophet, "
            "очистка буферов) с отчётом о времени, пиковой RAM, OOF и MD5"
        ),
    )
    parser.add_argument(
        "--no-memory-safe",
        action="store_true",
        help="с --retrain-from-scratch: не включать memory-safe надстройку (диагностика)",
    )
    available = (
        "prepare_data",
        "train_forecast",
        "detect_shocks",
        "generate_report",
        "submission",
        "build_presentation",
    )
    for stage in available:
        parser.add_argument("--" + stage.replace("_", "-"), action="store_true")
    args = parser.parse_args()
    if args.retrain_from_scratch:
        # Полный цикл: чистый старт снимает все кэши, --all прогоняет все этапы.
        args.all = True
        args.clean_start = True
    stages = [stage for stage in available if args.all or getattr(args, stage)]
    if not stages and not args.clean_start and not args.from_cache:
        parser.error("Выберите --all, --clean-start, --from-cache или хотя бы один этап")
    if args.fast and not args.from_cache:
        parser.error("--fast имеет смысл только вместе с --from-cache")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    resources = ExitStack()
    handler = None
    try:
        # PYTHONHASHSEED применяется только при старте процесса.
        import yaml

        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        repro = raw["reproducibility"]
        seed = int(repro["seed"])
        if os.environ.get("PYTHONHASHSEED") != str(seed):
            environment = dict(os.environ, PYTHONHASHSEED=str(seed))
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                env=environment,
                check=False,
            ).returncode
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = str(repro["threads"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import pandas as pd
        from src.configuration import load_config

        config = load_config(args.config)
        from src.clean_start import pipeline_lock

        resources.enter_context(pipeline_lock(Path.cwd()))
        if args.retrain_from_scratch and not args.no_memory_safe:
            memory = config.memory.model_copy(
                update={
                    "safe_mode": True,
                    "downcast_features": True,
                    "prophet_batch_size": config.memory.prophet_batch_size or 128,
                    "chronos_batch_size": config.memory.chronos_batch_size or 128,
                }
            )
            config = config.model_copy(update={"memory": memory})
            LOGGER.info("MEMORY-SAFE режим включён: %s", memory.model_dump(mode="json"))
        if args.clean_start:
            from src.clean_start import clean_generated_outputs

            removed = clean_generated_outputs(config, project_root=Path.cwd())
            LOGGER.info("Clean start: удалено производных путей: %d", len(removed))
            for path in removed:
                LOGGER.info("Clean start removed: %s", path)
        if not stages and not args.from_cache:
            return 0
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
        # В быстром кэшированном режиме изолируется и диагностика: ни один файл
        # боевого каталога artifacts не должен быть тронут.
        isolated = args.from_cache and args.fast and not args.dry_run
        diag_dir = Path("reports/fast_run") if isolated else config.paths.artifacts
        diag_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(diag_dir / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        (diag_dir / "config.resolved.json").write_text(
            config.model_dump_json(indent=2), encoding="utf-8"
        )
        if args.from_cache:
            from src.fast_run import format_comparison, load_weights, run_from_cache

            LOGGER.info("CACHE MODE: пропуск этапа обучения CatBoost, Prophet, Chronos")
            LOGGER.info("Загрузка верифицированных frozen-origin прогнозов из %s", args.cache_dir)
            summary = run_from_cache(
                config,
                cache_dir=args.cache_dir,
                fast=args.fast,
                dry_run=args.dry_run,
                weights=load_weights(args.weights_json),
                tag=args.tag,
                fast_submission=args.fast_submission,
            )
            print(format_comparison(summary["comparison"]))

            submission = summary["submission"]
            if submission.get("md5"):
                print(
                    f"submission md5={submission['md5']} "
                    f"matches_expected={submission.get('matches_expected')}"
                )
            mode = "FAST" if args.fast else "PRODUCTION"
            LOGGER.info(
                "%s from-cache: %d МО, %.2fs (ансамбль %.2fs), артефакты: %s",
                mode,
                summary["mo_count"],
                summary["seconds_total"],
                summary["seconds_ensemble"],
                summary["written"] or "не записывались",
            )
            if args.fast and not summary["guardrail"]["protected_unchanged"]:
                LOGGER.error("ГАРДРЕЙЛ: боевые артефакты изменились")
                return 1
            if args.dry_run:
                return 0
            if not args.fast and submission.get("matches_expected") is False:
                LOGGER.error("MD5 сабмита не совпал с ожидаемым — проверьте веса/кэш")
                return 1
            return 0

        from src.memory import MemoryMonitor, available_mib, memory_percent

        monitor = MemoryMonitor("retrain" if args.retrain_from_scratch else "pipeline")
        monitor.sample()
        timings: dict[str, float] = {}
        started_all = perf_counter()
        overall = tqdm_iter(stages, args.retrain_from_scratch)
        for stage in overall:
            if args.retrain_from_scratch:
                percent = memory_percent()
                available = available_mib()
                overall.set_postfix_str(
                    f"RAM {percent:.0f}% / free {available:.0f}MiB"
                    if percent is not None
                    else "RAM n/a"
                )
            started = perf_counter()
            LOGGER.info("START %s", stage)
            if stage == "prepare_data":
                from src.features import build_dataset

                frame = build_dataset(
                    config.data.directory,
                    config.data,
                    config.features,
                    tda_config=config.tda,
                    downcast=config.memory.downcast_features,
                )
                config.paths.supervised.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(config.paths.supervised, index=False)
                del frame
            elif stage == "train_forecast":
                from src.forecasting import (
                    forecast_multi_horizon,
                    metric_table,
                    save_horizon_metrics,
                )

                oof = forecast_multi_horizon(config, config.validation.horizons)
                metrics = metric_table(oof.loc[oof["lead_months"].eq(1)])
                horizons = save_horizon_metrics(
                    oof,
                    config.paths.artifacts,
                    Path(config.changepoint_detection.figures),
                    config.validation.horizons,
                )
                pooled = horizons.loc[
                    horizons["fold"].eq("pooled") & horizons["scope"].eq("exact")
                ].copy()
                pooled["fold"] = pooled["horizon"].map(lambda h: f"h={h}")
                metrics = pd.concat([metrics, pooled], ignore_index=True)
                metrics.to_csv(config.paths.artifacts / "forecast_metrics.csv", index=False)
                del oof, metrics, horizons, pooled
            elif stage == "detect_shocks":
                from src.evaluate_changepoints import run_benchmark

                run_benchmark(config.changepoint_detection)
            elif stage == "generate_report":
                from src.reporting import generate_report

                generate_report(config)
            elif stage == "build_presentation":
                from src.presentation_builder import build_presentation

                build_presentation(
                    config.paths.artifacts, Path(config.changepoint_detection.figures)
                )
            elif stage == "submission":
                from src.forecasting import forecast_submission
                from src.submission import build_submission

                predictions_path = forecast_submission(
                    config, reuse_base_forecasts=args.reuse_submission_base
                )
                roster = (
                    pd.read_parquet(predictions_path, columns=["mo"])["mo"]
                    .drop_duplicates()
                    .tolist()
                )
                build_submission(
                    predictions_path, prediction_column="pred_regime_aware", entity_universe=roster
                )
            timings[stage] = perf_counter() - started
            monitor.sample()
            LOGGER.info("DONE %s %.3fs", stage, timings[stage])
            (config.paths.artifacts / "run.json").write_text(
                json.dumps(
                    {
                        "python": sys.version,
                        "platform": platform.platform(),
                        "seed": seed,
                        "completed_stages_seconds": timings,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if args.retrain_from_scratch:
            run_path = config.paths.artifacts / "run.json"
            payload = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
            payload.update(
                {
                    "mode": "retrain_from_scratch",
                    "memory_safe": not args.no_memory_safe,
                    "weights_path": str(config.ensemble.weights_path)
                    if config.ensemble.weights_path
                    else None,
                    "seconds_total": perf_counter() - started_all,
                    "peak_rss_mib": round(monitor.peak_rss_mib, 1),
                    "peak_system_memory_percent": round(monitor.peak_percent, 1),
                }
            )
            run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            _print_retrain_summary(config, timings, monitor)
        elif args.all:
            _print_retrain_summary(config, timings, monitor)
        return 0
    except Exception:
        LOGGER.exception("Пайплайн остановлен: последующие этапы не запускались")
        return 1
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        resources.close()


def tqdm_iter(stages: list[str], enabled: bool) -> list[str]:
    """Возвращает этапы; для полного retrain добавляет прогресс-бар tqdm."""
    if not enabled:
        return stages
    from tqdm.auto import tqdm

    return tqdm(stages, desc="СберИндекс retrain", unit="этап", dynamic_ncols=True)  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
