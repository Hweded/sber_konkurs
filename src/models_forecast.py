"""Параллельный причинный Prophet-прогноз по муниципальным образованиям.

Модуль работает в двух режимах.  Обычный режим запускает все МО одним вызовом
joblib.  Memory-safe режим (``--retrain-from-scratch``) режет 2 094 ряда на
мини-батчи, при необходимости сбрасывает каждый батч на диск и явно освобождает
Stan-буферы между батчами, чтобы пиковое потребление RAM не зависело от числа МО.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from src.memory import MemoryMonitor, empty_torch_cache, free, spill_frame

# Каждый loky-worker уже занят одним Prophet; BLAS-потоки только жрут память.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")


def _silence_prophet_logging() -> None:
    """Подавляет служебный INFO-вывод Prophet/CmdStan во всех loky-процессах.

    CmdStan буферизует stdout/err; на 2 094 рядах это заметная доля RSS.  Уровень
    строго ERROR — требование memory-safe режима.
    """
    logger_names = {
        "prophet",
        "prophet.models",
        "prophet.forecaster",
        "cmdstanpy",
        "cmdstanpy.model",
        "cmdstanpy.utils",
        "pystan",
        "stan",
    }
    logger_names.update(
        name
        for name in logging.root.manager.loggerDict
        if name.startswith(("prophet", "cmdstanpy", "pystan", "stan"))
    )
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        for handler in logger.handlers:
            handler.setLevel(logging.ERROR)


_silence_prophet_logging()
warnings.filterwarnings("ignore", category=UserWarning, module=r"prophet(\..*)?")

LOGGER = logging.getLogger(__name__)
PROPHET_OUTPUT_COLUMNS = ("period", "mo", "prophet_prediction")
CHRONOS_BATCH_SIZE = 128
PROPHET_BATCH_SIZE = 128


def _empty_prediction_frame() -> pd.DataFrame:
    """Возвращает пустой результат с устойчивой схемой Prophet."""
    return pd.DataFrame(
        {
            "period": pd.Series(dtype="datetime64[ns]"),
            "mo": pd.Series(dtype="object"),
            "prophet_prediction": pd.Series(dtype="float64"),
            "_test_order": pd.Series(dtype="int64"),
        }
    )


def _fallback_value(history: pd.DataFrame) -> float:
    """Выбирает последнее конечное значение или безопасный нулевой прогноз."""
    if "y" not in history.columns:
        return 0.0
    values = pd.to_numeric(history["y"], errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if len(finite) else 0.0


def _constant_predictions(
    mo: object,
    test: pd.DataFrame,
    value: float,
) -> pd.DataFrame:
    """Формирует константный прогноз, сохраняя порядок строк test."""
    if test.empty:
        return _empty_prediction_frame()
    return pd.DataFrame(
        {
            "period": pd.to_datetime(test["period"], errors="coerce").to_numpy(),
            "mo": np.repeat(mo, len(test)),
            "prophet_prediction": np.full(len(test), value, dtype=np.float64),
            "_test_order": test["_test_order"].to_numpy(dtype=np.int64),
            "_prophet_fallback": True,
        }
    )


@contextmanager
def _cmdstan_windows_probe() -> Iterator[None]:
    if sys.platform != "win32":
        yield
        return
    from cmdstanpy import model as cmdstan_model

    original_command = cmdstan_model.do_command

    def run_command(command: list[str], cwd: str | None = None, **kwargs: Any) -> None:
        if command != ["where.exe", "tbb.dll"]:
            return original_command(command, cwd=cwd, **kwargs)
        # От CmdStan нужен только код возврата; текст Windows локализован.
        try:
            subprocess.run(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Windows TBB lookup failed: {error}") from error

    cmdstan_model.do_command = run_command
    try:
        yield
    finally:
        cmdstan_model.do_command = original_command


def _create_prophet(prophet_params: Mapping[str, Any] | None) -> Any:
    parameters = dict(prophet_params or {})
    if parameters.get("stan_backend") is None:
        # Auto-detection hides the original CmdStan error in Prophet 1.1.6.
        parameters["stan_backend"] = "CMDSTANPY"
    parameters.update(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_changepoints=2,
        changepoint_prior_scale=0.05,
    )
    try:
        from prophet import Prophet

        _silence_prophet_logging()
        with _cmdstan_windows_probe():
            model = Prophet(**parameters)
    except Exception as error:
        raise RuntimeError(
            f"Prophet: не удалось инициализировать Stan backend: {error}. "
            "Проверьте установку prophet/cmdstanpy в Python, запускающем pipeline; "
            "для восстановления: python -m pip install --force-reinstall --no-deps "
            "prophet==1.1.6 cmdstanpy==1.2.5"
        ) from error
    _silence_prophet_logging()
    return model


def _fit_predict_single_mo(
    mo: object,
    df_train_mo: pd.DataFrame,
    df_test_mo: pd.DataFrame,
    prophet_params: Mapping[str, Any] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Обучает Prophet одного МО с константным fallback на плохой истории.

    Модель освобождается в ``finally``: внутри долгоживущего loky-worker объект
    Prophet со Stan-компилятором иначе накапливался бы до конца процесса.
    """
    test = df_test_mo.copy()
    if "_test_order" not in test.columns:
        test["_test_order"] = np.arange(len(test), dtype=np.int64)
    if test.empty:
        return _empty_prediction_frame()

    raw_history = _normalise_prophet_train(df_train_mo)
    fallback = _fallback_value(raw_history)

    history = raw_history[["period", "y"]].rename(columns={"period": "ds"})
    history["ds"] = pd.to_datetime(history["ds"], errors="coerce")
    if history["ds"].dt.tz is not None:
        history["ds"] = history["ds"].dt.tz_localize(None)
    history["y"] = pd.to_numeric(history["y"], errors="coerce")
    history = history.loc[
        history["ds"].notna() & np.isfinite(history["y"].to_numpy(dtype=np.float64))
    ].sort_values("ds", kind="stable")

    future = pd.DataFrame({"ds": pd.to_datetime(test["period"], errors="coerce")})
    if future["ds"].dt.tz is not None:
        future["ds"] = future["ds"].dt.tz_localize(None)
    valid_future = future["ds"].notna().to_numpy()
    if len(history) < 3:
        return _constant_predictions(mo, test, fallback)
    if not valid_future.all():
        LOGGER.warning("Prophet МО %s: невалидные даты test, применён fallback", mo)
        return _constant_predictions(mo, test, fallback)

    model = _create_prophet(prophet_params)
    try:
        model.fit(history, seed=seed)
        # Prophet сортирует ds внутри predict; верни прогноз к исходному порядку.
        order = np.argsort(future["ds"].to_numpy(), kind="stable")
        sorted_prediction = pd.to_numeric(
            model.predict(future.iloc[order])["yhat"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        predicted = np.empty(len(test), dtype=np.float64)
        predicted[order] = sorted_prediction
        if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
            raise ValueError("Prophet вернул неконечный прогноз или неверную длину")
        result = _constant_predictions(mo, test, fallback)
        result["prophet_prediction"] = predicted
        result["_prophet_fallback"] = False
        return result
    except Exception as error:
        LOGGER.warning(
            "Prophet МО %s: %s; применён константный fallback",
            mo,
            error,
        )
        return _constant_predictions(mo, test, fallback)
    finally:
        # Stan держит C++-буферы до сборки мусора.
        del model
        free()


@contextmanager
def _tqdm_joblib(progress: tqdm[Any]) -> Iterator[None]:
    """Обновляет tqdm по завершённым joblib batch и восстанавливает callback."""
    import joblib.parallel

    original_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(original_callback):  # type: ignore[misc, valid-type]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            progress.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback


def release_prophet_workers() -> None:
    """Shut the reusable loky pool down so Prophet/Stan memory leaves the process.

    Four live workers each keep a Prophet/Stan runtime resident.  On this host the
    Windows commit limit is RAM + a 2 GiB pagefile, and the Chronos checkpoint
    load fails with WinError 1455 unless that headroom is returned first.
    """
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
        LOGGER.info("Prophet: loky-пул остановлен, память воркеров освобождена")
    except Exception as error:  # pragma: no cover - best effort
        LOGGER.debug("loky shutdown пропущен: %s", error)
    free()


def predict_chronos_batched(
    pipeline: Any,
    contexts: Sequence[Any],
    *,
    prediction_length: int,
    batch_size: int = CHRONOS_BATCH_SIZE,
    device: Any | None = None,
    mixed_precision: bool = False,
) -> np.ndarray:
    """Пакетно получает медианный прогноз Chronos для нескольких рядов.

    Инференс идёт без графа вычислений (``inference_mode`` сильнее ``no_grad``),
    а после каждого батча освобождаются тензоры и torch-кэш: на 2 094 рядах это
    единственное место, где промежуточные активации реально уходят из RAM.
    """
    if not contexts:
        return np.empty(0, dtype=np.float64)
    if prediction_length <= 0 or batch_size <= 0:
        raise ValueError("prediction_length и batch_size должны быть положительными")
    import torch

    target_device = (
        device if device is not None else getattr(pipeline, "device", torch.device("cpu"))
    )
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    tensors = [
        (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=torch.float32)
        ).to(target_device)
        for value in contexts
    ]
    lengths = [int(tensor.numel()) for tensor in tensors]
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        # Chronos требует равные длины; padding исказит ряд, поэтому группируем.
        grouped_predictions: list[np.ndarray | None] = [None] * len(tensors)
        for length in unique_lengths:
            indices = [index for index, value in enumerate(lengths) if value == length]
            group_predictions = predict_chronos_batched(
                pipeline,
                [tensors[index] for index in indices],
                prediction_length=prediction_length,
                batch_size=batch_size,
                device=target_device,
                mixed_precision=mixed_precision,
            )
            for index, prediction in zip(indices, group_predictions, strict=True):
                grouped_predictions[index] = prediction
        if any(prediction is None for prediction in grouped_predictions):
            raise RuntimeError("Chronos не вернул прогноз для каждой группы контекстов")
        return np.stack(
            [prediction for prediction in grouped_predictions if prediction is not None], axis=0
        )
    predictions: list[np.ndarray] = []
    progress = tqdm(total=len(tensors), desc="Chronos Inference", unit="series", dynamic_ncols=True)
    try:
        for start in range(0, len(tensors), batch_size):
            batch = tensors[start : start + batch_size]
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=target_device.type
                    if hasattr(target_device, "type")
                    else str(target_device),
                    enabled=mixed_precision and str(target_device).startswith("cuda"),
                ),
            ):
                if hasattr(pipeline, "predict"):
                    # В Chronos 2.3.x batch_size уже задан внешним циклом.
                    raw = pipeline.predict(batch, prediction_length=prediction_length)
                    samples = raw[0] if isinstance(raw, tuple) else raw
                    sample_tensor = (
                        samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
                    )
                    if sample_tensor.ndim == 3:
                        values_tensor = sample_tensor.median(dim=1).values
                    elif sample_tensor.ndim == 2:
                        values_tensor = sample_tensor
                    else:
                        raise ValueError("Chronos predict вернул неожиданную размерность")
                else:
                    quantiles, _ = pipeline.predict_quantiles(
                        batch,
                        prediction_length=prediction_length,
                        quantile_levels=[0.5],
                    )
                    values_tensor = quantiles[:, 0, :]
            values = np.asarray(values_tensor.detach().cpu().numpy(), dtype=np.float64)
            if values.shape != (len(batch), prediction_length) or not np.isfinite(values).all():
                raise ValueError("Chronos вернул неконечный прогноз или неверную форму")
            predictions.append(values)
            progress.update(len(batch))
            # Батч и его тензоры живут только внутри одной итерации.
            del batch, values_tensor, values
            empty_torch_cache()
            free()
    finally:
        progress.close()
        del tensors
        free()
    return np.concatenate(predictions, axis=0)


def _clear_cuda_cache() -> None:
    """Освобождает CUDA-кэш только после подтверждённого CUDA OOM."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        LOGGER.info("Chronos: CUDA-кэш очищен после нехватки памяти")


def _resource_error(error: BaseException) -> bool:
    """Отличает нехватку памяти/ресурсов от ошибок API или данных."""
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()


def chronos_fallback_predictions(
    contexts: Sequence[Any],
    prediction_length: int,
    fallback_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Последнее конечное значение ряда как прозрачный baseline fallback."""
    if prediction_length <= 0:
        raise ValueError("prediction_length должен быть положительным")
    values: list[float] = []
    for index, context in enumerate(contexts):
        if fallback_values is not None:
            value = float(fallback_values[index])
        else:
            array = np.asarray(context, dtype=np.float64).reshape(-1)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise ValueError(
                    "Невозможно построить Chronos fallback: контекст не содержит конечных значений"
                )
            value = float(finite[-1])
        if not np.isfinite(value):
            raise ValueError("Невозможно построить Chronos fallback: baseline не конечен")
        values.append(value)
    return np.repeat(np.asarray(values, dtype=np.float64)[:, None], prediction_length, axis=1)


def cached_pipeline_factory(
    factory: Callable[[str, Any], Any],
) -> Callable[[str, Any], Any]:
    """Кэширует тяжёлый pipeline отдельно для каждой пары device/dtype."""
    cache: dict[tuple[str, Any], Any] = {}

    def get_pipeline(device: str, dtype: Any) -> Any:
        key = (device, dtype)
        if key not in cache:
            cache[key] = factory(device, dtype)
        return cache[key]

    return get_pipeline


def predict_chronos_resilient(
    pipeline_factory: Callable[[str, Any], Any],
    contexts: Sequence[Any],
    *,
    prediction_length: int,
    batch_size: int,
    device: Any,
    dtype: Any,
    fallback_to_cpu: bool = True,
    fallback_values: Sequence[float] | None = None,
) -> tuple[np.ndarray, str]:
    """Запускает Chronos по цепочке GPU → CPU → last-value baseline."""
    import torch

    target = str(device)
    try:
        pipeline = pipeline_factory(target, dtype)
        result = predict_chronos_batched(
            pipeline,
            contexts,
            prediction_length=prediction_length,
            batch_size=batch_size,
            device=torch.device(target),
        )
        return result, target
    except (RuntimeError, MemoryError) as error:
        if not _resource_error(error):
            raise
        LOGGER.warning("Chronos: нехватка ресурсов на %s: %s", target, error)
        if target.startswith("cuda"):
            _clear_cuda_cache()
        if fallback_to_cpu and target.startswith("cuda"):
            try:
                pipeline = pipeline_factory("cpu", torch.float32)
                result = predict_chronos_batched(
                    pipeline,
                    contexts,
                    prediction_length=prediction_length,
                    batch_size=max(1, batch_size // 2),
                    device=torch.device("cpu"),
                )
                LOGGER.warning("Chronos fallback: CPU, batch_size=%d", max(1, batch_size // 2))
                return result, "cpu"
            except (RuntimeError, MemoryError) as cpu_error:
                if not _resource_error(cpu_error):
                    raise
                LOGGER.error("Chronos: CPU также не смог выполнить инференс: %s", cpu_error)
        LOGGER.warning("Chronos fallback: last-value baseline; model inference skipped")
        return chronos_fallback_predictions(
            contexts, prediction_length, fallback_values
        ), "baseline"


@contextmanager
def _prophet_worker_cleanup(enabled: bool) -> Iterator[None]:
    """Освободи Stan-пул даже при ошибке воркера или записи батча."""
    try:
        yield
    finally:
        if enabled:
            release_prophet_workers()


def _normalise_prophet_train(train: pd.DataFrame) -> pd.DataFrame:
    """Приведи поддерживаемые колонки даты и цели к period/y."""
    date_column = next((name for name in ("ds", "period", "date") if name in train), None)
    target_column = next((name for name in ("y", "target", "value") if name in train), None)
    if date_column is None or target_column is None:
        raise ValueError(f"Prophet: не найдены колонки даты/таргета среди {train.columns.tolist()}")
    columns = [date_column, target_column]
    if "mo" in train:
        columns.append("mo")
    return train[columns].rename(columns={date_column: "period", target_column: "y"}).copy()


def predict_prophet_parallel(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prophet_params: Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    batch_size: int | None = None,
    spill_dir: Path | None = None,
    monitor: MemoryMonitor | None = None,
    release_workers: bool = True,
    max_fallback_fraction: float = 0.05,
) -> pd.DataFrame:
    """Параллельно прогнозирует МО и восстанавливает исходный порядок test.

    Memory-safe режим (``batch_size`` и/или ``spill_dir``) режет МО на мини-батчи:
    после каждого батча вызывается ``gc.collect()``, а результат при
    необходимости сбрасывается в ``spill_dir/prophet_batch_*.parquet``.  Итоговая
    таблица собирается из уже готовых батчей, поэтому пик RAM не растёт вместе с
    числом муниципалитетов.
    """
    if test.empty:
        return pd.DataFrame(
            {
                "period": pd.Series(dtype="datetime64[ns]"),
                "mo": pd.Series(dtype="object"),
                "prophet_prediction": pd.Series(dtype="float64"),
            }
        )
    if not {"period", "mo"}.issubset(test.columns):
        raise ValueError("test Prophet должен содержать period и mo")
    LOGGER.info("Prophet train matrix: cols=%s, rows=%d", train.columns.tolist(), len(train))
    train = _normalise_prophet_train(train)
    if "mo" not in train:
        raise ValueError("Prophet: нет колонки mo в обучающей матрице")
    if not 0 <= max_fallback_fraction <= 1:
        raise ValueError("max_fallback_fraction должен лежать в [0, 1]")

    ordered_test = test.copy()
    ordered_test["_test_order"] = np.arange(len(ordered_test), dtype=np.int64)
    entities: Sequence[object] = list(ordered_test["mo"].drop_duplicates())
    if {"period", "mo", "y"}.issubset(train.columns):
        valid_history = (
            train["mo"].isin(entities)
            & pd.to_datetime(train["period"], errors="coerce").notna()
            & np.isfinite(pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=float))
        )
        if train.loc[valid_history].groupby("mo", observed=True).size().ge(3).any():
            # Проверяем установку один раз: ошибка общая для всех workers.
            _create_prophet(prophet_params)
    # Больше четырёх Stan-процессов на Windows забивают память и stdout.
    workers = min(len(entities), max(1, min(4, (os.cpu_count() or 1) - 2)))
    chunk = int(batch_size) if batch_size is not None else 150
    if chunk <= 0:
        raise ValueError("batch_size должен быть положительным")
    if spill_dir is not None:
        spill_dir = Path(spill_dir)
        spill_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(entities), desc="Prophet folds", unit="MO")
    parts: list[pd.DataFrame] = []
    spilled: list[Path] = []
    fallback_entities = 0
    train_groups = dict(tuple(train.groupby("mo", sort=False, observed=True)))
    test_groups = dict(tuple(ordered_test.groupby("mo", sort=False, observed=True)))
    LOGGER.info("Prophet: workers=%d, batch_size=%d, МО=%d", workers, chunk, len(entities))
    with progress, _tqdm_joblib(progress), _prophet_worker_cleanup(release_workers):
        for start in range(0, len(entities), chunk):
            batch_entities = entities[start : start + chunk]
            tasks = (
                delayed(_fit_predict_single_mo)(
                    entity,
                    train_groups.get(entity, train.iloc[:0]),
                    test_groups[entity],
                    prophet_params,
                    seed,
                )
                for entity in batch_entities
            )
            batch_parts = Parallel(
                n_jobs=workers,
                batch_size=1,
                backend="loky",
                inner_max_num_threads=1,
                verbose=0,
            )(tasks)
            fallback_entities += sum(bool(part["_prophet_fallback"].any()) for part in batch_parts)
            if fallback_entities > max_fallback_fraction * len(entities):
                raise RuntimeError(
                    "Более 5% МО ушли в fallback. Проверьте структуру входных данных! "
                    f"fallback={fallback_entities}/{len(entities)}, "
                    f"limit={max_fallback_fraction:.1%}"
                )
            combined_batch = (
                pd.concat(batch_parts, ignore_index=True)
                if batch_parts
                else _empty_prediction_frame()
            )
            if spill_dir is not None:
                spilled.append(
                    spill_frame(
                        combined_batch, spill_dir, f"prophet_batch_{start // chunk:04d}.parquet"
                    )
                )
            else:
                parts.append(combined_batch)
            del batch_parts, combined_batch, tasks
            free()
            if monitor is not None:
                monitor.log(
                    f"prophet batch {start // chunk + 1}/{(len(entities) + chunk - 1) // chunk}"
                )

    if spilled:
        LOGGER.info("Prophet: %d батчей сохранено в %s", len(spilled), spill_dir)
        parts = [pd.read_parquet(path) for path in spilled]
    combined = pd.concat(parts, ignore_index=True) if parts else _empty_prediction_frame()
    combined = combined.sort_values("_test_order", kind="stable")
    result = combined.loc[:, list(PROPHET_OUTPUT_COLUMNS)].reset_index(drop=True)
    result.attrs["prophet_diagnostics"] = {
        "entities": len(entities),
        "fallback_entities": fallback_entities,
        "fitted_entities": len(entities) - fallback_entities,
        "fallback_fraction": fallback_entities / len(entities),
    }
    LOGGER.info("Prophet итог: %s", result.attrs["prophet_diagnostics"])
    del combined, parts
    free()
    return result
