"""Параллельный причинный Prophet-прогноз по муниципальным образованиям."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# NOTE: каждый loky-worker уже занят одним Prophet; BLAS-потоки только жрут память.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

def _silence_prophet_logging() -> None:
    """Подавляет служебный INFO-вывод Prophet/CmdStan во всех loky-процессах."""
    logger_names = {
        "prophet",
        "prophet.models",
        "prophet.forecaster",
        "cmdstanpy",
        "cmdstanpy.model",
        "cmdstanpy.utils",
    }
    logger_names.update(
        name
        for name in logging.root.manager.loggerDict
        if name.startswith(("prophet", "cmdstanpy"))
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
        # CmdStan needs only the exit code; Windows emits localized OEM text.
        try:
            subprocess.run(
                command, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
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
    """Обучает Prophet одного МО с константным fallback на плохой истории."""
    test = df_test_mo.copy()
    if "_test_order" not in test.columns:
        test["_test_order"] = np.arange(len(test), dtype=np.int64)
    if test.empty:
        return _empty_prediction_frame()

    raw_history = df_train_mo.copy()
    fallback = _fallback_value(raw_history)
    if not {"period", "y"}.issubset(raw_history.columns):
        LOGGER.warning("Prophet МО %s: отсутствуют period/y, применён fallback", mo)
        return _constant_predictions(mo, test, fallback)

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
        predicted = pd.to_numeric(model.predict(future)["yhat"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
            raise ValueError("Prophet вернул неконечный прогноз или неверную длину")
        result = _constant_predictions(mo, test, fallback)
        result["prophet_prediction"] = predicted
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        LOGGER.warning(
            "Prophet МО %s: %s; применён константный fallback",
            mo,
            error,
        )
        return _constant_predictions(mo, test, fallback)


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
        progress.close()


def predict_chronos_batched(
    pipeline: Any,
    contexts: Sequence[Any],
    *,
    prediction_length: int,
    batch_size: int = CHRONOS_BATCH_SIZE,
    device: Any | None = None,
    mixed_precision: bool = False,
) -> np.ndarray:
    """Пакетно получает медианный прогноз Chronos для нескольких рядов."""
    if not contexts:
        return np.empty(0, dtype=np.float64)
    if prediction_length <= 0 or batch_size <= 0:
        raise ValueError("prediction_length и batch_size должны быть положительными")
    import torch

    target_device = device if device is not None else getattr(pipeline, "device", torch.device("cpu"))
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    tensors = [
        (value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)).to(target_device)
        for value in contexts
    ]
    lengths = [int(tensor.numel()) for tensor in tensors]
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        # NOTE: Chronos требует равные длины; padding исказит ряд, поэтому группируем.
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
        return np.stack([prediction for prediction in grouped_predictions if prediction is not None], axis=0)
    predictions: list[np.ndarray] = []
    progress = tqdm(total=len(tensors), desc="Chronos Inference", unit="series", dynamic_ncols=True)
    try:
        for start in range(0, len(tensors), batch_size):
            batch = tensors[start:start + batch_size]
            with torch.inference_mode(), torch.autocast(
                device_type=target_device.type if hasattr(target_device, "type") else str(target_device),
                enabled=mixed_precision and str(target_device).startswith("cuda"),
            ):
                if hasattr(pipeline, "predict"):
                    # NOTE: в Chronos 2.3.x batch_size уже задан внешним циклом.
                    raw = pipeline.predict(batch, prediction_length=prediction_length)
                    samples = raw[0] if isinstance(raw, tuple) else raw
                    sample_tensor = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
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
    finally:
        progress.close()
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
                raise ValueError("Невозможно построить Chronos fallback: контекст не содержит конечных значений")
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
            pipeline, contexts, prediction_length=prediction_length,
            batch_size=batch_size, device=torch.device(target),
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
                    pipeline, contexts, prediction_length=prediction_length,
                    batch_size=max(1, batch_size // 2), device=torch.device("cpu"),
                )
                LOGGER.warning("Chronos fallback: CPU, batch_size=%d", max(1, batch_size // 2))
                return result, "cpu"
            except (RuntimeError, MemoryError) as cpu_error:
                if not _resource_error(cpu_error):
                    raise
                LOGGER.error("Chronos: CPU также не смог выполнить инференс: %s", cpu_error)
        LOGGER.warning("Chronos fallback: last-value baseline; model inference skipped")
        return chronos_fallback_predictions(contexts, prediction_length, fallback_values), "baseline"


def predict_prophet_parallel(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prophet_params: Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Параллельно прогнозирует МО и восстанавливает исходный порядок test."""
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
            # Check once before launching workers; installation failures are global.
            _create_prophet(prophet_params)
    tasks = (
        delayed(_fit_predict_single_mo)(
            entity,
            train.loc[train["mo"].eq(entity)].copy(),
            ordered_test.loc[ordered_test["mo"].eq(entity)].copy(),
            prophet_params,
            seed,
        )
        for entity in entities
    )
    # NOTE: больше четырёх Stan-процессов на Windows забивают память и stdout.
    workers = min(len(entities), max(1, min(4, os.cpu_count() or 1)))
    progress = tqdm(total=len(entities), desc="Prophet folds", unit="MO")
    with _tqdm_joblib(progress):
        parts = Parallel(
            n_jobs=workers,
            batch_size=1,
            backend="loky",
            inner_max_num_threads=1,
            verbose=0,
        )(tasks)

    combined = pd.concat(parts, ignore_index=True) if parts else _empty_prediction_frame()
    combined = combined.sort_values("_test_order", kind="stable")
    return combined.loc[:, list(PROPHET_OUTPUT_COLUMNS)].reset_index(drop=True)
