from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models_forecast import (
    cached_pipeline_factory,
    chronos_fallback_predictions,
    predict_chronos_resilient,
)


def test_chronos_fallback_uses_last_value() -> None:
    result = chronos_fallback_predictions([[1.0, 2.5]], prediction_length=2)
    np.testing.assert_allclose(result, [[2.5, 2.5]])


def test_chronos_resource_fallback_retries_cpu_then_baseline() -> None:
    calls: list[str] = []

    class FakePipeline:
        def __init__(self, device: str) -> None:
            self.device = device

        def predict(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("out of memory")

    def factory(device: str, _: Any) -> FakePipeline:
        calls.append(device)
        return FakePipeline(device)

    result, mode = predict_chronos_resilient(
        factory,
        [[3.0, 4.0]],
        prediction_length=1,
        batch_size=4,
        device="cpu",
        dtype=None,
        fallback_to_cpu=True,
    )
    assert calls == ["cpu"]
    assert mode == "baseline"
    np.testing.assert_allclose(result, [[4.0]])


def test_cached_pipeline_factory_reuses_pipeline_per_device_and_dtype() -> None:
    calls: list[tuple[str, Any]] = []

    def factory(device: str, dtype: Any) -> object:
        calls.append((device, dtype))
        return object()

    cached = cached_pipeline_factory(factory)
    cpu_first = cached("cpu", "float32")
    cpu_second = cached("cpu", "float32")
    cuda = cached("cuda", "float32")

    assert cpu_first is cpu_second
    assert cuda is not cpu_first
    assert calls == [("cpu", "float32"), ("cuda", "float32")]


def test_chronos_non_resource_error_is_not_hidden() -> None:
    class FakePipeline:
        def predict(self, *_: Any, **__: Any) -> Any:
            raise ValueError("invalid Chronos API")

    with pytest.raises(ValueError, match="invalid Chronos API"):
        predict_chronos_resilient(
            lambda _device, _dtype: FakePipeline(),
            [[1.0, 2.0]],
            prediction_length=1,
            batch_size=1,
            device="cpu",
            dtype=None,
        )
