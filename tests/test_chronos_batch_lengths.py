from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models_forecast import predict_chronos_batched


def test_chronos_batches_different_context_lengths_without_padding() -> None:
    torch = pytest.importorskip("torch")
    calls: list[int] = []

    class FakePipeline:
        def predict(self, batch: list[Any], *, prediction_length: int) -> Any:
            calls.append(len(batch[0]))
            return torch.stack([value[-1:].repeat(prediction_length) for value in batch]).unsqueeze(1)

    result = predict_chronos_batched(
        FakePipeline(),
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])],
        prediction_length=1,
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert calls == [2, 3]
    np.testing.assert_allclose(result, [[2.0], [5.0]])


def test_chronos_empty_contexts_return_empty() -> None:
    assert predict_chronos_batched(object(), [], prediction_length=1).shape == (0,)


def test_chronos_invalid_prediction_length_rejected() -> None:
    with pytest.raises(ValueError, match="prediction_length"):
        predict_chronos_batched(object(), [[1.0]], prediction_length=0)


if __name__ == "__main__":
    raise SystemExit(0)
