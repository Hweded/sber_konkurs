from __future__ import annotations

import pytest

from src.device import resolve_device


def test_cpu_device_is_available_without_cuda() -> None:
    info = resolve_device("cpu")
    assert info.device == "cpu"


def test_auto_device_is_cpu_or_cuda() -> None:
    info = resolve_device("auto")
    assert info.device in {"cpu", "cuda"}


def test_cuda_mode_requires_available_runtime() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed")
    if torch.cuda.is_available():
        assert resolve_device("cuda").device == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA недоступна"):
            resolve_device("cuda")


def test_invalid_device_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Неизвестный режим"):
        resolve_device("invalid")  # type: ignore[arg-type]
