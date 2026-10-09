"""Единый выбор вычислительного устройства и диагностика CUDA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

LOGGER = logging.getLogger(__name__)
DeviceMode = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class DeviceInfo:
    """Разрешённое устройство и сведения о runtime."""

    mode: DeviceMode
    device: str
    cuda_available: bool
    gpu_name: str | None
    cuda_version: str | None
    fallback_reason: str | None = None


def resolve_device(mode: DeviceMode = "auto") -> DeviceInfo:
    """Разрешает auto/cuda/cpu без импорта Torch на CPU-only ETL раньше времени."""
    if mode not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Неизвестный режим устройства: {mode}")
    try:
        import torch
    except ImportError as error:
        if mode == "cuda":
            raise RuntimeError("Запрошен CUDA, но PyTorch не установлен") from error
        reason = "PyTorch не установлен"
        info = DeviceInfo(
            mode=mode,
            device="cpu",
            cuda_available=False,
            gpu_name=None,
            cuda_version=None,
            fallback_reason=reason,
        )
        LOGGER.info("Устройство: cpu; fallback: %s", reason)
        return info

    available = bool(torch.cuda.is_available())
    if mode == "cuda" and not available:
        detail = "CUDA недоступна: проверьте CUDA-сборку PyTorch, драйвер и NVIDIA GPU"
        raise RuntimeError(detail)
    if mode == "auto" and not available:
        reason = "CUDA недоступна, выбран CPU"
        info = DeviceInfo(
            mode=mode,
            device="cpu",
            cuda_available=False,
            gpu_name=None,
            cuda_version=getattr(torch.version, "cuda", None),
            fallback_reason=reason,
        )
        LOGGER.warning("Устройство: cpu; %s", reason)
        return info
    if mode == "cpu":
        info = DeviceInfo(
            mode=mode,
            device="cpu",
            cuda_available=available,
            gpu_name=None,
            cuda_version=getattr(torch.version, "cuda", None),
        )
        LOGGER.info("Устройство: cpu (CUDA намеренно отключена)")
        return info

    name = torch.cuda.get_device_name(0)
    info = DeviceInfo(
        mode=mode,
        device="cuda",
        cuda_available=True,
        gpu_name=name,
        cuda_version=getattr(torch.version, "cuda", None),
    )
    LOGGER.info("Устройство: cuda; GPU=%s; CUDA runtime=%s", name, info.cuda_version or "unknown")
    return info


def cuda_memory_error(error: BaseException) -> bool:
    """Распознаёт OOM CUDA, не скрывая прочие ошибки."""
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        return False
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()
