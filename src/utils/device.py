"""Hardware-agnostic PyTorch device selection (CUDA -> MPS -> CPU)."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the optimal available PyTorch compute device.

    Priority order:
        1. NVIDIA GPU (`cuda`) if `torch.cuda.is_available()`
        2. Apple Silicon GPU (`mps`) if `torch.backends.mps.is_available()`
        3. CPU (`cpu`) fallback

    Returns:
        torch.device: Selected compute device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Selected compute device: %s (%s)", device, gpu_name)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Selected compute device: %s (Apple Silicon Metal Performance Shaders)", device)
    else:
        device = torch.device("cpu")
        logger.info("Selected compute device: %s (CPU fallback)", device)

    return device
