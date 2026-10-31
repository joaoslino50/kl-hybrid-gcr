"""Shared hardware utilities and data contracts for the KGQA pipeline."""

from src.utils.device import get_device
from src.utils.types import ReasoningOutput, SubgraphData, Triple

__all__ = [
    "get_device",
    "Triple",
    "SubgraphData",
    "ReasoningOutput",
]
