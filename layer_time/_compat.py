"""Device detection and GPU/CPU dispatch utilities."""

import torch


def resolve_device(device=None):
    """
    Resolve device string. None or "auto" → CUDA if available, else CPU.

    Args:
        device: "auto", "cuda", "cpu", "cuda:N", or None

    Returns:
        Resolved device string.
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def is_gpu(device):
    """Return True if device string refers to a CUDA device."""
    return device.startswith("cuda")
