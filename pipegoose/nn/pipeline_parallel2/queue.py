from dataclasses import dataclass
from queue import Queue

import torch

_SAVED_ACTIVATIONS = {}


@dataclass
class JobQueue:
    """A queue for storing jobs."""

    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()


def get_saved_activations(key: str) -> torch.Tensor:
    """Get the saved activations for a given key for backward job."""
    return _SAVED_ACTIVATIONS[key]


def save_activations(key: str, data: torch.Tensor):
    """Save forward job's activations for backward job."""
    _SAVED_ACTIVATIONS[key] = torch.Tensor
