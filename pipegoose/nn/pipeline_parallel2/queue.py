from dataclasses import dataclass
from queue import Queue

import torch

# NOTE: save activations from forward job for backward job
_SAVED_ACTIVATIONS = {}


@dataclass
class JobQueue:
    """A queue for storing jobs."""

    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()


def get_saved_activations(microbatch_idx: int) -> torch.Tensor:
    """Get the saved activations for a given key for backward job."""
    # NOTE: because a partition can have multiple microbatches,
    return _SAVED_ACTIVATIONS.pop(microbatch_idx)


def save_activations(microbatch_idx: int, data: torch.Tensor):
    """Save forward job's activations for backward job."""
    _SAVED_ACTIVATIONS[microbatch_idx] = data
