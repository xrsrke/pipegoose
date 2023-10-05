from dataclasses import dataclass
from queue import Queue
from typing import Dict, NewType, Tuple

import torch

ActivationKey = NewType("ActivationKey", Tuple[int, int])

# NOTE: the activations that received from earlier stages
_INPUT_ACTIVATIONS: Dict[ActivationKey, torch.Tensor] = {}

# NOTE: save activations from forward job for backward job
_SAVED_ACTIVATIONS: Dict[ActivationKey, torch.Tensor] = {}


@dataclass
class JobQueue:
    """A queue for storing jobs."""

    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()


class SavedActivation:
    """A class for saving activations from forward job for backward job."""

    @staticmethod
    def get_key(microbatch_idx: int, partition_idx: int) -> ActivationKey:
        return (microbatch_idx, partition_idx)

    @staticmethod
    def get_saved_activations(key: ActivationKey) -> torch.Tensor:
        """Get the saved activations for a given key for backward job."""
        # NOTE: because a partition can have multiple microbatches,
        return _SAVED_ACTIVATIONS.pop(key)

    def save_activations(key: ActivationKey, data: torch.Tensor):
        """Save forward job's activations for backward job."""
        _SAVED_ACTIVATIONS[key] = data


class InputActivations:
    """A class for saving activations from forward job for backward job."""

    @staticmethod
    def get_key(microbatch_idx: int, partition_idx: int) -> ActivationKey:
        return (microbatch_idx, partition_idx)

    @staticmethod
    def get_saved_activations(key: ActivationKey) -> torch.Tensor:
        """Get the saved activations for a given key for backward job."""
        # NOTE: because a partition can have multiple microbatches,
        return _INPUT_ACTIVATIONS.pop(key)

    def save_activations(key: ActivationKey, data: torch.Tensor):
        """Save forward job's activations for backward job."""
        _INPUT_ACTIVATIONS[key] = data
