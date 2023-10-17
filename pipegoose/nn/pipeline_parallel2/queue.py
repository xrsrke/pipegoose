from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, NewType, Tuple

import torch

from pipegoose.nn.pipeline_parallel2.exception import (
    PipelineNoSavedActivationError,
    PipelineNoSavedInput,
)

ActivationKey = NewType("ActivationKey", Tuple[int, int])

# NOTE: the activations that received from earlier stages
_INPUT_ACTIVATIONS: Dict[ActivationKey, torch.Tensor] = {}

# NOTE: save activations from forward job for backward job
_SAVED_ACTIVATIONS: Dict[ActivationKey, torch.Tensor] = {}

_SAVED_SCHEDULED_ACTIVATIONS: Dict[ActivationKey, torch.Tensor] = {}

_SAVED_GRAD_LOSS: Dict[ActivationKey, torch.Tensor] = {}

_SAVED_METADATA_of_GRAD_LOSS: Dict[ActivationKey, Any] = {}


@dataclass
class JobQueue:
    """A queue for storing jobs."""

    PENDING_JOBS = Queue()
    SELECTED_JOBS = Queue()
    FINISHED_JOBS = Queue()


class SavedActivation:
    """A class for saving activations from forward job for backward job."""

    @staticmethod
    def is_saved(microbatch_idx: int, partition_idx: int) -> bool:
        key = SavedActivation.get_key(microbatch_idx, partition_idx)
        return key in _SAVED_ACTIVATIONS

    @staticmethod
    def get_key(microbatch_idx: int, partition_idx: int) -> ActivationKey:
        return (microbatch_idx, partition_idx)

    @staticmethod
    def get_saved_activations(key: ActivationKey) -> torch.Tensor:
        """Get the saved activations for a given key for backward job."""
        # NOTE: because a partition can have multiple microbatches,
        return _SAVED_ACTIVATIONS.pop(key)

    def save_activations(key: ActivationKey, data: torch.Tensor, is_by_schedule: bool = False):
        """Save forward job's activations for backward job."""
        # if is_by_schedule is True:
        #     # TODO: why is this the case
        #     # NOTE: if create a backward job by schedule,
        #     # it requires the data to be detached
        #     # but directly create backward job doesn't require
        #     data = data.detach().requires_grad_(True)
        _SAVED_ACTIVATIONS[key] = data


class InputActivations:
    """A class for saving activations from forward job for backward job."""

    @staticmethod
    def get_key(microbatch_idx: int, partition_idx: int) -> ActivationKey:
        return (microbatch_idx, partition_idx)

    @staticmethod
    def is_saved(microbatch_idx: int, partition_idx: int) -> bool:
        key = InputActivations.get_key(microbatch_idx, partition_idx)
        return key in _INPUT_ACTIVATIONS

    @staticmethod
    def get_saved_activations(key: ActivationKey) -> torch.Tensor:
        """Get the saved activations for a given key for backward job."""
        # NOTE: because a partition can have multiple microbatches,
        # return _INPUT_ACTIVATIONS.pop(key)
        return _INPUT_ACTIVATIONS[key]

    def save_activations(key: ActivationKey, data: torch.Tensor):
        """Save forward job's activations for backward job."""
        _INPUT_ACTIVATIONS[key] = data


def save_input_activations(input: torch.Tensor, microbatch_idx: int, partition_idx: int):
    # input.requires_grad = True
    key = InputActivations.get_key(microbatch_idx, partition_idx)
    InputActivations.save_activations(key, input)


def get_input_activations(microbatch_idx: int, partition_idx: int) -> torch.Tensor:
    key = InputActivations.get_key(microbatch_idx, partition_idx)
    try:
        return InputActivations.get_saved_activations(key)
    except KeyError:
        raise PipelineNoSavedInput(
            f"Can't find the input activations to return the gradients for \
            microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"
        )


def save_output_activations(output: torch.Tensor, microbatch_idx: int, partition_idx: int):
    key = SavedActivation.get_key(microbatch_idx, partition_idx)
    SavedActivation.save_activations(key, output)


def get_output_activations(microbatch_idx: int, partition_idx: int, is_pipeline: bool = False) -> torch.Tensor:
    key = SavedActivation.get_key(microbatch_idx, partition_idx)

    try:
        output = _SAVED_ACTIVATIONS[key]
        # return output
        if is_pipeline is True:
            # return output.detach().requires_grad_(True)
            return output.requires_grad_(True)
        else:
            # return output.requires_grad_(True)
            return output.detach().requires_grad_(True)
    except KeyError:
        raise PipelineNoSavedActivationError(
            f"Can't find saved activations to do backpropogation for \
            microbatch_idx={microbatch_idx}, partition_idx={partition_idx}"
        )
