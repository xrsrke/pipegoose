import torch
from torch import nn

from pipegoose.nn.pipeline_parallel2.queue import (
    SavedActivation,
    get_input_activations,
    get_output_activations,
    save_input_activations,
    save_output_activations,
)


def test_save_and_retrieve_activations():
    MICROBATCH_DIX = 1
    PARTITION_IDX = 0

    activations = torch.randn(2, 4)

    key = SavedActivation.get_key(MICROBATCH_DIX, PARTITION_IDX)
    SavedActivation.save_activations(key, activations)

    saved_activations = SavedActivation.get_saved_activations(key)

    assert torch.equal(activations, saved_activations)


def test_save_and_get_output_activations():
    MICROBATCH_DIX = 1
    PARTITION_IDX = 0

    input = torch.randn(2, 4)
    linear = nn.Linear(4, 2)
    output = linear(input)

    save_output_activations(output, MICROBATCH_DIX, PARTITION_IDX)

    retrieved_output = get_output_activations(MICROBATCH_DIX, PARTITION_IDX)
    assert torch.equal(output, retrieved_output)
    # NOTE: for the pipeline engine do backward, the input should require grad
    assert retrieved_output.requires_grad is True


def test_save_and_get_input_activations():
    MICROBATCH_DIX = 1
    PARTITION_IDX = 0

    input = torch.randn(2, 4)
    save_input_activations(input, MICROBATCH_DIX, PARTITION_IDX)

    retrieved_input = get_input_activations(MICROBATCH_DIX, PARTITION_IDX)
    assert torch.equal(input, retrieved_input)
    # NOTE: for the pipeline engine do backward, the input should require grad
    assert retrieved_input.requires_grad is True
