import torch

from pipegoose.nn.pipeline_parallel2.queue import SavedActivation


def test_save_and_retrieve_activations():
    MICROBATCH_DIX = 1
    PARTITION_IDX = 0

    activations = torch.randn(2, 4)

    key = SavedActivation.get_key(MICROBATCH_DIX, PARTITION_IDX)
    SavedActivation.save_activations(key, activations)

    saved_activations = SavedActivation.get_saved_activations(key)

    assert torch.equal(activations, saved_activations)
