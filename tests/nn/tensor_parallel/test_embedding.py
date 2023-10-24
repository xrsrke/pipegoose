from copy import deepcopy

import pytest
import torch
from torch import nn

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding
from pipegoose.testing.utils import get_partition, init_parallel_context, spawn


def run_parallel_embedding(
    rank,
    world_size,
    port,
    tensor_parallel_size,
    pipeline_parallel_size,
    data_parallel_size,
    input,
    output,
    orig_weight,
    ref_weight,
    grads,
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    local_rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR)
    ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode=ParallelMode.TENSOR)

    if local_rank in ranks_in_group:
        NUM_EMBEDDING = orig_weight.shape[0]
        EMBEDDING_DIM = orig_weight.shape[1]
        parallel_embedding = ParallelEmbedding(NUM_EMBEDDING, EMBEDDING_DIM, parallel_context=parallel_context)

        parallel_embedding.weight.data = get_partition(orig_weight, dim=0, parallel_context=parallel_context)
        parallel_output = parallel_embedding(input)

        assert torch.allclose(parallel_output, output)

        parallel_output.sum().backward()

        REF_GRAD = get_partition(grads, dim=0, parallel_context=parallel_context)
        assert torch.allclose(parallel_embedding.weight.grad.data, REF_GRAD)

        REF_WEIGHT = get_partition(ref_weight, dim=0, parallel_context=parallel_context)
        assert torch.allclose(parallel_embedding.weight.data, REF_WEIGHT)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallel_embedding(tensor_parallel_size):
    PIPELINE_PARALLEL_SIZE = 1
    DATA_PARALLEL_SIZE = 1

    NUM_EMBEDDING = 100
    EMBEDDING_DIM = 10

    input = torch.randint(0, NUM_EMBEDDING, (10, 5))
    embedding = nn.Embedding(NUM_EMBEDDING, EMBEDDING_DIM)
    weight = deepcopy(embedding.weight.data)
    output = embedding(input)
    output.sum().backward()
    grads = deepcopy(embedding.weight.grad.data)

    ref_weight = deepcopy(embedding.weight.data)

    spawn(
        run_parallel_embedding,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        input=input.detach(),
        output=output.detach(),
        orig_weight=weight.detach(),
        ref_weight=ref_weight,
        grads=grads.detach(),
    )
