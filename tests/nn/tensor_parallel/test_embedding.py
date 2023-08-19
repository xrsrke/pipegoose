from copy import deepcopy

import torch
from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext, ParallelMode
from pipegoose.nn.tensor_parallel.embedding import ParallelEmbedding

# from pipegoose.nn.tensor_parallel.utils import get_vocab_range_idx
from pipegoose.testing.utils import spawn


def run_parallel_embedding(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    NUM_EMBEDDING = 100
    EMBEDDING_DIM = 10

    local_rank = parallel_context.get_local_rank(parallel_mode=ParallelMode.TENSOR)
    ranks_in_group = parallel_context.get_ranks_in_group(parallel_mode=ParallelMode.TENSOR)

    if local_rank in ranks_in_group:
        # input = torch.tensor([[0, 10, 60, 30, 40], [1, 20, 70, 50, 40]])
        input = torch.tensor([[0, 0], [20, 20]])
        embedding = nn.Embedding(NUM_EMBEDDING, EMBEDDING_DIM)
        weight = deepcopy(embedding.weight.data)
        embedding(input)

        parallel_embedding = ParallelEmbedding(NUM_EMBEDDING, EMBEDDING_DIM, parallel_context=parallel_context)

        def get_partition_weight(weight):
            local_world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)
            num_embeddings_per_partition = NUM_EMBEDDING // local_world_size
            chunks = torch.split(weight, num_embeddings_per_partition, dim=0)
            return chunks[local_rank]

        parallel_embedding.weight.data = get_partition_weight(weight)
        parallel_output, out = parallel_embedding(input)

        # assert torch.allclose(parallel_output, output)


def test_parallel_embedding():
    spawn(
        run_parallel_embedding,
        world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
