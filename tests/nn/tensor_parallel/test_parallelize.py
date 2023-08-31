import pytest
import torch
from transformers import AutoModel

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._utils import VocabUtility, is_splitable
from pipegoose.nn.tensor_parallel.parallelize import ParallelizeEmbedding
from pipegoose.testing.utils import spawn


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
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

    return parallel_context


def run_parallelize_embedding(
    rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, embedding, input, output
):
    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

    def get_new_embedding_size(vocab_size):
        padding_size = 0
        while not is_splitable(vocab_size + padding_size, parallel_context):
            padding_size += 1

        new_vocab_size = vocab_size + padding_size
        new_partition_size = new_vocab_size // world_size
        return new_vocab_size, new_partition_size

    vocab_size, embedding_dim = embedding.weight.size()
    new_vocab_size, new_partition_size = get_new_embedding_size(vocab_size)
    vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_from_global_vocab_size(world_size, rank, new_vocab_size)

    parallelized_embedding = ParallelizeEmbedding(embedding, parallel_context).parallelize()
    parallel_output = parallelized_embedding(input)

    assert parallelized_embedding.vocab_start_idx == vocab_start_idx
    assert parallelized_embedding.vocab_end_idx == vocab_end_idx
    assert parallelized_embedding.weight.shape == (new_partition_size, embedding_dim)
    assert torch.allclose(parallel_output, output)


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_parallelize_embedding(tensor_parallel_size):
    model = AutoModel.from_pretrained("gpt2")
    input = torch.arange(0, 10)
    embedding = model.get_input_embeddings()
    output = embedding(input)

    spawn(
        run_parallelize_embedding,
        world_size=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        embedding=embedding,
        input=input.detach(),
        output=output.detach(),
    )
