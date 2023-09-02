import pytest
from transformers import AutoModel, AutoTokenizer

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel.tensor_parallel import TensorParallel


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


@pytest.mark.skip
def test_parallelize_a_transformers():
    parallel_context = init_parallel_context()
    # world_size = parallel_context.get_world_size(parallel_mode=ParallelMode.TENSOR)

    model = AutoModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input = tokenizer.tokenize("Persistence is all you need.", return_tensors="pt")

    parallelized_model = TensorParallel(model, parallel_context)
    parallelized_model.parallelize()

    generated_ids = parallelized_model(**input)
