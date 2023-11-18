import pytest
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from pipegoose.nn.pipeline_parallel.partitioner import (  # PartitionPolicy,; get_model_partition,
    UniformPartitioner,
)
from pipegoose.testing.utils import init_parallel_context, spawn


def get_small_gpt2_and_tokenizer(n_layer=12):
    return GPT2LMHeadModel(
        GPT2Config(
            n_layer=n_layer
        )
    ), AutoTokenizer.from_pretrained("gpt2")


# def run_model_partitioner(
#     rank,
#     world_size,
#     port,
#     tensor_parallel_size,
#     pipeline_parallel_size,
#     data_parallel_size,
# ):
#     parallel_context = init_parallel_context(
#         rank,
#         world_size,
#         port,
#         tensor_parallel_size,
#         pipeline_parallel_size,
#         data_parallel_size,
#     )
#     model = get_small_gpt2()
#     partitions = UniformPartitioner(model, parallel_context).split(["input_ids"])

#     # partition = get_model_partition(module, policy, parallel_context)

#     for p in partitions:
#         print("==================")
#         print(sum([x.numel() for x in p.parameters()]))
#         print("==================")
    
#     assert False


@pytest.mark.parametrize("pipeline_parallel_size", [4])
def test_naive_partitioning(pipeline_parallel_size):
    # TENSOR_PARALLEL_SIZE = 1
    # DATA_PARALLEL_SIZE = 1

    # spawn(
    #     run_model_partitioner,
    #     world_size=pipeline_parallel_size,
    #     tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    #     pipeline_parallel_size=pipeline_parallel_size,
    #     data_parallel_size=DATA_PARALLEL_SIZE,
    # )

    batch_sentences = ["hello world from pipegoose"]

    model, tokenizer = get_small_gpt2_and_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    partitioned_model = UniformPartitioner(model, pipeline_parallel_size).split(["input_ids"])

    for p in partitioned_model:
        print("==================")
        # print(p)
        print(sum([x.numel() for x in p.parameters()]))
        print("==================")

    inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")

    partitioned_model_result = inputs["input_ids"]
    for partition_id in range(pipeline_parallel_size):
        partitioned_model_result = partitioned_model[partition_id](partitioned_model_result)

    gt_result = model(**inputs)

    assert torch.allclose(gt_result, partitioned_model_result)

    assert False, "Debug"
