import pytest
import torch
from transformers import (
    AutoTokenizer,
    BloomConfig,
    BloomForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
)

from pipegoose.nn.pipeline_parallel.partitioner import UniformPartitioner


def get_gpt2_and_tokenizer():
    model = GPT2LMHeadModel(config=GPT2Config(n_layer=6))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


def get_bloom_and_tokenizer_with_6_layers():
    model = BloomForCausalLM(BloomConfig(n_layer=6))
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    return model, tokenizer


@pytest.mark.parametrize("n_partitions", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "model_retrieval_func",
    [
        get_gpt2_and_tokenizer,
        get_bloom_and_tokenizer_with_6_layers,
    ],
)
def test_naive_partitioning(n_partitions, model_retrieval_func):
    print(f"Running test with n_partitions={n_partitions}")

    torch.manual_seed(0)
    batch_sentences = ["hello world from pipegoose"]
    model, tokenizer = model_retrieval_func()
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")
    gt_logits = model(**inputs).logits

    partitioned_model = UniformPartitioner(model, n_partitions).split()

    assert len(partitioned_model) == n_partitions, f"Received model with {len(partitioned_model)} instead of {n_partitions}"

    print("Start printing partitioned model")
    for i, shard in enumerate(partitioned_model):
        shard_param_count = 0
        print("==================")
        print(f"Shard {i + 1}")
        for _, module in shard.named_children():
            # Sum the parameters of each module in the shard
            shard_param_count += sum(p.numel() for p in module.parameters())
            print(f"Layer type: {type(module).__name__}")
            print(module)
        print(f"Total parameters in Shard {i + 1}: {shard_param_count}")
        print("==================")
    print("End printing partitioned model")

    partitioned_model_result = inputs
    for partition_id in range(n_partitions):
        if type(partitioned_model_result) in (list, tuple):
            partitioned_model_result = partitioned_model[partition_id](*partitioned_model_result)
        else:
            partitioned_model_result = partitioned_model[partition_id](**partitioned_model_result)

    assert torch.allclose(gt_logits, partitioned_model_result), "Results are not close"
