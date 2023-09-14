from pipegoose.nn.tensor_parallel.parallel_mapping import ParallelMapping


def test_is_column_parallel_mapping(model):
    BLOOM_DENSE_H_TO_4H_NAME = "transformer.h.{}.mlp.dense_h_to_4h"
    BLOOM_QKV_NAME = "transformer.h.{}.self_attention.query_key_value"
    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_column_parallel(name)

    for layer_idx in range(len(model.transformer.h)):
        assert mappings[BLOOM_DENSE_H_TO_4H_NAME.format(layer_idx)] is True
        assert mappings[BLOOM_QKV_NAME.format(layer_idx)] is True


def test_is_row_parallel_mapping(model):
    BLOOM_DENSE_4H_TO_H_NAME = "transformer.h.{}.mlp.dense_4h_to_h"
    BLOOM_ATTENTION_DENSE_NAME = "transformer.h.{}.self_attention.dense"

    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_row_parallel(name)

    for layer_idx in range(len(model.transformer.h)):
        assert (
            mappings[BLOOM_DENSE_4H_TO_H_NAME.format(layer_idx)] is True
        ), f"{BLOOM_DENSE_4H_TO_H_NAME.format(layer_idx)} is not row parallelized"
        assert (
            mappings[BLOOM_ATTENTION_DENSE_NAME.format(layer_idx)] is True
        ), f"{BLOOM_ATTENTION_DENSE_NAME.format(layer_idx)} is not row parallelized"


def test_is_lm_head_mapping(model):
    BLOOM_LM_HEAD_NAME = "lm_head"

    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_lm_head(name)

    assert mappings[BLOOM_LM_HEAD_NAME] is True, f"{BLOOM_LM_HEAD_NAME} is not language model head"
