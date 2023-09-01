from pipegoose.nn.tensor_parallel.parallel_mapping import ParallelMapping


def test_is_column_parallel_mapping(model):
    BLOOM_DENSE_H_TO_4H_NAME = "h.{}.mlp.dense_h_to_4h"
    BLOOM_QKV_NAME = "h.{}.self_attention.query_key_value"
    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_column_parallel(name)

    for layer_idx in range(len(model.h)):
        assert mappings[BLOOM_DENSE_H_TO_4H_NAME.format(layer_idx)] is True
        assert mappings[BLOOM_QKV_NAME.format(layer_idx)] is True


def test_is_row_parallel_mapping(model):
    BLOOM_DENSE_4H_TO_H_NAME = "h.{}.mlp.dense_4h_to_h"
    BLOOM_ATTENTION_DENSE_NAME = "h.{}.self_attention.dense"

    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_row_parallel(name)

    for layer_idx in range(len(model.h)):
        # TODO: add check attention layer
        assert mappings[BLOOM_DENSE_4H_TO_H_NAME.format(layer_idx)] is True
        assert mappings[BLOOM_ATTENTION_DENSE_NAME.format(layer_idx)] is True
