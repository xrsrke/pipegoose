from pipegoose.nn.tensor_parallel.parallel_mapping import ParallelMapping


def test_is_column_parallel_mapping(model):
    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_column_parallel(name)

    for layer_idx in range(len(model.h)):
        # TODO: add check attention layer
        assert mappings[f"h.{layer_idx}.mlp.dense_h_to_4h"] is True


def test_is_row_parallel_mapping(model):
    mappings = {}

    for name, _ in model.named_modules():
        mappings[name] = ParallelMapping.is_row_parallel(name)

    for layer_idx in range(len(model.h)):
        # TODO: add check attention layer
        assert mappings[f"h.{layer_idx}.mlp.dense_4h_to_h"] is True
