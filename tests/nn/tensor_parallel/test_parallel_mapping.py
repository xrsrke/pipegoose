# from transformers import AutoModel

# from pipegoose.nn.tensor_parallel.parallel_mapping import Column, ParallelMapping, Row


def test_parallel_mapping():
    # model = AutoModel.from_pretrained("bigscience/bloom-560m")
    # EXPECTED_MAPPING = {
    #     "dense_h_to_4h": Column,
    #     "dense_4h_to_h": Row,
    # }

    # mappings = {}

    # for name, module in model.named_modules():
    #     mappings[name] = ParallelMapping.is_column_parallel(module)

    # for name, parallel_type in mappings.items():
    #     assert isinstance(parallel_type, EXPECTED_MAPPING[name])

    # module = model.h[-1].mlp.dense_4h_to_h
    # output = ParallelMapping.is_row_parallel(module)
    # assert isinstance(output, Row)
    pass
