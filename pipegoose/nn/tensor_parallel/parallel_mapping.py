from pipegoose.nn.parallel_mapping import ParallelInfo, ParallelMapping


class Column(ParallelInfo):
    pass


class Row(ParallelInfo):
    pass


class LMHead(ParallelInfo):
    pass


class TensorParallelMapping(ParallelMapping):
    """
    NOTE: Inspired from OSLO's Parallel Mapping
    https://github.com/EleutherAI/oslo/blob/d7c4e32e766a99cc9d56533bc090570360dc8b2a/oslo/torch/nn/parallel/tensor_parallel/mapping.py#L43
    """

    # TODO: make this extendable
    # so user can define their own mapping
    __MAPPING__ = {
        "albert-base-v2": [Column(("query", "key", "value")), Row("attention.dense")],
        "bloom-560m": [
            Column(("mlp.dense_h_to_4h", "self_attention.query_key_value")),
            Row(("mlp.dense_4h_to_h", "self_attention.dense")),
            LMHead(("lm_head",)),
        ],
    }

    @staticmethod
    def is_column_parallel(module_name: str) -> bool:
        item = TensorParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, Column)

    @staticmethod
    def is_row_parallel(module_name: str) -> bool:
        item = TensorParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, Row)

    @staticmethod
    def is_lm_head(module_name: str) -> bool:
        item = TensorParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, LMHead)
