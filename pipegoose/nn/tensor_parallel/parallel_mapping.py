from typing import Tuple, Optional


class TensorParallelInformation:
    def __init__(self, module_name: Tuple[str], **kwargs):
        self.module_name = module_name
        self.kwargs = kwargs


class Column(TensorParallelInformation):
    pass


class Row(TensorParallelInformation):
    pass


class LMHead(TensorParallelInformation):
    pass


class ParallelMapping:

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
        item = ParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, Column)

    @staticmethod
    def is_row_parallel(module_name: str) -> bool:
        item = ParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, Row)

    @staticmethod
    def is_lm_head(module_name: str) -> bool:
        item = ParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, LMHead)

    @staticmethod
    def _search(module_name: str) -> Optional[TensorParallelInformation]:
        """
        Search for module_name in mappings.
        """
        module_name = ParallelMapping._extract_module_name(module_name)
        for items in ParallelMapping.__MAPPING__.values():
            for item in items:
                if any(module_name in mapping_name for mapping_name in item.module_name):
                    return item
        return None

    @staticmethod
    def _extract_module_name(module_name: str) -> str:
        if "." in module_name:
            # NOTE: transformer.h.0.self_attention.dense -> self_attention.dense
            SEPARATOR = "."
            sections = module_name.split(SEPARATOR)
            return SEPARATOR.join(sections[-2:])

        return module_name
