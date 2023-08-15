from typing import Tuple


class TensorParallelInformation:
    def __init__(self, module_name: Tuple[str], **kwargs):
        self.module_name = module_name
        self.kwargs = kwargs


class Column(TensorParallelInformation):
    pass


class Row(TensorParallelInformation):
    pass


class ParallelMapping:
    # TODO: make this extendable
    # so user can define their own mapping

    __MAPPING__ = {
        "albert-base-v2": [Column(("query", "key", "value")), Row("attention.dense")],
        "bloom-560m": [
            Column(("dense_h_to_4h",)),
            Row(("dense_4h_to_h",)),
        ],
    }

    def _search(cls, module_name: str) -> TensorParallelInformation:
        """
        Search for module_name in mappings.
        """
        for _, items in cls.__MAPPING__.items():
            for item in items:
                if module_name in item.module_name:
                    return item
        return None

    @classmethod
    def is_column_parallel(cls, module_name: str) -> bool:
        item = cls._search(module_name)
        if item is None:
            return False
        return isinstance(item, Column)

    @classmethod
    def is_row_parallel(cls, module_name: str) -> bool:
        item = cls._search(module_name)
        if item is None:
            return False
        return isinstance(item, Row)
