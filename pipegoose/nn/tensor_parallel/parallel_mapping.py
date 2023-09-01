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
            Column(("dense_h_to_4h", "query_key_value")),
            Row(("dense_4h_to_h", "dense")),
        ],
    }

    @staticmethod
    def _search(module_name: str) -> TensorParallelInformation:
        """
        Search for module_name in mappings.
        """
        module_name = ParallelMapping._extract_module_name(module_name)
        for _, items in ParallelMapping.__MAPPING__.items():
            for item in items:
                if module_name in item.module_name:
                    return item
        return None

    @staticmethod
    def _extract_module_name(module_name: str) -> str:
        def _check_module_name_in_named_modules(module_name: str) -> bool:
            return '.' in module_name

        if _check_module_name_in_named_modules(module_name) is True:
            return module_name.split('.')[-1]

        return module_name

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
