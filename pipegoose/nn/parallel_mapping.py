from typing import Optional, Tuple, cast


class ParallelInfo:
    def __init__(self, module_name: Tuple[str], **kwargs):
        self.module_name = module_name
        self.kwargs = kwargs


class ParallelMapping:
    # def __init__(self, mapping: Dict):
    #     self.mapping = mapping

    @staticmethod
    def _search(module_name: str) -> Optional[ParallelInfo]:
        """
        Search for module_name in mappings.
        """
        module_name = ParallelMapping._extract_module_name(module_name)
        for child_class in ParallelMapping.__subclasses__():
            from pipegoose.nn.tensor_parallel.parallel_mapping import (
                TensorParallelMapping,
            )

            if child_class == TensorParallelMapping:
                continue

            if hasattr(child_class, "__MAPPING__"):
                for items in child_class.__MAPPING__.values():
                    for item in items:
                        item = cast(ParallelInfo, item)
                        if any(module_name in mapping_name for mapping_name in item.module_name):
                            return item
                # NOTE: only search the first subclass of the current instance
                break

        # for items in self.mapping.values():
        #     for item in items:
        #         item = cast(ParallelInfo, item)
        #         if any(module_name in mapping_name for mapping_name in item.module_name):
        #             return item

        return None

    @staticmethod
    def _extract_module_name(module_name: str) -> str:
        if "." in module_name:
            # NOTE: transformer.h.0.self_attention.dense -> self_attention.dense
            SEPARATOR = "."
            sections = module_name.split(SEPARATOR)
            return SEPARATOR.join(sections[-2:])

        return module_name
