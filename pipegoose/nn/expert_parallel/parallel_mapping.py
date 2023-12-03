from pipegoose.nn.parallel_mapping import ParallelInfo, ParallelMapping


class MLP(ParallelInfo):
    pass


class ExpertParallelMapping(ParallelMapping):
    __MAPPING__ = {
        "bloom-560m": [MLP("mlp")],
    }

    @staticmethod
    def is_mlp(module_name: str) -> bool:
        item = ExpertParallelMapping._search(module_name)
        if item is None:
            return False
        return isinstance(item, MLP)
