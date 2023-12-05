from enum import Enum


class ParallelMode(Enum):
    GLOBAL = "global"

    TENSOR = "tensor"
    PIPELINE = "pipeline"
    DATA = "data"

    # NOTE: for expert data parallelism
    EXPERT = "expert"
