from torch import nn


def is_linear_parallelizable(module: nn.Module) -> bool:
    pass


def _is_column_parallel(module: nn.Module) -> bool:
    pass


def _is_row_parallel(module: nn.Module) -> bool:
    pass
