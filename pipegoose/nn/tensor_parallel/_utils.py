from typing import Tuple

from torch import nn


def is_linear_parallelizable(module: nn.Module) -> bool:
    pass


def _is_column_parallel(module: nn.Module) -> bool:
    pass


def _is_row_parallel(module: nn.Module) -> bool:
    pass


def get_vocab_range_idx(partition_size: int, rank: int) -> Tuple[int, int]:
    start_idx = rank * partition_size
    end_idx = start_idx + partition_size
    return start_idx, end_idx
