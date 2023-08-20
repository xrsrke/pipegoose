from typing import Tuple

from torch import nn


def is_linear_parallelizable(module: nn.Module) -> bool:
    pass


def _is_column_parallel(module: nn.Module) -> bool:
    pass


def _is_row_parallel(module: nn.Module) -> bool:
    pass


def get_vocab_range_idx(num_embeddings: int, rank: int, world_size: int) -> Tuple[int, int]:
    num_embeddings_per_partition = num_embeddings // world_size
    start_idx = rank * num_embeddings_per_partition
    end_idx = start_idx + num_embeddings_per_partition
    return start_idx, end_idx
