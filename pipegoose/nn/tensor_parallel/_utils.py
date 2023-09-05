from typing import Tuple


class VocabUtility:
    @staticmethod
    def get_vocab_range_idx_from_partition_size(partition_size: int, rank: int) -> Tuple[int, int]:
        start_idx = rank * partition_size
        end_idx = start_idx + partition_size
        return start_idx, end_idx

    @staticmethod
    def get_vocab_range_from_global_vocab_size(world_size, rank, vocab_size):
        partition_size = vocab_size // world_size
        return VocabUtility.get_vocab_range_idx_from_partition_size(partition_size, rank)
