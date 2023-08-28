import pytest

from pipegoose.nn.tensor_parallel._utils import get_vocab_range_idx


@pytest.mark.parametrize("partition_size, rank, start_idx, end_idx", [(5, 1, 5, 10), (10, 1, 10, 20)])
def test_get_vocab_range_idx(partition_size, rank, start_idx, end_idx):
    vocab_start_idx, vocab_end_idx = get_vocab_range_idx(partition_size, rank)

    assert vocab_start_idx == start_idx
    assert vocab_end_idx == end_idx
