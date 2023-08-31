from pipegoose.nn.tensor_parallel._utils import VocabUtility


def test_get_vocab_range_from_global_vocab_size_from_vocab_utility():
    world_size = 2
    rank = 1
    vocab_size = 10

    vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_from_global_vocab_size(world_size, rank, vocab_size)

    assert vocab_start_idx == 5
    assert vocab_end_idx == 10


def test_get_vocab_range_from_partition_size_from_vocab_utility():
    rank = 1
    partition_size = 5

    vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_idx_from_partition_size(partition_size, rank)

    assert vocab_start_idx == 5
    assert vocab_end_idx == 10
