import copy

import torch

from pipegoose.nn.data_parallel.sharding import GreedySharding


def test_greedy_sharding(parallel_context):
    rank = parallel_context.get_rank()

    params = torch.randn(10, 5)
    params_data = params.clone()
    params_mid = copy.deepcopy(params.storage().data_ptr())

    # TODO: add parallel_context
    sharder = GreedySharding(params, parallel_context)
    shared_param = sharder.shard()

    assert params.storage().size() == 0
    assert params.storage().data_ptr() == 0
    assert shared_param.storage().data_ptr() != params_mid
    assert shared_param.storage().size() == 10
    assert shared_param == params_data[rank]
