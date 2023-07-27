from copy import deepcopy

from torch import nn

from pipegoose.nn.data_parallel.sharding import GreedySharding


# TODO: fix this
class FakeParallelContext:
    def get_rank(self):
        return 1

    def get_world_size(self):
        return 4


def test_greedy_sharding():
    parallel_context = FakeParallelContext()
    world_size = parallel_context.get_world_size()

    def get_numel(module, param_name):
        return getattr(module, param_name).numel()

    def get_mid(module, param_name):
        return getattr(module, param_name).storage().data_ptr()

    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
    )
    copy_model = deepcopy(model)

    # TODO: add parallel_context
    sharder = GreedySharding(model, parallel_context)
    sharded_model = sharder.shard()

    for idx in [0, 2]:
        assert get_numel(sharded_model[idx], "weight") * world_size == get_numel(copy_model[idx], "weight")
        assert get_numel(sharded_model[idx], "bias") * world_size == get_numel(copy_model[idx], "bias")

        assert get_mid(sharded_model[idx], "weight") != get_mid(copy_model[idx], "weight")
        assert get_mid(sharded_model[idx], "bias") != get_mid(copy_model[idx], "bias")
