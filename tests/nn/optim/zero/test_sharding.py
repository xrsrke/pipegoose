from torch.optim import SGD
from transformers import AutoModel

from pipegoose.nn.optim.zero.sharding import ParameterSharding


class FakeParalellContext:
    def get_world_size(self):
        return 5

    def get_rank(self):
        return 1


def test_shard_optimizer_states():
    parallel_context = FakeParalellContext()
    world_size = parallel_context.get_world_size()

    model = AutoModel.from_pretrained("gpt2")
    optim = SGD(model.parameters(), lr=0.01)
    param_groups = optim.param_groups

    sharder = ParameterSharding(param_groups, parallel_context)
    sharded_params = sharder.shard()

    assert len(sharded_params) == world_size

    for shard in sharded_params:
        assert isinstance(shard, list)
        # for each rank, expect to have the same number of parameter groups
        assert len(shard) == len(optim.param_groups)

    total_elements = sum(param.numel() for param in model.parameters())

    def calculate_total_sharded_elements(sharded_params):
        total = 0
        for param_groups in sharded_params:
            for param_group in param_groups:
                for param in param_group["params"]:
                    total += param.numel()
        return total

    total_sharded_elements = calculate_total_sharded_elements(sharded_params)

    assert total_sharded_elements == total_elements
