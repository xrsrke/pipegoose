from torch._utils import _flatten_dense_tensors
from torch.optim import Optimizer

from pipegoose.distributed.functional import broadcast
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.optim import BaseDistributedOptimizer
from pipegoose.optim.zero.sharding import OptimizerStateSharding


class DistributedOptimizer(BaseDistributedOptimizer):
    """ZeRO-1 optimizer that works natively in 3D parallelism."""

    def __init__(self, optim: Optimizer, parallel_context: ParallelContext):
        self.optim = optim
        self.parallel_context = parallel_context

        self._master_params = {}
        self._setup_local_optim()

    def _setup_local_optim(self):
        """Setup local optimizer."""

        # NOTE: shard and assign the corresponding local parameters to the local optimizer
        for i, param_groups in enumerate(self.optim.param_groups):
            self._master_params[i] = param_groups["params"]

        sharded_param_groups = OptimizerStateSharding(
            self.optim.param_groups, self.parallel_context, ParallelMode.DATA
        ).shard()
        ranks_in_group = self.parallel_context.get_ranks_in_group(ParallelMode.DATA)
        self._rank_to_params = {rank: params for rank, params in zip(ranks_in_group, sharded_param_groups)}

        local_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        self.optim.param_groups = self._rank_to_params[local_rank]

    @property
    def defaults(self):
        """Return the default hyperparameters."""
        return self.optim.defaults

    @property
    def param_groups(self):
        """Return the parameter groups."""
        return self.optim.param_groups

    def add_param_group(self, *args, **kwargs):
        """Add a new parameter group to the optimizer."""
        self.optim.add_param_group(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Load the optimizer state."""
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Return the state of the optimizer"""
        return self.optim.state_dict(*args, **kwargs)

    # def _update_master_params(self):
    #     """Update the master parameters from the updated local parameters."""
    #     local_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)

    #     for i, param_groups in enumerate(self.optim.param_groups):
    #         updated_params = all_gather(param_groups["params"], self.parallel_context, ParallelMode.DATA)
    #         self._master_params[i] = updated_params[local_rank]

    def step(self, *args, **kwargs):
        # NOTE: each rank updates its subset of parameters using the local optimizer
        self.optim.step(*args, **kwargs)

        # NOTE: update the master parameters from the updated local parameters
        # self._update_master_params()

        # NOTE: gather the full updated parameters from all ranks

        # NOTE: each model replicas broadcast the updated parameters to other model replicas
        for rank, param_groups in self._rank_to_params.items():
            flatten_params = _flatten_dense_tensors(param_groups[0]["params"])
            broadcast(flatten_params, src=rank, parallel_context=self.parallel_context, parallel_mode=ParallelMode.DATA)
        assert 1 == 1

    def zero_grad(self, *args, **kwargs):
        """Zero out gradients."""
        self.optim.zero_grad(*args, **kwargs)
