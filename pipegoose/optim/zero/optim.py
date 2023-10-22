from torch.optim import Optimizer

from pipegoose.distributed.functional import broadcast
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.optim.base_optim import BaseDistributedOptimizer
from pipegoose.optim.zero.sharding import OptimizerStateSharding
from pipegoose.optim.zero.utils import flatten_a_list_tensor


class DistributedOptimizer(BaseDistributedOptimizer):
    """ZeRO-1 optimizer that works natively in 3D parallelism."""

    def __init__(self, optim: Optimizer, parallel_context: ParallelContext):
        self.optim = optim
        self.parallel_context = parallel_context

        self._setup_local_optim()

    def _setup_local_optim(self):
        """Setup local optimizer."""
        sharded_param_groups = OptimizerStateSharding(
            self.optim.param_groups, self.parallel_context, ParallelMode.DATA
        ).shard()
        ranks_in_group = self.parallel_context.get_ranks_in_group(ParallelMode.DATA)
        self._rank_to_param_groups = {rank: params for rank, params in zip(ranks_in_group, sharded_param_groups)}

        dp_local_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        dp_global_rank = self.parallel_context.get_global_rank_from_local_rank(dp_local_rank, ParallelMode.DATA)
        self.optim.param_groups = self._rank_to_param_groups[dp_global_rank]

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

    def step(self, *args, **kwargs):
        # NOTE: each rank updates its subset of parameters using the local optimizer
        self.optim.step(*args, **kwargs)

        # NOTE: gather the full updated parameters from all ranks

        # NOTE: each model replicas broadcast the updated parameters to other model replicas
        for rank, param_groups in self._rank_to_param_groups.items():
            for param_group in param_groups:
                flatten_params = flatten_a_list_tensor(param_group["params"])
                broadcast(flatten_params, src=rank, parallel_context=self.parallel_context, parallel_mode=ParallelMode.DATA)

    def zero_grad(self):
        """Zero out gradients."""
        # NOTE: we zero out the gradients of the all parameters
        for param_groups in self._rank_to_param_groups.values():
            for param_group in param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        param.grad = None
