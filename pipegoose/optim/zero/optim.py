from torch.optim import Optimizer

from pipegoose.distributed.functional import broadcast
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.optim import BaseDistributedOptimizer
from pipegoose.optim.zero.sharding import ParameterSharding


class DistributedOptimizer(BaseDistributedOptimizer):
    """ZeRO-1 optimizer that works natively in 3D parallelism."""

    def __init__(
        self,
        optim: Optimizer,
        parallel_context: ParallelContext,
    ):
        self.optim = optim
        self.parallel_context = parallel_context

        self._master_params = None

        self._setup_local_optim()

    # def _sync_hyperparams(self, source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]):
    #     for source_group, destination_group in zip(source, destination):
    #         for k in source_group.keys():
    #             if k != "params":
    #                 destination_group[k] = source_group[k]

    def _setup_local_optim(self):
        """Setup local optimizer."""
        local_rank = self.parallel_context.get_local_rank(ParallelMode.DATA)

        # optim = self._optim_constructor(self.params, **self.default)
        # NOTE: shard and assign the corresponding local parameters to the local optimizer
        sharded_param_groups = ParameterSharding(self.optim.param_groups, self.parallel_context, ParallelMode.DATA).shard()
        self._master_params = {rank: params for rank, params in enumerate(sharded_param_groups)}
        self.optim.param_groups = sharded_param_groups[local_rank]

    # def _construct_local_optim(self, local_params: Dict[str, torch.Tensor]) -> Optimizer:
    #     optim = self._optim_constructor(local_params, **self.default)
    #     return optim

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

    def state_dict(self):
        """Return the state of the optimizer"""
        return self.optim.state_dict()

    def step(self, *args, **kwargs):
        # NOTE: each rank updates its subset of parameters using the local optimizer
        self.optim.step(*args, **kwargs)

        # NOTE: update the master parameters from the updated local parameters

        # NOTE: each model replicas broadcast the updated parameters to other model replicas
        rank = self.parallel_context.get_local_rank(ParallelMode.DATA)
        for group in self.optim.param_groups:
            for p in group["params"]:
                if p.requires_grad is True:
                    broadcast(p, src=rank, parallel_context=self.parallel_context, parallel_mode=ParallelMode.DATA)

    def zero_grad(self, *args, **kwargs):
        """Zero out gradients."""
        self.optim.zero_grad(*args, **kwargs)
