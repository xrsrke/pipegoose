import os

import torch
from torch import nn

from pipegoose.constants import CHECKPOINT_PATH_NAME, CHECKPOINT_WEIGHTS_NAME
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


def from_pretrained(module: nn.Module, ckp_path: str, parallel_context: ParallelContext):
    """Load the weights of a pretrained parallelized model."""
    tp_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    pp_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)

    ckp_name = CHECKPOINT_WEIGHTS_NAME.format(tp_rank, pp_rank)
    ckp_path = os.path.join(ckp_path, ckp_name)

    if os.path.exists(ckp_path):
        state_dict = torch.load(ckp_path)
        module.load_state_dict(state_dict)
    else:
        raise ValueError(f"ckp_path {ckp_path} does not exist")


def save_pretrained(
    module: nn.Module,
    ckp_name: str = CHECKPOINT_WEIGHTS_NAME,
    ckp_path: str = CHECKPOINT_PATH_NAME,
    parallel_context: ParallelContext = None,
):
    """
    Save the weights of a pretrained parallelized model.

    NOTE: Assume that the model is already parallelized and discarded
    the weights of parts that a node is not responsible for.
    """
    assert isinstance(
        parallel_context, ParallelContext
    ), f"parallel_context must be an instance of ParallelContext, got {type(parallel_context)}"

    tp_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    pp_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
    ckp_name = ckp_name.format(tp_rank, pp_rank)

    if os.path.isdir(ckp_path):
        state_dict = module.state_dict()
        torch.save(state_dict, os.path.join(ckp_path, ckp_name))
    else:
        raise ValueError(f"ckp_path {ckp_path} does not exist")
