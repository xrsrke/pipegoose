import torch

_PIPELINE_SCHEDULER_SYNC = {}


def get_execution_plan():
    return _PIPELINE_SCHEDULER_SYNC


def recv_execution_plan(task):
    microbatch_idx, partition_idx = torch.unbind(task, dim=0)
    microbatch_idx = microbatch_idx.item()
    partition_idx = partition_idx.item()
    key = (microbatch_idx, partition_idx)
    _PIPELINE_SCHEDULER_SYNC[key] = False
