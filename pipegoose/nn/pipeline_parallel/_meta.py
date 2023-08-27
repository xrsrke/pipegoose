from dataclasses import dataclass

import torch


@dataclass
class Data:
    # data
    microbatch_idx: int
    partition_idx: int
    data: torch.Tensor  # the actual data

    # job
    is_forward: bool
    is_training: bool
    is_grad_enabled: bool  # not implemented yet
    is_fp16: bool  # not implemented yet

    # communication
    src: int
    dst: int
