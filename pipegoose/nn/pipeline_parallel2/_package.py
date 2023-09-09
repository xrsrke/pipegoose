from dataclasses import dataclass


import torch


@dataclass
class Metadata:
    # pipeline
    microbatch_idx: int
    partition_idx: int

    # job
    is_forward: bool
    is_training: bool
    is_grad_enabled: bool

    # global rank
    src: int
    dst: int


class Package:
    """A data package that will be sent from one pipeline stage to another."""
    def __init__(self, data: torch.Tensor, metadata: Metadata):
        self.data = data
        self.metadata = metadata
