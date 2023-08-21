from typing import List

import torch
from torch import nn

from pipegoose.nn.pipeline_parallel.partrition import BasePartitioner


class GPipe:
    def __init__(self, module: nn.Sequential, devices: List[torch.device], partritioner: BasePartitioner) -> None:
        pass
