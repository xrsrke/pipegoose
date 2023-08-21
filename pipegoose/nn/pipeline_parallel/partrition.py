import time
from abc import ABC, abstractclassmethod
from typing import List

import torch
from torch import nn

from pipegoose.nn.pipeline_parallel.batch import Batch


class BasePartitioner(ABC):
    @abstractclassmethod
    def split(self):
        raise NotImplementedError


class TimePartritioner(BasePartitioner):
    def __init__(self, model: nn.Module, devices: List[torch.device]):
        self.model = model

    def split(self, data):
        for layer in self.model.named_children():
            pass


def profile_elasped_time_per_layer(model: nn.Module, batches: List[Batch]) -> List[float]:
    records = []

    for layer in model.children():
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        start_time = time.time()

        # start_event.record()
        outputs = [layer(batch) for batch in batches]
        output_with_grad = [x for x in outputs if x.requires_grad]

        if len(output_with_grad) > 0:
            for output in output_with_grad:
                torch.autograd.backward(output, output)

        # end_event.record()
        end_time = time.time()
        records.append(end_time - start_time)
