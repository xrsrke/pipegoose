import copy
import time

import torch
from torch import nn

from pipegoose.pipeline import Pipeline


def test_init_pipeline():
    N_MICROBATCHES = 3
    N_PARTRITIONS = 4

    microbatches = [torch.randn(3, 3) for _ in range(N_MICROBATCHES)]
    partritions = [nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3)) for _ in range(N_PARTRITIONS)]

    pipeline = Pipeline(microbatches, partritions)

    assert pipeline.batches == microbatches
    assert pipeline.partritions == partritions


def test_forward_and_backward_pipeline():
    N_MICROBATCHES = 3
    N_PARTRITIONS = 2

    forward_timeline = []
    backward_timeline = []

    def backward_hook(module, grad_input, grad_output):
        backward_timeline.append((module.microbatch_idx - 1, module.partrition_idx))
        module.microbatch_idx -= 1

    class AddOne(nn.Module):
        def __init__(self, partrition_idx):
            super().__init__()
            self.microbatch_idx = 0
            self.partrition_idx = partrition_idx
            self.net = nn.Linear(1, 1)
            self.register_backward_hook(backward_hook)

        def forward(self, x):
            # TODO: do this necessary
            time.sleep(0.5)
            forward_timeline.append((self.microbatch_idx, self.partrition_idx))
            self.microbatch_idx += 1
            return self.net(x)

    def loss_func(x):
        return x.mean()

    # CASE 1
    # batch = torch.arange(0, N_MICROBATCHES, dtype=torch.float32)
    # microbatches = [x.unsqueeze(0) for x in batch.unbind()]
    # microbatches = [x.requires_grad_() for x in microbatches]

    # CASE 2
    batch = torch.arange(0, N_MICROBATCHES, dtype=torch.float32, requires_grad=True)
    microbatches = [x.unsqueeze(0) for x in batch.unbind()]

    partritions = [nn.Sequential(AddOne(x)) for x in range(N_PARTRITIONS)]
    devices = [torch.device("cpu") for _ in range(N_PARTRITIONS)]

    def create_non_parallel_model(partritions):
        non_parallel_model = nn.Sequential(*[copy.deepcopy(x[0]) for x in partritions])
        for param in non_parallel_model.parameters():
            param.grad = None
        return non_parallel_model

    def create_non_parallel_batch(batch):
        non_parallel_batch = copy.deepcopy(batch)
        non_parallel_batch.grad = None
        return non_parallel_batch

    non_parallel_model = create_non_parallel_model(partritions)
    non_parallel_batch = create_non_parallel_batch(batch)

    pipeline = Pipeline(microbatches, partritions, devices)
    pipeline.fit()

    assert forward_timeline == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1)]

    outputs = torch.cat([microbatch.unsqueeze(0) for microbatch in microbatches])
    non_parallel_outputs = [non_parallel_model(x.unsqueeze(0)) for x in non_parallel_batch.unbind()]

    for x, y in zip(outputs, non_parallel_outputs):
        assert torch.allclose(x, y)

    # for x in non_parallel_outputs:
    #     loss = loss_func(x)
    #     loss.backward()

    loss_func(outputs[0]).backward()

    # TODO: why does sometime (1, 0), then (0, 1)
    # sometime (0, 1), then (1, 0)
    assert backward_timeline == [(2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)] or backward_timeline == [
        (2, 1),
        (2, 0),
        (1, 1),
        (0, 1),
        (1, 0),
        (0, 0),
    ]

    # TODO: there are couple between parallel and non-parallel
    loss_func(non_parallel_outputs[0]).backward()

    assert batch.grad is not None
    for partrition in partritions:
        for param in partrition.parameters():
            assert param.grad is not None

    assert partritions[0][0].net.weight.grad == non_parallel_model[0].net.weight.grad
