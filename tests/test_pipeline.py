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
            forward_timeline.append((self.microbatch_idx, self.partrition_idx))
            self.microbatch_idx += 1
            return self.net(x)

    microbatches = [x.unsqueeze(0) for x in torch.arange(0, N_MICROBATCHES, dtype=torch.float32).unbind()]
    microbatches = [x.requires_grad_() for x in microbatches]
    # microbatches_with_grads = [x.clone() for x in microbatches]
    partritions = [nn.Sequential(AddOne(x)) for x in range(N_PARTRITIONS)]
    devices = [torch.device("cpu") for _ in range(N_PARTRITIONS)]

    pipeline = Pipeline(microbatches, partritions, devices)
    pipeline.fit()

    assert forward_timeline == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1)]

    outputs = torch.cat([microbatch.unsqueeze(0) for microbatch in microbatches])
    loss = (outputs + 69.0).mean()
    loss.backward()

    # assert backward_timeline == [(2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)]
    assert backward_timeline == [(2, 1), (2, 0), (1, 1), (0, 1), (1, 0), (0, 0)]
    # assert all([x.grad is not None for x in microbatches])
