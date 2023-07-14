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


def test_run_pipeline():
    N_MICROBATCHES = 3
    N_PARTRITIONS = 2

    timeline = []

    class AddOne(nn.Module):
        def __init__(self, partrition_idx):
            super().__init__()
            self.microbatch_idx = 0
            self.partrition_idx = partrition_idx

        def forward(self, x):
            import time

            time.sleep(1)
            timeline.append((self.microbatch_idx, self.partrition_idx))
            self.microbatch_idx += 1
            return x + 1

    microbatches = [x for x in torch.arange(0, N_MICROBATCHES).unbind()]
    partritions = [nn.Sequential(AddOne(x)) for x in range(N_PARTRITIONS)]
    devices = [torch.device("cpu") for _ in range(N_PARTRITIONS)]

    pipeline = Pipeline(microbatches, partritions, devices)

    pipeline.fit()

    assert timeline == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1)]
