import time

import torch
from torch import nn

from pipegoose.pipeline import Pipeline


class TestPipeline:
    N_MICROBATCHES = 3
    N_partitions = 2

    def setup_method(self):
        forward_timeline = []
        backward_timeline = []

        def backward_hook(module, grad_input, grad_output):
            backward_timeline.append((module.microbatch_idx - 1, module.partition_idx))
            module.microbatch_idx -= 1

        class AddOne(nn.Module):
            def __init__(self, partition_idx, is_logging):
                super().__init__()
                self.microbatch_idx = 0
                self.partition_idx = partition_idx
                self.is_logging = is_logging
                self.net = nn.Linear(1, 1)
                self.register_backward_hook(backward_hook)

            def forward(self, x):
                if self.is_logging:
                    time.sleep(0.5)
                    forward_timeline.append((self.microbatch_idx, self.partition_idx))
                    self.microbatch_idx += 1

                return self.net(x)

        self.forward_timeline = forward_timeline
        self.backward_timeline = backward_timeline
        self.AddOne = AddOne
        self.batch = torch.arange(0, self.N_MICROBATCHES, dtype=torch.float32, requires_grad=True)
        self.microbatches = [x.unsqueeze(0) for x in self.batch.unbind()]
        self.partitions = [nn.Sequential(self.AddOne(partition_idx=x, is_logging=True)) for x in range(self.N_partitions)]
        self.devices = [torch.device("cpu") for _ in range(self.N_partitions)]
        self.non_parallel_model = self.create_non_parallel_model(self.partitions)
        self.non_parallel_batch = self.create_non_parallel_batch(self.batch)

        self.results = {}

    def create_non_parallel_model(self, partitions):
        non_parallel_model = nn.Sequential(*[self.AddOne(partition_idx=x, is_logging=False) for x in range(self.N_partitions)])
        for non_parallel_layer, original_partition in zip(non_parallel_model, partitions):
            non_parallel_layer.load_state_dict(original_partition[0].state_dict())
        return non_parallel_model

    def create_non_parallel_batch(self, batch):
        non_parallel_batch = batch.detach().clone()
        non_parallel_batch.grad = None
        return non_parallel_batch

    def test_forward_pass_and_backward(self):
        pipeline = Pipeline(self.microbatches, self.partitions, self.devices)

        assert pipeline.batches == self.microbatches
        assert pipeline.partitions == self.partitions

        pipeline.fit()

        assert self.forward_timeline == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1)]

        outputs = self.microbatches
        non_parallel_outputs = [self.non_parallel_model(x.unsqueeze(0)) for x in self.non_parallel_batch.unbind()]
        for x, y in zip(outputs, non_parallel_outputs):
            assert torch.allclose(x, y)

        def loss_func(x):
            return x.mean()

        for x in outputs:
            loss = loss_func(x)
            loss.backward()
        assert self.backward_timeline == [(2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)] or self.backward_timeline == [
            (2, 1),
            (2, 0),
            (1, 1),
            (0, 1),
            (1, 0),
            (0, 0),
        ]

        for x in non_parallel_outputs:
            loss = loss_func(x)
            loss.backward()

        assert self.batch.grad is not None
        for partrition in self.partitions:
            for param in partrition.parameters():
                assert param.grad is not None

        for partition_idx in range(self.N_partitions):
            for w1, w2 in zip(
                self.partitions[partition_idx].parameters(), self.non_parallel_model[partition_idx].parameters()
            ):
                assert torch.allclose(w1.grad, w2.grad)
