import os

import pytest

from torch import nn

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.utils import from_pretrained, save_pretrained
from pipegoose.constants import CHECKPOINT_WEIGHTS_NAME
from pipegoose.testing.utils import spawn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    parallel_context = ParallelContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        host="localhost",
        port=port,
        seed=69,
        backend="gloo",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    return parallel_context


def run_save_and_load_pretrained(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    # TODO: add automatically create the directory if it does not exist
    # and then delete it after the test is done

    CHECKPOINT_WEIGHTS_PATH = "./downloads"

    def zero_weights(m):
        """Sets all model weights to zero."""
        if isinstance(m, nn.Module):
            if hasattr(m, 'weight'):
                nn.init.constant_(m.weight, 0)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)

            for layer in m.children():
                zero_weights(layer)

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )

    model = SimpleModel()

    save_pretrained(model, ckp_path=CHECKPOINT_WEIGHTS_PATH, parallel_context=parallel_context)

    tp_rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    pp_rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)

    assert os.path.exists(os.path.join(CHECKPOINT_WEIGHTS_PATH, CHECKPOINT_WEIGHTS_NAME.format(tp_rank, pp_rank)))

    zero_weights(model)
    assert model.fc.weight.sum() == 0

    from_pretrained(model, ckp_path=CHECKPOINT_WEIGHTS_PATH, parallel_context=parallel_context)
    assert model.fc.weight.sum() != 0


@pytest.mark.parametrize(
    "tensor_parallel_size, pipeline_parallel_size, data_parallel_size",
    [(1, 1, 1), (2, 2, 2)]
)
def test_save_and_load_pretrained(tensor_parallel_size, pipeline_parallel_size, data_parallel_size):
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    spawn(
        run_save_and_load_pretrained,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
