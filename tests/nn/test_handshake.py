import pytest
import torch
import torch.distributed.rpc as rpc

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.func import recv_execution_plan
from pipegoose.nn.pipeline_parallel2.sync.handshake import SchedulerHandshake
from pipegoose.testing.utils import init_parallel_context, spawn

RPC_RECEIVE_QUEUE = list()


# def recv_execution_plan(value):
#     tensor = torch.Tensor(value)
#     RPC_RECEIVE_QUEUE.append(tensor)


def recv_rpc_call(value):
    tensor = torch.Tensor(value)
    RPC_RECEIVE_QUEUE.append(tensor)


def recv_execution_plan(*args, **kwargs):
    # microbatch_idx, partition_idx = torch.unbind(task, dim=0)
    # key = (microbatch_idx, partition_idx)
    # _PIPELINE_SCHEDULER_SYNC[key] = False
    pass


def run_handshake(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_mode):
    NUM_SECONDS_IDLE = 1
    EXPECTED_TASKS = {}
    MICROBATCH_IDX = 0

    VALUE = 2

    parallel_context = init_parallel_context(
        rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    )
    local_rank = parallel_context.get_local_rank(parallel_mode)
    # local_world_size = parallel_context.get_world_size(parallel_mode)
    handshake = SchedulerHandshake(parallel_context, parallel_mode)

    # for partition_idx in range(pipeline_parallel_size):
    #     for slice_idx in range(tensor_parallel_size):
    #         EXPECTED_TASKS[(MICROBATCH_IDX, partition_idx, slice_idx)] = False

    for partition_idx in range(pipeline_parallel_size):
        EXPECTED_TASKS[(MICROBATCH_IDX, partition_idx)] = False

    # if pipeline_parallel_size > 1:
    #     assert rpc._is_current_rpc_agent_set() is True

    # if rank == 0:
    #     # tensor = torch.tensor(VALUE)

    #     rpc.rpc_sync(to=parallel_context.get_worker_name(rank=1), func=recv_execution_plan, args=(VALUE,))

    # else:
    #     while len(RPC_RECEIVE_QUEUE) < 1:
    #         time.sleep(0.1)

    #     tensor = RPC_RECEIVE_QUEUE.pop()

    #     assert tensor == VALUE

    # def get_pipeline_stage_idx():
    #     rank = parallel_context.get_local_rank(ParallelMode.PIPELINE)
    #     n_ranks_per_group = len(parallel_context.get_ranks_in_group(ParallelMode.PIPELINE))
    #     pipeline_stage_idx = rank // n_ranks_per_group
    #     return pipeline_stage_idx

    # QUEUES = {}

    # def set_session_id(session_id):
    #     handshake._session_id = session_id

    # IN_SESSION = False

    # def start():
    #     nonlocal IN_SESSION
    #     IN_SESSION = True
    # from pipegoose.nn.pipeline_parallel2.sync.func import recv_execution_plan

    if rank == 0:
        # handshake.initiate(EXPECTED_TASKS)
        # assert handshake.is_initiated() is True
        # trigger()
        worker_name = "RPC_GLOBAL_WORKER_1"
        # task = torch.tensor([1, 2])
        # from pipegoose.nn.pipeline_parallel2.sync.func import recv_execution_plan
        rpc.rpc_sync(
            to=worker_name,
            func=recv_execution_plan,
            # args=(task,)
            # func=torch.add,
            args=(torch.ones(2), 3),
        )

    # if parallel_context.is_first_rank(parallel_mode):
    #     # assert handshake.is_initiated() is False
    #     # assert handshake.session_id is None

    #     import torch
    #     import torch.distributed.rpc as rpc

    #     # handshake.initiate(EXPECTED_TASKS)
    #     worker_name = parallel_context.get_worker_name(rank=1)
    #     tensor = torch.tensor(1)
    #     rpc.rpc_sync(
    #         to=worker_name,
    #         func=recv_execution_plan,
    #         args=(tensor,)
    #     )

    # assert handshake.is_initiated() is True
    # assert handshake.session_id is not None

    # while True:
    #     if handshake.wait_until_all_confirmed() is True:
    #         for task in EXPECTED_TASKS:
    #             assert handshake.is_confirmed(task) is True
    #         break
    #     else:
    #         sleep(NUM_SECONDS_IDLE)

    # else:

    #     sleep(3)

    #     partition_idx = get_pipeline_stage_idx()
    #     TASK = (0, partition_idx)

    #     handshake.confirm(TASK)

    #     while True:
    #         if handshake.is_initiated() is True:
    #             break
    #         else:
    #             # assert handshake.session_id is None
    #             sleep(NUM_SECONDS_IDLE)

    #     # assert handshake.session_id is not None

    #     handshake.confirm(TASK)
    #     assert handshake.is_confirmed(TASK) is True

    # while True:
    #     if handshake.is_all_confirmed() is True:
    #         break
    #     else:
    #         sleep(NUM_SECONDS_IDLE)

    # assert handshake.is_all_confirmed() is True
    # assert handshake.num_confirmed() == local_world_size


@pytest.mark.parametrize("tensor_parallel_size", [1])
@pytest.mark.parametrize("pipeline_parallel_size", [2])
# @pytest.mark.parametrize("data_parallel_size", [2])
@pytest.mark.parametrize("parallel_mode", [ParallelMode.GLOBAL])
def test_syncronous_scheduler(tensor_parallel_size, pipeline_parallel_size, parallel_mode):
    DATA_PARALLEL_SIZE = 1

    WORLD_SIZE = tensor_parallel_size * pipeline_parallel_size * DATA_PARALLEL_SIZE

    spawn(
        run_handshake,
        world_size=WORLD_SIZE,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=DATA_PARALLEL_SIZE,
        parallel_mode=parallel_mode,
    )
