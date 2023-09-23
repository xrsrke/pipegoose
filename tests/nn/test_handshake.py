import pytest

from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.pipeline_parallel2.sync.handshake import SchedulerHandshake
from pipegoose.testing.utils import init_parallel_context, spawn


def run_handshake(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_mode):
    NUM_SECONDS_IDLE = 1
    EXPECTED_TASKS = {}
    MICROBATCH_IDX = 0

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

    if parallel_context.is_first_rank(parallel_mode):
        # assert handshake.is_initiated() is False
        # assert handshake.session_id is None

        handshake.initiate(EXPECTED_TASKS)
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


# def run_handshake(
#     rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, parallel_mode
# ):
#     N_HANDSHAKE = 1
#     parallel_context = init_parallel_context(
#         rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
#     )
#     local_world_size = parallel_context.get_world_size(parallel_mode)

#     handshake = Handshake(parallel_context)

#     for _ in range(N_HANDSHAKE):
#         assert handshake.is_initiated() is False
#         assert handshake.is_all_confirmed() is False

#         if parallel_context.is_first_rank() is True:
#             handshake.initiate()
#             assert handshake.is_initiated() is True
#             assert handshake.session_id is not None

#             with pytest.raises(Exception):
#                 # NOTE: can't initiate twice
#                 # have to wait for the other ranks to confirm first
#                 # then end the current handshake session
#                 handshake.initiate()
#         else:
#             # TODO: test that the other ranks cannot initiate
#             assert handshake.is_confirmed() is False
#             handshake.confirm()
#             assert handshake.is_confirmed() is True

#             with pytest.raises(Exception):
#                 # cannot confirm twice
#                 handshake.confirm()

#         while True:
#             if handshake.is_all_confirmed() is True:
#                 break
#             else:
#                 sleep(0.1)

#         assert handshake.is_all_confirmed() is True
#         assert handshake.num_confirmed() == local_world_size


@pytest.mark.parametrize("tensor_parallel_size", [2])
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
