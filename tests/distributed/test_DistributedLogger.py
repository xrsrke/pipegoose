import os
from multiprocessing import Process
from unittest.mock import mock_open, patch

import pytest
import torch.distributed as dist

from pipegoose.distributed import ParallelMode
from pipegoose.distributed.logger import DistributedLogger
from pipegoose.testing.utils import find_free_port, init_parallel_context


@pytest.fixture
def logger(parallel_context):
    return DistributedLogger("test_logger", parallel_context)


@pytest.fixture(params=[1, 1])
def tensor_parallel_size(request):
    return request.param


def should_log_test(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, logger_name):
    context = init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size)
    logger = DistributedLogger(logger_name, context)
    assert logger._should_log(rank, ParallelMode.GLOBAL) == True
    dist.destroy_process_group()


@pytest.mark.parametrize("tensor_parallel_size", [1, 1])
@pytest.mark.parametrize("data_parallel_size", [1, 1])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 1])
def test_should_log(tensor_parallel_size, data_parallel_size, pipeline_parallel_size):
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    port = find_free_port()
    processes = []

    for rank in range(world_size):
        p = Process(
            target=should_log_test,
            args=(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, "test_logger"),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def test_save_log(logger):
    # Test when path directory does not exist
    with patch.object(os.path, "exists", return_value=False):
        with patch.object(os, "makedirs") as mock_makedirs:
            with patch("builtins.open", mock_open()) as mock_file:
                logger._save_log("logs/", "test log")
                mock_makedirs.assert_called_once_with("logs/")
                mock_file.assert_called_once_with("logs/test_logger.txt", "a")
                mock_file().write.assert_called_once_with("test log")

    # Test when log file does not exist
    with patch.object(os.path, "exists", return_value=True):
        with patch("builtins.open", mock_open()) as mock_file:
            logger._save_log("logs/", "test log")
            mock_file.assert_called_once_with("logs/test_logger.txt", "a")
            mock_file().write.assert_called_once_with("test log")

    # Test when log file exists
    with patch.object(os.path, "exists", return_value=True):
        with patch("builtins.open", mock_open()) as mock_file:
            with patch.object(os.path, "isfile", return_value=True):
                logger._save_log("logs/", "test log")
                mock_file.assert_called_once_with("logs/test_logger.txt", "a")
                mock_file().write.assert_called_once_with(", test log")


def test_log_message(logger):
    # Test when should_log is True
    with patch.object(logger, "_should_log", return_value=True):
        with patch("builtins.print") as mock_print:
            with patch.object(logger, "_save_log") as mock_save_log:
                logger._log_message("test message", "INFO")
                mock_print.assert_called_once_with("[INFO] test message")
                mock_save_log.assert_called_once_with("logs/", "[INFO] test message")

    with patch.object(logger, "_should_log", return_value=False) as mock_should_log:
        with patch("builtins.print") as mock_print:
            with patch.object(logger, "_save_log") as mock_save_log:
                logger._log_message("test message", "INFO")
                mock_should_log.assert_called()  # Ensure _should_log is being called
                mock_print.assert_not_called()
                mock_save_log.assert_not_called()


def test_log_message_with_rank_zero_global(logger):
    # Test _log_message with rank=0 and parallel_mode=ParallelMode.GLOBAL
    with patch.object(logger, "_should_log", return_value=True):
        with patch("builtins.print") as mock_print:
            with patch.object(logger, "_save_log") as mock_save_log:
                logger._log_message("test message", "INFO", rank=0, parallel_mode=ParallelMode.GLOBAL)
                mock_print.assert_called_once_with("[INFO] test message")
                mock_save_log.assert_called_once_with("logs/", "[INFO] test message")

    # You can also test the behavior when _should_log returns False
    with patch.object(logger, "_should_log", return_value=False):
        with patch("builtins.print") as mock_print:
            with patch.object(logger, "_save_log") as mock_save_log:
                logger._log_message("test message", "INFO", rank=0, parallel_mode=ParallelMode.GLOBAL)
                mock_print.assert_not_called()
                mock_save_log.assert_not_called()


def test_info(logger):
    with patch.object(logger, "_log_message") as mock_log_message:
        logger.info("test message")
        mock_log_message.assert_called_once_with("test message", "INFO", None, ParallelMode.GLOBAL)


def test_warning(logger):
    with patch.object(logger, "_log_message") as mock_log_message:
        logger.warning("test message")
        mock_log_message.assert_called_once_with("test message", "WARNING", None, ParallelMode.GLOBAL)


def test_debug(logger):
    with patch.object(logger, "_log_message") as mock_log_message:
        logger.debug("test message")
        mock_log_message.assert_called_once_with("test message", "DEBUG", None, ParallelMode.GLOBAL)


def test_error(logger):
    with patch.object(logger, "_log_message") as mock_log_message:
        logger.error("test message")
        mock_log_message.assert_called_once_with("test message", "ERROR", None, ParallelMode.GLOBAL)
