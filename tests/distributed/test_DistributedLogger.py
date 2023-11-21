import os
import pytest
from unittest.mock import patch, mock_open
from pipegoose.testing.utils import spawn, init_parallel_context, find_free_port
from pipegoose.distributed import ParallelMode, ParallelContext
# from pipegoose.distributed.logger import DistributedLogger
import torch.distributed as dist
from multiprocessing import Process

from pipegoose.distributed import ParallelMode
import os


import socket
import random

def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    for _ in range(max_port - min_port):  # Limit the number of attempts
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))  # Binding to 0.0.0.0
                return port
        except OSError:
            continue  # Ignore the error and try a different port
    raise RuntimeError("No free port found in the specified range")


class DistributedLogger:
    def __init__(self, name, parallel_context):
        self.name = name
        self.parallel_context = parallel_context
        # Initialize file handling and logging configurations

    def _should_log(self, rank, parallel_mode):
        current_rank = self.parallel_context.get_global_rank()
        rank_check = (rank is None or rank == current_rank)

        # Check if the current parallel mode is initialized and if the current process is part of it
        mode_check = self.parallel_context.is_initialized(parallel_mode)

        return rank_check and mode_check

    
    def _save_log(self, path, log):
        # Add code to save the log file to the specified path
        log_name = self.name + ".txt"

        # check if path directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        
        # check if log file exists
        if not os.path.isfile(path + log_name):
            with open(os.path.join(path, log_name), 'w') as f:
                f.write(log)
        else:
            with open(os.path.join(path, log_name), 'a') as f:
                f.write(", " +log)


    def _log_message(self, message, level, rank=None, parallel_mode=ParallelMode.GLOBAL):
        if self._should_log(rank, parallel_mode):
            # Print and save the message
            log = f"[{level}] {message}"
            print(log)
            # Add code to save the message to a file
            self._save_log("logs/", log)
        # else:
        #     print(f"Process {self.parallel_context.get_global_rank()} is not part of the {parallel_mode} parallel mode")
            


    # The logging methods (info, warning, debug, error) remain the same

    def info(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "INFO", rank, parallel_mode)

    def warning(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "WARNING", rank, parallel_mode)

    def debug(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "DEBUG", rank, parallel_mode)

    def error(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "ERROR", rank, parallel_mode)
    
    
    




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
        p = Process(target=should_log_test, args=(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, "test_logger"))
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
                mock_file.assert_called_once_with("logs/test_logger.txt", "w")
                mock_file().write.assert_called_once_with("test log")

    # Test when log file does not exist
    with patch.object(os.path, "exists", return_value=True):
        with patch("builtins.open", mock_open()) as mock_file:
            logger._save_log("logs/", "test log")
            mock_file.assert_called_once_with("logs/test_logger.txt", "a")
            mock_file().write.assert_called_once_with(", test log")

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