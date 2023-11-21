import os
import pytest
import os
from unittest.mock import patch, mock_open
from pipegoose.testing.utils import spawn, init_parallel_context, find_free_port
from pipegoose.distributed import ParallelMode, ParallelContext
# from pipegoose.distributed.logger import DistributedLogger
import torch.distributed as dist

class DistributedLogger:
    def __init__(self, name, parallel_context):
        self.name = name
        self.parallel_context = parallel_context
        # Initialize file handling and logging configurations

    def _should_log(self, rank, parallel_mode):
        current_rank = self.parallel_context.get_global_rank()
        rank_check = (rank is None or rank == current_rank)
        mode_check = self.parallel_context.is_initialized(parallel_mode)
        return rank_check and mode_check

    def _save_log(self, path, log):
        log_name = self.name + ".txt"
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isfile(path + log_name):
            with open(path + log_name, 'w') as f:
                f.write(log)
        else:
            with open(path + log_name, 'a') as f:
                f.write(", " + log)

    def _log_message(self, message, level, rank=None, parallel_mode=ParallelMode.GLOBAL):
        if self._should_log(rank, parallel_mode):
            log = f"[{level}] {message}"
            print(log)
            self._save_log("logs/", log)

    def info(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "INFO", rank, parallel_mode)

    def warning(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "WARNING", rank, parallel_mode)

    def debug(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "DEBUG", rank, parallel_mode)

    def error(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "ERROR", rank, parallel_mode)




@pytest.fixture(scope="function")
def parallel_context(tensor_parallel_size, data_parallel_size, pipeline_parallel_size):
    port = find_free_port()
    rank = 0
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    context = init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size)
    return context

@pytest.fixture
def logger(parallel_context):
    return DistributedLogger("test_logger", parallel_context)

@pytest.fixture(params=[1, 2])
def tensor_parallel_size(request):
    return request.param

def should_log_test(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size, logger_name):
    context = init_parallel_context(rank, world_size, port, tensor_parallel_size, pipeline_parallel_size, data_parallel_size)
    logger = DistributedLogger(logger_name, context)
    assert logger._should_log(None, None) == True
    dist.destroy_process_group()  # Ensure cleanup

@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("data_parallel_size", [1, 2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_should_log(tensor_parallel_size, data_parallel_size, pipeline_parallel_size):
    world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    spawn(should_log_test, world_size=world_size, tensor_parallel_size=tensor_parallel_size,
          data_parallel_size=data_parallel_size, pipeline_parallel_size=pipeline_parallel_size, logger_name="test_logger")





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

    # Test when should_log is False
    with patch.object(logger, "_should_log", return_value=False):
        with patch("builtins.print") as mock_print:
            with patch.object(logger, "_save_log") as mock_save_log:
                logger._log_message("test message", "INFO")
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