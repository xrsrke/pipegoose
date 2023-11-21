from pipegoose.distributed import ParallelMode
import os


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
            with open(path + log_name, 'w') as f:
                f.write(log)
        else:
            with open(path + log_name, 'a') as f:
                f.write(", " +log)


    def _log_message(self, message, level, rank=None, parallel_mode=ParallelMode.GLOBAL):
        if self._should_log(rank, parallel_mode):
            # Print and save the message
            log = f"[{level}] {message}"
            print(log)
            # Add code to save the message to a file
            self._save_log("logs/", log)
        else:
            print(f"Process {self.parallel_context.get_global_rank()} is not part of the {parallel_mode} parallel mode")
            


    # The logging methods (info, warning, debug, error) remain the same

    def info(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "INFO", rank, parallel_mode)

    def warning(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "WARNING", rank, parallel_mode)

    def debug(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "DEBUG", rank, parallel_mode)

    def error(self, message, rank=None, parallel_mode=ParallelMode.GLOBAL):
        self._log_message(message, "ERROR", rank, parallel_mode)
    
    
    
