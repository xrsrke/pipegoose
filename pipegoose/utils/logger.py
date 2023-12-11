import datetime
import inspect
import sys
import os
import wandb
import glob
import re
import os

class Logger:
    # https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/lib/logger.py
    """ The Logger class is a singleton. It contains all the utilities
        for logging variables in a key-value dictionary.
        It can also be considered as a replacement for the print function.

        .. code-block:: python

            Logger(dir_logs='logs/mnist')
            Logger().flush() # write the logs.json
            Logger()("Launching training procedures") # written to logs.txt
            > [I 2018-07-23 18:58:31] ...trap/engines/engine.py.80: Launching training procedures
    """

    DEBUG = -1
    INFO = 0
    SUMMARY = 1
    WARNING = 2
    ERROR = 3
    SYSTEM = 4
    _instance = None
    indicator = {DEBUG: 'D', INFO: 'I', SUMMARY: 'S', WARNING: 'W', ERROR: 'E', SYSTEM: 'S'}

    class Colors:
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        GREY = 30
        RED = 31
        GREEN = 32
        YELLOW = 33
        BLUE = 34
        PURPLE = 35
        SKY = 36
        WHITE = 37
        BACKGROUND = 10
        LIGHT = 60

        @staticmethod
        def code(value):
            return '\033[{}m'.format(value)

    colorcode = {
        DEBUG: Colors.code(Colors.GREEN),
        INFO: Colors.code(Colors.GREY + Colors.LIGHT),
        SUMMARY: Colors.code(Colors.BLUE + Colors.LIGHT),
        WARNING: Colors.code(Colors.YELLOW + Colors.LIGHT),
        ERROR: Colors.code(Colors.RED + Colors.LIGHT),
        SYSTEM: Colors.code(Colors.WHITE + Colors.LIGHT)
    }

    compactjson = True
    log_level = None  # log level
    dir_logs = None
    path_json = None
    path_txt = None
    file_txt = None
    name = None
    max_lineno_width = 3
    
    def __new__(cls, dir_logs=None, name='logs'):
        if Logger._instance is None:
            Logger._instance = object.__new__(Logger)

            if dir_logs:
                Logger._instance.name = name
                Logger._instance.dir_logs = dir_logs
                Logger._instance.path_txt = os.path.join(dir_logs, '{}.txt'.format(name))
                Logger._instance.file_txt = open(os.path.join(dir_logs, '{}.txt'.format(name)), 'a+')
                # NOTE: Support json or CSV ? 
                # Logger._instance.path_json = os.path.join(dir_logs, '{}.json'.format(name))
                # Logger._instance.reload_json()
            else:
                Logger._instance.log_message('No logs files will be created (dir_logs attribute is empty)',
                                             log_level=Logger.WARNING)

        return Logger._instance

    def __call__(self, *args, **kwargs):
        return self.log_message(*args, **kwargs, stack_displacement=2)

    def log_message(self, *message, log_level=INFO, break_line=True, print_header=True, stack_displacement=1,
                    raise_error=True, adaptive_width=True):

        if self.dir_logs and not self.file_txt:
            raise Exception('Critical: Log file not defined. Do you have write permissions for {}?'.format(self.dir_logs))

        caller_info = inspect.getframeinfo(inspect.stack()[stack_displacement][0])
        message = ' '.join([str(m) for m in list(message)])

        if print_header:
            message_header = '[{} {:%Y-%m-%d %H:%M:%S}]'.format(self.indicator[log_level],
                                                                datetime.datetime.now())
            filename = caller_info.filename
            if adaptive_width:
                # allows the lineno_width to grow when necessary
                lineno_width = len(str(caller_info.lineno))
                self.max_lineno_width = max(lineno_width, self.max_lineno_width)
            else:
                # manually fix it to 3 numbers
                lineno_width = 3

            if len(filename) > 28 - self.max_lineno_width:
                filename = '...{}'.format(filename[-22 - (self.max_lineno_width - lineno_width):])

            message_locate = '{}.{}:'.format(filename, caller_info.lineno)
            message_logger = '{} {} {}'.format(message_header, message_locate, message)
            message_screen = '{}{}{}{} {} {}'.format(self.Colors.BOLD,
                                                     self.colorcode[log_level],
                                                     message_header,
                                                     self.Colors.END,
                                                     message_locate,
                                                     message)
        else:
            message_logger = message
            message_screen = message

        if break_line:
            print(message_screen)
            if self.dir_logs:
                self.file_txt.write('%s\n' % message_logger)
        else:
            print(message_screen, end='')
            sys.stdout.flush()
            if self.dir_logs:
                self.file_txt.write(message_logger)

        if self.dir_logs:
            self.file_txt.flush()
        if log_level == self.ERROR and raise_error:
            raise Exception(message)

    def update_log_file(self, path_src, path_dst):
        """
        Append content of file at path_src to file at path_dst
        """

        with open(path_src, 'r') as f:
            lines_src = f.readlines()
        
        with open(path_dst, 'r') as f:
            lines_dst = f.readlines()

        with open(path_dst, 'w') as f:
            f.writelines(lines_src + ["\n"] + lines_dst)