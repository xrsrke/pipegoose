class BucketFullError(Exception):
    """Exception raised when a bucket is full and a new item is added."""


class BucketClosedError(Exception):
    """Exception raised when a bucket is closed and a new item is added."""
