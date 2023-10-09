from typing import Dict, Tuple, Union

import torch
import torch.distributed as dist

from pipegoose.core.bucket.bucket import Bucket
from pipegoose.core.bucket.utils import mb_size_to_num_elements
from pipegoose.distributed.functional import all_reduce
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode

OPERATOR_MAPPING = {dist.all_reduce: all_reduce}

DistOperator = Union[
    dist.broadcast,
    dist.all_reduce,
    dist.reduce,
    dist.all_gather,
    dist.gather,
    dist.scatter,
    dist.reduce_scatter,
    dist.all_to_all,
]


class BucketDistributor:
    """
    Perform an asynchronous, distributed operation on a bucket,
    filling it until full before executing the operation.

    NOTE: Inspired from the design of FairScale's ReduceScatterBucketer
    https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/internal/reduce_scatter_bucketer.py#L74
    """

    # DIST_OPERATOR = [dist.broadcast, dist.all_reduce, dist.reduce, dist.all_gather, dist.gather, dist.scatter, dist.reduce_scatter, dist.all_to_all]

    def __init__(self, op: DistOperator, bucket_size_mb: Union[int, float], parallel_context: ParallelContext = None):
        assert op in OPERATOR_MAPPING, f"Operation must be one of {OPERATOR_MAPPING}."
        assert bucket_size_mb > 0, "Bucket size must be greater than 0."

        self.op = op
        self.bucket_size_mb = bucket_size_mb
        # NOTE: the number of elements in the bucket
        self.bucket_size = mb_size_to_num_elements(bucket_size_mb, torch.float32)
        self.parallel_context = parallel_context
        self.buckets: Dict[Tuple[torch.dtype, ParallelMode], Bucket] = {}

    @torch.no_grad()
    def execute(self, tensor: torch.Tensor, parallel_mode: ParallelMode):
        # NOTE: execute the operation if the tensor is larger than the bucket size
        if tensor.numel() > self.bucket_size:
            OPERATOR_MAPPING[self.op](tensor, parallel_context=self.parallel_context, parallel_mode=parallel_mode)
            return

        # NOTE: execute the bucket if the tensor is larger than the available space,
        # then empty and refill the bucket with the tensor
        key = (tensor.dtype, parallel_mode)
        if key not in self.buckets:

            self.buckets[key] = Bucket(self.bucket_size, tensor.dtype, self.parallel_context)
        else:
            bucket = self.buckets[key]

        bucket.add_tensor(tensor)

    def _create_bucket(self):
        pass
