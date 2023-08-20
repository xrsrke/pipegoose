from typing import Any

import torch

from pipegoose.distributed.functional import all_reduce
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode


class VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, parallel_logits: torch.Tensor, targets: torch.Tensor, parallel_context: ParallelContext):
        def normalize_logits(parallel_logits):
            logit_max = torch.max(parallel_logits, dim=-1)[0]
            logit_max = all_reduce(
                logit_max,
                op=torch.distributed.ReduceOp.MAX,
                parallel_context=parallel_context,
                parallel_mode=ParallelMode.TENSOR,
            )
            normalized_parallel_logits = parallel_logits - logit_max.unsqueeze(-1)
            return normalized_parallel_logits

        def get_vocab_start_end_idx(parallel_logits):
            partition_vocab_size = parallel_logits.shape[-1]
            rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
            vocab_start_idx = rank * partition_vocab_size
            vocab_end_idx = vocab_start_idx + partition_vocab_size
            return vocab_start_idx, vocab_end_idx

        parallel_logits = normalize_logits(parallel_logits)
        vocab_start_idx, vocab_end_idx = get_vocab_start_end_idx(parallel_logits)

        target_mask = (targets < vocab_start_idx) | (targets >= vocab_end_idx)
        masked_target = targets.clone() - vocab_start_idx
        masked_target[target_mask] = 0

        parallel_logits = parallel_logits.view(-1, parallel_logits.shape[-1])
        masked_target = masked_target.view(-1)
        predicted_logits = parallel_logits[:, masked_target]
        predicted_logits[target_mask.view(-1)] = 0.0

        predicted_logits = all_reduce(predicted_logits, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

        return predicted_logits

    @staticmethod
    def backward(ctx):
        pass
