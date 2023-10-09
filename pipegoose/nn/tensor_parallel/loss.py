from typing import Any, Tuple

import torch
from einops import rearrange
from torch import nn
from torchtyping import TensorType

from pipegoose.distributed.functional import all_reduce
from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn.tensor_parallel._utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        parallel_logits: TensorType["batch_size", "n_samples", "vocab_size"],
        targets: TensorType["batch_size", "n_samples"],
        parallel_context: ParallelContext,
    ) -> torch.Tensor:
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

        def get_predicted_logits(parallel_logits, targets):
            # TODO: split get masked targets into another function
            rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
            partition_size = parallel_logits.shape[-1]
            vocab_start_idx, vocab_end_idx = VocabUtility.get_vocab_range_idx_from_partition_size(partition_size, rank)

            target_mask = (targets < vocab_start_idx) | (targets >= vocab_end_idx)
            masked_targets = targets.clone() - vocab_start_idx
            masked_targets[target_mask] = 0

            parallel_logits = rearrange(parallel_logits, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size")
            masked_targets_1d = rearrange(masked_targets, "batch_size seq_len -> (batch_size seq_len)")
            predicted_logits = parallel_logits[torch.arange(masked_targets_1d.size(0)), masked_targets_1d]
            predicted_logits = torch.where(
                rearrange(target_mask, "batch_size seq_len -> (batch_size seq_len)") == False, predicted_logits, 0.0
            )
            predicted_logits = all_reduce(
                predicted_logits, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR
            )
            return predicted_logits, target_mask, masked_targets_1d

        # NOTE: parallel cross entropy still works without normalizing logits
        parallel_logits = normalize_logits(parallel_logits)
        predicted_logits, target_mask, masked_targets_1d = get_predicted_logits(parallel_logits, targets)

        exp_logits = torch.exp(parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = all_reduce(sum_exp_logits, parallel_context=parallel_context, parallel_mode=ParallelMode.TENSOR)

        loss = torch.log(sum_exp_logits) - predicted_logits
        ctx.save_for_backward(exp_logits, target_mask, masked_targets_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Source: https://github.com/NVIDIA/Megatron-LM/blob/f7727433293427bef04858f67b2889fe9b177d88/megatron/core/tensor_parallel/cross_entropy.py#L98C5-L98C5
        """
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size(-1)
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size(0))
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


class VocabParallelCrossEntropy(nn.Module):
    def __init__(self, parallel_context: ParallelContext):
        super().__init__()
        # TODO: support reduce_mean, ignore_index
        self.parallel_context = parallel_context

    def forward(
        self, logits: TensorType["batch_size", "n_samples", "vocab_size"], targets: TensorType["batch_size", "n_samples"]
    ) -> torch.Tensor:
        loss = _VocabParallelCrossEntropy.apply(logits, targets, self.parallel_context)
        loss = loss.mean() / len(targets)
        return loss
