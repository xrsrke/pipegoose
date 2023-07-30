import torch
from torch import nn

from pipegoose.distributed.context import ParallelContext
from pipegoose.nn.tensor_parallel.layers import (
    ParallelizeAttention,
    ParallelizeEmbedding,
    ParallelizeLayerNorm,
    ParallelizeLinear,
)

# from pipegoose.distributed.mode import ParallelMode


class TensorParallel:
    """Turn a sequential model into a tensor-parallel model.

    Inspired by OSLO's TensorParallel: https://github.com/EleutherAI/oslo/blob/00e3be56446df37a0372a93a094863ffc89a2f8b/oslo/torch/nn/parallel/tensor_parallel/tensor_parallel.py#L51
    """

    def __init__(self, module: nn.Module, parallel_context: ParallelContext):
        super().__init__()
        self.module = module
        self.parallel_context = parallel_context

    @torch.no_grad()
    def parallelize(self):
        paralleler = {
            "linear": ParallelizeLinear,
            "embedding": ParallelizeEmbedding,
            "layer_norm": ParallelizeLayerNorm,
            "attention": ParallelizeAttention,
        }

        for name, module in self.module.named_modules():
            if name in paralleler:
                paralleler[name](module, self.parallel_context).parallelize()

    def _parallelize_embedding(self):
        pass

    def _parallize_layernorm(self):
        for _, module in self.module.named_modules():
            if isinstance(module, nn.LayerNorm):
                pass

    def _resize_vocab_size(self, module: nn.Module):
        pass
