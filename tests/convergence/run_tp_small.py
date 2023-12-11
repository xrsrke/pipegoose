from copy import deepcopy

import torch
import torch.distributed as dist
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, transforms

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn import TensorParallel
from pipegoose.utils.logger import Logger
import torch.nn.functional as F
import numpy as np
import random

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.debug_single_mlp = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.debug_single_mlp(x)
        return x

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    NUM_EPOCHS = 30
    LR = 2e-1
    SEED = 42

    torch.cuda.empty_cache()

    Logger()(f"device_count: {torch.cuda.device_count()}")
    Logger()(f"is available: {torch.cuda.is_available()}")

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        seed=SEED,
        backend="nccl"
    )

    rank = parallel_context.get_local_rank(ParallelMode.TENSOR)
    set_random_seed(SEED + rank)

    Logger()(f"rank={rank}, initialized parallel_context")

    BATCH_SIZE = 1
    IN_FEATURES = 4
    OUT_FEATURES = 6

    X = torch.randn(BATCH_SIZE, IN_FEATURES, device="cuda", requires_grad=True)
    L_weight = torch.randn(BATCH_SIZE, OUT_FEATURES, device="cuda")

    # Rank 0 brodcast X and W to other rank
    dist.broadcast(X, src=0)
    dist.broadcast(L_weight, src=0)

    Logger()(f"[rank {rank}]: {X}")
    Logger()(f"[rank {rank}]: {L_weight}")

    model = NN(input_size=IN_FEATURES, output_size=OUT_FEATURES)
    model_ref = deepcopy(model)

    dist.barrier()

    model = TensorParallel(model, parallel_context).parallelize()
    model.to("cuda")
    device = next(model.parameters()).device
    model_ref.to(device)
    Logger()(f"[rank {rank}]: model is moved to device: {device}")

    # Reference
    Y_ref = model_ref(X)
    L_ref = torch.mul(Y_ref, L_weight).sum()        
    # Manually compute the gradient
    dLdW_ref = torch.matmul(L_weight.t(), X)
    dLdX_ref = torch.matmul(L_weight, model_ref.debug_single_mlp.weight)

    dist.barrier()

    # Distributed
    Logger()("===========FORWARD===========")
    Y = model(X)
    L = torch.mul(Y, L_weight).sum()
    Y.retain_grad()
    
    Logger()("===========BACKWARD===========")
    L.backward()

    #HACK: we need to divide by world size because we are calling L.backward() on rank 0 and 1
    # Too lazy to find a way to merge into a single matrix
    dLdX = X.grad / dist.get_world_size() 

    if rank == 0:
        #NOTE: tests inspired from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/tests/test_layers.py#L173
        Logger()(f"error Y_ref - Y: {Y_ref.sub(Y).abs().max()}")
        Logger()(f"error L_ref - L: {L_ref.sub(L).abs().max()}")
        Logger()(f"error dLdX_ref - dLdX: {dLdX_ref.sub(dLdX).abs().max()}")
    
    dist.barrier()
    
    dLdW_ref = torch.split(dLdW_ref, OUT_FEATURES // dist.get_world_size(), dim=0)[rank].contiguous()
    dLdW = model.debug_single_mlp.weight.grad
    Logger()(f"error dLdW_ref - dLdW (rank {rank}): {dLdW_ref.sub(dLdW).abs().max()}")