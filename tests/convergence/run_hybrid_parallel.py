from copy import deepcopy

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn import DataParallel, TensorParallel


def get_model_params_size(model, fp_bytes=4):
    params_size = 0
    for p in model.parameters():
        params_size += p.numel()
    params_gb = params_size * fp_bytes / 2**30
    return params_gb


if __name__ == "__main__":
    import wandb

    DATA_PARALLEL_SIZE = 2
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    MODEL = "bigscience/bloom-560m"
    DATASET = "imdb"
    NUM_EPOCHS = 100
    LR = 1e-3
    SEED = 69
    BATCH_SIZE = 4
    CONTEXT_LENGTH = 1024

    torch.cuda.empty_cache()
    print(f"device_count: {torch.cuda.device_count()}")
    print(f"is available: {torch.cuda.is_available()}")

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
    )
    rank = parallel_context.get_global_rank()

    print(f"rank={rank}, initialized parallel_context")

    dataset = load_dataset("imdb", split="train[:100]")
    dataset = dataset.map(lambda x: {"text": x["text"][:30]})  # for demonstration purposes

    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    sampler = DistributedSampler(dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE, shuffle=False, sampler=sampler)

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    ref_model = deepcopy(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"rank={rank}, model size before parallelizing: {round(get_model_params_size(model), 3)} GB")

    model = TensorParallel(model, parallel_context).parallelize()
    model = DataParallel(model, parallel_context).parallelize()
    optim = SGD(model.parameters(), lr=LR)
    model.to("cuda")
    device = next(model.parameters()).device

    print(f"rank={rank}, model size after parallelizing: {round(get_model_params_size(model), 3)} GB")
    print(f"rank={rank}, model is moved to device: {device}")

    ref_model.to(device)
    if DATA_PARALLEL_SIZE > 1:
        ref_model = torch.nn.parallel.DistributedDataParallel(ref_model, device_ids=[device])

    ref_optim = SGD(ref_model.parameters(), lr=LR)

    if rank == 0:

        def get_time_name():
            import datetime

            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        wandb.init(
            project="pipegoose",
            name=f"{get_time_name()}.test_tp_dp_zero1_converegence",
            config={
                "data_parallel_size": DATA_PARALLEL_SIZE,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
                "model": MODEL,
                "dataset": DATASET,
                "epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
            },
        )

    dist.barrier()
    step = 0

    for epoch in range(NUM_EPOCHS):

        sampler.set_epoch(epoch)
        print(f"rank={rank}, epoch={epoch}")

        for batch in dataloader:
            inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=CONTEXT_LENGTH, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = inputs["input_ids"]

            outputs = model(**inputs, labels=labels)
            ref_outputs = ref_model(**inputs, labels=labels)

            optim.zero_grad()
            outputs.loss.backward()
            optim.step()

            ref_optim.zero_grad()
            ref_outputs.loss.backward()
            ref_optim.step()

            print(f"rank={rank}, loss={outputs.loss}, ref_loss={ref_outputs.loss}, step={step}")

            if rank == 0:
                wandb.log({"loss": outputs.loss, "ref_loss": ref_outputs.loss, "step": step, "epoch": epoch})

            step += 1

    wandb.finish()
    model.cpu()
