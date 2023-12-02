from copy import deepcopy

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.nn import PipelineParallel
from pipegoose.nn.pipeline_parallel._utils import is_last_stage


def get_model_params_size(model, fp_bytes=4):
    params_size = 0
    for p in model.parameters():
        params_size += p.numel()
    params_gb = params_size * fp_bytes / 2**30
    return params_gb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import wandb

    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_PARALLEL_SIZE = 2
    MODEL = "bigscience/bloom-560m"
    DATASET = "imdb"
    NUM_EPOCHS = 4
    LR = 1e-3
    SEED = 69
    BATCH_SIZE = 36
    CONTEXT_LENGTH = 1024

    NUM_MICROBATCHES = 6

    torch.cuda.empty_cache()
    set_seed(SEED)

    print(f"device_count: {torch.cuda.device_count()}")
    print(f"is available: {torch.cuda.is_available()}")

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        backend="nccl",
    )
    rank = parallel_context.get_global_rank()

    print(f"rank={rank}, initialized parallel_context")

    train_dataset = load_dataset("imdb", split="train[:130]")
    train_dataset = train_dataset.map(lambda x: {"text": x["text"][:10]})  # for demonstration purposes

    # dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=SEED)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE,
        shuffle=False,
        # sampler=train_sampler
    )

    val_dataset = load_dataset("imdb", split="test[:130]")
    val_dataset = val_dataset.map(lambda x: {"text": x["text"][:4]})  # for demonstration purposes
    # val_sampler = DistributedSampler(val_dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=SEED)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE,
        shuffle=False,
        # sampler=val_sampler
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    ref_model = deepcopy(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"rank={rank}, model size before parallelizing: {round(get_model_params_size(model), 3)} GB")

    dist.barrier()

    # model = TensorParallel(model, parallel_context).parallelize()
    # model = DataParallel(model, parallel_context).parallelize()
    model = PipelineParallel(model, NUM_MICROBATCHES, parallel_context).parallelize()
    optim = SGD(model.parameters(), lr=LR)
    # optim = DistributedOptimizer(optim, parallel_context)
    model.to("cuda")
    device = next(model.parameters()).device

    print(f"rank={rank}, model size after parallelizing: {round(get_model_params_size(model), 3)} GB")
    print(f"rank={rank}, model is moved to device: {device}")

    ref_model.to(device)
    if DATA_PARALLEL_SIZE > 1:
        ref_model = torch.nn.parallel.DistributedDataParallel(ref_model, device_ids=[device])

    ref_optim = SGD(ref_model.parameters(), lr=LR)

    model.train()
    ref_model.train()
    step = 0
    dist.barrier()

    if rank == 0:

        def get_time_name():
            import datetime

            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        wandb.init(
            project="pipegoose",
            name=f"{get_time_name()}.test_pp_convergence",
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
                "num_microbatches": NUM_MICROBATCHES,
            },
        )

    for epoch in range(NUM_EPOCHS):
        # train_sampler.set_epoch(epoch)
        print(f"rank={rank}, epoch={epoch}")

        for batch in train_dataloader:
            inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=CONTEXT_LENGTH, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = inputs["input_ids"]

            outputs = model(**inputs)

            if is_last_stage(parallel_context):
                loss = torch.cat(outputs, dim=0).sum()

            optim.zero_grad()

            for output in outputs:
                output.sum().backward(retain_graph=True)

            optim.step()

            if is_last_stage(parallel_context):
                # TODO: auto concat outputs
                # assert torch.allclose(torch.cat(outputs, dim=0), ref_logits)
                # torch.allclose(torch.cat(outputs, dim=0).sum(), ref_loss)

                ref_outputs = ref_model(**inputs)
                ref_loss = ref_outputs.sum()

                ref_optim.zero_grad()
                ref_loss.backward()
                ref_optim.step()

                print(f"epoch={epoch}, step={step}, rank={rank}, train_loss={loss}, ref_train_loss={ref_loss}")

                wandb.log({"train_loss": loss, "ref_train_loss": ref_loss, "step": step, "epoch": epoch})

            step += 1

    model.eval()
    ref_model.eval()
    dist.barrier()

    step = 0
    # val_sampler.set_epoch(1)

    for batch in val_dataloader:
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=CONTEXT_LENGTH, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        labels = inputs["input_ids"]

        outputs = model(**inputs)
        ref_outputs = ref_model(**inputs)

        loss = outputs.sum()
        ref_loss = ref_outputs.sum()

        print(f"rank={rank}, val_loss={loss}, ref_val_loss={ref_loss}, step={step}")

        if rank == 0:
            wandb.log({"val_loss": loss, "ref_val_loss": ref_loss, "step": step})

        step += 1

    wandb.finish()
    model.cpu()
