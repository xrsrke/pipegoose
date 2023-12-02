from copy import deepcopy

import torch
import torch.distributed as dist
from datasets import load_dataset
from einops import rearrange

# from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed.parallel_context import ParallelContext
from pipegoose.distributed.parallel_mode import ParallelMode
from pipegoose.nn import ExpertParallel
from pipegoose.nn.expert_parallel import ExpertLoss, SwitchNoisePolicy, Top1Router


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
    PIPELINE_PARALLEL_SIZE = 1
    MODEL = "bigscience/bloom-560m"
    DATASET = "imdb"
    NUM_EPOCHS = 100
    LR = 1e-3
    SEED = 69
    BATCH_SIZE = 32
    CONTEXT_LENGTH = 20

    NUM_EXPERTS = 4
    AUX_WEIGHT = 0.01
    Z_WEIGHT = 0.01

    torch.cuda.empty_cache()
    set_seed(SEED)

    print(f"device_count: {torch.cuda.device_count()}")
    print(f"is available: {torch.cuda.is_available()}")

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
    )
    rank = parallel_context.get_global_rank()

    print(f"rank={rank}, initialized parallel_context")

    train_dataset = load_dataset("imdb", split="train[:1000]")
    train_dataset = train_dataset.map(lambda x: {"text": " ".join(x["text"].split()[:10])})  # for demonstration purposes

    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=SEED)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE,
        shuffle=True,
        # sampler=train_sampler
    )

    val_dataset = load_dataset("imdb", split="test")
    val_dataset = val_dataset.map(lambda x: {"text": " ".join(x["text"].split()[:10])})  # for demonstration purposes
    # val_sampler = DistributedSampler(val_dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=SEED)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE,
        shuffle=True,
        # sampler=val_sampler
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.init_weights()
    # config = BloomConfig(n_layer=4)
    # model = BloomForCausalLM(config)
    ref_model = deepcopy(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"rank={rank}, model size before parallelizing: {round(get_model_params_size(model), 3)} GB")

    dist.barrier()

    # model = TensorParallel(model, parallel_context).parallelize()
    # model = DataParallel(model, parallel_context).parallelize()
    D_MODEL = model.config.hidden_size
    noise_policy = SwitchNoisePolicy()
    router = Top1Router(noise_policy, NUM_EXPERTS, D_MODEL)

    model = ExpertParallel(
        model, num_experts=NUM_EXPERTS, mapping=[0, 2], router=router, parallel_context=parallel_context
    ).parallelize()

    optim = SGD(model.parameters(), lr=LR)
    # optim = DistributedOptimizer(optim, parallel_context)
    model.to("cuda")
    device = next(model.parameters()).device

    # print(f"rank={rank}, model size after parallelizing: {round(get_model_params_size(model), 3)} GB")
    # print(f"rank={rank}, model is moved to device: {device}")

    # ref_model.to(device)
    # if DATA_PARALLEL_SIZE > 1:
    #     ref_model = torch.nn.parallel.DistributedDataParallel(ref_model, device_ids=[device])

    ref_model.cuda()
    ref_optim = SGD(ref_model.parameters(), lr=LR)

    model.train()
    ref_model.train()
    step = 0
    dist.barrier()

    loss_func = ExpertLoss(nn.CrossEntropyLoss(), aux_weight=AUX_WEIGHT, z_weight=Z_WEIGHT)
    ref_loss_func = nn.CrossEntropyLoss()

    def compute_loss_func(logits, targets):
        logits = rearrange(logits, "batch_size seq_len d_model -> (batch_size seq_len) d_model")
        targets = rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")
        return loss_func(logits, targets)

    def compute_ref_loss_func(logits, targets):
        logits = rearrange(logits, "batch_size seq_len d_model -> (batch_size seq_len) d_model")
        targets = rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")
        return ref_loss_func(logits, targets)

    if rank == 0:

        def get_time_name():
            import datetime

            today = datetime.datetime.now()
            return today.strftime("%d/%m/%Y_%H:%M:%S")

        wandb.init(
            project="pipegoose",
            name=f"{get_time_name()}.test_ep_convergence",
            config={
                "data_parallel_size": DATA_PARALLEL_SIZE,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
                "model": MODEL,
                "dataset": DATASET,
                "context_length": CONTEXT_LENGTH,
                "epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "num_experts": NUM_EXPERTS,
                "aux_weight": AUX_WEIGHT,
                "z_weight": Z_WEIGHT,
            },
        )

    for epoch in range(NUM_EPOCHS):
        # train_sampler.set_epoch(epoch)
        print(f"rank={rank}, epoch={epoch}")

        for batch in train_dataloader:
            inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=CONTEXT_LENGTH, return_tensors="pt")
            # inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
            labels = inputs["input_ids"]

            outputs = model(**inputs)
            ref_outputs = ref_model(**inputs)

            if rank == 0:
                wandb.log(
                    {
                        "expert_0_aux_loss": loss_func.aux_loss[0],
                        "expert_1_aux_loss": loss_func.aux_loss[1],
                        "expert_0_z_loss": loss_func.z_loss[0],
                        "expert_1_z_loss": loss_func.z_loss[1],
                        "step": step,
                        "epoch": epoch,
                    }
                )

            loss = compute_loss_func(outputs.logits[..., :-1, :], inputs["input_ids"][..., 1:])
            ref_loss = compute_ref_loss_func(ref_outputs.logits[..., :-1, :], inputs["input_ids"][..., 1:])

            optim.zero_grad()
            loss.backward()
            optim.step()

            ref_optim.zero_grad()
            ref_loss.backward()
            ref_optim.step()

            print(f"epoch={epoch}, step={step}, rank={rank}, train_loss={loss}, ref_train_loss={ref_loss}")

            if rank == 0:
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

        loss = compute_loss_func(outputs.logits[..., :-1, :], inputs["input_ids"][..., 1:])
        ref_loss = compute_ref_loss_func(ref_outputs.logits[..., :-1, :], inputs["input_ids"][..., 1:])

        print(f"rank={rank}, val_loss={loss}, ref_val_loss={ref_loss}, step={step}")

        if rank == 0:
            wandb.log({"val_loss": loss, "ref_val_loss": ref_loss, "step": step})

        step += 1

    wandb.finish()
    model.cpu()
    ref_model.cpu()

    torch.save(model.state_dict(), "./moe.pth")
    torch.save(ref_model.state_dict(), "./not_a_moe.pth")
