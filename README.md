# 🚧 pipegoose: Decentralized large-scale 4D parallelism multi-modal pre-training for 🤗 `transformers` in Mixture of Experts

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/xrsrke/pipegoose) [![tests](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml/badge.svg)](https://github.com/xrsrke/pipegoose/actions/workflows/tests.yaml) [<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/s9ZS9VXZ3p) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [<img alt="Codecov" src="https://img.shields.io/codecov/c/github/xrsrke/pipegoose">](https://app.codecov.io/gh/xrsrke/pipegoose) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

![pipeline](3d-parallelism.png)

<!-- [![docs](https://img.shields.io/github/deployments/Production?label=docs&logo=vercel)](https://docs.dev/) -->

We're building a library for an end-to-end framework for **training multi-modal MoE in a decentralized way, as proposed by the paper [DiLoCo](https://arxiv.org/abs/2311.08105)**. The core papers that we are replicating are:
- DiLoCo: Distributed Low-Communication Training of Language Models [[link]](https://arxiv.org/abs/2311.08105)
- Pipeline MoE: A Flexible MoE Implementation with Pipeline Parallelism [[link]](https://arxiv.org/abs/2304.11414)
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity [[link]](https://arxiv.org/abs/2101.03961)
- Flamingo: a Visual Language Model for Few-Shot Learning [[link]](https://arxiv.org/abs/2204.14198)
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism  [[link]](https://arxiv.org/abs/1909.08053)


**If you're interested in contributing, check out [[CONTRIBUTING.md]](./CONTRIBUTING.md) [[good first issue]](https://github.com/xrsrke/pipegoose/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) [[roadmap]](https://github.com/users/xrsrke/projects/5). Come join us: [[discord link]](https://discord.gg/s9ZS9VXZ3p)**

⚠️ **Currently only parallelize `transformers`'s `bloom` is supported.**

```diff
from torch.utils.data import DataLoader
+ from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

+ from pipegoose.distributed import ParallelContext, ParallelMode
+ from pipegoose.nn import DataParallel, TensorParallel
+ from pipegoose.optim import DistributedOptimizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
tokenizer.pad_token = tokenizer.eos_token

BATCH_SIZE = 4
+ DATA_PARALLEL_SIZE = 2
+ parallel_context = ParallelContext.from_torch(
+    tensor_parallel_size=2,
+    data_parallel_size=2,
+    pipeline_parallel_size=1
+ )
+ model = TensorParallel(model, parallel_context).parallelize()
+ model = DataParallel(model, parallel_context).parallelize()
model.to("cuda")
+ device = next(model.parameters()).device

optim = Adam(model.parameters(), lr=1e-3)
+ optim = DistributedOptimizer(optim, parallel_context)

dataset = load_dataset("imdb", split="train")
+ dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
+ sampler = DistributedSampler(dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=42)
+ dataloader = DataLoader(dataset, batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE, shuffle=False, sampler=sampler)

for epoch in range(100):
+    sampler.set_epoch(epoch)

    for batch in dataloader:
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        labels = inputs["input_ids"]

        outputs = model(**inputs, labels=labels)

        optim.zero_grad()
        outputs.loss.backward()
        optim.step()
```

**Installation and try it out**

You can install the package through the following command:

```bash
git clone https://github.com/xrsrke/pipegoose.git
cd pipegoose && pip install -e .
```

And try out a hybrid tensor and data parallelism training script (You must have at least 4 GPUs in order to try hybrid 2D parallelism).

```bash
cd pipegoose/examples
torchrun --standalone --nnodes=1 --nproc-per-node=4 hybrid_parallelism.py
```

We did a small scale correctness test by comparing the validation losses between a paralleized transformer and one kept by default, starting at identical checkpoints and training data. We will conduct rigorous large scale convergence and weak scaling law benchmarks against Megatron and DeepSpeed in the near future if we manage to make it.
- Data Parallelism [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/t5cr56xd?workspace)
- ~~Tensor Parallelism [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/iz17f50n)~~ (We've found a bug in convergence, and we are fixing it)
- ~~Hybrid 2D Parallelism (TP+DP) [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/us31p3q1)~~
- Distributed Optimizer ZeRO-1 Convergence: [[sgd link]](https://wandb.ai/xariusdrake/pipegoose/runs/fn4t9as4?workspace) [[adam link]](https://wandb.ai/xariusdrake/pipegoose/runs/yn4m2sky)
- Mixture of Experts [[link]](https://wandb.ai/xariusdrake/pipegoose/runs/ac30wbfy/workspace)

**Features**
- End-to-end multi-modal including in 3D parallelism including distributed CLIP..
- Sequence parallelism and Mixture of Experts that work in 3D parallelism
- ZeRO-1: Distributed Optimizer
- Kernel fusion
- ...

**Appreciation**

- Big thanks to 🤗 [Hugging Face](https://huggingface.co/) for sponsoring this project with GPUs for testing!

- The library's APIs are inspired by [OSLO](https://github.com/EleutherAI/oslo)'s and [ColossalAI](https://github.com/hpcaitech/ColossalAI)'s APIs.

**Citation**

```
@software{pipegoose,
  title = {{pipegoose: Large-scale 4D parallelism pre-training for `transformers`}},
  author = {},
  url = {https://github.com/xrsrke/pipegoose},
  doi = {},
  month = {},
  year = {2024},
  version = {},
}
```
