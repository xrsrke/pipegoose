### Hybrid tensor parallelism and data parallelism training

Hybrid 3D parallelism for 🤗 `transformers` will be available in the upcoming weeks (it's basically done, but it doesn't support 🤗 `transformers` yet)

**You must have at least 4 GPUs to run 2D parallelism.**. `nproc-per-node` is equal to `tensor_parallel_size` * `pipeline_parallel_size` * `data_parallel_size`.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 hybrid_parallelism.py
```
