from datasets import load_dataset
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipegoose.distributed import ParallelContext, ParallelMode
from pipegoose.nn import DataParallel, TensorParallel

if __name__ == "__main__":
    DATA_PARALLEL_SIZE = 2
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    BATCH_SIZE = 4

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
    )
    rank = parallel_context.get_global_rank()

    dataset = load_dataset("imdb", split="train[:100]")
    dataset = dataset.map(lambda x: {"text": x["text"][:30]})  # for demonstration purposes, you can remove this line

    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    sampler = DistributedSampler(dataset, num_replicas=DATA_PARALLEL_SIZE, rank=dp_rank, seed=69)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE // DATA_PARALLEL_SIZE, shuffle=False, sampler=sampler)

    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    tokenizer.pad_token = tokenizer.eos_token

    model = TensorParallel(model, parallel_context).parallelize()
    model = DataParallel(model, parallel_context).parallelize()
    optim = SGD(model.parameters(), lr=1e-3)
    model.to("cuda")
    device = next(model.parameters()).device

    print(f"rank={rank}, moved to device: {device}")

    for epoch in range(100):
        sampler.set_epoch(epoch)

        for batch in dataloader:
            inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=1024, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            labels = inputs["input_ids"]

            outputs = model(**inputs, labels=labels)

            optim.zero_grad()
            outputs.loss.backward()
            optim.step()

            print(f"rank={rank}, loss={outputs.loss}")

    model.cpu()
