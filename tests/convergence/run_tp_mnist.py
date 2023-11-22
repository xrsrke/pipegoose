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

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.debug_single_mlp = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.debug_single_mlp(x)
        return x

class MNISTloader:
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "./data/",
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        train_val_split: float = 0.1,
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_split = train_val_split

        self.setup()

    def setup(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        val_split = int(len(self.train_dataset) * self.train_val_split)
        train_split = len(self.train_dataset) - val_split

        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_split, val_split]
        )
        self.test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )

        print(
            "Image Shape:    {}".format(self.train_dataset[0][0].numpy().shape),
            end="\n\n",
        )
        print("Training Set:   {} samples".format(len(self.train_dataset)))
        print("Validation Set: {} samples".format(len(self.val_dataset)))
        print("Test Set:       {} samples".format(len(self.test_dataset)))

    def load(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

        return train_loader, val_loader, test_loader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    import wandb

    DATA_PARALLEL_SIZE = 1
    TENSOR_PARALLEL_SIZE = 2
    PIPELINE_PARALLEL_SIZE = 1
    NUM_EPOCHS = 10
    LR = 2e-1
    SEED = 42
    BATCH_SIZE = 4

    torch.cuda.empty_cache()
    set_seed(SEED)

    Logger()(f"device_count: {torch.cuda.device_count()}")
    Logger()(f"is available: {torch.cuda.is_available()}")

    parallel_context = ParallelContext.from_torch(
        data_parallel_size=DATA_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
    )
    rank = parallel_context.get_global_rank()

    Logger()(f"rank={rank}, initialized parallel_context")

    dp_rank = parallel_context.get_local_rank(ParallelMode.DATA)
    train_dataloader, _, _ = MNISTloader(train_val_split=0.99).load()

    model = NN(input_size=32 * 32, output_size=10)
    model.load_state_dict(torch.load("model.pt"))
    ref_model = deepcopy(model)

    dist.barrier()

    model = TensorParallel(model, parallel_context).parallelize()
    optim = SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.to("cuda")
    device = next(model.parameters()).device

    Logger()(f"rank={rank}, model is moved to device: {device}")

    ref_model.to(device)
    ref_optim = SGD(ref_model.parameters(), lr=LR)
    ref_criterion = nn.CrossEntropyLoss()


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
            name=f"{get_time_name()}.test_tp_mnist_converegence",
            config={
                "data_parallel_size": DATA_PARALLEL_SIZE,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
                "model": "NN",
                "dataset": "MNIST",
                "epochs": NUM_EPOCHS,
                "learning_rate": LR,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
            },
        )

    for epoch in range(NUM_EPOCHS):
        Logger()(f"rank={rank}, epoch={epoch}")

        train_loss_running, train_acc_running = 0, 0

        for inputs, labels in train_dataloader:
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            ref_outputs = ref_model(inputs)
            _, ref_predictions = torch.max(ref_outputs, dim=1)
            ref_loss = ref_criterion(ref_outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            ref_optim.zero_grad()
            ref_loss.backward()
            ref_optim.step()

            Logger()(f"epoch={epoch}, step={step}, rank={rank}, train_loss={loss}, ref_train_loss={ref_loss}")

            if rank == 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "ref_train_loss": ref_loss,
                        "step": step,
                        "epoch": epoch,
                    }
                )

            step += 1

    dist.barrier()
    wandb.finish()
    model.cpu()
