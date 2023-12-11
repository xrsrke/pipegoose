import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from pipegoose.utils.logger import Logger

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

if __name__ == "__main__":
    seed_everything(42)
    LR = 0.001
    EPOCHS = 30

    model = NN(input_size=32 * 32, output_size=10)
    device = torch.device("cuda")
    optimizer = optim.SGD(model.parameters(), LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, _, _ = MNISTloader(train_val_split=0.).load()

    model = model.to(device)

    for epoch in range(EPOCHS):

        train_loss_running, train_acc_running = 0, 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_running += loss.item() * inputs.shape[0]
            train_acc_running += torch.sum(predictions == labels.data)

        train_loss = train_loss_running / len(train_loader.sampler)
        train_acc = train_acc_running / len(train_loader.sampler)

        info = "Epoch: {:3}/{} \t train_loss: {:.3f} \t train_acc: {:.3f}"
        Logger()(info.format(epoch + 1, EPOCHS, train_loss, train_acc))
        torch.save(model.state_dict(), "model.pt")