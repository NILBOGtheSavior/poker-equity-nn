import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PokerEquityNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)


class TrainPokerEquity():
    def __init__(self, data, batch_size, epochs, load, output):
        print("Initialized training class")
        self.device = self.get_device()
        self.batch_size = batch_size
        self.epochs = epochs
        self.output = output
        self.training_data = self.get_data(data)
        self.validation_data = self.get_data('data/validation_1k.pt')

        self.model = PokerEquityNN().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-6)

        self.epoch = 0
        self.loss = 0

        self.training_loss = []
        self.validation_loss = []

        self.model = self.train_model()

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self
        }, self.output)
        print(f"Training complete. Model saved to {self.output}")

    def train_model(self):
        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            self.training_loop()
            self.validation_loop()
            self.epoch = epoch

        plot_data(self.training_loss, self.validation_loss)

        return self.model

    def training_loop(self):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in self.training_data:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze(-1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.loss = loss.item()
            running_loss += loss.item()

        avg_loss = running_loss / len(self.training_data)

        self.training_loss.append(avg_loss)

    def validation_loop(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for inputs, targets in self.validation_data:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.validation_data)

        self.validation_loss.append(avg_loss)

    def get_device(self):
        device = torch.accelerator.current_accelerator(
        ).type if torch.accelerator.is_available() else "cpu"
        print(f"Using {device} device")
        return device

    def get_data(self, datafile):
        data = torch.load(datafile)
        X = data['X']
        y = data['y']
        print(f"Loaded {len(X)} examples")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader


def plot_data(training_loss, validation_loss):
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Train poker equity neural network'
    )
    parser.add_argument('-d', '--data', type=str, default='data/sample_1k.pt',
                        help='Path to data file (default: data/sample_1k.pt)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of training itterations (default: 50)')
    parser.add_argument('-l', '--load', type=str,
                        help='Optional: Include model checkpoint')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output filename (saved in data/*.pth)')

    args = parser.parse_args()

    load = args.load

    output = f"models/{args.output}.pth"
    o = Path(output)
    if o.exists():
        print(f"Caution: {output} already exists.")
        i = input("Would you like to overwrite the file? y/N\n")
        if i == "n" or i == "N" or "":
            filename = input(
                "Enter new filename:\n")
            output = f"models/{filename}.pth"
        elif i == "y" or i == "Y":
            print(f"Overwriting {output}")
        else:
            print("Unknown value. Quitting...")
            exit()

    TrainPokerEquity(args.data, args.batch_size, args.epochs, load, output)


if __name__ == "__main__":
    print("Running: train_model.py")
    print("NILBOGtheSavior\n")

    main()
