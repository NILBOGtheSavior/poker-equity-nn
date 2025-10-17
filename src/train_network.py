import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PokerEquityNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(53, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)


class TrainPokerEquity():
    def __init__(self, data, batch_size, epochs, output):
        print("Initialized training class")
        self.device = self.get_device()
        self.batch_size = batch_size
        self.epochs = epochs
        self.output = output
        self.training_data = self.get_data(data)
        self.validation_data = self.get_data('data/validation_1k.pt')

        self.model = PokerEquityNN().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.model = train_model(self.model, self.training_data,
                                 self.validation_data, self.criterion,
                                 self.optimizer, self.device, self.epochs)

        torch.save(self.model.state_dict(), self.output)
        print(f"Training complete. Model saved to {self.output}")

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


def train_model(model, training_data, validation_data,
                criterion, optimizer, device, epochs):
    training_loss = []
    validation_loss = []
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Training loop
        model.train()
        running_loss = 0.0
        for inputs, targets in training_data:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in validation_data:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_loss = running_loss / len(training_data)
        avg_val_loss = val_loss / len(validation_data)

        training_loss.append(avg_loss)
        validation_loss.append(avg_val_loss)

    plot_data(training_loss, validation_loss)

    return model


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
    parser.add_argument('-o', '--output', type=str,
                        default='models/default.pth',
                        help='Output filepath (default: models/default.pth)')

    args = parser.parse_args()

    TrainPokerEquity(args.data, args.batch_size, args.epochs, args.output)


if __name__ == "__main__":
    print("Running: train_model.py")
    print("NILBOGtheSavior\n")

    main()
