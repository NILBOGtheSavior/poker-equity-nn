import argparse
from tqdm import tqdm
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


def train_model(model, training_data, validation_data,
                criterion, optimizer, device, epochs):
    log = ""
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
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

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in validation_data:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs).squeeze(-1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_loss = running_loss / len(training_data)
        log += f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}\n"

        avg_val_loss = val_loss / len(validation_data)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    print(log)
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

    device = torch.accelerator.current_accelerator(
    ).type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    print("Loading data...")  # TODO Clean up this mess
    data = torch.load(args.data)
    X = data['X']
    y = data['y']
    val_data = torch.load('data/sample_1k.pt')
    val_X = val_data['X']
    val_y = val_data['y']
    print(f"Loaded {len(X)} examples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    training_dataset = TensorDataset(X, y)
    validation_dataset = TensorDataset(val_X, val_y)

    training_data = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True)
    validation_data = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=True)

    model = PokerEquityNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model = train_model(model, training_data, validation_data, criterion,
                        optimizer, device, args.epochs)

    torch.save(model.state_dict(), args.output)
    print(f"Training complete. Model saved to {args.output}")


if __name__ == "__main__":
    print("Running: train_model.py")
    print("NILBOGtheSavior\n")

    main()
