import os
import torch
from torch import nn
from torch.utils.data import DataLoader


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
            return logits


def main():
    device = torch.accelerator.current_accelerator(
    ).type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = PokerEquityNN().to(device)
    print(model)


if __name__ == "__main__":
    main()
