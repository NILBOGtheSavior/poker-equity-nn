# Documentation Overview

This section is intended to document the process of developing the neural network, including research, experimentation, and takeaways.

## Preliminary Research

### Understanding Poker Hand Equity

- Poker hand equity is the percentage chance your hand has to win against others.
- To calculate hand equity, a hand is compared against the range of possible hands that opponents might hold.

`Equity = (The Number of Ways to Win) / (Total Number of Possible Outcomes)`

To calculate this, I will be implementing a **Monte Carlo Simulation**.

### PyTorch

Open source ML framework with:

- Deep learning primitives
- NN layer types
- Activation and loss functions
- Optimizers

## Basic Plan

1. Create a `generate_data.py` program that simulates a specific number of poker hands and their probabilities using Monte Carlo simulations.
2. Create a `train_model.py` program that trains the model on the generated data.
3. Train the data and save checkpoints along the way.
4. Use trained model to predict new poker hands.
