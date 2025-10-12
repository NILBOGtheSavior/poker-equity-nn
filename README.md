# Poker AI Win Probability Predictor

The goal of this project is to build a neural network to estimate poker equity.

This README is intended to document the process of developing the neural network, including research, experimentation, and takeaways.

## Preliminary Research

### Understanding Poker Hand Equity

- Poker hand equity is the percentage chance your hand has to win against others.
- To calculate hand equity, a hand is compared against the range of possible hands that opponents might hold.

`Equity = (The Number of Ways to Win) / (Total Number of Possible Outcomes)`

To calculate this, I will be implementing a **Monte Carlo simulation**.

### PyTorch

Open source ML framework with:

- Deep learning primitives
- NN layer types
- Activation and loss functions
- Optimizers

Tensors are multi-dimensional arrays that are optimized for ML. They are an important data structure to understand for ML projects.

## Basic Plan

1. Create a `generate_data.py` program that simulates a specific number of poker hands and their probabilities using Monte Carlo simulations.
2. Create a `train_model.py` program that trains the model on the generated data.
3. Train the data and save checkpoints along the way.
4. Use trained model to predict new poker hands.

## Designing the data

To effectively train a neural network, relevant data must be selected and provided for the training algorithm to be able to learn. Due to this, I made the following considerations:

- I will initially build this neural network to train only on the **river**, meaning when all 5 community cards are on the table.
- The more opponents there are at a table, the lower the probability of winning, therefore, the number of opponents must be passed to the training algorithm.
- The cards should be able to be understood by the ML algorithm, so each card will be encoded as a 52-digit binary vector using **one-hot encoding**.

## Generating the data

1. A function that takes in data such as `hand`, `board`, `num_opponents` calculates the normalized probability.
2. Another function converts a card using one hot encoding. Each card in the hand and on the board will be encoded
