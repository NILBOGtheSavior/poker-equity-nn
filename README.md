# Poker Equity Neural Network

The goal of this project is to build a neural network to estimate poker equity.

This README is intended to document the process of developing the neural network, including research, experimentation, and takeaways.

## Preliminary Research

### Understanding Poker Hand Equity

- Poker hand equity is the percentage chance your hand has to win against others.
- To calculate hand equity, a hand is compared against the range of possible hands that opponents might hold.

$$
\text{Equity}=\frac{\text{The Number of Ways to Win}}{\text{Total Number of Possible Outcomes}}
$$

To calculate this, I will be implementing a **Monte Carlo simulation**.

### PyTorch

Open source ML framework with:

- Deep learning primitives
- Neural network layer types
- Activation and loss functions
- Optimizers

#### Tensors

Tensors are multidimensional arrays that are optimized for ML. They are an important data structure to understand for ML projects.

## Basic Plan

1. Create a `generate_data.py` program that simulates a specific number of poker hands and their probabilities using Monte Carlo simulations.
2. Create a `train_model.py` program that trains the model on the generated data.
3. Train the data and save checkpoints along the way.
4. Use trained model to predict new poker hands.

## Designing the data

To effectively train a neural network, relevant data must be selected and provided for the training algorithm to be able to learn. Due to this, I made the following considerations:

- I will initially build this neural network to train only on the **river**, meaning when all 5 community cards are on the table.
- The more opponents there are at a table, the lower the probability of winning, therefore, the number of opponents must be passed to the training algorithm.
- The cards must be encoded so that our model can understand what cards are in the hand. To do this, a 52-dimensional binary vector is used.

## Generating the data

1. A function that takes in data such as `hand`, `board`, `num_opponents` returns the normalized probability.
2. Another function converts the hand to a binary vector and stores the array along with a normalized value of `num_opponents`.
3. Each feature (the binary vector and normalized `num_opponents`) and label is stored as a tensor.
4. The data is stored in an output file.
