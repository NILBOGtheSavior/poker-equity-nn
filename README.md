# Poker Equity Neural Network

The goal of this project is to build a neural network to estimate poker equity.

This README is intended to document the process of developing the neural network, including research, experimentation, and takeaways.

## Instructions

If you would like to run this neural network locally, follow these instructions to generate your datasets and train your model:

1. Ensure you have **Python 3.9** installed.
2. Clone this repository into your local directory and enter it.
3. Create a virtual environment with the `python -m venv venv` command.
4. Run `pip3 install -r requirements.txt` in your virtual environment.
5. Run `python3 src/generate_data.py`.
    - Use the `-h` tag to view the available options.
6. Run `python3 src/train_model.py`.
    - Use the `-h` tag to view the available options.
    - Ensure the correct data file is selected with the `-d` tag.

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

## Designing the Data

To effectively train a neural network, relevant data must be selected and provided for the training algorithm to be able to learn. Due to this, I made the following considerations:

- I will initially build this neural network to train only on the **river**, meaning when all 5 community cards are on the table.
- The more opponents there are at a table, the lower the probability of winning, therefore, the number of opponents must be passed to the training algorithm.
- The cards must be encoded so that our model can understand what cards are in the hand. To do this, a 52-dimensional binary vector is used.

## Generating the Data

### Overview

1. A function that takes in data such as `hand`, `board`, `num_opponents` returns the normalized probability.
2. Another function converts the hand to a binary vector and stores the array along with a normalized value of `num_opponents`.
3. Each feature (the binary vector and normalized `num_opponents`) and label (equity) is stored as a tensor.
4. The data is stored in an output file.

### Observations and Improvements

- The program was able to run in approximately 10 minutes with 10,000 examples and 1,000 sims.
- A larger data file with 50,000 examples will be created.
- A small set of 1,000 examples will be created with more simulation runs with the goal of reducing noise for validation sets.
    
    - A graph showing equity estimates versus the number of simulations will be plotted to determine the optimal number of simulations that yields stable and low-noise results.

## Training the Model

### Overview

1. A function loads the training dataset and the validation dataset.
2. A neural network is defined.
3. The data is trained over multiple epochs and track loss/accuracy
4. The trained model is saved in an output file.

### Observations and Improvements

- Compared to the 1k sample, training the 10k sample has worse loss. This suggests that overfitting occurred. The model will be retrained with 50k samples to see if the pattern continues.
- Loss will be logged in the future to ensure that results can be compared.
- The Adam optimizer algorithm was used in initial training runs. AdamW and some others will be tested to determine the best optimizer to use for this project.
