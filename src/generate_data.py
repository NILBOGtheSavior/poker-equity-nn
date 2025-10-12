import argparse
import random
import torch
from tqdm import tqdm
from treys import Deck, Card
from treys import Evaluator


SUIT_TO_INDEX = {1: 0, 2: 1, 4: 2, 8: 3}
evaluator = Evaluator()


def generate_data(num_examples, sims, output):
    print("Generating data...")
    all_features = []
    all_labels = []

    for i in tqdm(range(num_examples)):
        num_ops = random.randint(1, 10)  # Generate the number of opponents

        # Generate the hand that will be evaluated using simulation
        deck = Deck()
        board = deck.draw(5)
        hand = deck.draw(2)

        equity = calculate_equity(sims, num_ops, board, hand)

        for card in board + hand:
            encode(card)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("\tSet seed to " + str(seed))


def encode(card):
    enc = [0] * 52

    # Rank returns: 0 - 12
    rank = Card.get_rank_int(card)
    # Suite returns:
    #   SPADE: 1
    #   HEART: 2
    #   DIAMO: 4
    #   CLUBS: 8
    suit = SUIT_TO_INDEX[Card.get_suit_int(card)]

    index = rank * 4 + suit
    enc[index] = 1

    return enc


def calculate_equity(sims, ops, board, hand):
    wins = 0

    score = evaluator.evaluate(hand, board)
    for i in range(sims):
        temp_deck = Deck()
        for card in board + hand:
            temp_deck.cards.remove(card)
        table = []
        for _ in range(ops):
            op_hand = temp_deck.draw(2)
            op_score = evaluator.evaluate(op_hand, board)
            table.append(op_score)
        op_high_score = min(table)
        if score < op_high_score:
            wins += 1
        if score == op_high_score:
            wins += 0.5
    return wins / sims


def main():
    parser = argparse.ArgumentParser(
        description='Generate poker equity training data'
    )
    parser.add_argument('-n', '--examples', type=int, default=10000,
                        help='Number of training examples (default: 10000)')
    parser.add_argument('-s', '--sims', type=int, default=1000,
                        help='Simulations per example (default: 1000)')
    parser.add_argument('-o', '--output', type=str, default='data/default.pt',
                        help='Output filepath (default: data/default.pt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    print("Running: generate_data.py")
    print("NILBOGtheSavior\n")
    set_seed(args.seed)

    generate_data(
        num_examples=args.examples,
        sims=args.sims,
        output=args.output,
    )


if __name__ == "__main__":
    main()
