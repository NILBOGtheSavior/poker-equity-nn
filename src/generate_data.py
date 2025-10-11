import argparse
import random
import torch
from treys import Deck
from treys import Evaluator


evaluator = Evaluator()


def generate_data(num_examples, sims, output):
    print("Generating data...")
    for i in range(num_examples):
        print("Example " + str(i))
        num_ops = random.randint(1, 10)  # Generates the number of opponents
        # This is the hand that will be evaluated using simulation
        print(calculate_equity(sims, num_ops))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Set seed to " + str(seed))


def calculate_equity(sims, ops):
    wins = 0

    # Determine simulation scenario
    deck = Deck()
    board = deck.draw(5)
    hand = deck.draw(2)

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

    set_seed(args.seed)

    generate_data(
        num_examples=args.examples,
        sims=args.sims,
        output=args.output,
    )


if __name__ == "__main__":
    main()
