"""Training data: random scrambles labelled with the move that undoes the last turn.

Run directly to write a sample dataset and the fixed set of test scrambles:

    python data.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cube_env import INVERSE, MOVES, NUM_MOVES, SOLVED, apply_moves

DATA_DIR = Path(__file__).parent / "data"
# God's number in the quarter-turn metric is 26, so scrambles this long reach every depth.
SCRAMBLE_DEPTH = 26


def random_moves(rng, previous):
    """One random move per scramble, never the inverse of that scramble's previous move."""
    moves = rng.integers(0, NUM_MOVES - 1, len(previous))
    return np.where(moves >= INVERSE[previous], moves + 1, moves)


def generate_batch(num_scrambles, rng, depth=SCRAMBLE_DEPTH):
    """Returns (states, labels) for num_scrambles scrambles of `depth` moves each.

    Every intermediate state becomes a sample, labelled with the inverse of the move that
    produced it, so a batch holds num_scrambles * depth samples.
    """
    states = np.tile(SOLVED, (num_scrambles, 1))
    all_states = np.empty((depth, num_scrambles, SOLVED.size), dtype=np.uint8)
    labels = np.empty((depth, num_scrambles), dtype=np.int64)
    moves = rng.integers(0, NUM_MOVES, num_scrambles)
    for d in range(depth):
        if d > 0:
            moves = random_moves(rng, moves)
        states = apply_moves(states, moves)
        all_states[d] = states
        labels[d] = INVERSE[moves]
    return all_states.reshape(-1, SOLVED.size), labels.reshape(-1)


def scramble_sequences(count, length, rng):
    sequences = np.empty((length, count), dtype=np.int64)
    sequences[0] = rng.integers(0, NUM_MOVES, count)
    for i in range(1, length):
        sequences[i] = random_moves(rng, sequences[i - 1])
    return [[MOVES[m] for m in seq] for seq in sequences.T]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000_000, help="samples in the sample dataset")
    parser.add_argument("--test-scrambles", type=int, default=100)
    parser.add_argument("--test-length", type=int, default=100, help="moves per test scramble")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    DATA_DIR.mkdir(exist_ok=True)

    states, labels = generate_batch(args.samples // SCRAMBLE_DEPTH, rng)
    np.savez_compressed(DATA_DIR / "sample.npz", states=states, labels=labels)
    unique = len(np.unique(states, axis=0))
    print(f"sample.npz: {len(states):,} samples, {unique:,} unique states")
    print("label counts:", dict(zip(MOVES, np.bincount(labels, minlength=NUM_MOVES).tolist())))

    scrambles = scramble_sequences(args.test_scrambles, args.test_length, rng)
    (DATA_DIR / "test_scrambles.json").write_text(json.dumps(scrambles))
    print(f"test_scrambles.json: {len(scrambles)} scrambles of {args.test_length} moves")


if __name__ == "__main__":
    main()
