"""Beam search guided by the policy network, and an evaluation on the test scrambles.

    python search.py --beam 1000
"""

import argparse
import json
import time

import numpy as np
import torch

from cube_env import INVERSE, MOVES, NUM_MOVES, SOLVED, apply_moves, apply_sequence, is_solved
from data import DATA_DIR
from model import load, one_hot
from train import CHECKPOINT_DIR


@torch.no_grad()
def beam_search(model, state, beam_width=1000, max_depth=60):
    """Returns the solving move list, or None if no solution is found within max_depth."""
    if is_solved(state):
        return []
    states = state[None]
    scores = np.zeros(1)
    paths = np.zeros((1, 0), dtype=np.int64)

    for _ in range(max_depth):
        log_probs = torch.log_softmax(model(one_hot(states)), dim=1).numpy()
        candidate_scores = scores[:, None] + log_probs
        if paths.shape[1]:
            # Never undo the move that was just made.
            candidate_scores[np.arange(len(states)), INVERSE[paths[:, -1]]] = -np.inf

        flat = candidate_scores.ravel()
        keep = np.argsort(-flat)[: beam_width]
        keep = keep[np.isfinite(flat[keep])]
        parents, moves = np.divmod(keep, NUM_MOVES)

        states = apply_moves(states[parents], moves)
        scores = flat[keep]
        paths = np.concatenate([paths[parents], moves[:, None]], axis=1)

        solved = np.flatnonzero(is_solved(states))
        if solved.size:
            return [MOVES[m] for m in paths[solved[0]]]

        # Different paths often reach the same state; keep only the best-scoring copy.
        _, first = np.unique(states, axis=0, return_index=True)
        first.sort()
        states, scores, paths = states[first], scores[first], paths[first]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "latest.pt"))
    parser.add_argument("--beam", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--count", type=int, default=20, help="test scrambles to try")
    args = parser.parse_args()

    model, checkpoint = load(args.checkpoint)
    print(f"checkpoint step {checkpoint['step']}, beam {args.beam}")
    scrambles = json.loads((DATA_DIR / "test_scrambles.json").read_text())[: args.count]

    lengths, times = [], []
    for i, scramble in enumerate(scrambles):
        state = apply_sequence(SOLVED.copy(), scramble)
        started = time.time()
        solution = beam_search(model, state, args.beam, args.max_depth)
        times.append(time.time() - started)
        if solution is not None:
            assert is_solved(apply_sequence(state, solution))
            lengths.append(len(solution))
        print(f"{i + 1:>3}: {'solved in ' + str(len(solution)) if solution else 'FAILED'} ({times[-1]:.1f}s)", flush=True)

    print(
        f"\nsolved {len(lengths)}/{len(scrambles)}, "
        f"mean length {np.mean(lengths) if lengths else float('nan'):.1f}, mean time {np.mean(times):.1f}s"
    )


if __name__ == "__main__":
    main()
