"""Policy network: 54 sticker colors in, a score for each of the 12 moves out."""

import torch
from torch import nn

from cube_env import FACE_ORDER, NUM_MOVES, SOLVED

INPUT_SIZE = SOLVED.size * len(FACE_ORDER)


def one_hot(states):
    """(N, 54) color indices -> (N, 324) float one-hot encoding."""
    states = torch.as_tensor(states, dtype=torch.int64)
    return nn.functional.one_hot(states, len(FACE_ORDER)).flatten(1).float()


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class PolicyNet(nn.Module):
    def __init__(self, hidden=512, blocks=3):
        super().__init__()
        self.config = {"hidden": hidden, "blocks": blocks}
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            *[ResidualBlock(hidden) for _ in range(blocks)],
            nn.Linear(hidden, NUM_MOVES),
        )

    def forward(self, x):
        """x: (N, 324) one-hot states. Returns (N, 12) move logits."""
        return self.net(x)


def load(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = PolicyNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
