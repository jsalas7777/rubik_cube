"""Policy network: a transformer that reads the 54 stickers and scores each of the 12 moves.

Every sticker is one token. A token's embedding is its color plus a learned embedding for
its slot on the cube, so attention can learn which stickers belong to the same cubie or
move together. A learned [CLS] token gathers the whole cube and predicts the move.

The network still takes the (N, 324) one-hot input the MLP used, so export.py and the
page's solver.js don't change: a linear layer on a one-hot vector is a color embedding.
"""

import torch
from torch import nn

from cube_env import FACE_ORDER, NUM_MOVES, SOLVED

NUM_STICKERS = SOLVED.size
NUM_COLORS = len(FACE_ORDER)
INPUT_SIZE = NUM_STICKERS * NUM_COLORS


def one_hot(states):
    """(N, 54) color indices -> (N, 324) float one-hot encoding."""
    states = torch.as_tensor(states, dtype=torch.int64)
    return nn.functional.one_hot(states, NUM_COLORS).flatten(1).float()


class TransformerBlock(nn.Module):
    """Pre-norm block: self-attention, then a feed-forward layer, each with a residual."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        n, t, d = x.shape
        # (N, T, 3D) -> 3 x (N, heads, T, D / heads)
        q, k, v = self.qkv(self.attn_norm(x)).reshape(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attended = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.attn_out(attended.transpose(1, 2).reshape(n, t, d))
        return x + self.mlp(self.mlp_norm(x))


class PolicyNet(nn.Module):
    def __init__(self, dim=128, layers=4, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.config = {"dim": dim, "layers": layers, "heads": heads}
        self.color_embed = nn.Linear(NUM_COLORS, dim)
        self.slot_embed = nn.Parameter(torch.zeros(1, NUM_STICKERS, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_MOVES)
        nn.init.trunc_normal_(self.slot_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """x: (N, 324) one-hot states. Returns (N, 12) move logits."""
        tokens = self.color_embed(x.reshape(-1, NUM_STICKERS, NUM_COLORS)) + self.slot_embed
        x = torch.cat([self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


def load(path):
    checkpoint = torch.load(path, map_location="cpu")
    try:
        model = PolicyNet(**checkpoint["config"])
    except TypeError as error:
        raise SystemExit(
            f"{path} was saved by a different architecture ({checkpoint['config']}). "
            "Train a new model instead of using --resume."
        ) from error
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
