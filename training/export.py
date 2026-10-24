"""Export a checkpoint for the web page: model/solver.onnx plus model/solver.json.

solver.json holds the cube layout the page needs to read its state and run the search:
the sticker slot order, color order, move names, move permutations and inverses.

    python export.py
"""

import argparse
import json
from pathlib import Path

import torch

from cube_env import FACE_ORDER, INVERSE, MOVES, PERMS, STICKERS
from model import INPUT_SIZE, load
from train import CHECKPOINT_DIR

OUTPUT_DIR = Path(__file__).parent.parent / "model"


class LogSoftmaxPolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return torch.log_softmax(self.model(state), dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "latest.pt"))
    args = parser.parse_args()

    model, checkpoint = load(args.checkpoint)
    OUTPUT_DIR.mkdir(exist_ok=True)

    torch.onnx.export(
        LogSoftmaxPolicy(model),
        (torch.zeros(2, INPUT_SIZE),),
        OUTPUT_DIR / "solver.onnx",
        input_names=["state"],
        output_names=["log_probs"],
        dynamic_axes={"state": {0: "batch"}, "log_probs": {0: "batch"}},
        external_data=False,
    )

    layout = {
        "step": checkpoint["step"],
        "faces": FACE_ORDER,
        "moves": MOVES,
        "inverse": INVERSE.tolist(),
        "perms": PERMS.tolist(),
        "stickers": [[*position, *normal] for position, normal in STICKERS],
    }
    (OUTPUT_DIR / "solver.json").write_text(json.dumps(layout))
    print(f"exported step {checkpoint['step']} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
