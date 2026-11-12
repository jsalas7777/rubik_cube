"""Write a model for the web page: model/solver.onnx plus model/solver.json.

train.py calls export_for_web every time it saves a checkpoint, so the page always runs
the latest model. Run this script directly only to export a specific checkpoint:

    python export.py --checkpoint checkpoints/latest.pt

solver.json holds the cube layout the page needs to read its state and run the search:
the sticker slot order, color order, move names, move permutations and inverses.
"""

import argparse
import copy
import json
import logging
import os
import warnings
from pathlib import Path

import torch

from cube_env import FACE_ORDER, INVERSE, MOVES, PERMS, STICKERS
from model import CHECKPOINT_DIR, INPUT_SIZE, load

OUTPUT_DIR = Path(__file__).parent.parent / "model"

# The ONNX exporter logs harmless "torchvision is not installed" notices on every export.
logging.getLogger("torch.onnx").setLevel(logging.ERROR)


class LogSoftmaxPolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        return torch.log_softmax(self.model(state), dim=1)


def export_for_web(model, step, output_dir=OUTPUT_DIR):
    """Writes solver.onnx and solver.json. The model is copied to the CPU, so training can go on."""
    output_dir.mkdir(exist_ok=True)
    model = copy.deepcopy(model).cpu().eval()

    # Write to temporary files and rename them, so the page never loads a half-written model.
    onnx_tmp = output_dir / "solver.onnx.tmp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            LogSoftmaxPolicy(model),
            (torch.zeros(2, INPUT_SIZE),),
            onnx_tmp,
            input_names=["state"],
            output_names=["log_probs"],
            dynamic_axes={"state": {0: "batch"}, "log_probs": {0: "batch"}},
            external_data=False,
            verbose=False,
        )

    layout = {
        "step": step,
        "faces": FACE_ORDER,
        "moves": MOVES,
        "inverse": INVERSE.tolist(),
        "perms": PERMS.tolist(),
        "stickers": [[*position, *normal] for position, normal in STICKERS],
    }
    json_tmp = output_dir / "solver.json.tmp"
    json_tmp.write_text(json.dumps(layout))

    os.replace(onnx_tmp, output_dir / "solver.onnx")
    os.replace(json_tmp, output_dir / "solver.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "latest.pt"))
    args = parser.parse_args()

    model, checkpoint = load(args.checkpoint)
    export_for_web(model, checkpoint["step"])
    print(f"exported step {checkpoint['step']} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
