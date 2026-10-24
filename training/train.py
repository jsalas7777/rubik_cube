"""Train the policy network on freshly generated scrambles.

Every step draws a new batch from data.generate_batch, so the network never sees the same
data twice and there is no dataset file to outgrow.

    python train.py --steps 20000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import SCRAMBLE_DEPTH, generate_batch
from model import PolicyNet, load, one_hot

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def save(path, model, optimizer, step):
    torch.save(
        {"config": model.config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-scrambles", type=int, default=160, help=f"scrambles per batch (x{SCRAMBLE_DEPTH} samples)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/latest.pt")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    latest = CHECKPOINT_DIR / "latest.pt"

    if args.resume:
        model, checkpoint = load(latest)
        model.train()
        start_step = checkpoint["step"]
    else:
        model, checkpoint = PolicyNet(args.hidden, args.blocks), None
        start_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    params = sum(p.numel() for p in model.parameters())
    print(f"model {model.config}, {params / 1e6:.1f}M parameters, {torch.get_num_threads()} threads")

    loss_fn = nn.CrossEntropyLoss()
    started = time.time()
    total_loss = total_correct = total_samples = 0
    for step in range(start_step + 1, args.steps + 1):
        states, labels = generate_batch(args.batch_scrambles, rng)
        x, y = one_hot(states), torch.from_numpy(labels)

        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += len(y)

        if step % args.log_every == 0:
            elapsed = time.time() - started
            print(
                f"step {step:>6}  loss {total_loss / total_samples:.4f}  "
                f"acc {total_correct / total_samples:.3f}  {elapsed / 60:.1f} min",
                flush=True,
            )
            total_loss = total_correct = total_samples = 0
        if step % args.save_every == 0 or step == args.steps:
            save(latest, model, optimizer, step)


if __name__ == "__main__":
    main()
