"""Train the policy network on freshly generated scrambles.

Every step draws a new batch from data.generate_batch, so the network never sees the same
data twice and there is no dataset file to outgrow.

    python train.py --steps 20000
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import SCRAMBLE_DEPTH, generate_batch
from model import PolicyNet, load, one_hot

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def learning_rate(step, args):
    """Linear warmup, then cosine decay to 10% of the peak by the final step."""
    if step <= args.warmup:
        return args.lr * step / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def save(path, model, optimizer, step):
    torch.save(
        {"config": model.config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-scrambles", type=int, default=160, help=f"scrambles per batch (x{SCRAMBLE_DEPTH} samples)")
    parser.add_argument("--dim", type=int, default=128, help="transformer embedding width")
    parser.add_argument("--layers", type=int, default=4, help="transformer blocks")
    parser.add_argument("--heads", type=int, default=4, help="attention heads per block")
    parser.add_argument("--lr", type=float, default=5e-4, help="peak learning rate")
    parser.add_argument("--warmup", type=int, default=1000, help="steps of linear learning-rate warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
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
        model, checkpoint = PolicyNet(args.dim, args.layers, args.heads), None
        start_step = 0
    device = pick_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    params = sum(p.numel() for p in model.parameters())
    print(f"model {model.config}, {params / 1e6:.2f}M parameters, device {device}")

    loss_fn = nn.CrossEntropyLoss()
    started = time.time()
    total_loss = total_correct = total_samples = 0
    # The bar draws only in a terminal; with output redirected to a file, the log lines remain.
    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        initial=start_step,
        total=args.steps,
        unit="step",
        dynamic_ncols=True,
        disable=None,
    )
    for step in progress:
        states, labels = generate_batch(args.batch_scrambles, rng)
        x, y = one_hot(states).to(device), torch.from_numpy(labels).to(device)

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += len(y)
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.3f}", refresh=False
        )

        if step % args.log_every == 0:
            elapsed = time.time() - started
            progress.write(
                f"step {step:>6}  loss {total_loss / total_samples:.4f}  "
                f"acc {total_correct / total_samples:.3f}  lr {learning_rate(step, args):.1e}  {elapsed / 60:.1f} min"
            )
            total_loss = total_correct = total_samples = 0
        if step % args.save_every == 0 or step == args.steps:
            save(latest, model, optimizer, step)
    progress.close()


if __name__ == "__main__":
    main()
