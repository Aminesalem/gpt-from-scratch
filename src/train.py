import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")  # no display needed (Colab, servers)
import matplotlib.pyplot as plt
import torch

from data import get_batch, vocab_size, decode
from model import BigramModel, GPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt", choices=["bigram", "gpt"])
parser.add_argument("--tag", default=None, help="name used for the files in results/")
parser.add_argument("--block_size", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_embd", type=int, default=128)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--eval_every", type=int, default=250)
parser.add_argument("--eval_iters", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
tag = args.tag or args.model

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"  # fp16 is a lot faster on the GPU

log_file = open(f"results/{tag}.txt", "w", encoding="utf-8")


def log(msg=""):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


if args.model == "bigram":
    model = BigramModel(vocab_size)
else:
    model = GPT(vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout)
model = model.to(device)
log(f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M parameters, device: {device}")
log(str(vars(args)))

# weight decay only on the weight matrices, not on biases and layernorm gains
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.99),
)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def get_lr(step):
    # short warmup, then cosine decay down to 10% of the base lr
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(iters):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(split, args.block_size, args.batch_size, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


start = torch.zeros((1, 1), dtype=torch.long, device=device)
steps_log, train_log, val_log = [], [], []
t0 = time.time()

for step in range(args.max_steps):
    for g in opt.param_groups:
        g["lr"] = get_lr(step)
    if step % args.eval_every == 0 or step == args.max_steps - 1:
        l = estimate_loss(args.eval_iters)
        steps_log.append(step)
        train_log.append(l["train"])
        val_log.append(l["val"])
        log(f"step {step}: train {l['train']:.4f}, val {l['val']:.4f} | {(time.time() - t0) / 60:.1f} min")
        log(decode(model.generate(start, 150)[0].tolist()))
        log("-" * 40)
    x, y = get_batch("train", args.block_size, args.batch_size, device)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

final = estimate_loss(200)  # more batches, so the final number is less noisy
log(f"final (200 batches each): train {final['train']:.4f}, val {final['val']:.4f}")
log(f"training time: {(time.time() - t0) / 60:.1f} min")

torch.save({"model": model.state_dict(), "args": vars(args)}, f"results/{tag}.pt")  # save first, in case sampling is slow

sample = decode(model.generate(start, 1000)[0].tolist())
with open(f"results/sample_{tag}.txt", "w", encoding="utf-8") as f:
    f.write(sample)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(steps_log, val_log, color="#2a78d6", linewidth=2, label="val")
ax.plot(steps_log, train_log, color="#2a78d6", linewidth=1.5, linestyle="--", label="train")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title(tag)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"results/loss_{tag}.png", dpi=150)