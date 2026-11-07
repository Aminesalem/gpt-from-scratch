import torch
import matplotlib.pyplot as plt

from data import get_batch, vocab_size, decode
from model import BigramModel

torch.manual_seed(0)

MODEL = "gpt"  # "bigram" or "gpt"
block_size = 64
batch_size = 32
max_steps = 3000
eval_every = 300
eval_iters = 50
lr = 1e-2 if MODEL == "bigram" else 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL == "bigram":
    model = BigramModel(vocab_size)
else:
    from model import GPT
    model = GPT(vocab_size, block_size, n_embd=128, n_head=4, n_layer=4, dropout=0.1)
model = model.to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M parameters")

opt = torch.optim.AdamW(model.parameters(), lr=lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


start = torch.zeros((1, 1), dtype=torch.long, device=device)
steps_log, train_log, val_log = [], [], []

for step in range(max_steps):
    if step % eval_every == 0 or step == max_steps - 1:
        l = estimate_loss()
        steps_log.append(step)
        train_log.append(l["train"])
        val_log.append(l["val"])
        print(f"step {step}: train {l['train']:.4f}, val {l['val']:.4f}")
        print(decode(model.generate(start, 150)[0].tolist()))
        print("-" * 40)
    x, y = get_batch("train", block_size, batch_size, device)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

sample = decode(model.generate(start, 1000)[0].tolist())
with open(f"results/sample_{MODEL}.txt", "w", encoding="utf-8") as f:
    f.write(sample)

plt.plot(steps_log, train_log, label="train")
plt.plot(steps_log, val_log, label="val")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig(f"results/loss_{MODEL}.png", dpi=150)