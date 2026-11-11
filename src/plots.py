import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (log file in results/, label, color) - a model keeps its color in every chart
RUNS = [
    ("bigram", "bigram", "#2a78d6"),
    ("gpt", "GPT v1", "#eb6834"),
    ("gpt_v2", "GPT v2", "#1baf7a"),
    ("gpt_v3", "GPT v3 (bigger)", "#6250d6"),
]


def read_text(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):  # PowerShell's Tee-Object saves utf-16
        return raw.decode("utf-16")
    return raw.decode("utf-8")


def read_log(path):
    steps, train, val = [], [], []
    for line in read_text(path).splitlines():
        m = re.match(r"step (\d+): train ([\d.]+), val ([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    return steps, train, val


fig, ax = plt.subplots(figsize=(8, 5))
for name, label, color in RUNS:
    path = f"results/{name}.txt"
    if not os.path.exists(path):
        continue
    steps, train, val = read_log(path)
    ax.plot(steps, val, color=color, linewidth=2, label=f"{label} (val {val[-1]:.3f})")
    ax.plot(steps, train, color=color, linewidth=1.5, linestyle="--")

ax.plot([], [], color="gray", linewidth=1.5, linestyle="--", label="dashed = train")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.grid(axis="y", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("results/loss_compare.png", dpi=150)