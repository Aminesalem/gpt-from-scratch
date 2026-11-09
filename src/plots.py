import re
import matplotlib.pyplot as plt


def read_log(path):
    steps, train, val = [], [], []
    for line in open(path, encoding="utf-16"):
        m = re.match(r"step (\d+): train ([\d.]+), val ([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    return steps, train, val


for name in ["bigram", "gpt", "gpt_v2"]:
    steps, train, val = read_log(f"results/{name}.txt")
    plt.plot(steps, val, label=f"{name} val")
    plt.plot(steps, train, "--", label=f"{name} train")

plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("results/loss_compare.png", dpi=150)