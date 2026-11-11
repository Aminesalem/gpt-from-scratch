import argparse

import torch

from data import encode, decode, vocab_size
from model import GPT

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="which trained model to load (results/<tag>.pt)")
parser.add_argument("--prompt", default="\n")
parser.add_argument("--length", type=int, default=400)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(f"results/{args.tag}.pt", map_location=device)
a = ckpt["args"]
model = GPT(vocab_size, a["block_size"], a["n_embd"], a["n_head"], a["n_layer"], a["dropout"]).to(device)
model.load_state_dict(ckpt["model"])

# (temperature, top_k): low temperature = safer text, high = more surprising
settings = [(1.0, None), (0.8, None), (0.8, 40), (0.5, 10)]
idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

out = []
for temperature, top_k in settings:
    torch.manual_seed(1)  # same seed, so the settings are comparable
    text = decode(model.generate(idx, args.length, temperature, top_k)[0].tolist())
    out.append(f"##### temperature {temperature}, top_k {top_k}\n{text}\n")

result = "\n".join(out)
print(result)
with open(f"results/samples_{args.tag}.txt", "w", encoding="utf-8") as f:
    f.write(result)