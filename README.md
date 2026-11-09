# GPT from scratch

A small character-level GPT in PyTorch, trained on Tiny Shakespeare (about 1.1M characters, 65 different characters). It learns to predict the next character, and then I use it to generate text.

I first trained a bigram model (it only looks at the current character) as a baseline, then the transformer, so I could see what attention actually adds. After that I made a second version of the GPT with better training settings.

## Results

| model | parameters | train loss | val loss |
|---|---|---|---|
| bigram | 0.004M | 2.460 | 2.488 |
| GPT v1 (4 layers, 4 heads, 128 dim) | 0.817M | 1.491 | 1.667 |
| GPT v2 (same model, better training) | 0.817M | 1.493 | 1.659 |

![loss curves](results/loss_compare.png)

The bigram stops at about 2.5, because one character is not enough context to guess the next one. The GPT looks at the previous 64 characters and gets down to about 1.66.

For v2 I kept the same model but raised dropout from 0.1 to 0.2, added weight decay and a learning rate schedule (short warmup, then decay), and trained for 5000 steps instead of 3000. Val loss only went from 1.667 to 1.659, and the gap between train and val is still about 0.17, so the model overfits a bit. For this small model and dataset these changes help very little. A bigger model with more context would be the next thing to try.

Text from the bigram (step 2999):
```
Whe
n,
Blle ldr:
'?
Thtry aroue gofe ry f y fout st spastinester bes easiritou itondanqu by bo it,
```

Text from GPT v2 (step 4999):
```
How brave my bales' own balk which beer been.
HAs Jotest we have refore Richard to--
Then think the rements oath and never.

MPnd RICHARIV:
Ay? thou m
```

The GPT has learned the format of a play (character names, line breaks) and many real words, but the sentences don't mean much yet.

## How to run

1. `pip install -r requirements.txt`
2. Download Tiny Shakespeare and save it as `data/input.txt`
3. In `src/train.py`, set `MODEL` to `"bigram"` or `"gpt"`, and `TAG` to the name you want for the output files. The file now has the v2 settings. v1 used dropout 0.1, a constant learning rate of 1e-3 and 3000 steps.
4. `python src/train.py`

It runs on CPU. The GPT v1 took about 12 minutes on my laptop, and v2 takes longer because of the extra steps. The bigram takes about a minute. Samples and loss curves are saved in `results/`.

## Files

```
src/data.py     character encoding and batches
src/model.py    bigram model and the transformer (attention, blocks, GPT)
src/train.py    training loop, evaluation, sampling
src/plots.py    loss comparison plot
results/        logs, samples and plots
```

## Next

Train a bigger model with a longer context on a GPU, and try temperature and top-k sampling.