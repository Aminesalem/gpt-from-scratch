# GPT from scratch

A small character-level GPT in PyTorch, trained on Tiny Shakespeare (about 1.1M characters, 65 different characters). It learns to predict the next character, and then I use it to generate text.

I first trained a bigram model (it only looks at the current character) as a baseline, then the transformer, so I could see what attention actually adds.

## Results

| model | parameters | train loss | val loss |
|---|---|---|---|
| bigram | 0.004M | 2.460 | 2.488 |
| GPT (4 layers, 4 heads, 128 dim) | 0.817M | 1.491 | 1.667 |

![loss curves](results/loss_compare.png)

The bigram stops at about 2.5, because one character is not enough context to guess the next one. The GPT looks at the previous 64 characters and gets down to 1.67. Its train loss is lower than val loss and the gap grows toward the end, so it starts to overfit a bit (small dataset).

Text from the bigram (step 2999):
```
Whe
n,
Blle ldr:
'?
Thtry aroue gofe ry f y fout st spastinester bes easiritou itondanqu by bo it,
```

Text from the GPT (step 2999):
```
I land your heaven, and keep.

JOHN OF GAUNT:
What he well stay have, roship, let's resubping contre
calle their hase into beheld.
```

The GPT has learned the format of a play (character names, line breaks) and many real words, but the sentences don't mean much yet.

## How to run

1. `pip install -r requirements.txt`
2. Download Tiny Shakespeare and save it as `data/input.txt`
3. In `src/train.py`, set `MODEL` to `"bigram"` or `"gpt"`
4. `python src/train.py`

It runs on CPU. The GPT takes a bit more than 10 minutes, the bigram about a minute. Samples and loss curves are saved in `results/`.

## Files

```
src/data.py     character encoding and batches
src/model.py    bigram model and the transformer (attention, blocks, GPT)
src/train.py    training loop, evaluation, sampling
src/plots.py    loss comparison plot
results/        logs, samples and plots
```

## Next

Try temperature and top-k sampling, and train longer with a bigger model on a GPU.