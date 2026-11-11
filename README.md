# GPT from scratch

A small character-level GPT in PyTorch, trained on Tiny Shakespeare (about 1.1M characters, 65 different characters). It learns to predict the next character, and then I use it to generate text.

I first trained a bigram model (it only looks at the current character) as a baseline, then the transformer, so I could see what attention actually adds. After that I made a second version of the GPT with better training settings, and a third, bigger one trained on a GPU.

## Results

| model | parameters | train loss | val loss |
|---|---|---|---|
| bigram | 0.004M | 2.460 | 2.488 |
| GPT v1 (4 layers, 4 heads, 128 dim, context 64) | 0.817M | 1.491 | 1.667 |
| GPT v2 (same model, better training) | 0.817M | 1.493 | 1.659 |
| GPT v3 (6 layers, 6 heads, 384 dim, context 256) | 10.789M | 0.975 | 1.503 |

![loss curves](results/loss_compare.png)

The bigram stops at about 2.5, because one character is not enough context to guess the next one. The GPT v1 looks at the previous 64 characters and gets down to about 1.66.

For v2 I kept the same model but raised dropout from 0.1 to 0.2, added weight decay and a learning rate schedule (short warmup, then decay), and trained for 5000 steps instead of 3000. Val loss only went from 1.667 to 1.659, which is within the noise of the measurement (about ±0.01 between evaluations of the same model), so these changes did not really help this model. The gap between train and val is still about 0.17, so it overfits a bit.

For v3 I made the model much bigger (about 10.8M parameters) and gave it a context of 256 characters. This one needs a GPU: I trained it on a free Colab T4 with fp16 mixed precision, and it took about 24 minutes. Val loss went down to 1.503, clearly better than v2.

The curves also show overfitting. Val loss was lowest at around 1.47 near step 3000, and after that it slowly went back up to 1.50 while train loss kept falling to 0.97. The model starts to memorize the training text, which is easy to do with only 1.1M characters. I report the final numbers (the last evaluation with 200 batches), not the best point. Stopping earlier, more dropout or more data would probably help.

Text from the bigram (step 2999):
```
Whe
n,
Blle ldr:
'?
Thtry aroue gofe ry f y fout st spastinester bes easiritou itondanqu by bo it,
```

Text from GPT v3 (final model, temperature 1.0):
```
O, call me to God, for the meo the Capitol,
And live, pardon, all thy proceedings' mouths,
Since to a plague of benefit had gone,
I follow'd acknowledge; had I waken!

BISHOP OF ELY:
The key-conscience was as my accused from Edward.
This young pursuit of this torture,
Give me my lord: when thou say'st for my queen
Your mother is again.
```

The v3 text has the format of a play, real words and mostly correct grammar, and it even makes up words that sound right ("malapel", "unmale"). The sentences still don't really mean anything, which is what you expect from a model this small trained only on characters.

## How to run

1. `pip install -r requirements.txt`
2. Download Tiny Shakespeare and save it as `data/input.txt`
3. Train with command line flags. The tag is the name used for the files in `results/`:
   - `python src/train.py --model bigram --lr 1e-2 --max_steps 3000 --tag bigram`
   - `python src/train.py --tag gpt_v2` (defaults: 4 layers, 128 dim, context 64)
   - `python src/train.py --tag gpt_v3 --n_embd 384 --n_head 6 --n_layer 6 --block_size 256 --batch_size 64 --lr 6e-4` (needs a GPU, I used a free Colab T4)
4. `python src/sample.py --tag gpt_v3 --prompt "ROMEO:"` to generate text with different temperature and top-k settings
5. `python src/plots.py` to redraw the loss comparison

v1 and v2 run on CPU (v1 took about 12 minutes on my laptop, v2 takes longer because of the extra steps, the bigram about a minute). v3 is too big for that, so it needs a GPU. Samples and loss curves are saved in `results/`.

## Files

```
src/data.py     character encoding and batches
src/model.py    bigram model and the transformer (attention, blocks, GPT)
src/train.py    training loop, evaluation, sampling
src/sample.py   load a trained model and generate text
src/plots.py    loss comparison plot
results/        logs, samples and plots
```

## Next

Try to reduce the overfitting (more dropout, stopping earlier), and try a tokenizer that works on pieces of words instead of single characters.