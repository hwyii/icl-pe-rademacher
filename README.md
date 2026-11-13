# Impact of Positional Encoding: Clean and Adversarial Rademacher Complexity for Transformers under In-Context Regression

Code for the simulations in

> Weiyi He, Yue Xing. **Impact of Positional Encoding: Clean and Adversarial Rademacher Complexity for Transformers under In-Context Regression.** AISTATS 2026, PMLR 300:4438–4446.
> [[paper]](https://proceedings.mlr.press/v300/he26d.html) [[pdf]](https://raw.githubusercontent.com/mlresearch/v300/main/assets/he26d/he26d.pdf) [[OpenReview]](https://openreview.net/forum?id=18OveXSw7d)

We train single-layer, single-head Transformers on in-context linear regression and compare the (clean and adversarial)
generalization gap of models **with a trainable positional encoding (PE)**, **without PE**, and **with RoPE**.

## Setup

```bash
pip install -r requirements.txt
```

`transformers` is pinned to 4.46.3 because the model patches GPT-2's eager attention of that version (see below).

## Reproduce the figures from the released results

`results/` holds the per-run numbers behind the paper, so the figures can be redrawn without a GPU:

```bash
python make_figures.py --results results --out figures
```

| Output | Paper |
|---|---|
| `figures/fig1_gap_pe_vs_nope.pdf` | Figure 1 — generalization gap vs. context length, with / without PE |
| `figures/fig2_attack_pe_vs_nope.pdf` | Figure 2 — clean vs. attacked gap (PGD, ε = 0.02) |
| `figures/fig3_attack_strength_nope.pdf` | Figure 3 — attacked gap for ε ∈ {0.05, 0.2, 0.3}, without PE |
| `figures/table1_rope.csv` (also printed) | Table 1 — Without PE / Trainable PE / RoPE |

## Reproduce from scratch

```bash
bash scripts/reproduce.sh
```

which runs

```bash
python train.py --pe none learned --seeds 42 123 256 567 777 2024 --out-dir outputs   # Figure 1, Table 1
python train.py --pe rope         --seeds 42 123 256 567 777 1024 --out-dir outputs   # Table 1
python run_attack.py --pe none learned --eps 0.02 0.05 0.2 0.3 --out-dir outputs     # Figures 2, 3
python make_figures.py --results outputs --out figures
```

On an RTX A6000 each training run takes about 80 seconds and each (model, ε) attack evaluation about 16 seconds. Single runs can be launched with e.g.
`python train.py --pe learned --context-lens 10 --seeds 42`.

## Files

```
pe_icl/model.py    single-layer GPT-2 backbone with --pe {none, learned, rope}
pe_icl/data.py     task generation, fixed training prompts, evaluation
pe_icl/attack.py   L2 PGD attack on the prompt inputs
train.py           training + clean generalization gap      -> outputs/clean.csv, outputs/checkpoints/
run_attack.py      adversarial evaluation of saved models   -> outputs/attack.csv
make_figures.py    Figures 1-3 and Table 1
data/              the dataset used in the paper (train: 309 tasks, validation: 9,991 tasks, d = 5)
results/           clean.csv and attack.csv from the paper's runs
```

## Setup details

- **Data.** `w ~ N(0, I_d/d)`, `x ~ N(0, I_d)`, `y = wᵀx`, `d = 5`; 10,300 tasks of 100 points each, split into 309
  training and 9,991 validation tasks. Each task stores a fixed permutation of its points. The training set is one
  fixed prompt of length `t` per training task (points `perm[:t]`); evaluation prompts use context `perm[:t-1]` and
  query `perm[-1]`. The released `data/*.pt` files are the exact data used in the paper; if they are missing,
  `train.py` regenerates a (different) dataset with a fixed seed.
- **Model.** Read-in layer → one GPT-2 block (1 head, `d_m = 1024`, bidirectional attention) → read-out. Token `i` is
  `[x_i, y_i]`, the query token is `[x_q, 0]`, and the loss is the squared error on the query prediction only.
  The trainable PE is GPT-2's learned absolute PE, initialised at 25× the GPT-2 default (`--pe-scale 25`) so that its
  norm matches the content embeddings. RoPE rotates queries and keys and has no trainable positional parameters.
- **Training.** Adam, learning rate 1e-5, 3,000 epochs, full-batch (batch size 309).
- **Generalization gap.** `val_loss − memorization_loss`, where the memorization loss is the loss on the fixed training
  prompts. The attacked gap replaces `val_loss` by the loss on attacked validation prompts.
- **Attack.** PGD on all prompt inputs `x_1, …, x_t, x_q` with `‖X′ − X‖₂ ≤ ε` over the whole prompt, 40 steps of
  size ε/10 along the L2-normalised gradient.
- **Seeds.** Each configuration is run with 6 seeds (`42 123 256 567 777 2024` for PE / no PE,
  `42 123 256 567 777 1024` for RoPE); figures report the mean over seeds.

Retraining with this code matches the paper's runs to about three decimal places (GPU non-determinism); evaluating
the paper's original checkpoints with this code reproduces the recorded clean and attacked losses exactly.

## Citation

```bibtex
@InProceedings{pmlr-v300-he26d,
  title     = {Impact of Positional Encoding: Clean and Adversarial Rademacher Complexity for Transformers under In-Context Regression},
  author    = {He, Weiyi and Xing, Yue},
  booktitle = {Proceedings of The 29th International Conference on Artificial Intelligence and Statistics},
  pages     = {4438--4446},
  year      = {2026},
  volume    = {300},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

## Acknowledgements

The training setup builds on [Garg et al. (2022)](https://github.com/dtsip/in-context-learning)
("What Can Transformers Learn In-Context? A Case Study of Simple Function Classes").
