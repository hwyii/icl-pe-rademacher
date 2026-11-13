# Impact of Positional Encoding: Clean and Adversarial Rademacher Complexity for Transformers under In-Context Regression

Code for the simulations in our AISTATS 2026 paper
([paper](https://proceedings.mlr.press/v300/he26d.html), [OpenReview](https://openreview.net/forum?id=18OveXSw7d)).
The setup is based on [Garg et al. (2022)](https://github.com/dtsip/in-context-learning).

## Usage

```bash
pip install -r requirements.txt

# Redraw Figures 1-3 and Table 1 from the released results
python make_figures.py --results results --out figures

# Or reproduce everything from scratch (train, attack, plot)
bash scripts/reproduce.sh
```

`train.py` trains single-layer Transformers with `--pe {none, learned, rope}`, `run_attack.py` evaluates them under
L2 PGD attacks, and `data/` contains the exact dataset used in the paper.

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
