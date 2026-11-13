#!/usr/bin/env bash
# Full reproduction: train all models, attack them, and redraw Figures 1-3 / Table 1 into figures/.
# Each training run takes ~80 s on an RTX A6000 (180 runs in total).
set -euo pipefail
cd "$(dirname "$0")/.."

python train.py --pe none learned --seeds 42 123 256 567 777 2024 --out-dir outputs
python train.py --pe rope         --seeds 42 123 256 567 777 1024 --out-dir outputs
python run_attack.py --pe none learned --eps 0.02 0.05 0.2 0.3 --out-dir outputs
python make_figures.py --results outputs --out figures
