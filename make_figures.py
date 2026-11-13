"""Reproduce Figures 1-3 and Table 1 of the paper from clean.csv / attack.csv (means over seeds)."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"learned": "#0072B2", "none": "#D55E00"}
MARKERS = {"learned": "o", "none": "s"}
LABELS = {"learned": "With PE", "none": "Without PE", "rope": "RoPE"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 5})


def new_axes(title):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.set_title(title)
    ax.set_xlabel("Context Length ($t$)")
    ax.set_ylabel("Generalization Gap")
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig, ax


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def figure1(clean, out):
    fig, ax = new_axes("Generalization Gap vs. Context Length ($t$)")
    for pe, ls in [("learned", "-"), ("none", "--")]:
        g = clean[clean.pe == pe].groupby("t").gap.mean()
        ax.plot(g.index, g.values, marker=MARKERS[pe], linestyle=ls, color=COLORS[pe], label=LABELS[pe],
                linewidth=1.5, markersize=3)
    ax.legend()
    save(fig, os.path.join(out, "fig1_gap_pe_vs_nope.pdf"))


def figure2(clean, attack, out, eps=0.02):
    fig, ax = new_axes(f"Generalization Gap under PGD Attack ($\\epsilon$={eps})")
    for pe in ["learned", "none"]:
        curves = [("Clean", "-", clean[clean.pe == pe].groupby("t").gap.mean()),
                  ("Attacked", "--", attack[(attack.pe == pe) & np.isclose(attack.eps, eps)].groupby("t").attacked_gap.mean())]
        for cond, ls, g in curves:
            ax.plot(g.index, g.values, marker=MARKERS[pe], linestyle=ls, color=COLORS[pe],
                    label=f"{LABELS[pe]}, {cond}", linewidth=1.5, markersize=3)
    ax.legend(ncol=2)
    save(fig, os.path.join(out, "fig2_attack_pe_vs_nope.pdf"))


def figure3(attack, out, pe="none", eps_list=(0.05, 0.2, 0.3)):
    fig, ax = new_axes(f"Different Attack Strength ({LABELS[pe]})")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(labelsize=5)
    for color, eps in zip(cm.viridis(np.linspace(0, 0.4, len(eps_list))), eps_list):
        g = attack[(attack.pe == pe) & np.isclose(attack.eps, eps)].groupby("t").attacked_gap.mean()
        ax.plot(g.index, g.values, marker="o", color=color, label=f"ε={eps}", linewidth=1.5, markersize=3.5)
    ax.legend(title="Attack Strength")
    fig.tight_layout()
    save(fig, os.path.join(out, "fig3_attack_strength_nope.pdf"))


def table1(clean, out):
    table = clean.groupby(["t", "pe"]).gap.mean().unstack("pe")[["none", "learned", "rope"]]
    table.columns = [LABELS[c] for c in table.columns]
    print("\nTable 1: generalization gap (mean over seeds)\n")
    print(table.round(4).to_string())
    table.round(4).to_csv(os.path.join(out, "table1_rope.csv"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results", help="directory with clean.csv and attack.csv")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clean = pd.read_csv(os.path.join(args.results, "clean.csv"))
    clean = clean[clean.t > 5]  # the analysis requires t > d = 5
    figure1(clean, args.out)
    table1(clean, args.out)
    attack_path = os.path.join(args.results, "attack.csv")
    if os.path.exists(attack_path):
        attack = pd.read_csv(attack_path)
        attack = attack[attack.t > 5]
        figure2(clean, attack, args.out)
        figure3(attack, args.out)


if __name__ == "__main__":
    main()
