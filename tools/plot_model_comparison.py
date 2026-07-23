#!/usr/bin/env python3
"""Compare les modeles livres (1/10/100/1000 sessions) en exploitation pure.

Charge chaque Q-table, joue N parties avec epsilon=0 (aucun apprentissage) et
trace la distribution de la longueur maximale atteinte et de la duree de survie
par modele. Sortie : docs/model_comparison.png.

Usage :
    .venv/bin/python tools/plot_model_comparison.py [--games 100] [--seed 0]
"""

import argparse
import os
import random
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from snakeai.core import Environment          # noqa: E402
from snakeai.learning import Agent            # noqa: E402
from snakeai.perception import Interpreter    # noqa: E402
from snakeai.training import run_session      # noqa: E402

# Modeles livres : (sessions d'entrainement, chemin). Axe ordinal croissant.
MODELS = [
    (1, "models/1sess.txt"),
    (10, "models/10sess.txt"),
    (100, "models/100sess.txt"),
    (1000, "models/1000sess.txt"),
]

# --- Palette (dataviz : rampe bleue ORDINALE, surface claire) --------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# 4 pas ordinaux clair->fonce (le plus clair >= step 250 pour rester lisible).
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def evaluate(path, games, seed):
    """Joue `games` parties d'un modele et retourne (lengths, durations).

    Renvoie (None, None) si le modele est introuvable ou illisible.
    """
    agent = Agent()
    if not agent.load(path):
        return None, None
    agent.epsilon = 0.0                       # exploitation pure
    interp = Interpreter()
    env = Environment()
    lengths, durations = [], []
    random.seed(seed)                         # reproductibilite par modele
    for _ in range(games):
        length, duration = run_session(
            env, interp, agent,
            learn=False, verbose=False, step_by_step=False,
        )
        lengths.append(length)
        durations.append(duration)
    return lengths, durations


def _panel(ax, series, labels, title, ylabel):
    """Trace un panneau : box plots ordinaux + nuage de points + moyenne."""
    positions = range(1, len(series) + 1)

    # Grille horizontale recessive uniquement, sous les marques.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)

    bp = ax.boxplot(
        series, positions=positions, widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color=SURFACE, linewidth=1.8),
        whiskerprops=dict(color=INK_SECONDARY, linewidth=1.2),
        capprops=dict(color=INK_SECONDARY, linewidth=1.2),
        boxprops=dict(linewidth=0),
    )
    for box, color in zip(bp["boxes"], RAMP):
        box.set_facecolor(color)
        box.set_edgecolor(SURFACE)      # anneau 2px de surface entre marques
        box.set_linewidth(2)

    # Nuage de points brut (jitter leger) : montre la dispersion reelle.
    for pos, values, color in zip(positions, series, RAMP):
        jitter = np.random.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(np.array([pos]) + jitter, values, s=10,
                   color=color, alpha=0.28, linewidths=0, zorder=1)

    # Moyenne : losange en encre primaire, valeur en label direct.
    means = [float(np.mean(v)) for v in series]
    ax.scatter(positions, means, marker="D", s=42, color=INK_PRIMARY,
               zorder=4, edgecolors=SURFACE, linewidths=1.2)
    top = max(max(v) for v in series)
    for pos, mean in zip(positions, means):
        ax.annotate("{:.1f}".format(mean), (pos, mean),
                    textcoords="offset points", xytext=(12, 0),
                    va="center", ha="left", fontsize=9,
                    color=INK_PRIMARY, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                              edgecolor="none", alpha=0.9))

    ax.set_title(title, fontsize=12, color=INK_PRIMARY,
                 fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=10, color=INK_PRIMARY)
    ax.set_xlabel("Sessions d'entrainement", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(0, top * 1.12)
    ax.margins(x=0.08)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="parties jouees par modele (defaut 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="graine aleatoire (defaut 0)")
    parser.add_argument("--out", default="docs/model_comparison.png",
                        help="chemin du PNG de sortie")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    labels, lengths, durations = [], [], []
    print("Evaluation de {} parties par modele (epsilon=0)...".format(
        args.games))
    for sessions, rel in MODELS:
        length, duration = evaluate(
            os.path.join(root, rel), args.games, args.seed)
        if length is None:
            print("  ! modele introuvable, ignore : {}".format(rel))
            continue
        labels.append(str(sessions))
        lengths.append(length)
        durations.append(duration)
        print("  {:>4} sessions : longueur moy {:.1f} (max {}), "
              "duree moy {:.0f}".format(
                  sessions, np.mean(length), max(length), np.mean(duration)))

    if len(labels) < 2:
        print("Pas assez de modeles a comparer.", file=sys.stderr)
        return 1

    np.random.seed(args.seed)   # jitter reproductible
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12,
                        wspace=0.22)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    _panel(axes[0], lengths, labels,
           "Longueur maximale atteinte", "Longueur (cellules)")
    _panel(axes[1], durations, labels,
           "Duree de survie", "Pas avant game over")

    fig.suptitle("Learn2Slither - performance par volume d'entrainement",
                 x=0.07, y=0.95, ha="left", fontsize=15,
                 color=INK_PRIMARY, fontweight="bold")
    fig.text(0.07, 0.885,
             "{} parties en exploitation pure (epsilon=0) par modele. "
             "Losange = moyenne, boite = quartiles, points = parties.".format(
                 args.games),
             ha="left", fontsize=9.5, color=INK_SECONDARY)

    out = os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("Figure ecrite : {}".format(os.path.normpath(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
