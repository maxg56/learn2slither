#!/usr/bin/env python3
"""Trace les courbes d'apprentissage d'un modele depuis son CSV lie.

Le fichier modele JSON porte un champ "data" pointant vers le CSV de metriques
ecrit en parallele a l'entrainement (longueur moyenne, record, duree, epsilon,
par generation). On resout ce lien, on lit le CSV et on trace l'evolution.

Usage :
    uv run python tools/plot_training_curve.py --model models/1000sess.json
    uv run python tools/plot_training_curve.py --csv data/1000sess.csv
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

# --- Palette (dataviz : surface claire, une teinte par metrique) -----------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# (colonne CSV, couleur, titre, format valeur). Panneaux independants : une
# seule serie chacun, donc pas de contrainte d'adjacence CVD.
PANELS = [
    ("avg_length", "#2a78d6", "Longueur moyenne (100 dernieres)", "{:.1f}"),
    ("max_length", "#4a3aa7", "Longueur record", "{:.0f}"),
    ("avg_duration", "#1baf7a", "Duree moyenne (pas)", "{:.0f}"),
    ("epsilon", "#e34948", "Epsilon (exploration)", "{:.3f}"),
]


def resolve_csv(model_path):
    """Retourne le chemin du CSV lie a un modele JSON (None si absent)."""
    try:
        with open(model_path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    ref = data.get("data")
    if not ref:
        return None
    # Le champ "data" est relatif au dossier du modele.
    return os.path.normpath(os.path.join(os.path.dirname(model_path), ref))


def load_csv(path):
    """Lit le CSV de metriques en un dict {colonne: [valeurs]}."""
    columns = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames:
            columns[name] = []
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(float(row[name]))
    return columns


def _panel(ax, xs, ys, color, title, value_fmt):
    """Trace une courbe (une metrique) sur un axe."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.plot(xs, ys, color=color, linewidth=2)
    ax.scatter([xs[-1]], [ys[-1]], s=26, color=color, zorder=3,
               edgecolors=SURFACE, linewidths=1.2)
    ax.annotate(value_fmt.format(ys[-1]), (xs[-1], ys[-1]),
                textcoords="offset points", xytext=(-6, 8),
                ha="right", fontsize=10, color=color, fontweight="bold")

    ax.set_title(title, fontsize=11.5, color=INK_PRIMARY,
                 fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("Generation", fontsize=9, color=INK_SECONDARY)
    ax.margins(x=0.02)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="modele JSON (suit son champ 'data')")
    parser.add_argument("--csv", help="CSV de metriques (prioritaire)")
    parser.add_argument("--out", default="docs/training_curve.png",
                        help="chemin du PNG de sortie")
    args = parser.parse_args()

    csv_path = args.csv or (resolve_csv(args.model) if args.model else None)
    if not csv_path:
        print("Fournir --csv, ou --model (JSON avec un champ 'data').",
              file=sys.stderr)
        return 1
    if not os.path.exists(csv_path):
        print("CSV introuvable : {}".format(csv_path), file=sys.stderr)
        return 1

    data = load_csv(csv_path)
    xs = data["episode"]
    if len(xs) < 2:
        print("Pas assez de points pour tracer une courbe.", file=sys.stderr)
        return 1

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.09,
                        wspace=0.18, hspace=0.35)
    for ax, (col, color, title, fmt) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        _panel(ax, xs, data[col], color, title, fmt)

    fig.suptitle("Learn2Slither - courbe d'apprentissage",
                 x=0.07, y=0.965, ha="left", fontsize=15,
                 color=INK_PRIMARY, fontweight="bold")
    fig.text(0.07, 0.905,
             "{} generations - source : {}".format(
                 int(xs[-1]), os.path.basename(csv_path)),
             ha="left", fontsize=9.5, color=INK_SECONDARY)

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("Figure ecrite : {}".format(os.path.normpath(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
