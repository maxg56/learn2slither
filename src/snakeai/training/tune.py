"""tune.py - recherche par grille des hyperparametres alpha/gamma/epsilon.

Bonus autonome, hors du flux principal `cli.py` -> `train()` : balaye une
petite grille de combinaisons (alpha, gamma, epsilon_decay), entraine pour
chacune un `Agent` + `Environment` + `Interpreter` frais sur un nombre modeste
de sessions via `training.trainer.train()`, classe les resultats et les
ecrit en CSV. Ne contient ni regles du jeu ni logique de Q-learning : il ne
fait qu'orchestrer des entrainements repetes et en comparer les resultats.

Usage :
    PYTHONPATH=src python -m snakeai.training.tune
    PYTHONPATH=src python -m snakeai.training.tune --sessions 200 \
        --output models/tuning_results.csv
"""

import argparse
import contextlib
import csv
import io
import os
import time

from snakeai.core import Environment
from snakeai.learning import Agent
from snakeai.perception import Interpreter
from snakeai.training.trainer import train

# Grille volontairement petite (18 combinaisons) pour rester rapide tout en
# couvrant les trois hyperparametres cles du Q-learning.
ALPHAS = (0.05, 0.1, 0.2)
GAMMAS = (0.8, 0.9, 0.95)
EPSILON_DECAYS = (0.98, 0.995)
SESSIONS_PER_COMBO = 200
DEFAULT_OUTPUT = "models/tuning_results.csv"

_FIELDNAMES = ["alpha", "gamma", "epsilon_decay", "sessions",
               "best_length", "best_duration", "mean_length",
               "mean_duration"]


def _train_quietly(env, interp, agent, sessions):
    """Enchaine `sessions` parties independantes via `train()` (une session
    par appel, pour recuperer le resultat de chaque partie) en supprimant
    l'affichage habituel de `train()`. Retourne (longueurs, durees), une
    valeur par session, pour permettre le calcul d'une moyenne."""
    lengths = []
    durations = []
    single_session = argparse.Namespace(
        sessions=1, visual="off", dontlearn=False, step_by_step=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(sessions):
            length, duration = train(env, interp, agent, single_session)
            lengths.append(length)
            durations.append(duration)
    return lengths, durations


def evaluate_combo(alpha, gamma, epsilon_decay, sessions=SESSIONS_PER_COMBO):
    """Entraine un agent frais pour une combinaison d'hyperparametres et
    retourne un dict de resultats (best_length, best_duration, moyennes)."""
    env = Environment()
    interp = Interpreter()
    agent = Agent(alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    lengths, durations = _train_quietly(env, interp, agent, sessions)
    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "sessions": sessions,
        "best_length": max(lengths),
        "best_duration": max(durations),
        "mean_length": sum(lengths) / len(lengths),
        "mean_duration": sum(durations) / len(durations),
    }


def run_grid_search(
        alphas=ALPHAS, gammas=GAMMAS, epsilon_decays=EPSILON_DECAYS,
        sessions=SESSIONS_PER_COMBO, output_path=DEFAULT_OUTPUT):
    """Balaye la grille alpha x gamma x epsilon_decay.

    Entraine un agent frais par combinaison, classe les resultats (par
    best_length puis best_duration decroissants), les ecrit en CSV et
    affiche un tableau resume. Retourne la liste triee des resultats.
    """
    combos = [(a, g, e)
              for a in alphas for g in gammas for e in epsilon_decays]
    start = time.time()
    results = [evaluate_combo(alpha, gamma, epsilon_decay, sessions)
               for alpha, gamma, epsilon_decay in combos]
    elapsed = time.time() - start

    results.sort(key=lambda r: (r["best_length"], r["best_duration"]),
                 reverse=True)
    _write_csv(results, output_path)
    _print_table(results, elapsed, output_path)
    return results


def _write_csv(results, output_path):
    """Ecrit les resultats en CSV (jamais de crash si le dossier manque)."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _print_table(results, elapsed, output_path):
    """Affiche un tableau classe des resultats sur stdout."""
    print("Recherche par grille terminee en {:.2f}s ({} combinaisons, "
          "{} sessions chacune)"
          .format(elapsed, len(results), results[0]["sessions"]))
    print("Resultats ecrits dans {}".format(output_path))
    print("{:>3}  {:>6} {:>6} {:>8}  {:>11} {:>13} {:>9}"
          .format("#", "alpha", "gamma", "eps_dec", "best_len",
                  "best_dur", "mean_len"))
    for rank, row in enumerate(results, start=1):
        print("{:>3}  {:>6.3f} {:>6.2f} {:>8.3f}  {:>11} {:>13} {:>9.2f}"
              .format(rank, row["alpha"], row["gamma"],
                      row["epsilon_decay"], row["best_length"],
                      row["best_duration"], row["mean_length"]))


def _parse_args(argv=None):
    description = ("Recherche par grille des hyperparametres "
                   "alpha/gamma/epsilon_decay.")
    parser = argparse.ArgumentParser(
        prog="snakeai.training.tune",
        description=description,
    )
    parser.add_argument("--sessions", type=int, default=SESSIONS_PER_COMBO,
                        help="sessions d'entrainement par combinaison")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="chemin du fichier CSV de resultats")
    return parser.parse_args(argv)


def main(argv=None):
    """Point d'entree pour `python -m snakeai.training.tune`."""
    args = _parse_args(argv)
    run_grid_search(sessions=args.sessions, output_path=args.output)


if __name__ == "__main__":
    main()
