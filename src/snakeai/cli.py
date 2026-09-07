"""cli.py - point d'entree : parsing CLI et cablage des composants.

Ne contient aucune logique de jeu, d'apprentissage ni de boucle : il se
contente d'interpreter les flags, d'assembler Agent/Interpreter/Environment
et de deleguer soit au trainer, soit au dashboard.
"""

import argparse
import os
import random
import sys
import tempfile

from snakeai import constants
from snakeai.core import Environment
from snakeai.learning import Agent, NNAgent
from snakeai.perception import Interpreter
from snakeai.training import MetricsRecorder, plot as plot_metrics
from snakeai.training import run_session, train
from snakeai.training.replay import replay as replay_recording
from snakeai.training.replay import save_recording

# Paliers de longueur du bonus "Records atteints" (cf. sujet 42).
LENGTH_MILESTONES = (15, 20, 25, 30, 35)


def parse_args(argv=None):
    """Parse les flags de la ligne de commande."""
    parser = argparse.ArgumentParser(
        prog="snake",
        description="Learn2Slither - Q-learning sur un Snake 10x10.",
    )
    parser.add_argument("-sessions", type=int, default=1,
                        help="nombre de sessions d'entrainement")
    parser.add_argument("-save", metavar="PATH",
                        help="chemin d'export du modele")
    parser.add_argument("-load", metavar="PATH",
                        help="chemin d'import du modele")
    parser.add_argument("-visual", choices=["on", "off"], default="on",
                        help="affichage graphique pygame")
    parser.add_argument("-dontlearn", action="store_true",
                        help="exploitation pure (epsilon=0, pas d'update)")
    parser.add_argument("-step-by-step", dest="step_by_step",
                        action="store_true",
                        help="avance action par action")
    parser.add_argument("-dashboard", action="store_true",
                        help="vue parallele : une grille de parties a la fois")
    parser.add_argument("-grid", type=int, default=6,
                        help="cote de la grille du dashboard (grid x grid)")
    parser.add_argument("-metrics", metavar="PATH",
                        help="export CSV des courbes d'entrainement")
    parser.add_argument("-plot", metavar="PATH",
                        help="export PNG des courbes (necessite matplotlib)")
    parser.add_argument("-record", metavar="PATH",
                        help="enregistre une session unique (frames) vers"
                             " PATH")
    parser.add_argument("-replay", metavar="PATH",
                        help="rejoue un enregistrement -record, sans agent")
    parser.add_argument("-model", choices=["qtable", "nn"], default="qtable",
                        help="fonction Q utilisee : Q-table ou reseau NN")
    parser.add_argument("-seed", type=int, default=None,
                        help="graine aleatoire pour des runs reproductibles")
    parser.add_argument("-board-size", dest="board_size", type=int,
                        default=None,
                        help="cote du board (defaut : constants.BOARD_SIZE)")
    parser.add_argument("-benchmark", action="store_true",
                        help="agrege longueur/duree de toutes les sessions")
    parser.add_argument("-reward-shaping", dest="reward_shaping",
                        choices=["default", "alt"], default="default",
                        help="schema de reward : historique ou alternatif "
                             "(anti demi-tour + bonus de survie)")
    parser.add_argument("-dashboard-lobby", dest="dashboard_lobby",
                        action="store_true",
                        help="affiche un lobby de choix de modele avant "
                             "de lancer le dashboard (-dashboard requis)")
    return parser.parse_args(argv)


def main():
    """Point d'entree du programme."""
    args = parse_args()
    if args.seed is not None:
        # Graine le module `random` global : utilise a la fois par
        # Environment (placement serpent/pommes) et Agent (epsilon-greedy),
        # donc suffisant pour rendre un run reproductible.
        random.seed(args.seed)

    if args.replay:
        _run_replay(args)
        return

    agent = _build_agent(args)
    learn = not args.dontlearn
    interp = Interpreter(reward_mode=args.reward_shaping)

    if args.dashboard:
        _run_dashboard(agent, interp, learn, args)
        return

    display = _make_display(args.visual == "on")
    size = args.board_size if args.board_size is not None \
        else constants.BOARD_SIZE
    env = Environment(size=size)
    recorder = MetricsRecorder() if (args.metrics or args.plot) else None

    if args.record:
        _run_recorded_session(env, interp, agent, learn, args, display)
    else:
        best_length, best_duration, lengths, durations = train(
            env, interp, agent, args, display, recorder)
        print("Game over, max length = {}, max duration = {}"
              .format(best_length, best_duration))
        _print_records(best_length)
        if args.benchmark:
            _print_benchmark(lengths, durations)
        _save_model(agent, args.save)
        _export_metrics(recorder, args.metrics, args.plot)

    if display is not None:
        display.close()


def _build_agent(args):
    """Cree l'agent, charge un modele et applique le mode -dontlearn."""
    agent = NNAgent() if args.model == "nn" else Agent()
    if args.load:
        if agent.load(args.load):
            print("Modele charge depuis {}".format(args.load))
        else:
            print("Avertissement : echec du chargement de {}"
                  .format(args.load), file=sys.stderr)
    if args.dontlearn:
        agent.epsilon = 0.0
    return agent


def _save_model(agent, path):
    """Sauvegarde le modele si un chemin est fourni (jamais de crash)."""
    if not path:
        return
    if agent.save(path):
        print("Modele sauvegarde dans {}".format(path))
    else:
        print("Avertissement : echec de la sauvegarde dans {}"
              .format(path), file=sys.stderr)


def _export_metrics(recorder, metrics_path, plot_path):
    """Ecrit le CSV des courbes et, si demande, le plot (jamais de crash).

    `-plot` sans `-metrics` passe par un CSV temporaire uniquement pour
    alimenter `plot()`, qui lit depuis un fichier comme le reste du module.
    """
    if recorder is None:
        return

    csv_path = metrics_path
    if metrics_path:
        if recorder.save(metrics_path):
            print("Metriques sauvegardees dans {}".format(metrics_path))
        else:
            print("Avertissement : echec de l'export des metriques dans {}"
                  .format(metrics_path), file=sys.stderr)
            csv_path = None

    if not plot_path:
        return

    tmp_csv = None
    if csv_path is None:
        tmp_handle = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False)
        tmp_handle.close()
        tmp_csv = tmp_handle.name
        if not recorder.save(tmp_csv):
            print("Avertissement : impossible de preparer les donnees "
                  "du plot", file=sys.stderr)
            os.unlink(tmp_csv)
            return
        csv_path = tmp_csv

    if plot_metrics(csv_path, plot_path):
        print("Graphique sauvegarde dans {}".format(plot_path))
    if tmp_csv is not None:
        os.unlink(tmp_csv)


def _run_recorded_session(env, interp, agent, learn, args, display):
    """Joue une session unique en enregistrant chaque frame vers -record.

    L'agent joue normalement (honore -dontlearn/-load) ; seule la boucle
    multi-sessions de `train()` est court-circuitee, puisqu'un
    enregistrement ne porte que sur une seule partie.
    """
    verbose = args.visual == "on" or args.step_by_step
    frames = []
    length, duration, _total_reward = run_session(
        env, interp, agent, learn, verbose, args.step_by_step, display,
        record=frames,
    )
    if learn:
        agent.decay_epsilon()
    print("Game over, max length = {}, max duration = {}"
          .format(length, duration))
    save_recording(args.record, frames, env.size)
    print("Partie enregistree dans {} ({} frames)"
          .format(args.record, len(frames)))
    _save_model(agent, args.save)


def _run_replay(args):
    """Rejoue un enregistrement -record : pure lecture, sans agent ni RNG."""
    display = _make_display(args.visual == "on")
    try:
        played = replay_recording(args.replay, display=display,
                                  step_by_step=args.step_by_step)
    finally:
        if display is not None:
            display.close()
    print("Replay termine, {} frames rejouees".format(played))


def _print_records(best_length):
    """Affiche les paliers de longueur (15/20/25/30/35) atteints ou non."""
    marks = " ".join(
        "{} [{}]".format(
            milestone, "OK" if best_length >= milestone else "--")
        for milestone in LENGTH_MILESTONES
    )
    print("Records atteints : {}".format(marks))


def _print_benchmark(lengths, durations):
    """Affiche les stats agregees (mean/min/max) du mode `-benchmark`."""
    if not lengths:
        return
    print("Benchmark ({} sessions) :".format(len(lengths)))
    print("  Longueur - mean = {:.2f}, min = {}, max = {}"
          .format(sum(lengths) / len(lengths), min(lengths), max(lengths)))
    print("  Duree    - mean = {:.2f}, min = {}, max = {}"
          .format(sum(durations) / len(durations), min(durations),
                  max(durations)))


def _run_dashboard(agent, interp, learn, args):
    """Lance la vue parallele puis sauvegarde le modele si demande."""
    try:
        from snakeai.ui.dashboard import Dashboard
        board = Dashboard(agent, interp, cols=args.grid, rows=args.grid,
                          learn=learn, save_path=args.save,
                          start_lobby=args.dashboard_lobby)
        board.run()
    except Exception as error:      # pragma: no cover - depend de l'env
        print("Avertissement : dashboard indisponible ({})".format(error),
              file=sys.stderr)
        return
    print("Game over, max length = {}, max duration = {}"
          .format(board.best_length, board.best_duration))
    _save_model(agent, args.save)


def _make_display(enabled):
    """Cree l'affichage pygame si demande ; None sinon ou en cas d'echec."""
    if not enabled:
        return None
    try:
        from snakeai.ui.display import Display
        return Display()
    except Exception as error:      # pragma: no cover - depend de l'env
        print("Avertissement : affichage graphique indisponible ({})"
              .format(error), file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
