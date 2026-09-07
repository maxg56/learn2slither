"""cli.py - point d'entree : parsing CLI et cablage des composants.

Ne contient aucune logique de jeu, d'apprentissage ni de boucle : il se
contente d'interpreter les flags, d'assembler Agent/Interpreter/Environment
et de deleguer soit au trainer, soit au dashboard.
"""

import argparse
import sys

from snakeai.core import Environment
from snakeai.learning import Agent, NNAgent
from snakeai.perception import Interpreter
from snakeai.training import train


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
    parser.add_argument("-model", choices=["qtable", "nn"], default="qtable",
                        help="fonction Q utilisee : Q-table ou reseau NN")
    return parser.parse_args(argv)


def main():
    """Point d'entree du programme."""
    args = parse_args()

    agent = _build_agent(args)
    learn = not args.dontlearn
    interp = Interpreter()

    if args.dashboard:
        _run_dashboard(agent, interp, learn, args)
        return

    display = _make_display(args.visual == "on")
    env = Environment()
    best_length, best_duration = train(env, interp, agent, args, display)
    if display is not None:
        display.close()

    print("Game over, max length = {}, max duration = {}"
          .format(best_length, best_duration))
    _save_model(agent, args.save)


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


def _run_dashboard(agent, interp, learn, args):
    """Lance la vue parallele puis sauvegarde le modele si demande."""
    try:
        from snakeai.ui.dashboard import Dashboard
        board = Dashboard(agent, interp, cols=args.grid, rows=args.grid,
                          learn=learn, save_path=args.save)
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
