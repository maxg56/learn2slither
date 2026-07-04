"""main.py - point d'entree : parsing CLI et boucle d'entrainement.

Assemble le flux Environment -> Interpreter -> Agent -> Environment,
gere les flags de la ligne de commande et affiche le bilan des sessions.
"""

import argparse
import sys

import constants
from agent import Agent
from environment import Environment
from interpreter import Interpreter


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
    return parser.parse_args(argv)


def run_session(env, interp, agent, learn, verbose, step_by_step,
                display=None):
    """Joue une partie complete et retourne (longueur_max, duree)."""
    env.reset()
    state = interp.get_state(env)
    max_length = len(env.snake)
    duration = 0

    while not env.is_game_over():
        if verbose:
            print(interp.render_vision(env))
        if display is not None:
            display.render(env)
            if display.should_quit():
                break
        action = agent.choose_action(state)
        if verbose:
            print("Action:", constants.ACTION_NAMES[action])
            print()
        if step_by_step:
            if display is not None:
                display.wait_step()
            else:
                _wait_step()

        event = env.step(action)
        reward = interp.get_reward(event)
        done = env.is_game_over()
        next_state = None if done else interp.get_state(env)
        if learn:
            agent.update(state, action, reward, next_state)
        if next_state is not None:
            state = next_state

        duration += 1
        max_length = max(max_length, len(env.snake))

    return max_length, duration


def _wait_step():
    """Attend l'appui sur Entree en mode pas a pas (tolerant a l'EOF)."""
    try:
        input("[Entree pour l'action suivante] ")
    except EOFError:
        pass


def main():
    """Point d'entree du programme."""
    args = parse_args()
    verbose = args.visual == "on" or args.step_by_step

    agent = Agent()
    if args.load:
        if agent.load(args.load):
            print("Modele charge depuis {}".format(args.load))
        else:
            print("Avertissement : echec du chargement de {}"
                  .format(args.load), file=sys.stderr)

    learn = not args.dontlearn
    if args.dontlearn:
        agent.epsilon = 0.0

    display = _make_display(args.visual == "on")

    env = Environment()
    interp = Interpreter()
    best_length = 0
    best_duration = 0
    for session in range(1, args.sessions + 1):
        length, duration = run_session(
            env, interp, agent,
            learn, verbose, args.step_by_step, display,
        )
        if learn:
            agent.decay_epsilon()
        best_length = max(best_length, length)
        best_duration = max(best_duration, duration)
        print("Session {}/{} - Game over, max length = {}, max duration = {}"
              .format(session, args.sessions, length, duration))
        if display is not None and display.should_quit():
            break

    if display is not None:
        display.close()

    print("Game over, max length = {}, max duration = {}"
          .format(best_length, best_duration))

    if args.save:
        if agent.save(args.save):
            print("Modele sauvegarde dans {}".format(args.save))
        else:
            print("Avertissement : echec de la sauvegarde dans {}"
                  .format(args.save), file=sys.stderr)


def _make_display(enabled):
    """Cree l'affichage pygame si demande ; None sinon ou en cas d'echec."""
    if not enabled:
        return None
    try:
        from display import Display
        return Display()
    except Exception as error:      # pragma: no cover - depend de l'env
        print("Avertissement : affichage graphique indisponible ({})"
              .format(error), file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
