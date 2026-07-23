"""trainer.py - la boucle d'entrainement, assemblage du flux du jeu.

Orchestre le flux impose : Environment -> Interpreter (state + reward) ->
Agent (action) -> Environment. Ne contient ni regles du jeu, ni logique de
Q-learning : il ne fait que les faire dialoguer, session apres session.
"""

from collections import deque

from snakeai import constants
from snakeai.training.stats import StatsHistory


def run_session(env, interp, agent, learn, verbose, step_by_step,
                display=None):
    """Joue une partie complete et retourne (longueur_max, duree)."""
    env.reset()
    state = interp.get_state(env)
    max_length = len(env.snake)
    duration = 0
    # Anti-blocage : borne le nombre de pas sans manger pour eviter qu'un
    # modele en exploitation pure ne tourne en rond indefiniment.
    stall = 0
    stall_limit = env.size * env.size * constants.STALL_STEPS_FACTOR

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

        dist_before = interp.green_distance(env)
        event = env.step(action)
        reward = interp.get_reward(event)
        # dist_after n'a de sens que sur un deplacement simple : apres avoir
        # mange/perdu, le serpent peut etre vide ou la pomme a bouge.
        dist_after = (interp.green_distance(env)
                      if event["type"] == "nothing" else None)
        reward += interp.approach_bonus(dist_before, dist_after, event["type"])
        stall = 0 if event["type"] == "green" else stall + 1
        done = env.is_game_over() or stall >= stall_limit
        next_state = None if done else interp.get_state(env)
        if learn:
            agent.update(state, action, reward, next_state)
        if next_state is not None:
            state = next_state

        duration += 1
        max_length = max(max_length, len(env.snake))
        if done:
            break

    return max_length, duration


def _wait_step():
    """Attend l'appui sur Entree en mode pas a pas (tolerant a l'EOF)."""
    try:
        input("[Entree pour l'action suivante] ")
    except EOFError:
        pass


def train(env, interp, agent, args, display=None):
    """Enchaine les sessions et retourne (best_len, best_dur, history).

    Applique le decay d'epsilon apres chaque session (sauf en `-dontlearn`),
    imprime le bilan de chaque partie et s'arrete si l'affichage est ferme.
    `history` est un `StatsHistory` echantillonne (une mesure par session) que
    la CLI sauvegarde en CSV a cote du modele pour la visualisation ulterieure.
    """
    learn = not args.dontlearn
    verbose = args.visual == "on" or args.step_by_step
    best_length = 0
    best_duration = 0
    history = StatsHistory()
    recent_lengths = deque(maxlen=100)
    recent_durations = deque(maxlen=100)

    for session in range(1, args.sessions + 1):
        length, duration = run_session(
            env, interp, agent,
            learn, verbose, args.step_by_step, display,
        )
        if learn:
            agent.decay_epsilon()
        best_length = max(best_length, length)
        best_duration = max(best_duration, duration)
        recent_lengths.append(length)
        recent_durations.append(duration)
        history.record(
            session,
            avg_length=sum(recent_lengths) / len(recent_lengths),
            max_length=best_length,
            avg_duration=sum(recent_durations) / len(recent_durations),
            epsilon=agent.epsilon,
        )
        print("Session {}/{} - Game over, max length = {}, max duration = {}"
              .format(session, args.sessions, length, duration))
        if display is not None and display.should_quit():
            break

    return best_length, best_duration, history
