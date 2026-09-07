"""trainer.py - la boucle d'entrainement, assemblage du flux du jeu.

Orchestre le flux impose : Environment -> Interpreter (state + reward) ->
Agent (action) -> Environment. Ne contient ni regles du jeu, ni logique de
Q-learning : il ne fait que les faire dialoguer, session apres session.
"""

from snakeai import constants


def run_session(env, interp, agent, learn, verbose, step_by_step,
                display=None, record=None):
    """Joue une partie complete et retourne (longueur_max, duree).

    Si `record` est une liste, une entree JSON-serialisable est ajoutee
    avant chaque action (avant que l'environnement n'avance), pour permettre
    un replay exact de la partie sans dependre de l'agent ni du RNG.
    """
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
        if record is not None:
            record.append({
                "snake": [list(cell) for cell in env.snake],
                "green_apples": [list(cell) for cell in env.green_apples],
                "red_apples": [list(cell) for cell in env.red_apples],
                "action": action,
            })
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
    """Enchaine les sessions d'entrainement et retourne (best_len, best_dur).

    Applique le decay d'epsilon apres chaque session (sauf en `-dontlearn`),
    imprime le bilan de chaque partie et s'arrete si l'affichage est ferme.
    """
    learn = not args.dontlearn
    verbose = args.visual == "on" or args.step_by_step
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

    return best_length, best_duration
