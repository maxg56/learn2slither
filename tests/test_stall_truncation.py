"""Tests de la coupure anti-blocage (truncation) vs etat terminal.

La coupure anti-blocage arrete la partie apres trop de pas sans manger,
mais l'etat atteint reste parfaitement viable : l'update de Q-learning doit
donc bootstrapper sur le vrai `next_state` et non recevoir `None`, qui
signale un etat terminal (game over) et apprendrait une valeur nulle fausse.

Meme style que les autres tests : fonctions simples, doublures a la main,
executables directement via `python tests/test_stall_truncation.py`.
"""

from snakeai import constants
from snakeai.ui.dashboard.simulation import Simulation


class FakeEnv:
    """Environnement minimal : avance sans jamais manger ni mourir."""

    def __init__(self, size=3):
        self.size = size
        self.snake = [(0, 0), (0, 1), (0, 2)]
        self.direction = constants.UP
        self.green_apples = [(1, 1), (2, 2)]
        self.red_apples = [(2, 0)]
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, action):
        self.direction = action
        return {"type": "nothing"}

    def is_game_over(self):
        return False


class FakeInterp:
    """Interpreter neutre : etat constant, rewards nuls."""

    def get_state(self, env):
        return (0, 0, 0, 0)

    def get_reward(self, event, prev_direction, action):
        return 0.0

    def green_distance(self, env):
        return 1

    def approach_bonus(self, dist_before, dist_after, event_type):
        return 0.0

    def render_vision(self, env):
        return ""


class SpyAgent:
    """Agent qui enregistre les `next_state` recus par `update`."""

    def __init__(self):
        self.epsilon = 0.0
        self.next_states = []

    def choose_action(self, state):
        return constants.UP

    def update(self, state, action, reward, next_state):
        self.next_states.append(next_state)

    def decay_epsilon(self):
        pass


def test_trainer_truncation_bootstraps_on_real_next_state():
    from snakeai.training.trainer import run_session

    env = FakeEnv(size=3)
    agent = SpyAgent()
    limit = env.size * env.size * constants.STALL_STEPS_FACTOR

    max_length, duration, _ = run_session(
        env, FakeInterp(), agent,
        learn=True, verbose=False, step_by_step=False,
    )

    # La partie s'arrete bien sur la coupure anti-blocage...
    assert duration == limit
    assert max_length == 3
    # ... mais aucune transition n'a ete presentee comme terminale.
    assert len(agent.next_states) == limit
    assert all(next_state is not None for next_state in agent.next_states)


def test_simulation_truncation_bootstraps_on_real_next_state():
    agent = SpyAgent()
    interp = FakeInterp()
    sim = Simulation(agent, interp, cols=1, rows=1, board_size=3)
    sim.envs = [FakeEnv(size=3)]
    sim.states = [interp.get_state(sim.envs[0])]

    limit = sim.stall_limit
    for _ in range(limit):
        sim.step_board(0)

    # L'episode a ete cloture par la troncature (board relance)...
    assert sim.episodes == 1
    assert sim.envs[0].resets == 1
    # ... sans jamais bootstrapper a zero sur un etat viable.
    assert len(agent.next_states) == limit
    assert all(next_state is not None for next_state in agent.next_states)


def test_simulation_game_over_still_terminal():
    """Un vrai game over doit, lui, rester une transition terminale."""
    class DyingEnv(FakeEnv):
        def __init__(self):
            FakeEnv.__init__(self, size=3)
            self.dead = False

        def step(self, action):
            self.dead = True
            return {"type": "death"}

        def is_game_over(self):
            return self.dead

    agent = SpyAgent()
    interp = FakeInterp()
    sim = Simulation(agent, interp, cols=1, rows=1, board_size=3)
    sim.envs = [DyingEnv()]
    sim.states = [interp.get_state(sim.envs[0])]

    sim.step_board(0)

    assert agent.next_states == [None]
    assert sim.episodes == 1


if __name__ == "__main__":
    test_trainer_truncation_bootstraps_on_real_next_state()
    test_simulation_truncation_bootstraps_on_real_next_state()
    test_simulation_game_over_still_terminal()
    print("OK - la coupure anti-blocage n'est plus traitee comme terminale")
