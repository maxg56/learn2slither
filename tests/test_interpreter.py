"""Tests unitaires pour `perception/interpreter.py`.

Meme style que `test_red_apple_fatal.py` : `Environment` instancie puis
force a la main (snake/pommes/direction), aucun fixture/classe.
"""

from snakeai import constants
from snakeai.core import Environment
from snakeai.perception import Interpreter


def _bare_env():
    """Environment() avec snake/pommes vides, pret a etre force a la main."""
    env = Environment()
    env.done = False
    env.snake = []
    env.green_apples = []
    env.red_apples = []
    return env


# -- get_vision() -----------------------------------------------------------

def test_get_vision_one_ray_per_action_ending_in_wall():
    env = Environment()
    interp = Interpreter()

    vision = interp.get_vision(env)

    assert set(vision.keys()) == set(constants.ACTIONS)
    for action in constants.ACTIONS:
        ray = vision[action]
        assert ray[-1] == constants.CELL_WALL
        assert ray.count(constants.CELL_WALL) == 1


# -- get_state() --------------------------------------------------------

def test_get_state_returns_state_size_binary_features():
    env = Environment()
    interp = Interpreter()

    state = interp.get_state(env)

    assert isinstance(state, tuple)
    assert len(state) == constants.STATE_SIZE == 16
    assert all(v in (0, 1) for v in state)


def test_get_state_reflects_danger_green_red_bits():
    env = _bare_env()
    env.snake = [(5, 5), (5, 4), (5, 3)]
    env.direction = constants.RIGHT
    env.green_apples = [(5, 7)]     # visible a droite, non adjacente
    env.red_apples = [(3, 5)]       # visible en haut, non adjacente
    interp = Interpreter()

    state = interp.get_state(env)

    up_danger, up_green, up_red, up_red_adj = state[0:4]
    right_danger, right_green, right_red, right_red_adj = state[12:16]

    assert up_red == 1
    assert up_red_adj == 0
    assert up_green == 0
    assert right_green == 1
    assert right_red == 0
    assert right_red_adj == 0
    assert up_danger == 0
    assert right_danger == 0


def test_get_state_red_adjacent_bit_distinguishes_glued_red_apple():
    """Regression #33 : une rouge collee a la tete doit etre distinguable
    d'une rouge lointaine sur le meme rayon (piege mortel a longueur 1)."""
    env = _bare_env()
    env.snake = [(5, 5), (5, 4), (5, 3)]
    env.direction = constants.RIGHT
    env.green_apples = []
    interp = Interpreter()

    env.red_apples = [(4, 5)]       # collee en haut
    up_danger, _, up_red, up_red_adj = interp.get_state(env)[0:4]
    assert (up_danger, up_red, up_red_adj) == (0, 1, 1)

    env.red_apples = [(1, 5)]       # meme rayon, loin
    up_danger, _, up_red, up_red_adj = interp.get_state(env)[0:4]
    assert (up_danger, up_red, up_red_adj) == (0, 1, 0)


# -- get_reward() -------------------------------------------------------

def test_get_reward_green():
    interp = Interpreter()
    assert interp.get_reward({"type": "green"}) == constants.REWARD_GREEN


def test_get_reward_red_non_fatal():
    interp = Interpreter()
    assert interp.get_reward({"type": "red"}) == constants.REWARD_RED


def test_get_reward_red_fatal():
    interp = Interpreter()
    event = {"type": "red", "fatal": True}
    assert interp.get_reward(event) == constants.REWARD_GAMEOVER


def test_get_reward_wall_collision_gameover():
    interp = Interpreter()
    for kind in ("wall", "collision", "gameover"):
        assert interp.get_reward({"type": kind}) == constants.REWARD_GAMEOVER


def test_get_reward_nothing():
    interp = Interpreter()
    assert interp.get_reward({"type": "nothing"}) == constants.REWARD_NOTHING


# -- get_reward() mode "alt" : demi-tour explicite ----------------------

def test_alt_uturn_penalty_from_explicit_directions():
    interp = Interpreter(reward_mode="alt")
    base = constants.REWARD_NOTHING + constants.REWARD_SURVIVAL_BONUS
    event = {"type": "nothing"}

    assert interp.get_reward(event, constants.UP, constants.DOWN) == \
        base + constants.REWARD_UTURN
    assert interp.get_reward(event, constants.UP, constants.LEFT) == base
    assert interp.get_reward(event, constants.UP, constants.UP) == base


def test_alt_no_uturn_penalty_without_directions():
    interp = Interpreter(reward_mode="alt")
    base = constants.REWARD_NOTHING + constants.REWARD_SURVIVAL_BONUS

    assert interp.get_reward({"type": "nothing"}) == base
    assert interp.get_reward({"type": "nothing"}, None, constants.UP) == base
    assert interp.get_reward({"type": "nothing"}, constants.UP, None) == base


def test_default_mode_ignores_uturn():
    interp = Interpreter()
    event = {"type": "nothing"}
    assert interp.get_reward(event, constants.UP, constants.DOWN) == \
        constants.REWARD_NOTHING


def test_interpreter_is_stateless_across_envs():
    """Une seule instance partagee par plusieurs envs (dashboard) : le
    reward d'un board ne depend jamais de ce qui a ete calcule sur un
    autre, ni de l'ordre des appels a green_distance()."""
    interp = Interpreter(reward_mode="alt")
    base = constants.REWARD_NOTHING + constants.REWARD_SURVIVAL_BONUS

    env_a = Environment()
    env_b = Environment()
    interp.green_distance(env_a)
    interp.green_distance(env_b)
    assert not hasattr(interp, "_env")
    assert not hasattr(interp, "_prev_direction")

    # Peu importe les appels precedents, seul l'argument compte.
    reward = interp.get_reward({"type": "nothing"},
                               constants.LEFT, constants.RIGHT)
    assert reward == base + constants.REWARD_UTURN


# -- green_distance() ----------------------------------------------------

def test_green_distance_returns_nearest_visible_apple():
    env = _bare_env()
    env.snake = [(5, 5), (5, 4), (5, 3)]
    env.direction = constants.RIGHT
    # Rayon droite : pomme a distance 2. Rayon haut : pomme a distance 3.
    env.green_apples = [(5, 7), (2, 5)]
    env.red_apples = [(0, 0)]
    interp = Interpreter()

    assert interp.green_distance(env) == 2


def test_green_distance_none_when_no_green_visible():
    env = _bare_env()
    env.snake = [(5, 5), (5, 4), (5, 3)]
    env.direction = constants.RIGHT
    env.green_apples = []
    env.red_apples = [(0, 0)]
    interp = Interpreter()

    assert interp.green_distance(env) is None


def test_green_distance_none_when_snake_empty():
    env = _bare_env()
    env.snake = []
    env.green_apples = [(5, 7)]
    interp = Interpreter()

    assert interp.green_distance(env) is None


# -- approach_bonus() -----------------------------------------------------

def test_approach_bonus_positive_when_getting_closer():
    interp = Interpreter()
    bonus = interp.approach_bonus(5, 3, "nothing")
    assert bonus > 0


def test_approach_bonus_negative_when_getting_further():
    interp = Interpreter()
    bonus = interp.approach_bonus(3, 5, "nothing")
    assert bonus < 0


def test_approach_bonus_zero_when_event_not_nothing():
    interp = Interpreter()
    assert interp.approach_bonus(5, 3, "green") == 0.0
    assert interp.approach_bonus(3, 5, "wall") == 0.0


def test_approach_bonus_zero_when_distance_missing():
    interp = Interpreter()
    assert interp.approach_bonus(None, 3, "nothing") == 0.0
    assert interp.approach_bonus(5, None, "nothing") == 0.0
    assert interp.approach_bonus(None, None, "nothing") == 0.0


if __name__ == "__main__":
    test_get_vision_one_ray_per_action_ending_in_wall()
    test_get_state_returns_state_size_binary_features()
    test_get_state_reflects_danger_green_red_bits()
    test_get_reward_green()
    test_get_reward_red_non_fatal()
    test_get_reward_red_fatal()
    test_get_reward_wall_collision_gameover()
    test_get_reward_nothing()
    test_alt_uturn_penalty_from_explicit_directions()
    test_alt_no_uturn_penalty_without_directions()
    test_default_mode_ignores_uturn()
    test_interpreter_is_stateless_across_envs()
    test_green_distance_returns_nearest_visible_apple()
    test_green_distance_none_when_no_green_visible()
    test_green_distance_none_when_snake_empty()
    test_approach_bonus_positive_when_getting_closer()
    test_approach_bonus_negative_when_getting_further()
    test_approach_bonus_zero_when_event_not_nothing()
    test_approach_bonus_zero_when_distance_missing()
    print("OK - tous les tests d'interpreter passent")
