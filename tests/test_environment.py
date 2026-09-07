"""Tests unitaires pour `core/environment.py`.

Meme style que `test_red_apple_fatal.py` : instanciation directe de
`Environment`, etat force a la main (snake/pommes/direction) plutot que
placement aleatoire, pour rendre chaque scenario deterministe.
"""

from snakeai import constants
from snakeai.core import Environment


def _bare_env():
    """Environment() avec snake/pommes vides, pret a etre force a la main."""
    env = Environment()
    env.done = False
    env.snake = []
    env.green_apples = []
    env.red_apples = []
    return env


# -- reset() ------------------------------------------------------------

def test_reset_places_contiguous_in_bounds_snake():
    env = Environment()
    assert len(env.snake) == constants.SNAKE_START_LENGTH
    assert all(env.in_bounds(cell) for cell in env.snake)
    assert len(set(env.snake)) == constants.SNAKE_START_LENGTH
    for (r1, c1), (r2, c2) in zip(env.snake, env.snake[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_reset_spawns_expected_apple_counts_on_distinct_cells():
    env = Environment()
    assert len(env.green_apples) == constants.GREEN_APPLES
    assert len(env.red_apples) == constants.RED_APPLES
    all_cells = env.snake + env.green_apples + env.red_apples
    assert len(set(all_cells)) == len(all_cells)


# -- step() : mur ---------------------------------------------------------

def test_step_into_wall_ends_game():
    env = _bare_env()
    env.snake = [(0, 5), (1, 5), (2, 5)]
    env.direction = constants.UP
    event = env.step(constants.UP)

    assert event["type"] == "wall"
    assert env.done is True
    assert env.is_game_over() is True


# -- step() : pomme verte --------------------------------------------------

def test_step_onto_green_apple_grows_snake_and_respawns():
    env = _bare_env()
    env.snake = [(5, 5), (5, 4), (5, 3)]
    env.direction = constants.RIGHT
    # Le nombre de pommes vertes en jeu est normalement GREEN_APPLES ; on en
    # place une a portee et le reste ailleurs, comme en jeu normal.
    env.green_apples = [(5, 6), (0, 0)]
    env.red_apples = [(0, 1)]

    event = env.step(constants.RIGHT)

    assert event["type"] == "green"
    assert env.done is False
    assert len(env.snake) == 4
    assert env.snake[0] == (5, 6)
    assert len(env.green_apples) == constants.GREEN_APPLES
    assert (5, 6) not in env.green_apples


# -- step() : collision avec le corps --------------------------------------

def test_step_into_own_body_ends_game():
    env = _bare_env()
    # Serpent en carre : avancer vers la gauche ramene la tete sur le
    # 2e segment du corps (pas la queue, qui elle se libere).
    env.snake = [(5, 5), (5, 4), (4, 4), (4, 5)]
    env.direction = constants.LEFT
    event = env.step(constants.LEFT)

    assert event["type"] == "collision"
    assert env.done is True
    assert env.is_game_over() is True


# -- step() : deplacement normal -------------------------------------------

def test_step_onto_empty_cell_keeps_length_and_pops_tail():
    env = _bare_env()
    env.snake = [(5, 5), (5, 4), (5, 3)]
    env.direction = constants.RIGHT
    env.green_apples = [(0, 0)]
    env.red_apples = [(0, 1)]

    event = env.step(constants.RIGHT)

    assert event["type"] == "nothing"
    assert env.done is False
    assert len(env.snake) == 3
    assert env.snake[0] == (5, 6)
    assert (5, 3) not in env.snake


# -- in_bounds() ------------------------------------------------------------

def test_in_bounds_corners_and_just_outside():
    env = Environment()
    size = env.size

    assert env.in_bounds((0, 0)) is True
    assert env.in_bounds((0, size - 1)) is True
    assert env.in_bounds((size - 1, 0)) is True
    assert env.in_bounds((size - 1, size - 1)) is True

    assert env.in_bounds((-1, 0)) is False
    assert env.in_bounds((0, -1)) is False
    assert env.in_bounds((size, 0)) is False
    assert env.in_bounds((0, size)) is False


# -- get_board() --------------------------------------------------------

def test_get_board_places_characters_correctly():
    env = _bare_env()
    env.snake = [(1, 1), (1, 0), (2, 0)]
    env.green_apples = [(0, 0)]
    env.red_apples = [(3, 3)]

    grid = env.get_board()

    assert grid[1][1] == constants.CELL_HEAD
    assert grid[1][0] == constants.CELL_BODY
    assert grid[2][0] == constants.CELL_BODY
    assert grid[0][0] == constants.CELL_GREEN
    assert grid[3][3] == constants.CELL_RED
    assert grid[0][1] == constants.CELL_EMPTY


if __name__ == "__main__":
    test_reset_places_contiguous_in_bounds_snake()
    test_reset_spawns_expected_apple_counts_on_distinct_cells()
    test_step_into_wall_ends_game()
    test_step_onto_green_apple_grows_snake_and_respawns()
    test_step_into_own_body_ends_game()
    test_step_onto_empty_cell_keeps_length_and_pops_tail()
    test_in_bounds_corners_and_just_outside()
    test_get_board_places_characters_correctly()
    print("OK - tous les tests d'environment passent")
