"""Regression : les bornes invalides doivent sortir proprement (issue #23).

Bug historique : `-board-size 1` construisait `Environment(size=1)` a nu,
laissant remonter la ValueError du garde-fou sous forme de traceback ;
`-grid 0` creait un dashboard sans aucune partie (`max()` sur une sequence
vide). `-sessions 0` ou negatif (issue #57) n'executait jamais la boucle
et affichait un bilan a zero comme un resultat. Les trois bornes sont
desormais validees au parsing et sortent via `parser.error()` : message
propre et code de retour 2.
"""

from snakeai import constants
from snakeai.cli import parse_args


def _exit_code(argv):
    """Retourne le code de sortie de `parse_args`, ou None si pas d'exit."""
    try:
        parse_args(argv)
    except SystemExit as error:
        return error.code
    return None


def test_board_size_below_snake_length_exits_with_code_2():
    assert _exit_code(["-board-size", "1"]) == 2


def test_grid_zero_exits_with_code_2():
    assert _exit_code(["-grid", "0"]) == 2


def test_sessions_zero_exits_with_code_2():
    assert _exit_code(["-sessions", "0"]) == 2


def test_sessions_negative_exits_with_code_2():
    assert _exit_code(["-sessions", "-3"]) == 2


def test_valid_bounds_are_accepted():
    args = parse_args(["-board-size", str(constants.SNAKE_START_LENGTH),
                       "-grid", "1", "-sessions", "1"])
    assert args.board_size == constants.SNAKE_START_LENGTH
    assert args.grid == 1
    assert args.sessions == 1


def test_replay_without_sessions_is_accepted():
    # -replay ignore -sessions : sa valeur par defaut doit rester valide.
    args = parse_args(["-replay", "recording.json"])
    assert args.replay == "recording.json"
    assert args.sessions >= 1


def test_default_bounds_are_accepted():
    args = parse_args([])
    assert args.board_size is None
    assert args.grid >= 1


if __name__ == "__main__":
    test_board_size_below_snake_length_exits_with_code_2()
    test_grid_zero_exits_with_code_2()
    test_sessions_zero_exits_with_code_2()
    test_sessions_negative_exits_with_code_2()
    test_valid_bounds_are_accepted()
    test_replay_without_sessions_is_accepted()
    test_default_bounds_are_accepted()
    print("OK - tous les tests de regression passent")
