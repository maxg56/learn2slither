"""Regression : `-board-size` doit atteindre les interfaces (issue #26).

Bug historique : `_make_display` et `_run_dashboard` ne transmettaient
jamais la taille demandee, si bien que `-board-size 20 -visual on` (ou
`-dashboard`) tournait dans une fenetre 10x10 : grille fausse et cellules
dessinees hors fenetre. Le replay souffrait du meme travers, l'affichage
etant cree avant la lecture de la taille enregistree.

Tests headless : SDL est force en mode `dummy`, aucune fenetre n'est
ouverte.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from snakeai import constants                                   # noqa: E402
from snakeai.cli import _make_display, parse_args               # noqa: E402
from snakeai.learning.agent import Agent                        # noqa: E402
from snakeai.perception.interpreter import Interpreter          # noqa: E402
from snakeai.ui.dashboard import Dashboard                      # noqa: E402


def test_display_uses_requested_board_size():
    display = _make_display(True, None, 20)
    assert display is not None
    assert display.size == 20
    side = 20 * display.cell_pixels
    assert display.screen.get_size() == (side, side)


def test_display_defaults_to_project_board_size():
    display = _make_display(True)
    assert display.size == constants.BOARD_SIZE


def test_display_window_stays_bounded_on_large_boards():
    display = _make_display(True, None, 60)
    width, height = display.screen.get_size()
    assert display.cell_pixels >= 1
    assert max(width, height) <= 800


def test_display_resize_follows_recorded_size():
    display = _make_display(True, None, constants.BOARD_SIZE)
    display.resize(25)
    assert display.size == 25
    side = 25 * display.cell_pixels
    assert display.screen.get_size() == (side, side)


def test_dashboard_uses_requested_board_size():
    dash = Dashboard(Agent(), Interpreter(), cols=2, rows=2, board_size=20)
    assert dash.renderer.board_size == 20
    assert all(env.size == 20 for env in dash.sim.envs)


def test_dashboard_window_size_is_independent_of_board_size():
    small = Dashboard(Agent(), Interpreter(), cols=2, rows=2, board_size=10)
    reference = small.renderer.screen.get_size()
    large = Dashboard(Agent(), Interpreter(), cols=2, rows=2, board_size=40)
    assert large.renderer.screen.get_size() == reference


def test_board_size_flag_is_parsed():
    assert parse_args(["-board-size", "20"]).board_size == 20


if __name__ == "__main__":
    test_display_uses_requested_board_size()
    test_display_defaults_to_project_board_size()
    test_display_window_stays_bounded_on_large_boards()
    test_display_resize_follows_recorded_size()
    test_dashboard_uses_requested_board_size()
    test_dashboard_window_size_is_independent_of_board_size()
    test_board_size_flag_is_parsed()
    print("OK - tous les tests de regression passent")
