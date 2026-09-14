"""Regression : un bug du dashboard n'est plus "dashboard indisponible".

Avant (issue #34), un seul `except Exception` englobait l'import de
pygame, la construction de `Dashboard` et toute la boucle `run()` : le
moindre `IndexError` interne ressortait sous le message "dashboard
indisponible", sans traceback. On verifie ici que les deux modes d'echec
restent bien distincts.
"""

import contextlib
import io
import sys
import types

import pygame

from snakeai import cli


class _Args:
    """Les seuls champs de `argparse.Namespace` lus par `_run_dashboard`."""

    grid = 2
    save = None
    dashboard_lobby = False


def _run_with_dashboard(dashboard_cls):
    """Joue `_run_dashboard` avec un faux module dashboard ; rend (out, err).

    Le vrai module ouvrirait une fenetre pygame : on le remplace le temps
    de l'appel par un module jetable exposant la classe voulue.
    """
    module = types.ModuleType("snakeai.ui.dashboard")
    module.Dashboard = dashboard_cls
    saved = sys.modules.get("snakeai.ui.dashboard")
    sys.modules["snakeai.ui.dashboard"] = module
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli._run_dashboard(None, None, True, _Args(), 10)
    finally:
        if saved is None:
            del sys.modules["snakeai.ui.dashboard"]
        else:
            sys.modules["snakeai.ui.dashboard"] = saved
    return out.getvalue(), err.getvalue()


class _Boom:
    """Dashboard qui se construit puis casse en pleine boucle."""

    best_length = 7
    best_duration = 42

    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        raise IndexError("board_at")


class _NoWindow:
    """Dashboard dont la fenetre ne peut pas s'ouvrir (pygame indispo)."""

    def __init__(self, *args, **kwargs):
        raise pygame.error("No available video device")


def test_internal_error_reports_traceback():
    out, err = _run_with_dashboard(_Boom)
    assert "indisponible" not in err
    assert "Erreur interne du dashboard" in err
    assert "IndexError: board_at" in err
    # Le bilan est tout de meme affiche : l'apprentissage n'est pas perdu.
    assert "max length = 7" in out


def test_window_failure_reports_unavailable():
    out, err = _run_with_dashboard(_NoWindow)
    assert "dashboard indisponible" in err
    assert "No available video device" in err
    assert out == ""


if __name__ == "__main__":
    test_internal_error_reports_traceback()
    test_window_failure_reports_unavailable()
    print("ok")
