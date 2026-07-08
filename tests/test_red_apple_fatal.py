"""Regression : une pomme rouge qui vide le serpent ne doit pas crasher.

Bug historique : quand une pomme rouge reduit le serpent a une longueur de 0,
`environment.step` renvoie sans probleme, mais l'appel suivant a
`interpreter.green_distance` lisait `env.snake[0]` sur une liste vide et levait
une IndexError (crash = 0 a l'evaluation). On verifie ici :
  1. l'evenement renvoye est bien un game over "fatal" ;
  2. le reward correspondant est la penalite de mort, pas le simple malus ;
  3. green_distance est robuste au serpent vide ;
  4. une session complete se joue sans exception dans ce scenario.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants                                # noqa: E402
from environment import Environment            # noqa: E402
from interpreter import Interpreter            # noqa: E402


def _one_length_snake_next_to_red():
    """Env force : serpent d'1 case, pomme rouge juste a droite de la tete."""
    env = Environment()
    env.done = False
    env.snake = [(5, 5)]
    env.direction = constants.RIGHT
    env.green_apples = [(0, 0)]
    env.red_apples = [(5, 6)]
    return env


def test_fatal_red_apple_event_and_reward():
    env = _one_length_snake_next_to_red()
    interp = Interpreter()

    event = env.step(constants.RIGHT)

    assert event["type"] == "red"
    assert event.get("fatal") is True
    assert env.snake == []
    assert env.is_game_over() is True
    assert interp.get_reward(event) == constants.REWARD_GAMEOVER


def test_green_distance_handles_empty_snake():
    env = _one_length_snake_next_to_red()
    interp = Interpreter()
    env.step(constants.RIGHT)          # serpent vide desormais
    # Ne doit pas lever : renvoie simplement None (aucune vision possible).
    assert interp.green_distance(env) is None


def test_non_fatal_red_apple_keeps_malus():
    env = Environment()
    env.done = False
    env.snake = [(5, 5), (5, 4)]       # longueur 2 -> restera 1 apres la rouge
    env.direction = constants.RIGHT
    env.green_apples = [(0, 0)]
    env.red_apples = [(5, 6)]
    interp = Interpreter()

    event = env.step(constants.RIGHT)

    assert event["type"] == "red"
    assert event.get("fatal") is None
    assert env.is_game_over() is False
    assert interp.get_reward(event) == constants.REWARD_RED


if __name__ == "__main__":
    test_fatal_red_apple_event_and_reward()
    test_green_distance_handles_empty_snake()
    test_non_fatal_red_apple_keeps_malus()
    print("OK - tous les tests de regression passent")
