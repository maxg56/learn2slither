"""Regression : un board plus petit que le serpent initial ne doit pas
faire tourner `_place_snake` en boucle infinie.

Bug historique : `Environment.__init__` ne verifiait pas que `size` etait
suffisant pour placer un serpent de `constants.SNAKE_START_LENGTH` cellules
alignees. Avec un board trop petit, aucun placement n'est jamais possible
dans les limites et la boucle `while True` de `_place_snake` tourne pour
toujours (freeze silencieux). On verifie ici :
  1. `Environment(size=1)` leve bien une ValueError explicite ;
  2. la taille par defaut reste fonctionnelle.
"""

from snakeai import constants
from snakeai.core import Environment


def test_size_below_snake_length_raises_value_error():
    try:
        Environment(size=1)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_default_size_still_works():
    env = Environment()
    assert env.size == constants.BOARD_SIZE
    assert len(env.snake) == constants.SNAKE_START_LENGTH


if __name__ == "__main__":
    test_size_below_snake_length_raises_value_error()
    test_default_size_still_works()
    print("OK - tous les tests de regression passent")
