"""replay.py - enregistrement et lecture d'une partie deja jouee.

Complement de `trainer.py` pour le bonus replay : une partie enregistree
par `run_session(..., record=frames)` est une simple liste de frames
JSON-serialisables (serpent, pommes, action choisie a chaque pas). Rejouer
cette liste ne necessite ni Agent, ni Interpreter.get_state, ni RNG : c'est
une pure lecture de donnees, reutilisant le rendu terminal (vision) et
l'affichage pygame deja utilises pendant l'entrainement.
"""

import json
import time

from snakeai import constants
from snakeai.core import Environment
from snakeai.perception import Interpreter

REPLAY_PAUSE_SECONDS = 0.2


def save_recording(path, frames, size):
    """Ecrit les frames enregistrees et la taille du board en JSON."""
    with open(path, "w") as handle:
        json.dump({"size": size, "frames": frames}, handle)


def load_recording(path):
    """Charge un enregistrement JSON. Retourne (frames, size)."""
    with open(path) as handle:
        data = json.load(handle)
    return data["frames"], data.get("size", constants.BOARD_SIZE)


def _wait_step(display):
    """Attend l'action suivante : touche (pygame) ou Entree (terminal)."""
    if display is not None:
        display.wait_step()
        return
    try:
        input("[Entree pour la frame suivante] ")
    except EOFError:
        pass


def replay(path, display=None, step_by_step=False, pause=REPLAY_PAUSE_SECONDS):
    """Rejoue un enregistrement frame par frame, sans agent ni RNG.

    Reconstruit a chaque frame un `Environment` jetable dont le serpent et
    les pommes sont fixes directement depuis les donnees enregistrees (meme
    principe que dans `tests/test_red_apple_fatal.py`), puis reutilise le
    rendu terminal existant (`Interpreter.render_vision`) et, si demande,
    l'affichage pygame (`Display.render` / `Display.wait_step`).

    Retourne le nombre de frames rejouees.
    """
    frames, size = load_recording(path)
    interp = Interpreter()
    env = Environment(size=size)

    played = 0
    for frame in frames:
        env.snake = [tuple(cell) for cell in frame["snake"]]
        env.green_apples = [tuple(cell) for cell in frame["green_apples"]]
        env.red_apples = [tuple(cell) for cell in frame["red_apples"]]
        env.done = False

        print(interp.render_vision(env))
        if display is not None:
            display.render(env)
            if display.should_quit():
                break
        print("Action:", constants.ACTION_NAMES[frame["action"]])
        print()
        played += 1

        if step_by_step:
            _wait_step(display)
        elif pause:
            time.sleep(pause)

    return played
