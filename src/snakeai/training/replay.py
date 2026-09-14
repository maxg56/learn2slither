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


class RecordingError(Exception):
    """Enregistrement illisible : fichier absent, JSON invalide ou forme
    incorrecte. Rattrapee dans `cli._run_replay` (jamais de crash)."""


def save_recording(path, frames, size):
    """Ecrit les frames enregistrees et la taille du board en JSON.

    Tolerant aux chemins invalides, comme `Agent.save` : retourne True si
    l'ecriture a reussi, False sinon (jamais de crash).
    """
    try:
        with open(path, "w") as handle:
            json.dump({"size": size, "frames": frames}, handle)
        return True
    except OSError:
        return False


def _validate_cells(frame, key):
    """Convertit `frame[key]` en liste de tuples (x, y) d'entiers."""
    cells = frame.get(key)
    if not isinstance(cells, list):
        raise RecordingError("cle '{}' absente ou invalide".format(key))
    result = []
    for cell in cells:
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2
                or not all(isinstance(v, int) for v in cell)):
            raise RecordingError("cellule invalide dans '{}'".format(key))
        result.append(tuple(cell))
    return result


def _validate_frame(frame):
    """Verifie une frame et la renvoie sous forme normalisee."""
    if not isinstance(frame, dict):
        raise RecordingError("frame invalide (objet attendu)")
    action = frame.get("action")
    if isinstance(action, bool) or action not in constants.ACTION_NAMES:
        raise RecordingError("action inconnue : {!r}".format(action))
    return {
        "snake": _validate_cells(frame, "snake"),
        "green_apples": _validate_cells(frame, "green_apples"),
        "red_apples": _validate_cells(frame, "red_apples"),
        "action": action,
    }


def load_recording(path):
    """Charge et valide un enregistrement JSON. Retourne (frames, size).

    Leve `RecordingError` (message lisible) si le fichier est absent,
    n'est pas du JSON, ou si les cles `frames`, `size` ou `action` sont
    absentes ou mal formees, plutot que de laisser remonter un traceback.
    """
    try:
        with open(path) as handle:
            data = json.load(handle)
    except OSError as error:
        raise RecordingError("lecture impossible ({})".format(error))
    except ValueError as error:
        raise RecordingError("JSON invalide ({})".format(error))

    if not isinstance(data, dict):
        raise RecordingError("enregistrement invalide (objet attendu)")
    size = data.get("size", constants.BOARD_SIZE)
    if (isinstance(size, bool) or not isinstance(size, int)
            or size < constants.SNAKE_START_LENGTH):
        raise RecordingError("taille de board invalide : {!r}".format(size))
    raw_frames = data.get("frames")
    if not isinstance(raw_frames, list):
        raise RecordingError("cle 'frames' absente ou invalide")
    return [_validate_frame(frame) for frame in raw_frames], size


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
        env.snake = frame["snake"]
        env.green_apples = frame["green_apples"]
        env.red_apples = frame["red_apples"]
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
