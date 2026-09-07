"""training - boucle d'entrainement assemblant le flux du jeu."""

from snakeai.training.trainer import run_session, train
from snakeai.training.replay import load_recording, replay, save_recording

__all__ = [
    "run_session", "train",
    "load_recording", "replay", "save_recording",
]
