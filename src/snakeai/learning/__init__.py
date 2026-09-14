"""learning - Q-learning : Q-table, politique epsilon-greedy, update.

``NNAgent`` depend de numpy : il est expose paresseusement pour que le
mode Q-table par defaut fonctionne sans cette dependance.
"""

from snakeai.learning.agent import Agent

__all__ = ["Agent", "NNAgent"]


def __getattr__(name):
    if name == "NNAgent":
        from snakeai.learning.nn_agent import NNAgent
        return NNAgent
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name))
