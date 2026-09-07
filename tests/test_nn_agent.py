"""Tests de l'agent Q-learning par reseau de neurones (NNAgent).

Verifie l'interface partagee avec Agent (Q-table) : choose_action renvoie
une action valide, update() effectue bien un pas de gradient (poids
modifies), et un cycle save/load restitue le meme comportement en
exploitation pure (epsilon=0).
"""

from snakeai import constants
from snakeai.learning.nn_agent import NNAgent

STATE = (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0)
NEXT_STATE = (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)


def test_choose_action_returns_valid_action():
    agent = NNAgent()
    action = agent.choose_action(STATE)
    assert action in constants.ACTIONS


def test_update_changes_weights_without_raising():
    agent = NNAgent()
    w1_before = agent.w1.copy()
    w2_before = agent.w2.copy()
    agent.update(STATE, constants.UP, constants.REWARD_GREEN, NEXT_STATE)
    assert not (agent.w1 == w1_before).all() or \
        not (agent.w2 == w2_before).all()


def test_update_handles_terminal_transition():
    agent = NNAgent()
    # next_state=None (transition terminale) ne doit pas crasher.
    agent.update(STATE, constants.DOWN, constants.REWARD_GAMEOVER, None)


def test_save_load_roundtrip_preserves_behavior(tmp_path):
    agent = NNAgent()
    agent.epsilon = 0.0
    for _ in range(5):
        agent.update(STATE, constants.RIGHT, constants.REWARD_GREEN,
                     NEXT_STATE)
    action_before = agent.choose_action(STATE)

    path = str(tmp_path / "nn_model.json")
    assert agent.save(path) is True

    reloaded = NNAgent()
    assert reloaded.load(path) is True
    assert reloaded.epsilon == 0.0
    action_after = reloaded.choose_action(STATE)

    assert action_after == action_before


def test_load_missing_file_returns_false():
    agent = NNAgent()
    assert agent.load("/chemin/inexistant/modele.json") is False


if __name__ == "__main__":
    test_choose_action_returns_valid_action()
    test_update_changes_weights_without_raising()
    test_update_handles_terminal_transition()
    test_load_missing_file_returns_false()
    print("OK - tests NNAgent (hors roundtrip qui necessite tmp_path)")
