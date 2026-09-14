"""Tests de l'agent Q-learning par reseau de neurones (NNAgent).

Verifie l'interface partagee avec Agent (Q-table) : choose_action renvoie
une action valide, update() effectue bien un pas de gradient (poids
modifies), et un cycle save/load restitue le meme comportement en
exploitation pure (epsilon=0).
"""

import json

from snakeai import constants
from snakeai.learning.nn_agent import (
    HIDDEN_UNITS, INPUT_SIZE, OUTPUT_SIZE, NNAgent,
)

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


def test_seed_makes_initial_weights_reproducible():
    """Regression : `-seed` ne grainait que `random`, pas numpy.

    Les poids initiaux venaient de `np.random.default_rng()` sans graine :
    deux runs `-model nn -seed 42` partaient de reseaux differents.
    """
    first = NNAgent(seed=42)
    second = NNAgent(seed=42)
    other = NNAgent(seed=7)

    assert (first.w1 == second.w1).all()
    assert (first.w2 == second.w2).all()
    assert not (first.w1 == other.w1).all()


def test_load_missing_file_returns_false():
    agent = NNAgent()
    assert agent.load("/chemin/inexistant/modele.json") is False


def _valid_payload():
    """Modele JSON bien forme, base pour les variantes corrompues."""
    return {
        "alpha": 0.1, "gamma": 0.9, "epsilon": 0.0,
        "w1": [[0.0] * HIDDEN_UNITS for _ in range(INPUT_SIZE)],
        "b1": [0.0] * HIDDEN_UNITS,
        "w2": [[0.0] * OUTPUT_SIZE for _ in range(HIDDEN_UNITS)],
        "b2": [0.0] * OUTPUT_SIZE,
    }


def _write(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


def test_load_rejects_corrupted_payloads(tmp_path):
    """Regression : un JSON bien forme mais incoherent etait accepte.

    `load()` ecrivait directement dans `self` sans verifier les dimensions
    des poids : le modele passait, puis le premier forward levait une
    ValueError (`matmul: ... size 8 is different from 12`).
    """
    bad_w1 = _valid_payload()
    bad_w1["w1"] = [[0.0] * HIDDEN_UNITS for _ in range(INPUT_SIZE - 4)]

    bad_b1 = _valid_payload()
    bad_b1["b1"] = [0.0] * (HIDDEN_UNITS + 1)

    bad_w2 = _valid_payload()
    bad_w2["w2"] = [[0.0] * (OUTPUT_SIZE + 1) for _ in range(HIDDEN_UNITS)]

    bad_b2 = _valid_payload()
    bad_b2["b2"] = 0.0

    bad_alpha = _valid_payload()
    bad_alpha["alpha"] = "vite"

    missing_key = _valid_payload()
    del missing_key["w2"]

    payloads = {
        "bad_w1": bad_w1, "bad_b1": bad_b1, "bad_w2": bad_w2,
        "bad_b2": bad_b2, "bad_alpha": bad_alpha,
        "missing_key": missing_key,
    }
    for name, payload in payloads.items():
        agent = NNAgent()
        path = _write(str(tmp_path / (name + ".json")), payload)
        assert agent.load(path) is False, name
        # L'agent reste utilisable : aucun forward ne doit crasher.
        assert agent.choose_action(STATE) in constants.ACTIONS


def test_failed_load_leaves_state_untouched(tmp_path):
    agent = NNAgent()
    agent.alpha = 0.42
    w1_before = agent.w1.copy()

    bad = _valid_payload()
    bad["alpha"] = 0.01
    bad["w1"] = [[0.0] * HIDDEN_UNITS for _ in range(INPUT_SIZE - 4)]
    assert agent.load(_write(str(tmp_path / "bad.json"), bad)) is False

    assert agent.alpha == 0.42
    assert (agent.w1 == w1_before).all()


def test_load_accepts_valid_payload(tmp_path):
    agent = NNAgent()
    path = _write(str(tmp_path / "ok.json"), _valid_payload())
    assert agent.load(path) is True
    assert agent.w1.shape == (INPUT_SIZE, HIDDEN_UNITS)
    assert agent.choose_action(STATE) in constants.ACTIONS


if __name__ == "__main__":
    test_choose_action_returns_valid_action()
    test_update_changes_weights_without_raising()
    test_update_handles_terminal_transition()
    test_load_missing_file_returns_false()
    print("OK - tests NNAgent (hors roundtrip qui necessite tmp_path)")
