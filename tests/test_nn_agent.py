"""Tests de l'agent Q-learning par reseau de neurones (NNAgent).

Verifie l'interface partagee avec Agent (Q-table) : choose_action renvoie
une action valide, update() effectue bien un pas de gradient (poids
modifies), et un cycle save/load restitue le meme comportement en
exploitation pure (epsilon=0).
"""

import argparse
import io
import json
import random
from contextlib import redirect_stdout

import numpy as np

from snakeai import constants
from snakeai.core import Environment
from snakeai.learning.nn_agent import (
    HIDDEN_UNITS, INPUT_SIZE, OUTPUT_SIZE, TARGET_SYNC_STEPS, NNAgent,
)
from snakeai.perception import Interpreter
from snakeai.training.trainer import train

STATE = (0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0)
NEXT_STATE = (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0)


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
        "learning_rate": 0.05, "gamma": 0.9, "epsilon": 0.0,
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
    ValueError (`matmul: ... size 12 is different from 16`).
    """
    bad_w1 = _valid_payload()
    bad_w1["w1"] = [[0.0] * HIDDEN_UNITS for _ in range(INPUT_SIZE - 4)]

    bad_b1 = _valid_payload()
    bad_b1["b1"] = [0.0] * (HIDDEN_UNITS + 1)

    bad_w2 = _valid_payload()
    bad_w2["w2"] = [[0.0] * (OUTPUT_SIZE + 1) for _ in range(HIDDEN_UNITS)]

    bad_b2 = _valid_payload()
    bad_b2["b2"] = 0.0

    bad_lr = _valid_payload()
    bad_lr["learning_rate"] = "vite"

    missing_key = _valid_payload()
    del missing_key["w2"]

    payloads = {
        "bad_w1": bad_w1, "bad_b1": bad_b1, "bad_w2": bad_w2,
        "bad_b2": bad_b2, "bad_lr": bad_lr,
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
    agent.learning_rate = 0.42
    w1_before = agent.w1.copy()

    bad = _valid_payload()
    bad["learning_rate"] = 0.01
    bad["w1"] = [[0.0] * HIDDEN_UNITS for _ in range(INPUT_SIZE - 4)]
    assert agent.load(_write(str(tmp_path / "bad.json"), bad)) is False

    assert agent.learning_rate == 0.42
    assert (agent.w1 == w1_before).all()


def test_load_accepts_valid_payload(tmp_path):
    agent = NNAgent()
    path = _write(str(tmp_path / "ok.json"), _valid_payload())
    assert agent.load(path) is True
    assert agent.w1.shape == (INPUT_SIZE, HIDDEN_UNITS)
    assert agent.choose_action(STATE) in constants.ACTIONS


def test_epsilon_decay_is_configurable_and_persisted(tmp_path):
    """Regression (#35) : NNAgent ignorait `epsilon_decay`.

    Le decay etait lu en dur dans constants et absent de save/load : un
    NNAgent n'etait pas substituable a Agent (tune.py ne pouvait pas le
    balayer) et un aller-retour save/load perdait le reglage.
    """
    agent = NNAgent(epsilon=1.0, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == 0.5

    path = str(tmp_path / "nn_model.json")
    assert agent.save(path) is True
    with open(path) as handle:
        assert json.load(handle)["epsilon_decay"] == 0.5

    reloaded = NNAgent()
    assert reloaded.load(path) is True
    assert reloaded.epsilon_decay == 0.5


def test_load_rejects_non_numeric_epsilon_decay(tmp_path):
    payload = _valid_payload()
    payload["epsilon_decay"] = "vite"
    path = tmp_path / "nn_model.json"
    path.write_text(json.dumps(payload))
    agent = NNAgent()
    assert agent.load(str(path)) is False
    assert agent.epsilon_decay == constants.EPSILON_DECAY


def test_target_network_is_synced_periodically():
    """Le reseau cible reste fige entre deux synchronisations.

    Sans cette stabilite, la cible `r + gamma * max Q(s')` derive a chaque
    pas et l'apprentissage ne converge pas (issue #56).
    """
    agent = NNAgent(seed=0)
    target_before = [w.copy() for w in agent._target]
    for _ in range(TARGET_SYNC_STEPS - 1):
        agent.update(STATE, constants.UP, constants.REWARD_NOTHING,
                     NEXT_STATE)
    assert all((a == b).all() for a, b in zip(agent._target, target_before))
    assert not (agent.w1 == target_before[0]).all()

    agent.update(STATE, constants.UP, constants.REWARD_NOTHING, NEXT_STATE)
    assert (agent._target[0] == agent.w1).all()
    assert (agent._target[2] == agent.w2).all()


def test_load_syncs_target_with_loaded_weights(tmp_path):
    trained = NNAgent(seed=0)
    for _ in range(3):
        trained.update(STATE, constants.RIGHT, constants.REWARD_GREEN,
                       NEXT_STATE)
    path = str(tmp_path / "nn_model.json")
    assert trained.save(path) is True

    reloaded = NNAgent(seed=1)
    assert reloaded.load(path) is True
    assert (reloaded._target[0] == reloaded.w1).all()
    assert (reloaded._target[2] == reloaded.w2).all()


def _mean_length(agent, sessions, learn):
    args = argparse.Namespace(
        sessions=sessions, visual="off", dontlearn=not learn,
        step_by_step=False, benchmark=True,
    )
    with redirect_stdout(io.StringIO()):
        _, _, lengths, _ = train(Environment(), Interpreter(), agent, args)
    return float(np.mean(lengths))


def test_nn_agent_learns_better_than_random():
    """Non-regression (#56) : `-model nn` doit reellement apprendre.

    Avant la correction, le reseau restait au niveau d'un agent aleatoire
    (longueur moyenne ~3) meme apres 500 sessions : learning rate confondu
    avec l'alpha de la Q-table, rewards bruts non normalises et SGD en
    ligne sans replay buffer ni reseau cible. On entraine quelques
    centaines de sessions avec une graine fixe, puis on compare en
    exploitation pure (epsilon=0) a une politique uniforme (epsilon=1).
    """
    random.seed(1)
    agent = NNAgent(seed=1)
    _mean_length(agent, sessions=400, learn=True)
    agent.epsilon = 0.0
    trained_mean = _mean_length(agent, sessions=30, learn=False)

    uniform = NNAgent(seed=1, epsilon=1.0)
    random_mean = _mean_length(uniform, sessions=30, learn=False)

    assert trained_mean > random_mean + 3, (trained_mean, random_mean)
