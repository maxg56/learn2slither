"""Tests unitaires pour `Agent.load()` (learning/agent.py).

Meme style que `test_red_apple_fatal.py` : pas de fixtures/classes,
instanciation directe de `Agent` et fichiers temporaires geres a la main.
Verifie que `load()` reste tolerant a un fichier absent, non-JSON, ou dont
le schema JSON est valide mais incorrect (q_table de mauvaise forme), sans
jamais corrompre l'etat de l'agent en cas d'echec.
"""

import os
import tempfile

from snakeai.learning import Agent


def _tmp_path():
    """Chemin d'un fichier temporaire qui n'existe pas encore."""
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    os.remove(path)
    return path


def _agent_with_state():
    """Agent avec une q_table connue, non vide, servant d'etat de reference."""
    agent = Agent(alpha=0.1, gamma=0.9, epsilon=0.5)
    agent.q_table = {
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): [1.0, 2.0, 3.0, 4.0],
        (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0): [-1.0, 0.5, 0.0, 9.0],
    }
    return agent


def test_load_missing_path_returns_false():
    agent = _agent_with_state()
    before = dict(agent.q_table)

    ok = agent.load("/tmp/this/path/does/not/exist/model.json")

    assert ok is False
    assert agent.q_table == before


def test_load_non_json_file_returns_false():
    agent = _agent_with_state()
    before = dict(agent.q_table)
    path = _tmp_path()
    try:
        with open(path, "w") as handle:
            handle.write("ceci n'est pas du json {{{")

        ok = agent.load(path)

        assert ok is False
        assert agent.q_table == before
    finally:
        os.remove(path)


def test_load_q_table_value_wrong_type_returns_false_and_keeps_state():
    agent = _agent_with_state()
    before = dict(agent.q_table)
    path = _tmp_path()
    try:
        with open(path, "w") as handle:
            handle.write(
                '{"alpha": 0.1, "gamma": 0.9, "epsilon": 1.0, '
                '"q_table": {"(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)": '
                '"abcd"}}'
            )

        ok = agent.load(path)

        assert ok is False
        assert agent.q_table == before
    finally:
        os.remove(path)


def test_load_q_table_value_wrong_length_returns_false_and_keeps_state():
    agent = _agent_with_state()
    before = dict(agent.q_table)
    path = _tmp_path()
    try:
        with open(path, "w") as handle:
            handle.write(
                '{"alpha": 0.1, "gamma": 0.9, "epsilon": 1.0, '
                '"q_table": {"(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)": '
                '[1.0, 2.0]}}'
            )

        ok = agent.load(path)

        assert ok is False
        assert agent.q_table == before
    finally:
        os.remove(path)


def test_load_q_table_key_not_tuple_returns_false_and_keeps_state():
    agent = _agent_with_state()
    before = dict(agent.q_table)
    path = _tmp_path()
    try:
        with open(path, "w") as handle:
            handle.write(
                '{"alpha": 0.1, "gamma": 0.9, "epsilon": 1.0, '
                '"q_table": {"42": [1.0, 2.0, 3.0, 4.0]}}'
            )

        ok = agent.load(path)

        assert ok is False
        assert agent.q_table == before
    finally:
        os.remove(path)


def test_load_non_numeric_hyperparameter_returns_false_and_keeps_state():
    agent = _agent_with_state()
    before = dict(agent.q_table)
    before_alpha = agent.alpha
    path = _tmp_path()
    try:
        with open(path, "w") as handle:
            handle.write(
                '{"alpha": "oops", "gamma": 0.9, "epsilon": 1.0, '
                '"q_table": {}}'
            )

        ok = agent.load(path)

        assert ok is False
        assert agent.q_table == before
        assert agent.alpha == before_alpha
    finally:
        os.remove(path)


def test_save_then_load_round_trip():
    saver = _agent_with_state()
    path = _tmp_path()
    try:
        assert saver.save(path) is True

        loader = Agent()
        ok = loader.load(path)

        assert ok is True
        assert loader.alpha == saver.alpha
        assert loader.gamma == saver.gamma
        assert loader.epsilon == saver.epsilon
        assert loader.q_table == saver.q_table
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_load_missing_path_returns_false()
    test_load_non_json_file_returns_false()
    test_load_q_table_value_wrong_type_returns_false_and_keeps_state()
    test_load_q_table_value_wrong_length_returns_false_and_keeps_state()
    test_load_q_table_key_not_tuple_returns_false_and_keeps_state()
    test_load_non_numeric_hyperparameter_returns_false_and_keeps_state()
    test_save_then_load_round_trip()
    print("OK - tous les tests de chargement d'agent passent")
