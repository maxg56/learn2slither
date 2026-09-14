"""Tests unitaires pour `save_recording` / `load_recording` (replay.py).

Meme style que `test_agent_load.py` : pas de fixtures/classes, fichiers
temporaires geres a la main. Verifie que la persistance des replays est
aussi tolerante que `Agent.save/load` : jamais de traceback sur un chemin
invalide, un JSON corrompu ou un enregistrement mal forme.
"""

import json
import os
import tempfile

from snakeai import constants
from snakeai.training.replay import (
    RecordingError, load_recording, save_recording,
)


def _tmp_path():
    """Chemin d'un fichier temporaire qui n'existe pas encore."""
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    os.remove(path)
    return path


def _write(path, payload):
    with open(path, "w") as handle:
        handle.write(payload)


def _frame(action=constants.UP):
    return {
        "snake": [[1, 1], [1, 2], [1, 3]],
        "green_apples": [[0, 0], [5, 5]],
        "red_apples": [[9, 9]],
        "action": action,
    }


def _expect_error(path):
    try:
        load_recording(path)
    except RecordingError:
        return True
    return False


def test_save_invalid_path_returns_false():
    ok = save_recording("/tmp/this/path/does/not/exist/rec.json",
                        [_frame()], 10)
    assert ok is False


def test_save_then_load_roundtrip():
    path = _tmp_path()
    try:
        assert save_recording(path, [_frame(constants.LEFT)], 7) is True
        frames, size = load_recording(path)
    finally:
        os.remove(path)
    assert size == 7
    assert frames == [{
        "snake": [(1, 1), (1, 2), (1, 3)],
        "green_apples": [(0, 0), (5, 5)],
        "red_apples": [(9, 9)],
        "action": constants.LEFT,
    }]


def test_load_missing_size_defaults_to_board_size():
    path = _tmp_path()
    try:
        _write(path, json.dumps({"frames": [_frame()]}))
        _frames, size = load_recording(path)
    finally:
        os.remove(path)
    assert size == constants.BOARD_SIZE


def test_load_missing_path_raises_recording_error():
    assert _expect_error("/tmp/this/path/does/not/exist/rec.json")


def test_load_invalid_json_raises_recording_error():
    path = _tmp_path()
    try:
        _write(path, '{"frames": [')
        assert _expect_error(path)
    finally:
        os.remove(path)


def test_load_malformed_recordings_raise_recording_error():
    bad_payloads = [
        "[]",                                       # pas un objet
        json.dumps({}),                             # sans frames
        json.dumps({"frames": {}}),                 # frames non liste
        json.dumps({"size": 2, "frames": []}),      # board trop petit
        json.dumps({"size": "10", "frames": []}),   # taille non entiere
        json.dumps({"frames": [42]}),               # frame non objet
        json.dumps({"frames": [dict(_frame(), action=9)]}),
        json.dumps({"frames": [dict(_frame(), action="UP")]}),
        json.dumps({"frames": [dict(_frame(), snake=None)]}),
        json.dumps({"frames": [dict(_frame(), snake=[[1]])]}),
        json.dumps({"frames": [dict(_frame(), red_apples=[["a", 1]])]}),
    ]
    path = _tmp_path()
    try:
        for payload in bad_payloads:
            _write(path, payload)
            assert _expect_error(path), payload
    finally:
        os.remove(path)
