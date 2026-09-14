"""Regression : `-record` doit honorer -metrics, -plot et -benchmark (#27).

Bug historique : dans `main()`, la branche `-record` ne transmettait pas le
`MetricsRecorder` a `_run_recorded_session` et n'appelait jamais
`_export_metrics` ni `_print_benchmark`. `./snake -record r.json -metrics
m.csv` ne produisait donc aucun CSV, sans avertissement. Les deux branches
passent desormais par `_report_run`, et `_run_recorded_session` alimente le
recorder puis retourne le meme quadruplet que `train()`.
"""

import csv
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

from snakeai import cli
from snakeai.core import Environment
from snakeai.learning import Agent
from snakeai.perception import Interpreter
from snakeai.training import MetricsRecorder


def _tmp_path(suffix):
    """Chemin d'un fichier temporaire qui n'existe pas encore."""
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.close()
    os.unlink(handle.name)
    return handle.name


def _args(**overrides):
    """Namespace CLI minimal, comme produit par `parse_args`."""
    base = ["-visual", "off", "-dontlearn"]
    for key, value in overrides.items():
        base.append("-" + key.replace("_", "-"))
        if value is not True:
            base.append(str(value))
    return cli.parse_args(base)


def _run_record(args, recorder):
    """Joue une partie enregistree et capture la sortie standard."""
    env = Environment()
    agent = Agent()
    agent.epsilon = 0.0
    out = io.StringIO()
    with redirect_stdout(out):
        result = cli._run_recorded_session(
            env, Interpreter(), agent, False, args, None, recorder)
        cli._report_run(agent, args, recorder, *result)
    return result, out.getvalue()


def test_record_feeds_recorder_and_exports_csv():
    record_path = _tmp_path(".json")
    metrics_path = _tmp_path(".csv")
    args = _args(record=record_path, metrics=metrics_path)
    recorder = MetricsRecorder()
    try:
        (length, duration, _l, _d), output = _run_record(args, recorder)
        assert len(recorder.records) == 1
        assert recorder.records[0]["length"] == length
        assert recorder.records[0]["duration"] == duration
        assert os.path.exists(metrics_path), "CSV -metrics non produit"
        with open(metrics_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert int(rows[0]["length"]) == length
        assert "Metriques sauvegardees" in output
        assert os.path.exists(record_path)
    finally:
        for path in (record_path, metrics_path):
            if os.path.exists(path):
                os.unlink(path)


def test_record_with_benchmark_prints_stats():
    record_path = _tmp_path(".json")
    args = _args(record=record_path, benchmark=True)
    try:
        (length, duration, lengths, durations), output = _run_record(
            args, None)
        assert lengths == [length]
        assert durations == [duration]
        assert "Benchmark (1 sessions)" in output
        assert "Game over, max length = {}".format(length) in output
    finally:
        if os.path.exists(record_path):
            os.unlink(record_path)


def test_record_without_benchmark_returns_empty_lists():
    record_path = _tmp_path(".json")
    args = _args(record=record_path)
    try:
        (_length, _duration, lengths, durations), output = _run_record(
            args, None)
        assert lengths == [] and durations == []
        assert "Benchmark" not in output
    finally:
        if os.path.exists(record_path):
            os.unlink(record_path)


if __name__ == "__main__":
    test_record_feeds_recorder_and_exports_csv()
    test_record_with_benchmark_prints_stats()
    test_record_without_benchmark_returns_empty_lists()
    print("OK - tous les tests de regression passent", file=sys.stderr)
