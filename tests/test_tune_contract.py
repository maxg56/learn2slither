"""Regression : tune.py doit rester aligne sur le contrat de `train()`.

Bug historique : `train()` est passe de 2 a 4 valeurs de retour lors de
l'ajout de `-benchmark` (best_length, best_duration, lengths, durations),
mais `tune._train_quietly` deballait toujours 2 valeurs :

    ValueError: too many values to unpack (expected 2)

Le module de recherche par grille etait donc totalement inutilisable. On
verifie ici :
  1. `train()` renvoie bien 4 champs, dont deux listes ;
  2. `evaluate_combo` s'execute de bout en bout sur une session ;
  3. `run_grid_search` produit un CSV complet sur une mini-grille.
"""

import argparse
import csv
import os
import tempfile

from snakeai.core import Environment
from snakeai.learning import Agent
from snakeai.perception import Interpreter
from snakeai.training import tune
from snakeai.training.trainer import train


def test_train_returns_four_fields():
    args = argparse.Namespace(
        sessions=1, visual="off", dontlearn=False, step_by_step=False,
        benchmark=False,
    )
    result = train(Environment(), Interpreter(), Agent(), args)

    assert len(result) == 4
    best_length, best_duration, lengths, durations = result
    assert isinstance(best_length, int)
    assert isinstance(best_duration, int)
    assert isinstance(lengths, list)
    assert isinstance(durations, list)


def test_evaluate_combo_single_session():
    row = tune.evaluate_combo(0.1, 0.9, 0.98, sessions=1)

    assert set(row) == set(tune._FIELDNAMES)
    assert row["alpha"] == 0.1
    assert row["gamma"] == 0.9
    assert row["epsilon_decay"] == 0.98
    assert row["sessions"] == 1
    assert row["best_length"] >= 1
    assert row["best_duration"] >= 0
    assert row["mean_length"] == row["best_length"]


def test_run_grid_search_writes_csv():
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "sub", "tuning.csv")
        results = tune.run_grid_search(
            alphas=(0.1,), gammas=(0.9,), epsilon_decays=(0.98,),
            sessions=1, output_path=output,
        )

        assert len(results) == 1
        with open(output, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert sorted(rows[0]) == sorted(tune._FIELDNAMES)


if __name__ == "__main__":
    test_train_returns_four_fields()
    test_evaluate_combo_single_session()
    test_run_grid_search_writes_csv()
    print("OK - tune.py est aligne sur le contrat de train()")
