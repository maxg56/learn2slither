"""persistence.py - sauvegarde liee : modele (JSON) + metriques (CSV).

Le modele est serialise en JSON (Q-table + hyperparametres). Les metriques
d'entrainement partent en parallele dans un CSV, place dans le dossier `data/`
(non versionne). Les deux fichiers restent separes mais relies : le JSON porte
un champ "data" pointant vers le CSV, chemin exprime relativement au dossier du
modele pour rester valide quel que soit le repertoire de travail. Un outil de
visualisation matplotlib retrouve ainsi les donnees depuis le seul modele.
"""

import csv
import os

DATA_DIR = "data"
# Ignore tout le contenu de data/ sauf le .gitignore lui-meme : le dossier est
# versionne (il doit exister) mais les CSV generes ne le sont jamais.
_GITIGNORE = ("# CSV de metriques generes a l'entrainement (non versionnes).\n"
              "*\n!.gitignore\n")


def to_json_path(path):
    """Force l'extension .json sur un chemin de modele (txt -> json)."""
    root, _ = os.path.splitext(path)
    return root + ".json"


def _ensure_data_dir():
    """Cree data/ et son .gitignore si besoin (jamais de crash)."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        gitignore = os.path.join(DATA_DIR, ".gitignore")
        if not os.path.exists(gitignore):
            with open(gitignore, "w") as handle:
                handle.write(_GITIGNORE)
    except OSError:
        pass


def write_stats_csv(model_path, history):
    """Ecrit les metriques dans data/<stem>.csv et retourne le chemin a lier.

    Le chemin retourne est relatif au dossier du modele (pour le champ "data"
    du JSON). Retourne None si l'historique est vide ou en cas d'echec.
    """
    if history is None or len(history) == 0:
        return None
    stem = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = os.path.join(DATA_DIR, stem + ".csv")
    _ensure_data_dir()
    try:
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode"] + list(history.FIELDS))
            for i, episode in enumerate(history.episodes):
                writer.writerow(
                    [episode] + [history.series[f][i] for f in history.FIELDS])
    except OSError:
        return None
    model_dir = os.path.dirname(model_path) or "."
    return os.path.relpath(csv_path, model_dir)


def save_model(agent, path, history=None):
    """Sauvegarde le modele en JSON + CSV de metriques lie.

    Retourne le chemin JSON ecrit en cas de succes, None sinon.
    """
    json_path = to_json_path(path)
    data_ref = write_stats_csv(json_path, history)
    return json_path if agent.save(json_path, data_path=data_ref) else None
