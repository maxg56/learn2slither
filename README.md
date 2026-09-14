# Learn2Slither

Learn2Slither est un projet de l'école 42 : un serpent évolue sur un plateau
10x10 et apprend à survivre et à grandir grâce au **Q-learning**, sans aucune
autre forme de modèle. L'agent choisit ses actions (`UP`, `LEFT`, `DOWN`,
`RIGHT`) à partir d'une Q-table mise à jour au fil des parties, en visant une
longueur d'au moins 10 et la durée de vie la plus longue possible.

## Installation

Le projet est packagé avec [uv](https://docs.astral.sh/uv/). Il demande
Python >= 3.10 (`requires-python` dans `pyproject.toml`) ; la version épinglée
pour le développement est celle de `.python-version` (3.11). L'installation la
plus fiable passe donc par `uv`, qui télécharge lui-même l'interpréteur voulu
si besoin :

```bash
# Installer les dépendances (numpy, pygame-ce, flake8, pytest) dans .venv
uv sync
```

Le wrapper `./snake` invoque directement `uv run python -m snakeai`, donc
aucune activation manuelle de venv n'est nécessaire : une fois `uv sync`
exécuté, `./snake ...` fonctionne tel quel.

Une installation classique reste possible sur tout Python >= 3.10
(`pip install -e .`, ou `pip install -r requirements.txt`).

Deux dépendances sont facultatives et seulement utiles à des options bonus :
`matplotlib` pour `-plot` et `Pillow` pour `-export-gif`. En leur absence, le
programme affiche un avertissement et continue sans crasher.

## Utilisation

```bash
# Entraîner un modèle sur 10 sessions, sans affichage (rapide), et le sauvegarder
./snake -sessions 10 -save models/10sess.txt -visual off

# Charger un modèle entraîné et l'évaluer en pur exploitation (epsilon=0,
# pas de mise à jour de la Q-table), avec affichage graphique pas à pas
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```

Flags du sujet :

| Flag | Rôle |
| --- | --- |
| `-sessions N` | nombre de sessions d'entraînement (défaut : 1) |
| `-save PATH` | sauvegarde le modèle (Q-table) à la fin |
| `-load PATH` | charge un modèle existant avant de démarrer |
| `-visual on\|off` | affichage graphique pygame (défaut : `on`) ; `off` coupe aussi l'affichage terminal de la vision |
| `-dontlearn` | exploitation pure : epsilon=0, aucune mise à jour |
| `-step-by-step` | avance action par action |

Options supplémentaires (bonus et outillage) :

| Flag | Rôle |
| --- | --- |
| `-model qtable\|nn` | fonction Q utilisée : Q-table ou réseau de neurones (défaut : `qtable`) |
| `-board-size N` | côté du board (défaut : 10) ; N >= 3 |
| `-seed N` | graine aléatoire pour des runs reproductibles |
| `-reward-shaping default\|alt` | schéma de reward : historique ou alternatif (anti demi-tour + bonus de survie) |
| `-benchmark` | agrège longueur/durée (mean/min/max) sur toutes les sessions |
| `-metrics PATH` | export CSV des courbes d'entraînement |
| `-plot PATH` | export PNG des courbes (nécessite `matplotlib`) |
| `-record PATH` | enregistre une partie (frames) vers PATH |
| `-replay PATH` | rejoue un enregistrement `-record`, sans agent |
| `-export-gif PATH` | exporte la partie jouée en GIF animé (nécessite `-visual on` et `Pillow`) |
| `-dashboard` | vue parallèle (bonus) : plusieurs parties simultanées |
| `-grid N` | côté de la grille du dashboard (`-dashboard`), défaut : 6 |
| `-dashboard-lobby` | lobby de choix de modèle avant le dashboard (`-dashboard` requis) |

À la fin de chaque exécution, le programme affiche
`Game over, max length = X, max duration = Y`.

## Développement

```bash
uv run flake8 .        # norme (obligatoire, le sujet impose flake8 sans erreur)
uv run pytest tests    # suite de tests
```

Les deux commandes tournent aussi en CI (GitHub Actions) sur chaque push et
pull request.

## Modèles livrés

Le dossier `models/` contient les modèles entraînés livrés avec le projet :

- `models/1sess.txt` — entraîné sur 1 session
- `models/10sess.txt` — entraîné sur 10 sessions
- `models/100sess.txt` — entraîné sur 100 sessions
- `models/1000sess.txt` — entraîné sur 1000 sessions

Seuls ces quatre fichiers sont versionnés. Les snapshots du dashboard
(`models/model_gen*.txt`) sont ignorés par Git, et les résultats
d'expérimentation ne vont pas dans `models/` : la recherche par grille
(`PYTHONPATH=src python -m snakeai.training.tune`) écrit par défaut dans
`data/tuning_results.csv`, dossier lui aussi ignoré par Git.

## Contrainte respectée

L'agent ne perçoit que la vision du serpent depuis sa tête, sous forme de
quatre rayons (haut/gauche/bas/droite) — jamais les coordonnées absolues, la
grille complète, ni la position des pommes hors de ces rayons.
