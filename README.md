# Learn2Slither

Learn2Slither est un projet de l'école 42 : un serpent évolue sur un plateau
10x10 et apprend à survivre et à grandir grâce au **Q-learning**, sans aucune
autre forme de modèle. L'agent choisit ses actions (`UP`, `LEFT`, `DOWN`,
`RIGHT`) à partir d'une Q-table mise à jour au fil des parties, en visant une
longueur d'au moins 10 et la durée de vie la plus longue possible.

## Installation

Le projet est packagé avec [uv](https://docs.astral.sh/uv/) et cible Python
3.14 (voir `pyproject.toml`). L'installation la plus fiable est donc via `uv`,
qui télécharge lui-même l'interpréteur Python requis si besoin :

```bash
# Installer les dépendances (numpy, pygame-ce, flake8) dans .venv
uv sync
```

Le wrapper `./snake` invoque directement `uv run python -m snakeai`, donc
aucune activation manuelle de venv n'est nécessaire : une fois `uv sync`
exécuté, `./snake ...` fonctionne tel quel.

> Remarque : `pip install -e .` échoue sur un Python < 3.12 (numpy>=2.5.0 et
> le `requires-python` du projet imposent une version récente) ; utiliser
> `uv sync` évite ce problème.

## Utilisation

```bash
# Entraîner un modèle sur 10 sessions, sans affichage (rapide), et le sauvegarder
./snake -sessions 10 -save models/10sess.txt -visual off

# Charger un modèle entraîné et l'évaluer en pur exploitation (epsilon=0,
# pas de mise à jour de la Q-table), avec affichage graphique pas à pas
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```

Flags disponibles :

| Flag | Rôle |
| --- | --- |
| `-sessions N` | nombre de sessions d'entraînement (défaut : 1) |
| `-save PATH` | sauvegarde le modèle (Q-table) à la fin |
| `-load PATH` | charge un modèle existant avant de démarrer |
| `-visual on\|off` | affichage graphique pygame (défaut : `on`) |
| `-dontlearn` | exploitation pure : epsilon=0, aucune mise à jour |
| `-step-by-step` | avance action par action |
| `-dashboard` | vue parallèle (bonus) : plusieurs parties simultanées |
| `-grid N` | taille de la grille du dashboard (`-dashboard`) |

À la fin de chaque exécution, le programme affiche
`Game over, max length = X, max duration = Y`.

## Modèles livrés

Le dossier `models/` contient les modèles entraînés livrés avec le projet :

- `models/1sess.txt` — entraîné sur 1 session
- `models/10sess.txt` — entraîné sur 10 sessions
- `models/100sess.txt` — entraîné sur 100 sessions
- `models/1000sess.txt` — entraîné sur 1000 sessions

## Contrainte respectée

L'agent ne perçoit que la vision du serpent depuis sa tête, sous forme de
quatre rayons (haut/gauche/bas/droite) — jamais les coordonnées absolues, la
grille complète, ni la position des pommes hors de ces rayons.
