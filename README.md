# Learn2Slither

Projet 42 de *reinforcement learning* : un serpent sur un board 10x10 apprend à
survivre et à grandir par **Q-learning** (Q-table, epsilon-greedy). L'agent ne
perçoit que ce que le serpent voit dans les 4 directions depuis sa tête.

## Installation

Le projet est un paquet Python installable géré avec [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # crée l'environnement et installe les dépendances
```

Dépendances : `numpy`, `pygame-ce` (affichage), `flake8` + `matplotlib` (dev).

## Utilisation

Point d'entrée : le wrapper `./snake` (équivaut à `uv run python -m snakeai`).

```bash
# Entraîner 10 sessions et sauvegarder le modèle, sans affichage (rapide)
./snake -sessions 10 -save models/10sess.json -visual off

# Rejouer un modèle appris, sans modifier son apprentissage, en pas à pas
./snake -visual on -load models/1000sess.json -sessions 10 -dontlearn -step-by-step
```

### Flags

| Flag | Description |
| --- | --- |
| `-sessions N` | nombre de parties d'entraînement |
| `-save PATH` | export du modèle (sérialisé en **JSON**) |
| `-load PATH` | import d'un modèle |
| `-visual on\|off` | affichage graphique pygame (`off` pour accélérer) |
| `-dontlearn` | exploitation pure : epsilon=0, aucune mise à jour |
| `-step-by-step` | avance action par action (Entrée pour continuer) |
| `-dashboard` | *(bonus)* vue parallèle d'une grille de parties |
| `-grid N` | côté de la grille du dashboard (`N`×`N`) |

> Les modèles sont sauvegardés en `.json` (Q-table + hyperparamètres). Le
> chargement accepte aussi les anciens fichiers au même format sous une autre
> extension. Un CSV de métriques lié est écrit en parallèle dans `data/`.

En fin de session, le programme affiche :
`Game over, max length = X, max duration = Y`.

## Affichage terminal

Avec `-visual on` ou `-step-by-step`, la vision du serpent est imprimée avant
chaque action (`W` mur, `H` tête, `S` corps, `G` pomme verte, `R` pomme rouge,
`0` vide), suivie de l'action choisie.

## Modèles livrés

`models/` contient des modèles entraînés à 1, 10, 100 et 1000 sessions,
montrant la progression de l'apprentissage (longueur maximale croissante).

## Développement

```bash
uv run flake8 .   # la norme doit passer sans erreur
```

## Architecture

```
src/snakeai/
├── core/environment.py       # board + règles du jeu (aucun apprentissage)
├── perception/interpreter.py # vision (4 rayons) + rewards
├── learning/agent.py         # Q-learning : Q-table, epsilon-greedy, update
├── training/                 # boucle d'entraînement + persistance + stats
├── ui/display.py             # affichage pygame
├── ui/dashboard/             # vue parallèle (bonus)
└── cli.py                    # parsing CLI + câblage des composants
```

Flux : `Environment` → `Interpreter` (state + reward) → `Agent` (action) → `Environment`.
