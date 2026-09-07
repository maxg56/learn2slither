# CLAUDE.md

Ce fichier guide Claude Code lorsqu'il travaille sur ce dépôt.

## Aperçu du projet

**Learn2Slither** — projet 42 de reinforcement learning. Un snake sur un board 10x10
est contrôlé par un agent qui apprend par **Q-learning** (Q-table ou réseau de
neurones, aucun autre modèle autorisé sous peine de 0).

## Architecture

Le code est un paquet Python installable sous `src/snakeai/`. La séparation
stricte imposée par le sujet est conservée, chaque responsabilité vivant dans
son sous-paquet :

- `core/environment.py` — le board et les règles du jeu (aucune logique d'apprentissage)
- `perception/interpreter.py` — calcule l'état (vision du snake) et les rewards
- `learning/agent.py` — Q-learning : Q-table, epsilon-greedy, update, choix de l'action
- `training/trainer.py` — boucle d'entraînement (`run_session`, `train`) assemblant le flux
- `ui/display.py` — interface graphique pygame : vitesse configurable + step-by-step
- `ui/dashboard/` — vue parallèle (bonus), découpée en `simulation` (logique, sans
  pygame), `renderer` (dessin), `theme` (couleurs/layout) et `app` (orchestration)
- `cli.py` — point d'entrée : parsing CLI et câblage des composants
- `models/` — modèles sauvegardés (obligatoire : 1, 10 et 100 sessions minimum)

Point d'entrée : `./snake` (wrapper) → `python -m snakeai` → `snakeai.cli:main`.

Flux : Environment → Interpreter (state + reward) → Agent (action) → Environment.

## Contraintes critiques (pénalités à l'évaluation)

1. **Vision limitée** : l'agent ne reçoit QUE ce que le snake voit dans les 4
   directions depuis sa tête (rayons UP/LEFT/DOWN/RIGHT). Fournir toute autre
   information du board à l'agent = pénalité de **-42**. Ne jamais passer les
   coordonnées absolues, la grille complète, ou la position des pommes hors rayons.
2. **Modèle** : uniquement une Q-function (Q-table ou NN). Rien d'autre.
3. **Affichage terminal obligatoire** : avant chaque action, afficher la vision
   (caractères `W` mur, `H` tête, `S` corps, `G` pomme verte, `R` pomme rouge,
   `0` vide) puis l'action choisie. Cette obligation s'applique aux runs
   `-visual on` / `-step-by-step` ; `-visual off` supprime volontairement
   tout affichage par pas (y compris ce texte, pas seulement la fenêtre
   pygame) pour garder l'entraînement en masse rapide.
4. **Norme** : le code Python doit passer `flake8` sans erreur.
5. Le programme ne doit jamais crasher (crash = 0 à l'évaluation).

## Règles du jeu

- Board 10x10, snake initial de 3 cellules contiguës placé aléatoirement.
- 2 pommes vertes + 1 pomme rouge, positions aléatoires.
- Pomme verte : longueur +1, nouvelle pomme verte apparaît.
- Pomme rouge : longueur -1, nouvelle pomme rouge apparaît.
- Game over : mur, collision avec sa queue, ou longueur 0.
- Actions possibles : UP, LEFT, DOWN, RIGHT uniquement.
- Objectif : longueur ≥ 10 et durée de vie maximale.

## Interface CLI attendue

```
./snake -sessions 10 -save models/10sess.txt -visual off
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```

Flags à supporter :
- `-sessions N` — nombre de sessions d'entraînement
- `-save PATH` / `-load PATH` — export/import du modèle (fichier unique contenant
  tout l'état d'apprentissage, principalement les Q-values)
- `-visual on|off` — affichage graphique (off pour accélérer l'entraînement) ;
  `off` coupe aussi l'affichage terminal de la vision par pas (voir
  contrainte 3), pas seulement la fenêtre pygame
- `-dontlearn` — exploitation pure : epsilon=0, aucune mise à jour de la Q-function
- `-step-by-step` — avance pas à pas

En fin de session, afficher : `Game over, max length = X, max duration = Y`.

## Commandes de développement

```bash
# Environnement virtuel
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # pygame, numpy, flake8

# Lint (obligatoire avant tout commit)
flake8 .

# Entraînement des modèles à livrer
./snake -sessions 1 -save models/1sess.txt -visual off
./snake -sessions 10 -save models/10sess.txt -visual off
./snake -sessions 100 -save models/100sess.txt -visual off
./snake -sessions 1000 -save models/1000sess.txt -visual off

# Évaluation d'un modèle sans altérer son apprentissage
./snake -load models/100sess.txt -sessions 10 -dontlearn -visual on
```

## Notes d'implémentation

- **Encodage de l'état** : réduire chaque rayon de vision en features compactes
  (par direction : danger adjacent, pomme verte visible, pomme rouge visible).
  Indépendant de la taille du board → permet le bonus "board size variable".
- **Q-learning** : `Q(s,a) += alpha * (r + gamma * max(Q(s')) - Q(s,a))`,
  alpha ≈ 0.1, gamma ≈ 0.9, epsilon-greedy avec decay (1.0 → 0.01).
- **Rewards** (ajustables) : pomme verte +20, pomme rouge -20, rien -1,
  game over -100.
- La Q-table est un dict `{state: [q_up, q_left, q_down, q_right]}`, sérialisée
  en JSON dans les fichiers modèles.

## Convention de commits

- Ne pas ajouter Claude comme co-auteur des commits. N'inclure aucune ligne
  `Co-Authored-By: Claude ...` ni mention de génération par Claude dans les
  messages de commit ou les descriptions de PR.

## Bonus (seulement si le mandatory est parfait)

- Longueur atteinte 15/20/25/30/35.
- Affichage avancé : lobby, panneau de configuration, statistiques.
- Taille de board paramétrable, avec le même modèle fonctionnel sur toute taille.
