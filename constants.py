"""Constantes globales du projet Learn2Slither.

Regroupe la taille du board, les actions, les caracteres d'affichage,
les hyperparametres de Q-learning et les rewards. Aucun autre module ne
doit redefinir ces valeurs.
"""

# --- Board -----------------------------------------------------------------
BOARD_SIZE = 10
SNAKE_START_LENGTH = 3
GREEN_APPLES = 2
RED_APPLES = 1

# --- Actions ---------------------------------------------------------------
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3
ACTIONS = (UP, LEFT, DOWN, RIGHT)
ACTION_NAMES = {UP: "UP", LEFT: "LEFT", DOWN: "DOWN", RIGHT: "RIGHT"}

# Vecteurs de deplacement (delta_ligne, delta_colonne).
# UP diminue l'indice de ligne (haut de la grille).
MOVES = {
    UP: (-1, 0),
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
}

# --- Caracteres d'affichage (terminal) -------------------------------------
CELL_EMPTY = "0"
CELL_WALL = "W"
CELL_HEAD = "H"
CELL_BODY = "S"
CELL_GREEN = "G"
CELL_RED = "R"

# --- Hyperparametres Q-learning --------------------------------------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

# --- Rewards ---------------------------------------------------------------
REWARD_GREEN = 20
REWARD_RED = -20
REWARD_NOTHING = -1
REWARD_GAMEOVER = -100

# --- Anti-blocage ----------------------------------------------------------
# En exploitation pure (epsilon=0, modele charge), le serpent suit une
# politique deterministe et peut tourner en rond indefiniment sans jamais
# mourir : la partie ne se termine plus et la generation reste figee. On
# force la fin de partie apres trop de pas consecutifs sans manger de pomme
# verte. Le seuil est proportionnel a l'aire du board (bonus taille variable).
STALL_STEPS_FACTOR = 4
