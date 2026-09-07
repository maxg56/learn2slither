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

# --- Reward shaping (rapprochement des pommes) -----------------------------
# L'etat ne code la pomme verte que par un bit "visible / pas visible" (sans
# distance) : avancer vers une pomme visible ne change pas l'etat et rapporte
# le meme -1 que s'en eloigner. Sans gradient, le serpent erre et tourne en
# rond au lieu de manger. On ajoute donc un bonus proportionnel a la reduction
# de distance a la pomme verte visible la plus proche (rapprochement +, recul
# -, distance inchangee 0). Calcule uniquement a partir des rayons de vision,
# donc conforme a la contrainte "vision seule".
REWARD_APPROACH = 1.0

# --- Reward shaping alternatif ("alt", selectionnable) ----------------------
# Schema alternatif optionnel, active via -reward-shaping alt / via
# Interpreter(reward_mode="alt"). Le mode "default" (aucun flag, comportement
# historique) reste strictement identique aux valeurs ci-dessus : ces deux
# constantes ne sont lues que lorsque reward_mode == "alt".

# Anti demi-tour : penalite quand l'action choisie est l'exact oppose de la
# direction courante du serpent (rebrousser chemin droit dans son propre
# cou). Cette information provient uniquement de la derniere action de
# l'agent et de la direction dans laquelle il se deplacait deja (un fait que
# l'agent connait de lui-meme, independant du board) : aucune coordonnee,
# grille ou position de pomme supplementaire n'est fournie, donc conforme a
# la contrainte "vision seule". En pratique, un demi-tour percute toujours
# immediatement le cou du serpent (collision deja sanctionnee par
# REWARD_GAMEOVER) : cette penalite additionnelle donne un signal associe a
# l'action elle-meme, qui se propage plus vite dans la Q-table que le seul
# reward terminal.
REWARD_UTURN = -10

# Bonus de survie : petit bonus ajoute a REWARD_NOTHING (pas un remplacement)
# sur chaque pas "nothing", pour alleger le cout de la simple survie. Reste
# tres inferieur a REWARD_GREEN (20) et tres petit face a REWARD_GAMEOVER
# (-100) : il ne peut donc pas inciter l'agent a eviter les pommes ou a
# provoquer sa propre mort pour "farmer" ce bonus. But : encourager une duree
# de vie plus longue (cf. objectif du sujet) sans desinciter la recherche de
# pommes vertes, qui restent bien plus rentables.
REWARD_SURVIVAL_BONUS = 0.2

# --- Anti-blocage ----------------------------------------------------------
# En exploitation pure (epsilon=0, modele charge), le serpent suit une
# politique deterministe et peut tourner en rond indefiniment sans jamais
# mourir : la partie ne se termine plus et la generation reste figee. On
# force la fin de partie apres trop de pas consecutifs sans manger de pomme
# verte. Le seuil est proportionnel a l'aire du board (bonus taille variable).
STALL_STEPS_FACTOR = 4
