"""nn_agent.py - Q-learning : reseau de neurones, epsilon-greedy, update.

Alternative a la Q-table de agent.py : la fonction Q est ici approximee par
un petit reseau feed-forward (constants.STATE_SIZE entrees -> couche
cachee -> 4 sorties), implemente a la main avec numpy (forward/backward
manuels, sans framework de deep learning). Meme interface publique que
Agent pour rester interchangeable depuis cli.py/trainer.py.

Contrairement a la Q-table, exacte par construction, un reseau entraine par
descente de gradient en ligne sur une seule transition ne converge pas :
la cible `r + gamma * max Q(s')` bouge a chaque pas puisqu'elle est calculee
par le reseau qu'on modifie, et les rewards bruts (-100 .. +20) saturent la
couche tanh. On applique donc les trois stabilisateurs classiques du DQN :

- un taux d'apprentissage propre (LEARNING_RATE), distinct de l'alpha de la
  Q-table qui est dix fois trop grand pour une descente de gradient ;
- des rewards ramenes dans [-1, 1] (division par REWARD_SCALE : la politique
  optimale est inchangee, seule l'echelle des Q-values change) ;
- un replay buffer echantillonne par mini-lots et un reseau cible (target
  network) resynchronise periodiquement, qui fige la cible entre deux
  copies.
"""

import json
import random
from collections import deque

import numpy as np

from snakeai import constants

HIDDEN_UNITS = 32
INPUT_SIZE = constants.STATE_SIZE
OUTPUT_SIZE = len(constants.ACTIONS)

# Taux d'apprentissage du gradient (voir docstring du module : ce n'est pas
# l'alpha de la Q-table).
LEARNING_RATE = 0.05
# Les rewards sont divises par cette valeur avant d'entrer dans la cible,
# pour que game over (-100) devienne -1 et pomme verte (+20) 0.2.
REWARD_SCALE = abs(constants.REWARD_GAMEOVER)
# Replay buffer : transitions conservees et taille des mini-lots.
REPLAY_CAPACITY = 5000
BATCH_SIZE = 32
# Nombre d'appels a update() entre deux copies des poids vers le reseau
# cible.
TARGET_SYNC_STEPS = 200
# Norme maximale du gradient (clipping global) : garde-fou contre les
# rares lots ou l'erreur explose.
GRAD_CLIP = 1.0


class NNAgent:
    """Agent Q-learning dont la fonction Q est un petit reseau de neurones."""

    def __init__(self, learning_rate=LEARNING_RATE, gamma=constants.GAMMA,
                 epsilon=constants.EPSILON_START,
                 epsilon_decay=constants.EPSILON_DECAY, seed=None):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # `seed` graine le generateur local plutot que l'etat global numpy :
        # sans lui, l'initialisation des poids reste aleatoire et `-seed`
        # ne suffit pas a rendre un run `-model nn` reproductible. Le meme
        # generateur tire les mini-lots du replay buffer.
        self._rng = np.random.default_rng(seed)
        # Initialisation "petite" pour eviter de saturer tanh des le depart.
        self.w1 = self._rng.normal(0, 0.1, (INPUT_SIZE, HIDDEN_UNITS))
        self.b1 = np.zeros(HIDDEN_UNITS)
        self.w2 = self._rng.normal(0, 0.1, (HIDDEN_UNITS, OUTPUT_SIZE))
        self.b2 = np.zeros(OUTPUT_SIZE)
        self._replay = deque(maxlen=REPLAY_CAPACITY)
        self._updates = 0
        self._sync_target()

    # --- Reseau ------------------------------------------------------------

    def _sync_target(self):
        """Copie les poids courants dans le reseau cible."""
        self._target = (self.w1.copy(), self.b1.copy(),
                        self.w2.copy(), self.b2.copy())

    def _forward(self, states, weights=None):
        """Passe avant : renvoie (q_values, activation_cachee).

        `states` est un vecteur (un etat) ou une matrice (un lot d'etats).
        hidden = tanh(x . w1 + b1) ; q = hidden . w2 + b2 (sorties lineaires,
        une regression de Q-value n'a pas besoin d'activation en sortie).
        `weights` permet d'evaluer le reseau cible au lieu du reseau
        courant.
        """
        w1, b1, w2, b2 = weights or (self.w1, self.b1, self.w2, self.b2)
        x = np.asarray(states, dtype=np.float64)
        hidden = np.tanh(x @ w1 + b1)
        q = hidden @ w2 + b2
        return q, hidden

    def _qvalues(self, state):
        q, _ = self._forward(state)
        return q

    def choose_action(self, state):
        """Choisit une action via epsilon-greedy sur les sorties du reseau."""
        if random.random() < self.epsilon:
            return random.choice(constants.ACTIONS)
        q = self._qvalues(state)
        best = q.max()
        candidates = [action for action, value in zip(constants.ACTIONS, q)
                      if value == best]
        return random.choice(candidates)

    # --- Apprentissage -----------------------------------------------------

    def update(self, state, action, reward, next_state):
        """Memorise la transition puis fait un pas de gradient sur un lot.

        La transition rejoint le replay buffer ; on echantillonne ensuite un
        mini-lot (la transition courante n'est donc pas forcement dedans) et
        on descend le gradient de l'erreur quadratique moyenne entre
        Q(s, a) et la cible reward/REWARD_SCALE (+ gamma * max_a' Q_cible(s',
        a') si la transition n'est pas terminale). Seule la sortie de
        l'action jouee recoit une erreur, les 3 autres restent inchangees.
        Tous les TARGET_SYNC_STEPS appels, le reseau cible est resynchronise.
        """
        self._replay.append((tuple(state), action, reward / REWARD_SCALE,
                             None if next_state is None else
                             tuple(next_state)))
        batch_size = min(BATCH_SIZE, len(self._replay))
        indices = self._rng.choice(len(self._replay), batch_size,
                                   replace=False)
        batch = [self._replay[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float64)
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        # Les transitions terminales n'ont pas d'etat suivant : on leur donne
        # un etat nul, dont la valeur est ensuite masquee par `alive`.
        alive = np.array([t[3] is not None for t in batch], dtype=np.float64)
        next_states = np.array([t[3] if t[3] is not None else (0,) * INPUT_SIZE
                                for t in batch], dtype=np.float64)

        next_q, _ = self._forward(next_states, self._target)
        targets = rewards + self.gamma * alive * next_q.max(axis=1)

        q, hidden = self._forward(states)
        rows = np.arange(batch_size)
        error = np.zeros_like(q)
        # Derivee de l'erreur quadratique moyenne 0.5 * mean((q - cible)^2).
        error[rows, actions] = (q[rows, actions] - targets) / batch_size

        grad_w2 = hidden.T @ error
        grad_b2 = error.sum(axis=0)
        # Retropropagation a travers tanh : d(tanh)/dz = 1 - tanh(z)^2.
        grad_hidden = (error @ self.w2.T) * (1 - hidden ** 2)
        grad_w1 = states.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)

        grads = [grad_w1, grad_b1, grad_w2, grad_b2]
        norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if norm > GRAD_CLIP:
            grads = [g * (GRAD_CLIP / norm) for g in grads]
        for param, grad in zip((self.w1, self.b1, self.w2, self.b2), grads):
            param -= self.learning_rate * grad

        self._updates += 1
        if self._updates % TARGET_SYNC_STEPS == 0:
            self._sync_target()

    def decay_epsilon(self):
        """Reduit epsilon vers sa valeur minimale."""
        self.epsilon = max(constants.EPSILON_MIN,
                           self.epsilon * self.epsilon_decay)

    # --- Persistance -------------------------------------------------------

    def save(self, path):
        """Serialise tout l'etat d'apprentissage dans un fichier JSON.

        Le replay buffer (de l'experience, pas de l'apprentissage) et le
        reseau cible (recopie des poids au chargement) ne sont pas
        sauvegardes. Tolerant aux chemins invalides (jamais de crash).
        Retourne True si la sauvegarde a reussi, False sinon.
        """
        data = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }
        try:
            with open(path, "w") as handle:
                json.dump(data, handle)
            return True
        except OSError:
            return False

    def load(self, path):
        """Recharge un etat d'apprentissage depuis un fichier JSON.

        Tolerant aux fichiers absents ou corrompus (jamais de crash), y
        compris un JSON syntaxiquement valide mais de forme incorrecte
        (poids aux mauvaises dimensions, hyperparametres non numeriques) :
        sans ce controle, un tel modele etait accepte puis faisait planter
        le premier forward. La validation se fait sur des variables
        locales : en cas d'echec, l'etat actuel de l'agent n'est jamais
        modifie. Retourne True si le chargement a reussi, False sinon.
        """
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
            learning_rate = data.get("learning_rate", self.learning_rate)
            gamma = data.get("gamma", self.gamma)
            epsilon = data.get("epsilon", self.epsilon)
            epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
            if not all(isinstance(v, (int, float)) for v in
                       (learning_rate, gamma, epsilon, epsilon_decay)):
                return False
            w1 = _as_array(data["w1"], (INPUT_SIZE, HIDDEN_UNITS))
            b1 = _as_array(data["b1"], (HIDDEN_UNITS,))
            w2 = _as_array(data["w2"], (HIDDEN_UNITS, OUTPUT_SIZE))
            b2 = _as_array(data["b2"], (OUTPUT_SIZE,))
            if any(array is None for array in (w1, b1, w2, b2)):
                return False
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.w1 = w1
            self.b1 = b1
            self.w2 = w2
            self.b2 = b2
            self._sync_target()
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False


def _as_array(values, shape):
    """Convertit `values` en tableau flottant de forme `shape`, sinon None.

    Retourne None (plutot que de lever) des que la conversion echoue ou que
    la forme ne correspond pas, pour que `load()` puisse refuser le fichier
    sans avoir touche a l'etat de l'agent.
    """
    try:
        array = np.array(values, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array
