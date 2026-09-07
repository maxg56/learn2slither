"""nn_agent.py - Q-learning : reseau de neurones, epsilon-greedy, update.

Alternative a la Q-table de agent.py : la fonction Q est ici approximee par
un petit reseau feed-forward (12 entrees -> couche cachee -> 4 sorties),
implemente a la main avec numpy (forward/backward manuels, sans framework
de deep learning). Meme interface publique que Agent pour rester
interchangeable depuis cli.py/trainer.py.
"""

import json
import random

import numpy as np

from snakeai import constants

HIDDEN_UNITS = 12
INPUT_SIZE = 12
OUTPUT_SIZE = len(constants.ACTIONS)


class NNAgent:
    """Agent Q-learning dont la fonction Q est un petit reseau de neurones."""

    def __init__(self, alpha=constants.ALPHA, gamma=constants.GAMMA,
                 epsilon=constants.EPSILON_START):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        rng = np.random.default_rng()
        # Initialisation "petite" pour eviter de saturer tanh des le depart.
        self.w1 = rng.normal(0, 0.1, (INPUT_SIZE, HIDDEN_UNITS))
        self.b1 = np.zeros(HIDDEN_UNITS)
        self.w2 = rng.normal(0, 0.1, (HIDDEN_UNITS, OUTPUT_SIZE))
        self.b2 = np.zeros(OUTPUT_SIZE)

    def _forward(self, state):
        """Passe avant : renvoie (q_values, activation_cachee) pour un etat.

        hidden = tanh(x . w1 + b1) ; q = hidden . w2 + b2 (sorties lineaires,
        une regression de Q-value n'a pas besoin d'activation en sortie).
        """
        x = np.asarray(state, dtype=np.float64)
        hidden = np.tanh(x @ self.w1 + self.b1)
        q = hidden @ self.w2 + self.b2
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

    def update(self, state, action, reward, next_state):
        """Un pas de descente de gradient vers la cible du Q-learning.

        Cible = reward (+ gamma * max_a' Q(next_state, a') si transition non
        terminale). Seule la sortie `action` est entrainee : l'erreur est
        nulle sur les 3 autres sorties (elle ne les influence donc pas).
        """
        if next_state is None:
            target = reward
        else:
            target = reward + self.gamma * self._qvalues(next_state).max()

        q, hidden = self._forward(state)
        error = np.zeros(OUTPUT_SIZE)
        error[action] = q[action] - target

        x = np.asarray(state, dtype=np.float64)
        # Gradients (derivee de l'erreur quadratique 0.5*(q-target)^2) :
        grad_w2 = np.outer(hidden, error)
        grad_b2 = error
        # Retropropagation a travers tanh : d(tanh)/dz = 1 - tanh(z)^2.
        grad_hidden = (error @ self.w2.T) * (1 - hidden ** 2)
        grad_w1 = np.outer(x, grad_hidden)
        grad_b1 = grad_hidden

        self.w2 -= self.alpha * grad_w2
        self.b2 -= self.alpha * grad_b2
        self.w1 -= self.alpha * grad_w1
        self.b1 -= self.alpha * grad_b1

    def decay_epsilon(self):
        """Reduit epsilon vers sa valeur minimale."""
        self.epsilon = max(constants.EPSILON_MIN,
                           self.epsilon * constants.EPSILON_DECAY)

    def save(self, path):
        """Serialise tout l'etat d'apprentissage dans un fichier JSON.

        Tolerant aux chemins invalides (jamais de crash). Retourne True si
        la sauvegarde a reussi, False sinon.
        """
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
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

        Tolerant aux fichiers absents ou corrompus (jamais de crash).
        Retourne True si le chargement a reussi, False sinon.
        """
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.w1 = np.array(data["w1"], dtype=np.float64)
            self.b1 = np.array(data["b1"], dtype=np.float64)
            self.w2 = np.array(data["w2"], dtype=np.float64)
            self.b2 = np.array(data["b2"], dtype=np.float64)
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
