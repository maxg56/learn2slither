"""agent.py - Q-learning : Q-table, epsilon-greedy, update.

Seul module contenant de la logique d'apprentissage. La Q-table est un
dictionnaire {state: [q_up, q_left, q_down, q_right]}, serialisee en JSON
pour save/load.
"""

import ast
import json
import random

from snakeai import constants


class Agent:
    """Agent Q-learning avec politique epsilon-greedy."""

    def __init__(self, alpha=constants.ALPHA, gamma=constants.GAMMA,
                 epsilon=constants.EPSILON_START):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _qvalues(self, state):
        """Retourne (en l'initialisant au besoin) les Q-values d'un etat."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(constants.ACTIONS)
        return self.q_table[state]

    def choose_action(self, state):
        """Choisit une action via epsilon-greedy.

        Les actions coincident avec les indices des Q-values
        (UP=0, LEFT=1, DOWN=2, RIGHT=3).
        """
        if random.random() < self.epsilon:
            return random.choice(constants.ACTIONS)
        q = self._qvalues(state)
        best = max(q)
        candidates = [action for action, value in zip(constants.ACTIONS, q)
                      if value == best]
        return random.choice(candidates)

    def update(self, state, action, reward, next_state):
        """Applique la regle de mise a jour du Q-learning.

        next_state a None signale une transition terminale (game over) :
        la cible se reduit alors a la seule recompense.
        """
        q = self._qvalues(state)
        if next_state is None:
            best_next = 0.0
        else:
            best_next = max(self._qvalues(next_state))
        q[action] += self.alpha * (reward + self.gamma * best_next - q[action])

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
            "q_table": {str(state): values
                        for state, values in self.q_table.items()},
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
        (q_table de mauvaise taille/type, hyperparametres non numeriques).
        La validation se fait sur des variables locales : en cas d'echec,
        l'etat actuel de l'agent n'est jamais modifie.
        Retourne True si le chargement a reussi, False sinon.
        """
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
            alpha = data.get("alpha", self.alpha)
            gamma = data.get("gamma", self.gamma)
            epsilon = data.get("epsilon", self.epsilon)
            if not all(isinstance(v, (int, float))
                       for v in (alpha, gamma, epsilon)):
                return False
            raw_q_table = data.get("q_table", {})
            if not isinstance(raw_q_table, dict):
                return False
            q_table = {}
            for state, values in raw_q_table.items():
                parsed_state = ast.literal_eval(state)
                if not isinstance(parsed_state, tuple):
                    return False
                values = list(values)
                if len(values) != len(constants.ACTIONS) or not all(
                        isinstance(v, (int, float)) for v in values):
                    return False
                q_table[parsed_state] = values
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.q_table = q_table
            return True
        except (OSError, ValueError, SyntaxError, TypeError):
            return False
