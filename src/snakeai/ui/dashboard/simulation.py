"""simulation.py - avance une grille de boards partageant un Agent.

Cur du dashboard, sans aucune dependance pygame : cette classe fait tourner
`count` parties en parallele sur la meme Q-table et cumule les statistiques
globales. Testable et reutilisable independamment de l'affichage.
"""

import os
from collections import deque

from snakeai import constants
from snakeai.core import Environment


class Simulation:
    """Grille de parties paralleles alimentant un meme Agent."""

    def __init__(self, agent, interp, cols, rows,
                 board_size=constants.BOARD_SIZE, learn=True, save_path=None):
        self.agent = agent
        self.interp = interp
        self.cols = cols
        self.rows = rows
        self.count = cols * rows
        self.board_size = board_size
        self.learn = learn
        self.save_path = save_path or "models/model.txt"
        self._saved_epsilon = agent.epsilon

        # Nombre de pas sans manger, par board, pour l'anti-blocage.
        self.stall = [0] * self.count
        self.stall_limit = (board_size * board_size
                            * constants.STALL_STEPS_FACTOR)

        # Un environnement par case de la grille.
        self.envs = [Environment(board_size) for _ in range(self.count)]
        self.states = [interp.get_state(env) for env in self.envs]
        self.cur_max_len = [len(env.snake) for env in self.envs]
        self.durations = [0] * self.count

        # Statistiques globales.
        self.episodes = 0
        self.best_length = max(self.cur_max_len)
        self.best_duration = 0
        self.recent_lengths = deque(maxlen=200)

    # -- Avancement --------------------------------------------------------
    def step_all(self, steps_per_frame):
        """Fait avancer toutes les parties de `steps_per_frame` pas."""
        for _ in range(steps_per_frame):
            for i in range(self.count):
                self.step_board(i)

    def step_board(self, i):
        """Fait avancer un board d'une action et gere le game over."""
        env = self.envs[i]
        state = self.states[i]
        action = self.agent.choose_action(state)
        dist_before = self.interp.green_distance(env)
        event = env.step(action)
        reward = self.interp.get_reward(event)
        dist_after = (self.interp.green_distance(env)
                      if event["type"] == "nothing" else None)
        reward += self.interp.approach_bonus(
            dist_before, dist_after, event["type"])
        done = env.is_game_over()

        # Anti-blocage : on remet le compteur a zero quand le serpent mange,
        # sinon on force la fin de partie s'il tourne en rond trop longtemps.
        if event["type"] == "green":
            self.stall[i] = 0
        else:
            self.stall[i] += 1
        if not done and self.stall[i] >= self.stall_limit:
            done = True

        next_state = None if done else self.interp.get_state(env)
        if self.learn:
            self.agent.update(state, action, reward, next_state)

        self.durations[i] += 1
        length = len(env.snake)
        self.cur_max_len[i] = max(self.cur_max_len[i], length)
        self.best_length = max(self.best_length, length)

        if done:
            self._end_episode(i)
        else:
            self.states[i] = next_state

    def _end_episode(self, i):
        """Cloture une partie : cumule les stats et relance un board neuf."""
        self.best_duration = max(self.best_duration, self.durations[i])
        self.recent_lengths.append(self.cur_max_len[i])
        self.episodes += 1
        if self.learn:
            self.agent.decay_epsilon()
        env = self.envs[i]
        env.reset()
        self.states[i] = self.interp.get_state(env)
        self.cur_max_len[i] = len(env.snake)
        self.durations[i] = 0
        self.stall[i] = 0

    # -- Requetes ----------------------------------------------------------
    def best_index(self):
        """Index du board le plus performant en cours (longueur, duree)."""
        return max(range(self.count),
                   key=lambda i: (len(self.envs[i].snake), self.durations[i]))

    def recent_average(self):
        """Longueur moyenne des dernieres parties (0.0 si aucune)."""
        if not self.recent_lengths:
            return 0.0
        return sum(self.recent_lengths) / len(self.recent_lengths)

    def survival_rate(self):
        """Fraction des dernieres parties ayant grandi (0.0 si aucune).

        Une partie compte comme "survie" si sa longueur maximale atteinte a
        depasse la longueur de depart (SNAKE_START_LENGTH), c'est a dire
        qu'elle a mange au moins une pomme verte nette avant de terminer.
        Proxy simple, calcule sur la meme fenetre que `recent_lengths`.
        """
        if not self.recent_lengths:
            return 0.0
        threshold = constants.SNAKE_START_LENGTH + 1
        grown = sum(1 for length in self.recent_lengths if length >= threshold)
        return grown / len(self.recent_lengths)

    # -- Apprentissage / persistance --------------------------------------
    def sync_epsilon(self):
        """Resynchronise l'epsilon fige apres un chargement externe.

        A appeler quand le modele de `agent` a ete remplace hors de cette
        classe (ex : lobby de selection), pour que `toggle_learn` restaure
        la bonne valeur au degel.
        """
        self._saved_epsilon = self.agent.epsilon

    def toggle_learn(self):
        """Gele/reactive l'apprentissage et retourne l'etat resultant."""
        if self.learn:
            self._saved_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0
            self.learn = False
        else:
            self.agent.epsilon = self._saved_epsilon
            self.learn = True
        return self.learn

    def save(self):
        """Enregistre le modele (nom tagge du nombre de generations).

        Retourne le chemin ecrit en cas de succes, None sinon.
        """
        root, ext = os.path.splitext(self.save_path)
        ext = ext or ".txt"
        path = "{}_gen{}{}".format(root, self.episodes, ext)
        return path if self.agent.save(path) else None
