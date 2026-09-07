"""interpreter.py - vision du serpent et calcul des rewards.

Seul module autorise a lire le board. Extrait les 4 rayons depuis la tete,
les reduit en features compactes et independantes de la taille du board,
et calcule la recompense de la transition. L'agent ne recoit QUE cette
vision (jamais les coordonnees absolues ni la grille complete).
"""

from snakeai import constants


class Interpreter:
    """Traduit le board en etat (vision) et en reward."""

    def __init__(self, reward_mode="default"):
        """Cree l'interpreter.

        reward_mode : "default" (comportement historique, inchange) ou "alt"
        (schema alternatif : anti demi-tour + bonus de survie, cf.
        constants.py). "default" reste le comportement si non precise.
        """
        if reward_mode not in ("default", "alt"):
            raise ValueError(
                "reward_mode invalide : {!r} (attendu 'default' ou 'alt')"
                .format(reward_mode))
        self.reward_mode = reward_mode
        # Direction du serpent et reference vers l'env, captures juste avant
        # env.step() (cf. green_distance) pour detecter un demi-tour au
        # prochain get_reward() : uniquement utilise en mode "alt".
        self._prev_direction = None
        self._env = None

    def _cell_char(self, env, cell):
        """Caractere d'affichage d'une case (hors tete)."""
        if cell in env.green_apples:
            return constants.CELL_GREEN
        if cell in env.red_apples:
            return constants.CELL_RED
        if cell in env.snake:
            return constants.CELL_BODY
        return constants.CELL_EMPTY

    def _ray(self, env, action):
        """Liste des cases vues depuis la tete dans une direction.

        Part de la case adjacente a la tete et avance jusqu'au mur inclus
        (le mur ferme toujours le rayon).
        """
        dr, dc = constants.MOVES[action]
        r, c = env.snake[0]
        cells = []
        r, c = r + dr, c + dc
        while env.in_bounds((r, c)):
            cells.append(self._cell_char(env, (r, c)))
            r, c = r + dr, c + dc
        cells.append(constants.CELL_WALL)
        return cells

    def get_vision(self, env):
        """Retourne les 4 rayons depuis la tete (dict par direction)."""
        return {action: self._ray(env, action)
                for action in constants.ACTIONS}

    @staticmethod
    def _first_distance(ray, char):
        """Distance en cases jusqu'a la 1re occurrence de char sur un rayon.

        None si char est absent du rayon. La case adjacente a la tete est a
        distance 1 (le rayon exclut la tete).
        """
        for i, cell in enumerate(ray):
            if cell == char:
                return i + 1
        return None

    def get_state(self, env):
        """Reduit la vision en une cle hashable pour la Q-table.

        Par direction : danger adjacent (mur ou corps), pomme verte visible,
        pomme rouge visible. Etat = tuple de 12 bits, independant de la taille
        du board. Le gradient vers la nourriture est fourni par le reward
        shaping (approach_bonus), pas par l'etat : coder la distance ici
        multiplie l'espace d'etats sans gain de perf mesurable.
        """
        features = []
        for action in constants.ACTIONS:
            ray = self._ray(env, action)
            danger = int(ray[0] in (constants.CELL_WALL,
                                    constants.CELL_BODY))
            green = int(constants.CELL_GREEN in ray)
            red = int(constants.CELL_RED in ray)
            features.extend((danger, green, red))
        return tuple(features)

    def green_distance(self, env):
        """Distance en cases jusqu'a la pomme verte visible la plus proche.

        Balaie les 4 rayons depuis la tete ; retourne le plus petit nombre de
        cases separant la tete d'une pomme verte alignee, ou None si aucune
        pomme verte n'est visible. Purement derive de la vision.
        """
        if not env.snake:
            return None
        # Appele juste avant env.step() dans le trainer/dashboard : c'est le
        # point d'entree ou l'on capture la direction "avant coup" et la
        # reference de l'env, pour que get_reward() puisse ensuite detecter
        # un demi-tour (mode "alt" uniquement, cf. _uturn_penalty).
        self._prev_direction = env.direction
        self._env = env
        best = None
        for action in constants.ACTIONS:
            ray = self._ray(env, action)
            distance = self._first_distance(ray, constants.CELL_GREEN)
            if distance is not None and (best is None or distance < best):
                best = distance
        return best

    def approach_bonus(self, dist_before, dist_after, event_type):
        """Bonus de reward pour le rapprochement d'une pomme verte visible.

        Applique seulement sur un deplacement simple (event "nothing") ou une
        pomme verte reste visible avant et apres : positif si le serpent s'est
        rapproche, negatif s'il s'est eloigne, nul a distance constante.
        Les pas ou l'on mange/meurt sont deja geres par les rewards de base.
        """
        if event_type != "nothing":
            return 0.0
        if dist_before is None or dist_after is None:
            return 0.0
        return constants.REWARD_APPROACH * (dist_before - dist_after)

    @staticmethod
    def _is_opposite(direction_a, direction_b):
        """Vrai si deux directions sont exactement opposees."""
        if direction_a is None or direction_b is None:
            return False
        dr_a, dc_a = constants.MOVES[direction_a]
        dr_b, dc_b = constants.MOVES[direction_b]
        return dr_a == -dr_b and dc_a == -dc_b

    def _uturn_penalty(self):
        """Penalite anti demi-tour (mode "alt" uniquement).

        Compare la direction capturee juste avant env.step() (self._prev_
        direction) a la direction courante de l'env juste apres (self._env.
        direction a deja ete mise a jour par env.step() avec l'action
        choisie). Se limite a la derniere action de l'agent : conforme a la
        contrainte "vision seule" (cf. constants.REWARD_UTURN).
        """
        if self.reward_mode != "alt" or self._env is None:
            return 0.0
        if self._is_opposite(self._env.direction, self._prev_direction):
            return constants.REWARD_UTURN
        return 0.0

    def _nothing_reward(self):
        """Reward d'un pas "nothing" : ajoute le bonus de survie en "alt"."""
        if self.reward_mode == "alt":
            return constants.REWARD_NOTHING + constants.REWARD_SURVIVAL_BONUS
        return constants.REWARD_NOTHING

    def get_reward(self, event):
        """Calcule la recompense associee a une transition de l'env."""
        kind = event["type"]
        if kind == "green":
            reward = constants.REWARD_GREEN
        elif kind == "red":
            # Une pomme rouge qui reduit le serpent a zero est une mort :
            # elle est penalisee comme un game over, pas comme un simple malus.
            if event.get("fatal"):
                reward = constants.REWARD_GAMEOVER
            else:
                reward = constants.REWARD_RED
        elif kind in ("wall", "collision", "gameover"):
            reward = constants.REWARD_GAMEOVER
        else:
            reward = self._nothing_reward()
        if self.reward_mode == "alt":
            reward += self._uturn_penalty()
        return reward

    def render_vision(self, env):
        """Affiche la vision en croix (W/H/S/G/R/0) avant chaque action."""
        size = env.size
        hr, hc = env.snake[0]
        canvas = [[" "] * (size + 2) for _ in range(size + 2)]
        canvas[hr + 1][0] = constants.CELL_WALL
        canvas[hr + 1][size + 1] = constants.CELL_WALL
        canvas[0][hc + 1] = constants.CELL_WALL
        canvas[size + 1][hc + 1] = constants.CELL_WALL
        for c in range(size):
            canvas[hr + 1][c + 1] = self._cell_char(env, (hr, c))
        for r in range(size):
            canvas[r + 1][hc + 1] = self._cell_char(env, (r, hc))
        canvas[hr + 1][hc + 1] = constants.CELL_HEAD
        return "\n".join(" ".join(row).rstrip() for row in canvas)
