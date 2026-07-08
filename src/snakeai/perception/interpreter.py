"""interpreter.py - vision du serpent et calcul des rewards.

Seul module autorise a lire le board. Extrait les 4 rayons depuis la tete,
les reduit en features compactes et independantes de la taille du board,
et calcule la recompense de la transition. L'agent ne recoit QUE cette
vision (jamais les coordonnees absolues ni la grille complete).
"""

from snakeai import constants


class Interpreter:
    """Traduit le board en etat (vision) et en reward."""

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

    def get_reward(self, event):
        """Calcule la recompense associee a une transition de l'env."""
        kind = event["type"]
        if kind == "green":
            return constants.REWARD_GREEN
        if kind == "red":
            # Une pomme rouge qui reduit le serpent a zero est une mort :
            # elle est penalisee comme un game over, pas comme un simple malus.
            if event.get("fatal"):
                return constants.REWARD_GAMEOVER
            return constants.REWARD_RED
        if kind in ("wall", "collision", "gameover"):
            return constants.REWARD_GAMEOVER
        return constants.REWARD_NOTHING

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
