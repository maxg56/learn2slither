"""environment.py - le board et les regles du jeu.

Gere l'etat du board 10x10, le placement et le deplacement du serpent,
l'apparition des pommes et la detection de game over. Ne contient AUCUNE
logique d'apprentissage : ne connait ni state ni reward.

Convention de coordonnees : (ligne, colonne), origine en haut a gauche.
La tete du serpent est self.snake[0].
"""

import random

from snakeai import constants


class Environment:
    """Board carre et regles du Snake."""

    def __init__(self, size=constants.BOARD_SIZE):
        self.size = size
        self.snake = []          # [(ligne, colonne), ...], tete en tete
        self.direction = None
        self.green_apples = []
        self.red_apples = []
        self.done = False
        self.reset()

    # -- Mise en place ------------------------------------------------------
    def reset(self):
        """Replace aleatoirement le serpent et les pommes, remet a zero."""
        self.done = False
        self.snake = []
        self.green_apples = []
        self.red_apples = []
        self._place_snake()
        for _ in range(constants.GREEN_APPLES):
            self._spawn_green()
        for _ in range(constants.RED_APPLES):
            self._spawn_red()

    def _place_snake(self):
        """Place un serpent de 3 cellules contigues et alignees."""
        while True:
            action = random.choice(constants.ACTIONS)
            dr, dc = constants.MOVES[action]
            head = (random.randrange(self.size),
                    random.randrange(self.size))
            # Le corps s'etend a l'oppose de la direction de la tete.
            cells = [(head[0] - i * dr, head[1] - i * dc)
                     for i in range(constants.SNAKE_START_LENGTH)]
            if all(self.in_bounds(cell) for cell in cells):
                self.snake = cells
                self.direction = action
                return

    def _spawn_green(self):
        """Fait apparaitre une pomme verte sur une case libre."""
        cell = self._random_empty_cell()
        if cell is not None:
            self.green_apples.append(cell)

    def _spawn_red(self):
        """Fait apparaitre une pomme rouge sur une case libre."""
        cell = self._random_empty_cell()
        if cell is not None:
            self.red_apples.append(cell)

    def _random_empty_cell(self):
        """Retourne une case libre au hasard, ou None si le board est plein."""
        occupied = set(self.snake)
        occupied.update(self.green_apples)
        occupied.update(self.red_apples)
        free = [(r, c)
                for r in range(self.size)
                for c in range(self.size)
                if (r, c) not in occupied]
        if not free:
            return None
        return random.choice(free)

    # -- Regles du jeu ------------------------------------------------------
    def step(self, action):
        """Applique une action et fait avancer le serpent d'une case.

        Retourne un evenement decrivant la transition (dict avec une cle
        "type" parmi : nothing, green, red, wall, collision, gameover),
        consomme ensuite par l'interpreter pour calculer le reward.
        """
        if self.done:
            return {"type": "gameover"}

        self.direction = action
        dr, dc = constants.MOVES[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # Collision avec un mur.
        if not self.in_bounds(new_head):
            self.done = True
            return {"type": "wall"}

        grows = new_head in self.green_apples
        # La queue se libere a chaque pas, sauf si le serpent grandit.
        body = self.snake if grows else self.snake[:-1]
        if new_head in body:
            self.done = True
            return {"type": "collision"}

        self.snake.insert(0, new_head)

        if grows:
            self.green_apples.remove(new_head)
            self._spawn_green()
            return {"type": "green"}

        if new_head in self.red_apples:
            self.red_apples.remove(new_head)
            self.snake.pop()   # deplacement normal
            self.snake.pop()   # retrait du a la pomme rouge
            self._spawn_red()
            # Une pomme rouge qui vide le serpent est un game over : on le
            # signale (fatal) pour que l'interpreter applique la penalite de
            # mort, et non le simple malus de pomme rouge.
            if len(self.snake) == 0:
                self.done = True
                return {"type": "red", "fatal": True}
            return {"type": "red"}

        self.snake.pop()
        return {"type": "nothing"}

    def in_bounds(self, cell):
        """Indique si une case (ligne, colonne) est a l'interieur du board.

        Publique : l'interpreter s'en sert pour tracer ses rayons de vision
        sans avoir a connaitre la representation interne du board.
        """
        r, c = cell
        return 0 <= r < self.size and 0 <= c < self.size

    def is_game_over(self):
        """Indique si la partie est terminee."""
        return self.done

    # -- Representation -----------------------------------------------------
    def get_board(self):
        """Retourne la grille courante sous forme de matrice de caracteres."""
        grid = [[constants.CELL_EMPTY] * self.size
                for _ in range(self.size)]
        for r, c in self.snake[1:]:
            grid[r][c] = constants.CELL_BODY
        for r, c in self.green_apples:
            grid[r][c] = constants.CELL_GREEN
        for r, c in self.red_apples:
            grid[r][c] = constants.CELL_RED
        if self.snake:
            hr, hc = self.snake[0]
            grid[hr][hc] = constants.CELL_HEAD
        return grid

    def render_board(self):
        """Retourne le board formate pour l'affichage terminal (debug)."""
        grid = self.get_board()
        return "\n".join(" ".join(row) for row in grid)
