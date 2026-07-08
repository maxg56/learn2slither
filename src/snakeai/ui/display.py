"""display.py - interface graphique pygame (optionnelle).

Rendu du board, du serpent et des pommes, avec vitesse configurable et
mode pas a pas. La logique de jeu ne depend jamais de ce module : il est
entierement desactivable via `-visual off`.
"""

import pygame

from snakeai import constants

# Couleurs (R, V, B).
COLOR_BG = (18, 18, 24)
COLOR_GRID = (34, 34, 44)
COLOR_HEAD = (80, 220, 120)
COLOR_BODY = (40, 160, 90)
COLOR_GREEN_APPLE = (60, 200, 60)
COLOR_RED_APPLE = (210, 60, 60)


class Display:
    """Rendu pygame du jeu."""

    def __init__(self, size=constants.BOARD_SIZE, cell_pixels=40, fps=10):
        self.size = size
        self.cell_pixels = cell_pixels
        self.fps = fps
        self.quit = False
        pygame.init()
        side = size * cell_pixels
        self.screen = pygame.display.set_mode((side, side))
        pygame.display.set_caption("Learn2Slither")
        self.clock = pygame.time.Clock()

    def _pump(self):
        """Traite les evenements ; memorise une demande de fermeture."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif (event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE):
                self.quit = True

    def _draw_cell(self, row, col, color):
        """Dessine une cellule pleine a la position (ligne, colonne)."""
        px = self.cell_pixels
        rect = pygame.Rect(col * px, row * px, px, px)
        pygame.draw.rect(self.screen, color, rect)

    def render(self, env):
        """Dessine l'etat courant du board."""
        self._pump()
        self.screen.fill(COLOR_BG)
        px = self.cell_pixels
        for i in range(1, self.size):
            pygame.draw.line(self.screen, COLOR_GRID,
                             (i * px, 0), (i * px, self.size * px))
            pygame.draw.line(self.screen, COLOR_GRID,
                             (0, i * px), (self.size * px, i * px))
        for r, c in env.green_apples:
            self._draw_cell(r, c, COLOR_GREEN_APPLE)
        for r, c in env.red_apples:
            self._draw_cell(r, c, COLOR_RED_APPLE)
        for r, c in env.snake[1:]:
            self._draw_cell(r, c, COLOR_BODY)
        if env.snake:
            hr, hc = env.snake[0]
            self._draw_cell(hr, hc, COLOR_HEAD)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def wait_step(self):
        """Bloque jusqu'a la touche suivante en mode pas a pas."""
        waiting = True
        while waiting and not self.quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit = True
                    else:
                        waiting = False
            self.clock.tick(30)

    def should_quit(self):
        """Indique si l'utilisateur a demande la fermeture."""
        return self.quit

    def close(self):
        """Ferme la fenetre graphique proprement."""
        pygame.quit()
