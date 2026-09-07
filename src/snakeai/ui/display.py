"""display.py - interface graphique pygame (optionnelle).

Rendu du board, du serpent et des pommes, avec vitesse configurable et
mode pas a pas. La logique de jeu ne depend jamais de ce module : il est
entierement desactivable via `-visual off`.
"""

import sys

import pygame

from snakeai import constants

# Nombre maximum de frames accumulees en memoire pour l'export GIF. Au-dela,
# les frames excedentaires sont sous-echantillonnees (une frame sur N) afin
# de garder une empreinte memoire raisonnable meme sur de longues parties.
GIF_MAX_FRAMES = 600

# Couleurs (R, V, B).
COLOR_BG = (18, 18, 24)
COLOR_GRID = (34, 34, 44)
COLOR_HEAD = (80, 220, 120)
COLOR_BODY = (40, 160, 90)
COLOR_GREEN_APPLE = (60, 200, 60)
COLOR_RED_APPLE = (210, 60, 60)


class Display:
    """Rendu pygame du jeu."""

    def __init__(self, size=constants.BOARD_SIZE, cell_pixels=40, fps=10,
                 export_path=None):
        self.size = size
        self.cell_pixels = cell_pixels
        self.fps = fps
        self.quit = False
        # Export GIF (optionnel) : accumule les frames dessinees par
        # render() en memoire, puis les assemble dans save_gif().
        self.export_path = export_path
        self._frames = []
        self._frame_stride = 1
        self._frame_tick = 0
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
        if self.export_path:
            self._capture_frame()
        self.clock.tick(self.fps)

    def _capture_frame(self):
        """Ajoute la frame courante au buffer d'export, avec sous-
        echantillonnage automatique si le nombre de frames devient trop
        grand (parties/sessions longues)."""
        self._frame_tick += 1
        if (self._frame_tick - 1) % self._frame_stride != 0:
            return
        frame = pygame.surfarray.array3d(self.screen).copy()
        self._frames.append(frame)
        if len(self._frames) >= GIF_MAX_FRAMES:
            self._frames = self._frames[::2]
            self._frame_stride *= 2

    def save_gif(self):
        """Assemble les frames accumulees en GIF anime a `export_path`.

        Ne fait rien si aucun export n'est demande ou si aucune frame n'a
        ete capturee. Ne crashe jamais : affiche un avertissement clair
        sur stderr si Pillow est absent ou si l'ecriture echoue.
        """
        if not self.export_path:
            return
        if not self._frames:
            print("Avertissement : aucune frame capturee, export GIF "
                  "annule ({})".format(self.export_path), file=sys.stderr)
            return
        try:
            from PIL import Image
        except ImportError:
            print("Avertissement : Pillow n'est pas installe, export GIF "
                  "ignore (pip install pillow) ; cible = {}"
                  .format(self.export_path), file=sys.stderr)
            return
        try:
            duration_ms = max(1000 // self.fps, 20)
            images = [
                Image.fromarray(frame.transpose(1, 0, 2), "RGB")
                for frame in self._frames
            ]
            images[0].save(
                self.export_path,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
            )
            print("GIF exporte dans {} ({} frames)"
                  .format(self.export_path, len(images)))
        except Exception as error:      # pragma: no cover - depend de l'env
            print("Avertissement : echec de l'export GIF dans {} ({})"
                  .format(self.export_path, error), file=sys.stderr)

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
        """Ferme la fenetre graphique proprement (et exporte le GIF)."""
        self.save_gif()
        pygame.quit()
