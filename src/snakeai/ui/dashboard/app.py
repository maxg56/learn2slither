"""app.py - orchestrateur du dashboard (pygame, optionnel).

Tient la boucle principale, l'etat d'affichage (spotlight, pause, vitesse)
et le traitement des entrees clavier/souris. Delegue le calcul du jeu a
`Simulation` et tout le dessin a `Renderer` : cette classe ne fait que les
coordonner.
"""

import pygame

from snakeai.ui.dashboard.renderer import Renderer
from snakeai.ui.dashboard.simulation import Simulation

# Duree d'affichage d'un message flash, en frames (~4 s a 30 fps).
FLASH_FRAMES = 120


class Dashboard:
    """Coordonne simulation et rendu d'une grille de parties paralleles."""

    def __init__(self, agent, interp, cols=6, rows=5,
                 board_size=None, cell_px=8, learn=True, save_path=None):
        if board_size is None:
            from snakeai import constants
            board_size = constants.BOARD_SIZE
        self.sim = Simulation(agent, interp, cols, rows,
                              board_size=board_size, learn=learn,
                              save_path=save_path)
        self.renderer = Renderer(cols, rows, board_size, cell_px)

        self.quit = False
        self.paused = False
        self.steps_per_frame = 1
        # Un modele charge (dontlearn) n'evolue plus : on met d'emblee une
        # partie en grand pour l'observer, plutot que la grille entiere.
        self.focus = not learn
        self.locked_index = None    # None = suit le leader ; int = fige 1
        self.show_stats = False     # ecran de courbes d'entrainement (live)
        self.status = ""
        self.status_frames = 0
        self.clock = pygame.time.Clock()

    # -- Statistiques exposees a la CLI ------------------------------------
    @property
    def best_length(self):
        return self.sim.best_length

    @property
    def best_duration(self):
        return self.sim.best_duration

    # -- Boucle ------------------------------------------------------------
    def run(self):
        """Boucle principale : anime tous les boards jusqu'a fermeture."""
        while not self.quit:
            self._pump()
            if not self.paused:
                self.sim.step_all(self.steps_per_frame)
            self.renderer.draw(self)
            if self.status_frames > 0:
                self.status_frames -= 1
            self.clock.tick(30)
        pygame.quit()

    def focus_index(self):
        """Index affiche en grand : board verrouille sinon le leader."""
        if self.locked_index is not None:
            return self.locked_index
        return self.sim.best_index()

    # -- Evenements --------------------------------------------------------
    def _pump(self):
        """Traite le clavier et la souris : pause, vitesse, focus, save."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                self._on_key(event.key)
            elif (event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1):
                self._on_click(event.pos)

    def _on_click(self, pos):
        """Dispatche un clic gauche : bouton de la sidebar ou board."""
        r = self.renderer
        if r.btn_save.collidepoint(pos):
            self._save()
        elif r.btn_prev.collidepoint(pos):
            self._cycle_focus(-1)
        elif r.btn_next.collidepoint(pos):
            self._cycle_focus(1)
        elif r.btn_best.collidepoint(pos):
            self._focus_best()
        elif r.btn_freeze.collidepoint(pos):
            self._toggle_freeze()
        elif r.btn_stats.collidepoint(pos):
            self.show_stats = not self.show_stats
        else:
            idx = r.board_at(*pos)
            if idx is not None:
                self.locked_index = idx
                self.focus = True

    def _on_key(self, key):
        """Reagit a une touche."""
        if key == pygame.K_ESCAPE:
            self.quit = True
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
            self.steps_per_frame = min(200, self.steps_per_frame + 1)
        elif key in (pygame.K_MINUS, pygame.K_DOWN):
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif key in (pygame.K_b, pygame.K_TAB):
            # Bascule le spotlight sur la MEILLEURE partie (leve le verrou).
            if self.focus and self.locked_index is None:
                self.focus = False
            else:
                self.focus = True
                self.locked_index = None
        elif key == pygame.K_LEFT:
            self._cycle_focus(-1)
        elif key == pygame.K_RIGHT:
            self._cycle_focus(1)
        elif key == pygame.K_l:
            self._toggle_freeze()
        elif key == pygame.K_g:
            self.show_stats = not self.show_stats
        elif key == pygame.K_s:
            self._save()

    def _cycle_focus(self, delta):
        """Passe le spotlight a la partie precedente/suivante de la grille."""
        self.focus = True
        if self.locked_index is None:
            self.locked_index = self.sim.best_index()
        self.locked_index = (self.locked_index + delta) % self.sim.count

    def _focus_best(self):
        """Remet le spotlight sur la meilleure partie (leve le verrou)."""
        self.focus = True
        self.locked_index = None

    def _toggle_freeze(self):
        """Gele/reactive l'apprentissage (fige la generation courante)."""
        if self.sim.toggle_learn():
            self._flash("Apprentissage REPRIS")
        else:
            self._flash("Generation GELEE (epsilon=0, pas d'apprentissage)")

    def _save(self):
        """Enregistre le modele, nom tagge avec le nombre de generations."""
        path = self.sim.save()
        if path is not None:
            self._flash("Sauvegarde : " + path)
            print("Modele sauvegarde dans {}".format(path))
        else:
            self._flash("Echec de la sauvegarde")

    def _flash(self, message):
        """Affiche un message temporaire dans le header."""
        self.status = message
        self.status_frames = FLASH_FRAMES
