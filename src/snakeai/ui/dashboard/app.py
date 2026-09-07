"""app.py - orchestrateur du dashboard (pygame, optionnel).

Tient la boucle principale, l'etat d'affichage (spotlight, pause, vitesse,
lobby, panneau de config) et le traitement des entrees clavier/souris.
Delegue le calcul du jeu a `Simulation` et tout le dessin a `Renderer` :
cette classe ne fait que les coordonner.
"""

import glob

import pygame

from snakeai.ui.dashboard.renderer import Renderer
from snakeai.ui.dashboard.simulation import Simulation

# Duree d'affichage d'un message flash, en frames (~4 s a 30 fps).
FLASH_FRAMES = 120

# Pas d'ajustement manuel de epsilon depuis le panneau de config.
EPSILON_STEP = 0.01


class Dashboard:
    """Coordonne simulation et rendu d'une grille de parties paralleles."""

    def __init__(self, agent, interp, cols=6, rows=5,
                 board_size=None, cell_px=8, learn=True, save_path=None,
                 start_lobby=False):
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
        self.status = ""
        self.status_frames = 0
        self.show_config = False
        self.clock = pygame.time.Clock()

        # -- Lobby de selection de modele (optionnel, voir `start_lobby`) --
        # Etat separe du reste : tant que `stage == "lobby"`, la grille ne
        # tourne pas et seul l'ecran de choix est dessine. La simulation et
        # le renderer existent deja (la taille de board n'y est pas
        # modifiable depuis le lobby, voir la note dans le README/PR) ; le
        # lobby se contente de (re)charger le modele dans l'agent partage.
        self.stage = "lobby" if start_lobby else "running"
        self.lobby_index = 0
        self._lobby_models = (self._scan_models() if start_lobby else [])

    @staticmethod
    def _scan_models():
        """Liste les modeles sauvegardes disponibles sous `models/`."""
        paths = glob.glob("models/*.txt") + glob.glob("models/*.json")
        return sorted(paths)

    # -- Statistiques exposees a la CLI ------------------------------------
    @property
    def best_length(self):
        return self.sim.best_length

    @property
    def best_duration(self):
        return self.sim.best_duration

    # -- Boucle ------------------------------------------------------------
    def run(self):
        """Boucle principale : lobby puis grille, jusqu'a fermeture."""
        while not self.quit:
            self._pump()
            if self.stage == "lobby":
                self.renderer.draw_lobby(self)
            else:
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

    # -- Lobby ---------------------------------------------------------
    def lobby_entries(self):
        """Libelles affiches dans le lobby (nouveau modele + fichiers)."""
        return ["(nouveau modele, sans apprentissage prealable)"] \
            + self._lobby_models

    def _confirm_lobby(self):
        """Charge le modele choisi (ou demarre a neuf) et lance la grille."""
        if self.lobby_index == 0 or not self._lobby_models:
            self._skip_lobby()
            return
        path = self._lobby_models[self.lobby_index - 1]
        if self.sim.agent.load(path):
            self.sim.sync_epsilon()
            self._flash("Modele charge : " + path)
        else:
            self._flash("Echec du chargement : " + path)
        self.stage = "running"

    def _skip_lobby(self):
        """Demarre la grille sans charger de modele (agent tel quel)."""
        self.stage = "running"

    # -- Evenements --------------------------------------------------------
    def _pump(self):
        """Traite le clavier et la souris : pause, vitesse, focus, save."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                if self.stage == "lobby":
                    self._on_key_lobby(event.key)
                else:
                    self._on_key(event.key)
            elif (event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1 and self.stage == "running"):
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
        else:
            idx = r.board_at(*pos)
            if idx is not None:
                self.locked_index = idx
                self.focus = True

    def _on_key_lobby(self, key):
        """Reagit a une touche pendant le lobby de selection de modele."""
        n = len(self._lobby_models) + 1
        if key == pygame.K_UP:
            self.lobby_index = (self.lobby_index - 1) % n
        elif key == pygame.K_DOWN:
            self.lobby_index = (self.lobby_index + 1) % n
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._confirm_lobby()
        elif key == pygame.K_ESCAPE:
            self._skip_lobby()

    def _on_key(self, key):
        """Reagit a une touche pendant la grille en cours d'execution."""
        if key == pygame.K_ESCAPE:
            self.quit = True
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.steps_per_frame = min(200, self.steps_per_frame + 1)
        elif key == pygame.K_MINUS:
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif key == pygame.K_UP:
            # Panneau de config ouvert : les fleches ajustent epsilon.
            # Sinon (comportement d'origine) elles jouent le role de +/-.
            if self.show_config:
                self._nudge_epsilon(EPSILON_STEP)
            else:
                self.steps_per_frame = min(200, self.steps_per_frame + 1)
        elif key == pygame.K_DOWN:
            if self.show_config:
                self._nudge_epsilon(-EPSILON_STEP)
            else:
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
        elif key == pygame.K_s:
            self._save()
        elif key == pygame.K_c:
            self.show_config = not self.show_config

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

    def _nudge_epsilon(self, delta):
        """Ajuste manuellement epsilon depuis le panneau de config."""
        agent = self.sim.agent
        agent.epsilon = min(1.0, max(0.0, agent.epsilon + delta))
        self.sim.sync_epsilon()

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
