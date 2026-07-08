"""dashboard.py - vue parallele "dashboard geant" (pygame, optionnelle).

Affiche une grille de nombreux boards qui jouent EN MEME TEMPS, tous
alimentant la meme Q-table (Agent partage). On voit ainsi defiler des
dizaines de generations simultanement et l'entrainement est accelere.

Purement cosmetique : la logique de jeu et d'apprentissage vit ailleurs.
Rendu volontairement minimaliste (des carres), desactivable.
"""

import os
from collections import deque

import pygame

import constants

# Couleurs (R, V, B) - minimaliste.
COLOR_BG = (12, 12, 16)
COLOR_PANEL = (22, 22, 30)
COLOR_BOARD_BG = (28, 28, 36)
COLOR_HEAD = (90, 230, 130)
COLOR_BODY = (45, 150, 90)
COLOR_GREEN_APPLE = (70, 200, 70)
COLOR_RED_APPLE = (215, 70, 70)
COLOR_TEXT = (220, 220, 230)
COLOR_DIM = (140, 140, 155)
COLOR_LEADER = (240, 210, 80)
COLOR_SELECT = (90, 200, 240)

SIDEBAR = 240
HEADER = 44
GAP = 6


class Dashboard:
    """Grille de boards paralleles partageant un meme Agent."""

    def __init__(self, agent, interp, cols=6, rows=5,
                 board_size=constants.BOARD_SIZE, cell_px=8, learn=True,
                 save_path=None):
        self.agent = agent
        self.interp = interp
        self.cols = cols
        self.rows = rows
        self.count = cols * rows
        self.board_size = board_size
        self.cell_px = cell_px
        self.learn = learn
        self.save_path = save_path or "models/model.txt"
        self._saved_epsilon = agent.epsilon

        self.quit = False
        self.paused = False
        self.steps_per_frame = 1
        # Un modele charge (dontlearn) n'evolue plus : on met d'emblee une
        # partie en grand pour l'observer, plutot que la grille entiere.
        self.focus = not learn
        self.locked_index = None    # None = suit le leader ; int = fige 1
        self.status = ""
        self.status_frames = 0

        # Nombre de pas sans manger, par board, pour l'anti-blocage.
        self.stall = [0] * self.count
        self.stall_limit = (board_size * board_size
                            * constants.STALL_STEPS_FACTOR)

        # Un environnement par case de la grille.
        from environment import Environment
        self.envs = [Environment(board_size) for _ in range(self.count)]
        self.states = [interp.get_state(env) for env in self.envs]
        self.cur_max_len = [len(env.snake) for env in self.envs]
        self.durations = [0] * self.count

        # Statistiques globales.
        self.episodes = 0
        self.best_length = max(self.cur_max_len)
        self.best_duration = 0
        self.recent_lengths = deque(maxlen=200)

        pygame.init()
        board_px = board_size * cell_px + GAP
        # Les boutons de la sidebar vivent sous les stats : la fenetre doit
        # etre assez haute pour les contenir, meme si la grille est petite.
        self.btn_y = HEADER + 16 + 7 * 46 + 8
        sidebar_needed = self.btn_y + 2 * 40 + 12
        width = SIDEBAR + cols * board_px + GAP
        height = max(HEADER + rows * board_px + GAP, sidebar_needed)
        self.board_px = board_px
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Learn2Slither - dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15)
        self.small = pygame.font.SysFont("monospace", 13)
        self._layout_buttons()

    def _layout_buttons(self):
        """Positionne les boutons cliquables de la sidebar."""
        bx = 14
        y = self.btn_y
        self.btn_prev = pygame.Rect(bx, y, 46, 30)
        self.btn_next = pygame.Rect(bx + 52, y, 46, 30)
        self.btn_best = pygame.Rect(bx + 104, y, 108, 30)
        self.btn_save = pygame.Rect(bx, y + 40, 100, 30)
        self.btn_freeze = pygame.Rect(bx + 106, y + 40, 106, 30)

    # -- Boucle ------------------------------------------------------------
    def run(self):
        """Boucle principale : anime tous les boards jusqu'a fermeture."""
        while not self.quit:
            self._pump()
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    for i in range(self.count):
                        self._step_board(i)
            self._draw()
            self.clock.tick(30)
        pygame.quit()

    def _step_board(self, i):
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
            self.best_duration = max(self.best_duration, self.durations[i])
            self.recent_lengths.append(self.cur_max_len[i])
            self.episodes += 1
            if self.learn:
                self.agent.decay_epsilon()
            env.reset()
            self.states[i] = self.interp.get_state(env)
            self.cur_max_len[i] = len(env.snake)
            self.durations[i] = 0
            self.stall[i] = 0
        else:
            self.states[i] = next_state

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
        if self.btn_save.collidepoint(pos):
            self._save()
        elif self.btn_prev.collidepoint(pos):
            self._cycle_focus(-1)
        elif self.btn_next.collidepoint(pos):
            self._cycle_focus(1)
        elif self.btn_best.collidepoint(pos):
            self._focus_best()
        elif self.btn_freeze.collidepoint(pos):
            self._toggle_freeze()
        else:
            idx = self._board_at(*pos)
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
        elif key == pygame.K_s:
            self._save()

    def _cycle_focus(self, delta):
        """Passe le spotlight a la partie precedente/suivante de la grille."""
        self.focus = True
        if self.locked_index is None:
            self.locked_index = self._best_index()
        self.locked_index = (self.locked_index + delta) % self.count

    def _focus_best(self):
        """Remet le spotlight sur la meilleure partie (leve le verrou)."""
        self.focus = True
        self.locked_index = None

    def _toggle_freeze(self):
        """Gele/reactive l'apprentissage (fige la generation courante)."""
        self.learn = not self.learn
        if not self.learn:
            self._saved_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0
            self._flash("Generation GELEE (epsilon=0, pas d'apprentissage)")
        else:
            self.agent.epsilon = self._saved_epsilon
            self._flash("Apprentissage REPRIS")

    def _save(self):
        """Enregistre le modele, nom tagge avec le nombre de generations."""
        root, ext = os.path.splitext(self.save_path)
        ext = ext or ".txt"
        path = "{}_gen{}{}".format(root, self.episodes, ext)
        if self.agent.save(path):
            self._flash("Sauvegarde : " + path)
            print("Modele sauvegarde dans {}".format(path))
        else:
            self._flash("Echec de la sauvegarde")

    def _flash(self, message):
        """Affiche un message temporaire dans le header."""
        self.status = message
        self.status_frames = 120

    def _board_at(self, mx, my):
        """Index du board sous le curseur, ou None hors zone."""
        col = (mx - SIDEBAR - GAP) // self.board_px
        row = (my - HEADER - GAP) // self.board_px
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return None
        side = self.board_size * self.cell_px
        lx = (mx - SIDEBAR - GAP) - col * self.board_px
        ly = (my - HEADER - GAP) - row * self.board_px
        if lx < 0 or ly < 0 or lx >= side or ly >= side:
            return None
        idx = row * self.cols + col
        return idx if idx < self.count else None

    def _best_index(self):
        """Index du board le plus performant en cours (longueur, duree)."""
        return max(range(self.count),
                   key=lambda i: (len(self.envs[i].snake), self.durations[i]))

    def _focus_index(self):
        """Index affiche en grand : board verrouille sinon le leader."""
        if self.locked_index is not None:
            return self.locked_index
        return self._best_index()

    # -- Rendu -------------------------------------------------------------
    def _draw(self):
        """Redessine le header, la sidebar et tous les boards."""
        self.screen.fill(COLOR_BG)
        self._draw_header()
        self._draw_sidebar()
        leader = self._best_index()
        side = self.board_size * self.cell_px
        for i, env in enumerate(self.envs):
            gx = SIDEBAR + (i % self.cols) * self.board_px + GAP
            gy = HEADER + (i // self.cols) * self.board_px + GAP
            self._draw_board_at(env, gx, gy, self.cell_px)
            frame = pygame.Rect(gx - 2, gy - 2, side + 4, side + 4)
            if i == self.locked_index:
                pygame.draw.rect(self.screen, COLOR_SELECT, frame, 2)
            elif i == leader:
                pygame.draw.rect(self.screen, COLOR_LEADER, frame, 2)
        if self.focus:
            self._draw_spotlight(self._focus_index())
        if self.status_frames > 0:
            self.status_frames -= 1
        pygame.display.flip()

    def _draw_board_at(self, env, gx, gy, px):
        """Dessine un board complet a la position et a l'echelle donnees."""
        side = self.board_size * px
        pygame.draw.rect(self.screen, COLOR_BOARD_BG,
                         pygame.Rect(gx, gy, side, side))
        cells = [(env.green_apples, COLOR_GREEN_APPLE),
                 (env.red_apples, COLOR_RED_APPLE),
                 (env.snake[1:], COLOR_BODY)]
        for group, color in cells:
            for r, c in group:
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(gx + c * px, gy + r * px, px, px))
        if env.snake:
            hr, hc = env.snake[0]
            pygame.draw.rect(self.screen, COLOR_HEAD,
                             pygame.Rect(gx + hc * px, gy + hr * px, px, px))

    def _draw_spotlight(self, index):
        """Affiche en grand, au centre, une seule partie (leader ou verrou)."""
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))

        env = self.envs[index]
        big_px = min(32, max(14, self.cell_px * 4))
        side = self.board_size * big_px
        pad = 14
        panel_w = side + 2 * pad
        panel_h = side + 2 * pad + 26
        px = (self.screen.get_width() - panel_w) // 2
        py = (self.screen.get_height() - panel_h) // 2
        locked = self.locked_index is not None
        color = COLOR_SELECT if locked else COLOR_LEADER
        pygame.draw.rect(self.screen, COLOR_PANEL,
                         pygame.Rect(px, py, panel_w, panel_h))
        pygame.draw.rect(self.screen, color,
                         pygame.Rect(px, py, panel_w, panel_h), 3)
        tag = "VERROUILLEE" if locked else "meilleure en cours"
        label = "Partie #{} ({})  long={}  duree={}".format(
            index, tag, len(env.snake), self.durations[index])
        self.screen.blit(self.small.render(label, True, COLOR_TEXT),
                         (px + pad, py + 6))
        self._draw_board_at(env, px + pad, py + pad + 24, big_px)

    def _draw_header(self):
        """Barre du haut : titre + etat pause/vitesse."""
        pygame.draw.rect(self.screen, COLOR_PANEL,
                         pygame.Rect(0, 0, self.screen.get_width(), HEADER))
        if self.status_frames > 0:
            title = self.status
            color = COLOR_LEADER
        else:
            title = "Learn2Slither  -  {} parties".format(self.count)
            color = COLOR_TEXT
        self.screen.blit(self.font.render(title, True, color), (12, 6))
        state = "PAUSE" if self.paused else "x{}".format(self.steps_per_frame)
        hint = ("[espace] pause  [+/-] vitesse  [fleches <>] partie  "
                "[B] meilleure  [L] geler  [S] save  " + state)
        self.screen.blit(self.small.render(hint, True, COLOR_DIM), (12, 26))

    def _draw_sidebar(self):
        """Panneau gauche : statistiques d'apprentissage globales."""
        pygame.draw.rect(self.screen, COLOR_PANEL,
                         pygame.Rect(0, HEADER, SIDEBAR,
                                     self.screen.get_height() - HEADER))
        if self.recent_lengths:
            avg = sum(self.recent_lengths) / len(self.recent_lengths)
        else:
            avg = 0.0
        lines = [
            ("Generations", str(self.episodes)),
            ("Epsilon", "{:.3f}".format(self.agent.epsilon)),
            ("Etats appris", str(len(self.agent.q_table))),
            ("Longueur max", str(self.best_length)),
            ("Duree max", str(self.best_duration)),
            ("Long. moy (200)", "{:.1f}".format(avg)),
            ("Apprentissage", "on" if self.learn else "off"),
        ]
        y = HEADER + 16
        for label, value in lines:
            self.screen.blit(self.small.render(label, True, COLOR_DIM),
                             (14, y))
            self.screen.blit(self.font.render(value, True, COLOR_TEXT),
                             (14, y + 16))
            y += 46
        self._draw_buttons()

    def _draw_buttons(self):
        """Dessine les boutons cliquables (navigation, sauvegarde, gel)."""
        self._draw_button(self.btn_prev, "<")
        self._draw_button(self.btn_next, ">")
        best_active = self.focus and self.locked_index is None
        self._draw_button(self.btn_best, "Meilleure", best_active)
        self._draw_button(self.btn_save, "Sauver")
        freeze_label = "Reprendre" if not self.learn else "Geler"
        self._draw_button(self.btn_freeze, freeze_label, not self.learn)

    def _draw_button(self, rect, label, active=False):
        """Dessine un bouton rectangulaire avec son libelle centre."""
        bg = COLOR_SELECT if active else COLOR_BOARD_BG
        pygame.draw.rect(self.screen, bg, rect, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_DIM, rect, 1, border_radius=4)
        txt = self.small.render(label, True, COLOR_TEXT)
        tx = rect.x + (rect.width - txt.get_width()) // 2
        ty = rect.y + (rect.height - txt.get_height()) // 2
        self.screen.blit(txt, (tx, ty))
