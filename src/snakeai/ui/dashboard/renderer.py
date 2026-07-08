"""renderer.py - rendu pygame du dashboard.

Possede la fenetre, les polices et toute la geometrie (grille de boards,
boutons de la sidebar). Lit l'etat a afficher depuis l'orchestrateur mais
ne le modifie jamais : le rendu est une projection, pas une source de verite.
"""

import pygame

from snakeai.ui.dashboard import theme
from snakeai.ui.dashboard.theme import GAP, HEADER, SIDEBAR


class Renderer:
    """Dessine header, sidebar et grille de boards ; gere la geometrie."""

    def __init__(self, cols, rows, board_size, cell_px):
        self.cols = cols
        self.rows = rows
        self.count = cols * rows
        self.board_size = board_size
        self.cell_px = cell_px
        self.board_px = board_size * cell_px + GAP

        pygame.init()
        # Les boutons vivent sous les 7 lignes de stats : la fenetre doit
        # etre assez haute pour les contenir, meme si la grille est petite.
        self.btn_y = HEADER + 16 + 7 * 46 + 8
        sidebar_needed = self.btn_y + 2 * 40 + 12
        width = SIDEBAR + cols * self.board_px + GAP
        height = max(HEADER + rows * self.board_px + GAP, sidebar_needed)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Learn2Slither - dashboard")
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

    # -- Geometrie ---------------------------------------------------------
    def board_at(self, mx, my):
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

    # -- Rendu -------------------------------------------------------------
    def draw(self, dash):
        """Redessine le header, la sidebar et tous les boards."""
        sim = dash.sim
        self.screen.fill(theme.COLOR_BG)
        self._draw_header(dash)
        self._draw_sidebar(dash)
        leader = sim.best_index()
        side = self.board_size * self.cell_px
        for i, env in enumerate(sim.envs):
            gx = SIDEBAR + (i % self.cols) * self.board_px + GAP
            gy = HEADER + (i // self.cols) * self.board_px + GAP
            self._draw_board_at(env, gx, gy, self.cell_px)
            frame = pygame.Rect(gx - 2, gy - 2, side + 4, side + 4)
            if i == dash.locked_index:
                pygame.draw.rect(self.screen, theme.COLOR_SELECT, frame, 2)
            elif i == leader:
                pygame.draw.rect(self.screen, theme.COLOR_LEADER, frame, 2)
        if dash.focus:
            self._draw_spotlight(dash, dash.focus_index())
        pygame.display.flip()

    def _draw_board_at(self, env, gx, gy, px):
        """Dessine un board complet a la position et a l'echelle donnees."""
        side = self.board_size * px
        pygame.draw.rect(self.screen, theme.COLOR_BOARD_BG,
                         pygame.Rect(gx, gy, side, side))
        cells = [(env.green_apples, theme.COLOR_GREEN_APPLE),
                 (env.red_apples, theme.COLOR_RED_APPLE),
                 (env.snake[1:], theme.COLOR_BODY)]
        for group, color in cells:
            for r, c in group:
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(gx + c * px, gy + r * px, px, px))
        if env.snake:
            hr, hc = env.snake[0]
            pygame.draw.rect(self.screen, theme.COLOR_HEAD,
                             pygame.Rect(gx + hc * px, gy + hr * px, px, px))

    def _draw_spotlight(self, dash, index):
        """Affiche en grand, au centre, une seule partie (leader ou verrou)."""
        sim = dash.sim
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))

        env = sim.envs[index]
        big_px = min(32, max(14, self.cell_px * 4))
        side = self.board_size * big_px
        pad = 14
        panel_w = side + 2 * pad
        panel_h = side + 2 * pad + 26
        px = (self.screen.get_width() - panel_w) // 2
        py = (self.screen.get_height() - panel_h) // 2
        locked = dash.locked_index is not None
        color = theme.COLOR_SELECT if locked else theme.COLOR_LEADER
        pygame.draw.rect(self.screen, theme.COLOR_PANEL,
                         pygame.Rect(px, py, panel_w, panel_h))
        pygame.draw.rect(self.screen, color,
                         pygame.Rect(px, py, panel_w, panel_h), 3)
        tag = "VERROUILLEE" if locked else "meilleure en cours"
        label = "Partie #{} ({})  long={}  duree={}".format(
            index, tag, len(env.snake), sim.durations[index])
        self.screen.blit(self.small.render(label, True, theme.COLOR_TEXT),
                         (px + pad, py + 6))
        self._draw_board_at(env, px + pad, py + pad + 24, big_px)

    def _draw_header(self, dash):
        """Barre du haut : titre + etat pause/vitesse."""
        pygame.draw.rect(self.screen, theme.COLOR_PANEL,
                         pygame.Rect(0, 0, self.screen.get_width(), HEADER))
        if dash.status_frames > 0:
            title = dash.status
            color = theme.COLOR_LEADER
        else:
            title = "Learn2Slither  -  {} parties".format(dash.sim.count)
            color = theme.COLOR_TEXT
        self.screen.blit(self.font.render(title, True, color), (12, 6))
        state = ("PAUSE" if dash.paused
                 else "x{}".format(dash.steps_per_frame))
        hint = ("[espace] pause  [+/-] vitesse  [fleches <>] partie  "
                "[B] meilleure  [L] geler  [S] save  " + state)
        self.screen.blit(self.small.render(hint, True, theme.COLOR_DIM),
                         (12, 26))

    def _draw_sidebar(self, dash):
        """Panneau gauche : statistiques d'apprentissage globales."""
        sim = dash.sim
        pygame.draw.rect(self.screen, theme.COLOR_PANEL,
                         pygame.Rect(0, HEADER, SIDEBAR,
                                     self.screen.get_height() - HEADER))
        lines = [
            ("Generations", str(sim.episodes)),
            ("Epsilon", "{:.3f}".format(sim.agent.epsilon)),
            ("Etats appris", str(len(sim.agent.q_table))),
            ("Longueur max", str(sim.best_length)),
            ("Duree max", str(sim.best_duration)),
            ("Long. moy (200)", "{:.1f}".format(sim.recent_average())),
            ("Apprentissage", "on" if sim.learn else "off"),
        ]
        y = HEADER + 16
        for label, value in lines:
            self.screen.blit(self.small.render(label, True, theme.COLOR_DIM),
                             (14, y))
            self.screen.blit(self.font.render(value, True, theme.COLOR_TEXT),
                             (14, y + 16))
            y += 46
        self._draw_buttons(dash)

    def _draw_buttons(self, dash):
        """Dessine les boutons cliquables (navigation, sauvegarde, gel)."""
        self._draw_button(self.btn_prev, "<")
        self._draw_button(self.btn_next, ">")
        best_active = dash.focus and dash.locked_index is None
        self._draw_button(self.btn_best, "Meilleure", best_active)
        self._draw_button(self.btn_save, "Sauver")
        freeze_label = "Reprendre" if not dash.sim.learn else "Geler"
        self._draw_button(self.btn_freeze, freeze_label, not dash.sim.learn)

    def _draw_button(self, rect, label, active=False):
        """Dessine un bouton rectangulaire avec son libelle centre."""
        bg = theme.COLOR_SELECT if active else theme.COLOR_BOARD_BG
        pygame.draw.rect(self.screen, bg, rect, border_radius=4)
        pygame.draw.rect(self.screen, theme.COLOR_DIM, rect, 1,
                         border_radius=4)
        txt = self.small.render(label, True, theme.COLOR_TEXT)
        tx = rect.x + (rect.width - txt.get_width()) // 2
        ty = rect.y + (rect.height - txt.get_height()) // 2
        self.screen.blit(txt, (tx, ty))
