"""stats.py - suivi borne des metriques d'entrainement (sans pygame).

Conserve des series paralleles (une valeur par echantillon, l'abscisse etant
le numero de generation) pretes a tracer en direct. Pour rester borne en
memoire quel que soit le nombre de parties jouees, l'historique se
sous-echantillonne : quand il depasse `max_points`, il decime un point sur
deux et double l'intervalle d'echantillonnage. Aucune dependance graphique :
la logique reste testable et reutilisable hors de l'affichage.
"""


class StatsHistory:
    """Historique echantillonne et borne des metriques d'entrainement."""

    # Metriques suivies, dans l'ordre. Chaque nom donne une serie temporelle.
    FIELDS = ("avg_length", "max_length", "avg_duration", "epsilon")

    def __init__(self, max_points=480):
        self.max_points = max_points
        self.interval = 1          # generations entre deux echantillons
        self._since = 0
        self.episodes = []         # abscisse : numero de generation
        self.series = {name: [] for name in self.FIELDS}

    def record(self, episode, **values):
        """Ajoute un echantillon si l'intervalle courant est atteint.

        `values` doit fournir une valeur pour chaque nom de `FIELDS`.
        """
        self._since += 1
        if self._since < self.interval:
            return
        self._since = 0
        self.episodes.append(episode)
        for name in self.FIELDS:
            self.series[name].append(float(values[name]))
        if len(self.episodes) > self.max_points:
            self._downsample()

    def _downsample(self):
        """Decime un point sur deux et double l'intervalle d'echantillon."""
        self.episodes[:] = self.episodes[::2]
        for name in self.FIELDS:
            self.series[name][:] = self.series[name][::2]
        self.interval *= 2

    def latest(self, name):
        """Derniere valeur d'une serie (0.0 si vide)."""
        values = self.series[name]
        return values[-1] if values else 0.0

    def __len__(self):
        return len(self.episodes)
