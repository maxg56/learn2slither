"""metrics.py - export CSV des courbes d'entrainement et plot optionnel.

Tenu a l'ecart de la boucle coeur de `trainer.py` : ce module ne fait que
collecter les statistiques par session (longueur, duree, epsilon, reward
cumule) puis les serialiser en CSV, avec un `plot` optionnel via matplotlib.
Suit le meme patron que `cli.py` (`_make_display`/`_run_dashboard`) pour les
imports optionnels : try/except autour de l'import, avertissement sur
stderr, jamais de crash.
"""

import csv
import sys

FIELDNAMES = ["session", "length", "duration", "epsilon", "total_reward"]


class MetricsRecorder:
    """Accumule les statistiques d'une session puis les exporte en CSV."""

    def __init__(self):
        self.records = []

    def record(self, session, length, duration, epsilon, total_reward):
        """Ajoute le bilan d'une session terminee."""
        self.records.append({
            "session": session,
            "length": length,
            "duration": duration,
            "epsilon": epsilon,
            "total_reward": total_reward,
        })

    def save(self, path):
        """Ecrit les enregistrements dans un fichier CSV.

        Tolerant aux chemins invalides (jamais de crash). Retourne True si
        l'ecriture a reussi, False sinon.
        """
        try:
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                for row in self.records:
                    writer.writerow(row)
            return True
        except OSError:
            return False


def plot(csv_path, out_png_path):
    """Trace longueur/reward/epsilon a partir du CSV, si matplotlib existe.

    Degrade sans crasher si matplotlib n'est pas installe : avertissement
    sur stderr puis retour, comme les imports optionnels de `cli.py`.
    Retourne True si le graphique a ete produit, False sinon.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print("Avertissement : matplotlib indisponible, pas de plot ({})"
              .format(error), file=sys.stderr)
        return False

    try:
        sessions, lengths, rewards, epsilons = _read_csv(csv_path)
    except OSError as error:
        print("Avertissement : lecture du CSV impossible ({})"
              .format(error), file=sys.stderr)
        return False

    try:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
        axes[0].plot(sessions, lengths, color="tab:green")
        axes[0].set_ylabel("Longueur max")
        axes[1].plot(sessions, rewards, color="tab:blue")
        axes[1].set_ylabel("Reward cumule")
        axes[2].plot(sessions, epsilons, color="tab:orange")
        axes[2].set_ylabel("Epsilon")
        axes[2].set_xlabel("Session")
        fig.tight_layout()
        fig.savefig(out_png_path)
        plt.close(fig)
        return True
    except OSError as error:
        print("Avertissement : ecriture du plot impossible ({})"
              .format(error), file=sys.stderr)
        return False


def _read_csv(csv_path):
    """Relit le CSV ecrit par `MetricsRecorder.save` en listes paralleles."""
    sessions, lengths, rewards, epsilons = [], [], [], []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sessions.append(int(row["session"]))
            lengths.append(float(row["length"]))
            rewards.append(float(row["total_reward"]))
            epsilons.append(float(row["epsilon"]))
    return sessions, lengths, rewards, epsilons
