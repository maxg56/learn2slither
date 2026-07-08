"""dashboard - vue parallele : une grille de parties partageant un Agent.

Decoupe en trois responsabilites :

    simulation -> avance les boards et cumule les stats (aucun pygame)
    renderer   -> dessine header, sidebar et boards (pygame)
    app        -> orchestre la boucle, l'etat d'affichage et les entrees

Purement cosmetique : la logique de jeu et d'apprentissage vit ailleurs.
"""

from snakeai.ui.dashboard.app import Dashboard

__all__ = ["Dashboard"]
