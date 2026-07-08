"""Learn2Slither - un Snake 10x10 pilote par Q-learning.

Paquet racine. L'architecture suit le flux impose par le sujet :

    core        -> le board et les regles du jeu (aucun apprentissage)
    perception  -> vision du serpent (etat) et rewards
    learning    -> Q-learning (Q-table, epsilon-greedy)
    training    -> boucle d'entrainement (assemble le flux)
    ui          -> affichages pygame optionnels (display, dashboard)
    cli         -> parsing des flags et point d'entree

Flux : core -> perception (state + reward) -> learning (action) -> core.
"""
