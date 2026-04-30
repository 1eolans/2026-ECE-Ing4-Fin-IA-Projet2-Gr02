"""
matchmaking.py - Sélection d'adversaires et calcul de probabilités de victoire
================================================================================
Ce module gère :
- Le calcul de la probabilité de victoire entre deux joueurs
- La sélection d'un adversaire équitable (matchmaking par niveau similaire)
"""

import math
import trueskill


def probabilite_victoire(joueur1, joueur2):
    """
    Calcule la probabilité que joueur1 batte joueur2 selon leurs ratings TrueSkill.

    La formule est tirée du papier TrueSkill de Microsoft (Herbrich et al., 2006).
    Elle utilise la fonction de répartition de la loi normale (Φ).

    Principe :
    - On modélise la performance de chaque joueur comme une gaussienne
      autour de son µ avec une variance σ² + β² (β = incertitude de performance)
    - La proba de victoire est la probabilité que perf(j1) > perf(j2)

    Paramètres
    ----------
    joueur1 : dict
    joueur2 : dict

    Retourne
    --------
    float : probabilité entre 0 et 1 que joueur1 gagne
    """
    env = trueskill.global_env()

    # β représente l'incertitude de performance par match (paramètre TrueSkill)
    beta = env.beta

    # Écart-type total de la différence de performance
    # Var(perf1 - perf2) = σ1² + σ2² + 2β²
    sigma_total = math.sqrt(
        joueur1['rating'].sigma ** 2 +
        joueur2['rating'].sigma ** 2 +
        2 * beta ** 2
    )

    # z-score : à quel point joueur1 est "meilleur" en unités d'écart-type
    z = (joueur1['rating'].mu - joueur2['rating'].mu) / sigma_total

    # P(joueur1 gagne) = Φ(z) où Φ est la CDF de la loi normale standard
    # trueskill utilise scipy en interne, on utilise math.erf pour éviter l'import scipy
    probabilite = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return probabilite


def trouver_adversaire(joueur, tous_les_joueurs):
    """
    Trouve l'adversaire le plus équitable pour un joueur donné.

    "Équitable" signifie : le joueur dont le µ TrueSkill est le plus proche.
    Cela maximise la qualité du match (match serré ≈ maximum d'information appris).

    Paramètres
    ----------
    joueur : dict
        Le joueur pour lequel on cherche un adversaire
    tous_les_joueurs : list[dict]
        La liste complète des joueurs

    Retourne
    --------
    dict : le meilleur adversaire (pas le joueur lui-même)
    """
    mu_cible = joueur['rating'].mu

    # On cherche le joueur différent dont le µ est le plus proche
    meilleur_adversaire = min(
        [j for j in tous_les_joueurs if j['nom'] != joueur['nom']],
        key=lambda j: abs(j['rating'].mu - mu_cible)
    )

    return meilleur_adversaire


def qualite_match(joueur1, joueur2):
    """
    Calcule la qualité d'un match potentiel entre deux joueurs.

    La qualité varie de 0 (match déséquilibré) à 1 (match parfaitement équilibré).
    Formule officielle TrueSkill : qualité ≈ exp(-distance²/2)

    Paramètres
    ----------
    joueur1 : dict
    joueur2 : dict

    Retourne
    --------
    float : score de qualité entre 0 et 1
    """
    # trueskill.quality_1vs1 retourne la probabilité de match nul
    # c'est un proxy pour "à quel point le match est équitable"
    qualite = trueskill.quality_1vs1(joueur1['rating'], joueur2['rating'])
    return qualite


def simuler_matchmaking(joueurs, nb_matchs=200):
    """
    Simule un tournoi en mode matchmaking : chaque joueur affronte son adversaire
    le plus proche en termes de µ TrueSkill.

    Cela accélère la convergence des ratings car les matchs sont plus informatifs.

    Paramètres
    ----------
    joueurs : list[dict]
    nb_matchs : int

    Retourne
    --------
    list[dict] : historique des matchs
    """
    import random
    from simulation import simuler_match

    historique = []

    for num_match in range(1, nb_matchs + 1):
        # Choisir un joueur de départ au hasard
        j1 = random.choice(joueurs)

        # Trouver son adversaire le plus équitable
        j2 = trouver_adversaire(j1, joueurs)

        # Simuler le match
        resultat = simuler_match(j1, j2)

        historique.append({
            'match_num': num_match,
            'joueur1': j1['nom'],
            'joueur2': j2['nom'],
            'resultat': resultat,
            'qualite': qualite_match(j1, j2),
        })

    return historique
