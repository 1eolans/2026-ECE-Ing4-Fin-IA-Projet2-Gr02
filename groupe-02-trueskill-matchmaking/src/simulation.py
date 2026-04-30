"""
simulation.py - Simulation de matchs 1v1 avec le système TrueSkill
=======================================================================
Ce module gère :
- La création des joueurs (avec compétence cachée et rating TrueSkill)
- La simulation de matchs 1v1
- La mise à jour des ratings après chaque match
"""

import random
import trueskill


def creer_joueurs(nb_joueurs=10, mu_min=10, mu_max=50):
    """
    Crée une liste de joueurs avec une compétence cachée aléatoire.

    Chaque joueur est un dictionnaire contenant :
    - 'nom'            : identifiant du joueur (ex: "Joueur_1")
    - 'competence'     : vrai niveau caché, tiré uniformément entre mu_min et mu_max
    - 'rating'         : objet TrueSkill Rating (µ=25, σ=8.333 par défaut)
    - 'historique_mu'  : liste des µ successifs après chaque match joué
    - 'historique_sigma' : liste des σ successifs après chaque match joué

    Paramètres
    ----------
    nb_joueurs : int
        Nombre de joueurs à créer (défaut : 10)
    mu_min : float
        Compétence minimale cachée
    mu_max : float
        Compétence maximale cachée

    Retourne
    --------
    list[dict] : liste des joueurs
    """
    env = trueskill.TrueSkill()  # Environnement TrueSkill avec paramètres par défaut

    joueurs = []
    for i in range(nb_joueurs):
        # Compétence cachée : le "vrai" niveau du joueur, inconnu du système
        competence_reelle = random.uniform(mu_min, mu_max)

        joueur = {
            'nom': f"Joueur_{i + 1}",
            'competence': competence_reelle,       # Valeur cachée (non utilisée pour décider les matchs)
            'rating': env.create_rating(),          # Rating initial : µ=25, σ=8.333
            'historique_mu': [env.mu],              # On enregistre le µ initial
            'historique_sigma': [env.sigma],        # On enregistre le σ initial
        }
        joueurs.append(joueur)

    return joueurs


def simuler_match(joueur1, joueur2):
    """
    Simule un match 1v1 entre deux joueurs et met à jour leurs ratings TrueSkill.

    Le résultat du match est déterminé par les compétences cachées :
    le joueur avec la compétence la plus élevée gagne avec une probabilité
    proportionnelle à la différence de niveau (modèle logistique simple).

    Paramètres
    ----------
    joueur1 : dict
    joueur2 : dict

    Retourne
    --------
    str : 'joueur1' si joueur1 gagne, 'joueur2' sinon
    """
    # Probabilité que joueur1 gagne basée sur la différence de compétences cachées
    diff = joueur1['competence'] - joueur2['competence']
    prob_joueur1_gagne = 1 / (1 + 10 ** (-diff / 10))  # Formule logistique (type ELO)

    # Tirage aléatoire pour déterminer le vainqueur
    if random.random() < prob_joueur1_gagne:
        vainqueur, perdant = joueur1, joueur2
        resultat = 'joueur1'
    else:
        vainqueur, perdant = joueur2, joueur1
        resultat = 'joueur2'

    # Mise à jour des ratings TrueSkill
    # trueskill.rate_1vs1 retourne les nouveaux ratings (vainqueur, perdant)
    nouveau_rating_vainqueur, nouveau_rating_perdant = trueskill.rate_1vs1(
        vainqueur['rating'], perdant['rating']
    )

    # Application des nouveaux ratings
    vainqueur['rating'] = nouveau_rating_vainqueur
    perdant['rating'] = nouveau_rating_perdant

    # Enregistrement de l'évolution de µ et σ pour les graphes
    joueur1['historique_mu'].append(joueur1['rating'].mu)
    joueur1['historique_sigma'].append(joueur1['rating'].sigma)
    joueur2['historique_mu'].append(joueur2['rating'].mu)
    joueur2['historique_sigma'].append(joueur2['rating'].sigma)

    return resultat


def simuler_tournoi(joueurs, nb_matchs=200):
    """
    Simule un tournoi de nb_matchs matchs 1v1 entre joueurs aléatoires.

    À chaque match, deux joueurs distincts sont choisis au hasard.

    Paramètres
    ----------
    joueurs : list[dict]
        Liste des joueurs (créés par creer_joueurs)
    nb_matchs : int
        Nombre total de matchs à simuler (défaut : 200)

    Retourne
    --------
    list[dict] : historique des matchs [{match_num, joueur1, joueur2, resultat}]
    """
    historique = []

    for num_match in range(1, nb_matchs + 1):
        # Sélection aléatoire de deux joueurs différents
        j1, j2 = random.sample(joueurs, 2)

        # Simulation du match et mise à jour des ratings
        resultat = simuler_match(j1, j2)

        historique.append({
            'match_num': num_match,
            'joueur1': j1['nom'],
            'joueur2': j2['nom'],
            'resultat': resultat,
        })

    return historique


def classement_final(joueurs):
    """
    Trie les joueurs par leur score TrueSkill (µ - 3σ, dit "score conservateur").

    Ce score représente le niveau minimum estimé avec 99.7% de confiance.
    C'est la métrique officielle utilisée par le système TrueSkill de Microsoft.

    Paramètres
    ----------
    joueurs : list[dict]

    Retourne
    --------
    list[dict] : joueurs triés par score décroissant
    """
    def score_trueskill(joueur):
        # µ - 3σ : borne inférieure de confiance à 99.7%
        return joueur['rating'].mu - 3 * joueur['rating'].sigma

    return sorted(joueurs, key=score_trueskill, reverse=True)
