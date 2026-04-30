"""
elo.py - Système de classement ELO classique
=============================================
Niveau BON : comparaison ELO vs TrueSkill

Le système ELO est l'ancêtre de TrueSkill.
Il met à jour un seul nombre (le rating) sans modéliser l'incertitude.
Ce module rejoue exactement les mêmes matchs que TrueSkill pour comparer.

Différence clé :
- ELO : un seul chiffre, pas d'incertitude modélisée, convergence lente
- TrueSkill : moyenne µ + incertitude σ, convergence rapide, Bayésien
"""

import math

# ─── Paramètres ELO ───────────────────────────────────────────────────────────
K_FACTOR    = 32      # Sensibilité : plus K est grand, plus le rating change vite
ELO_INITIAL = 1200   # Rating de départ pour tous les joueurs (standard FIDE)


# =============================================================================
# Création des joueurs ELO
# =============================================================================

def creer_joueurs_elo(joueurs_trueskill):
    """
    Crée des "clones ELO" à partir des joueurs TrueSkill existants.

    Chaque joueur ELO partage le même nom et la même compétence cachée,
    mais utilise un rating ELO (entier) au lieu du rating TrueSkill (µ, σ).

    Paramètres
    ----------
    joueurs_trueskill : list[dict]
        La liste des joueurs déjà créés par simulation.creer_joueurs()

    Retourne
    --------
    list[dict] : liste des joueurs avec rating ELO
    """
    joueurs_elo = []
    for j in joueurs_trueskill:
        joueurs_elo.append({
            'nom':              j['nom'],
            'competence':       j['competence'],      # Même compétence cachée
            'elo':              float(ELO_INITIAL),   # Rating ELO initial
            'historique_elo':   [float(ELO_INITIAL)], # Historique pour les graphes
        })
    return joueurs_elo


# =============================================================================
# Calculs ELO
# =============================================================================

def prob_victoire_elo(elo1, elo2):
    """
    Calcule la probabilité que le joueur 1 batte le joueur 2 selon ELO.

    Formule officielle ELO (FIDE) :
        P(j1 gagne) = 1 / (1 + 10^((elo2 - elo1) / 400))

    Si elo1 = elo2 → P = 0.5 (match équilibré)
    Si elo1 >> elo2 → P → 1.0 (j1 domine)

    Paramètres
    ----------
    elo1 : float   Rating ELO du joueur 1
    elo2 : float   Rating ELO du joueur 2

    Retourne
    --------
    float : probabilité de victoire entre 0 et 1
    """
    return 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))


def mettre_a_jour_elo(joueur1, joueur2, resultat):
    """
    Met à jour les ratings ELO des deux joueurs après un match.

    La mise à jour suit la règle :
        nouveau_elo = ancien_elo + K * (résultat_réel - résultat_attendu)

    Paramètres
    ----------
    joueur1  : dict  Joueur ELO 1
    joueur2  : dict  Joueur ELO 2
    resultat : str   'joueur1' si j1 gagne, 'joueur2' si j2 gagne
    """
    # Résultats attendus (probabilités ELO)
    p1 = prob_victoire_elo(joueur1['elo'], joueur2['elo'])
    p2 = 1.0 - p1

    # Résultats réels (1 = victoire, 0 = défaite)
    if resultat == 'joueur1':
        s1, s2 = 1.0, 0.0
    else:
        s1, s2 = 0.0, 1.0

    # Mise à jour des ratings
    joueur1['elo'] += K_FACTOR * (s1 - p1)
    joueur2['elo'] += K_FACTOR * (s2 - p2)

    # Enregistrement pour les graphes
    joueur1['historique_elo'].append(joueur1['elo'])
    joueur2['historique_elo'].append(joueur2['elo'])


# =============================================================================
# Simulation du tournoi ELO
# =============================================================================

def simuler_tournoi_elo(joueurs_elo, historique_matchs):
    """
    Rejoue exactement le même historique de matchs avec le système ELO.

    Cela permet une comparaison directe ELO vs TrueSkill sur les MÊMES matchs :
    - Même ordre de matchs
    - Même vainqueur à chaque match
    → Seule la méthode de mise à jour des scores diffère

    Paramètres
    ----------
    joueurs_elo      : list[dict]  Joueurs avec rating ELO
    historique_matchs : list[dict] Historique généré par simuler_tournoi()

    Retourne
    --------
    list[dict] : joueurs_elo mis à jour
    """
    # Index nom → joueur pour accès rapide
    dict_elo = {j['nom']: j for j in joueurs_elo}

    for match in historique_matchs:
        j1 = dict_elo[match['joueur1']]
        j2 = dict_elo[match['joueur2']]
        mettre_a_jour_elo(j1, j2, match['resultat'])

    return joueurs_elo


# =============================================================================
# Classement et statistiques
# =============================================================================

def classement_elo(joueurs_elo):
    """
    Trie les joueurs par rating ELO décroissant.

    Paramètres
    ----------
    joueurs_elo : list[dict]

    Retourne
    --------
    list[dict] : joueurs triés par ELO décroissant
    """
    return sorted(joueurs_elo, key=lambda j: j['elo'], reverse=True)


def comparer_classements(joueurs_ts_classes, joueurs_elo_classes):
    """
    Compare les classements TrueSkill et ELO.

    Calcule l'erreur de classement : combien de positions un joueur
    est déplacé entre les deux systèmes.

    Paramètres
    ----------
    joueurs_ts_classes  : list[dict]  Classement TrueSkill
    joueurs_elo_classes : list[dict]  Classement ELO

    Retourne
    --------
    list[dict] : comparaison [{nom, rang_ts, rang_elo, ecart}]
    """
    # Rang TrueSkill par joueur
    rang_ts  = {j['nom']: i + 1 for i, j in enumerate(joueurs_ts_classes)}
    # Rang ELO par joueur
    rang_elo = {j['nom']: i + 1 for i, j in enumerate(joueurs_elo_classes)}

    comparaison = []
    for nom in rang_ts:
        ecart = abs(rang_ts[nom] - rang_elo[nom])
        comparaison.append({
            'joueur':   nom,
            'rang_ts':  rang_ts[nom],
            'rang_elo': rang_elo.get(nom, '?'),
            'ecart':    ecart,
        })

    # Trier par rang TrueSkill
    return sorted(comparaison, key=lambda x: x['rang_ts'])
