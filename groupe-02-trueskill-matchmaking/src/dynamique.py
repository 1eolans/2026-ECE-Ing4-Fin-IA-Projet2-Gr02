"""
dynamique.py - Dynamique temporelle : saisons et dégradation de compétence
===========================================================================
Niveau EXCELLENT : modélisation du temps qui passe

Dans un vrai jeu en ligne :
- Les joueurs s'améliorent ou régressent entre deux saisons
- L'inactivité augmente l'incertitude (le système sait moins où en est le joueur)
- TrueSkill modélise cela par une augmentation de σ au fil du temps

Ce module simule :
1. Des saisons de matchs (blocs de matchs avec une période de "repos" entre eux)
2. Un "decay" (dégradation) inter-saison : σ augmente pendant la trêve
3. Des évolutions de compétence cachée (les joueurs progressent ou régressent)
"""

import random
import math
import trueskill


# =============================================================================
# Dégradation de l'incertitude (skill decay)
# =============================================================================

def appliquer_decay(joueurs, facteur_decay=0.15, sigma_max=None):
    """
    Applique un "drift" temporel : augmente σ de chaque joueur.

    Modélise l'inactivité entre deux saisons :
    → Le système "oublie" un peu le niveau du joueur (incertitude augmente)
    → σ_nouveau = σ_actuel × (1 + facteur_decay)
    → Mais σ ne dépasse jamais σ_max (valeur initiale TrueSkill)

    Ce concept est directement tiré du papier TrueSkill 2 :
    "Skill uncertainty increases with time since last game"

    Paramètres
    ----------
    joueurs       : list[dict]  Joueurs à mettre à jour
    facteur_decay : float       Pourcentage d'augmentation de σ (0.15 = +15%)
    sigma_max     : float       Limite haute de σ (défaut = sigma initial TrueSkill)
    """
    if sigma_max is None:
        sigma_max = trueskill.global_env().sigma  # σ initial ≈ 8.333

    for joueur in joueurs:
        sigma_actuel = joueur['rating'].sigma

        # Augmenter σ (le système devient moins certain)
        nouveau_sigma = min(sigma_actuel * (1.0 + facteur_decay), sigma_max)

        # Recréer le rating avec le nouveau σ (µ reste inchangé)
        joueur['rating'] = trueskill.Rating(
            mu=joueur['rating'].mu,
            sigma=nouveau_sigma
        )

        # Enregistrer ce "saut" dans l'historique pour les graphes
        joueur['historique_mu'].append(joueur['rating'].mu)
        joueur['historique_sigma'].append(nouveau_sigma)


def appliquer_evolution_competence(joueurs, amplitude=3.0):
    """
    Modifie légèrement la compétence cachée de chaque joueur entre deux saisons.

    Simule l'amélioration ou la régression naturelle des joueurs :
    - Un débutant peut s'améliorer beaucoup en pratiquant
    - Un expert peut légèrement régresser avec l'âge ou l'inactivité
    - La compétence reste dans [5, 55] pour éviter les extremes

    Paramètres
    ----------
    joueurs   : list[dict]  Les joueurs à modifier
    amplitude : float       Écart-type du bruit gaussien de progression
    """
    for joueur in joueurs:
        # Variation aléatoire gaussienne centrée en 0
        delta = random.gauss(0, amplitude)

        # Nouvelle compétence, bornée entre 5 et 55
        joueur['competence'] = max(5.0, min(55.0, joueur['competence'] + delta))


# =============================================================================
# Simulation de saisons
# =============================================================================

def simuler_saisons(joueurs, nb_saisons=3, matchs_par_saison=70,
                    decay_inter_saison=0.15, evolution_competence=True):
    """
    Simule plusieurs saisons de jeu avec inter-saison.

    Structure temporelle :
    ┌──────────┐  decay  ┌──────────┐  decay  ┌──────────┐
    │ Saison 1 │ ──────► │ Saison 2 │ ──────► │ Saison 3 │
    │ N matchs │         │ N matchs │         │ N matchs │
    └──────────┘         └──────────┘         └──────────┘

    Pendant le "decay" :
    - σ de chaque joueur augmente (incertitude inter-saison)
    - La compétence cachée peut légèrement évoluer (joueurs qui progressent)

    Paramètres
    ----------
    joueurs              : list[dict]  Liste des joueurs
    nb_saisons           : int         Nombre de saisons à simuler
    matchs_par_saison    : int         Matchs joués par saison
    decay_inter_saison   : float       Facteur de dégradation de σ entre saisons
    evolution_competence : bool        True = les compétences changent entre saisons

    Retourne
    --------
    tuple (historique_global, info_saisons) :
        historique_global : list[dict]   Tous les matchs de toutes les saisons
        info_saisons      : list[dict]   Résumé par saison (snapshot des ratings)
    """
    from simulation import simuler_match  # Import local pour éviter la circularité

    historique_global = []
    info_saisons      = []
    # breakpoints[i] = indice dans l'historique individuel où la saison i se termine
    # (utilisé par le dashboard pour placer les lignes verticales inter-saison)
    breakpoints       = []

    for num_saison in range(1, nb_saisons + 1):
        print(f"  [Saison {num_saison}/{nb_saisons}] Simulation de {matchs_par_saison} matchs...")

        historique_saison = []

        # ── Simulation des matchs de la saison ────────────────────────────────
        for num_match in range(1, matchs_par_saison + 1):
            j1, j2 = random.sample(joueurs, 2)
            resultat = simuler_match(j1, j2)

            historique_saison.append({
                'saison':    num_saison,
                'match_num': num_match,
                'joueur1':   j1['nom'],
                'joueur2':   j2['nom'],
                'resultat':  resultat,
            })

        historique_global.extend(historique_saison)

        # ── Enregistrer la longueur réelle de l'historique joueur à ce stade ──
        # On prend le 1er joueur comme référence (tous ont le même nombre de points)
        # len - 1 car on veut l'index du dernier point de la saison
        breakpoints.append(len(joueurs[0]['historique_mu']) - 1)

        # ── Snapshot de fin de saison (état des ratings) ──────────────────────
        snapshot = {
            'saison':   num_saison,
            'joueurs':  [
                {
                    'nom':        j['nom'],
                    'mu':         round(j['rating'].mu, 3),
                    'sigma':      round(j['rating'].sigma, 3),
                    'score_ts':   round(j['rating'].mu - 3 * j['rating'].sigma, 3),
                    'competence': round(j['competence'], 3),
                }
                for j in sorted(joueurs,
                                key=lambda x: x['rating'].mu - 3 * x['rating'].sigma,
                                reverse=True)
            ],
        }
        info_saisons.append(snapshot)

        # ── Inter-saison : decay + évolution (sauf après la dernière saison) ──
        if num_saison < nb_saisons:
            print(f"  [Inter-saison {num_saison}→{num_saison+1}] Decay σ +{decay_inter_saison*100:.0f}%...")
            appliquer_decay(joueurs, facteur_decay=decay_inter_saison)

            if evolution_competence:
                appliquer_evolution_competence(joueurs, amplitude=2.0)

    return historique_global, info_saisons, breakpoints


# =============================================================================
# Analyse de la dynamique temporelle
# =============================================================================

def analyser_progression(joueur, info_saisons):
    """
    Retrace l'évolution d'un joueur saison par saison.

    Paramètres
    ----------
    joueur      : dict        Le joueur à analyser
    info_saisons : list[dict]  Résultats retournés par simuler_saisons()

    Retourne
    --------
    list[dict] : [{saison, mu, sigma, score_ts, rang}] pour ce joueur
    """
    progression = []
    for snap in info_saisons:
        # Trouver le joueur dans ce snapshot
        for i, j_info in enumerate(snap['joueurs']):
            if j_info['nom'] == joueur['nom']:
                progression.append({
                    'saison':    snap['saison'],
                    'mu':        j_info['mu'],
                    'sigma':     j_info['sigma'],
                    'score_ts':  j_info['score_ts'],
                    'rang':      i + 1,  # Position dans le classement
                    'competence': j_info['competence'],
                })
                break

    return progression
