"""
equipes.py - Matchs par équipes avec marge de nul (draw margin)
================================================================
Niveau BON : équipes hétérogènes + matchs nuls

Ce module permet de :
- Former des équipes de taille variable à partir des joueurs
- Simuler des matchs d'équipe (2v2, 3v3...)
- Gérer les matchs nuls (draw) via la draw_probability de TrueSkill
- Mettre à jour les ratings individuels après un match d'équipe

Concept clé :
    La compétence d'une équipe = somme des µ de ses membres
    L'incertitude de l'équipe = √(somme des σ² de ses membres)
    TrueSkill met à jour CHAQUE joueur individuellement après le match.
"""

import random
import math
import trueskill


# =============================================================================
# Formation d'équipes
# =============================================================================

def creer_equipes_aleatoires(joueurs, taille=2):
    """
    Forme des équipes aléatoires à partir de la liste des joueurs.

    Les joueurs sont mélangés puis regroupés par paquets de 'taille'.
    Si le nombre de joueurs n'est pas divisible par 'taille',
    les joueurs restants sont ignorés pour ce match.

    Paramètres
    ----------
    joueurs : list[dict]   Liste complète des joueurs
    taille  : int          Nombre de joueurs par équipe (2 par défaut)

    Retourne
    --------
    tuple (equipe1, equipe2) : deux listes de joueurs, ou (None, None) si impossible
    """
    if len(joueurs) < taille * 2:
        return None, None  # Pas assez de joueurs

    # Mélanger aléatoirement les joueurs
    disponibles = joueurs.copy()
    random.shuffle(disponibles)

    # Prendre les 'taille' premiers pour chaque équipe
    equipe1 = disponibles[:taille]
    equipe2 = disponibles[taille:taille * 2]

    return equipe1, equipe2


def creer_equipes_equilibrees(joueurs, taille=2):
    """
    Forme deux équipes équilibrées en termes de µ TrueSkill.

    Algorithme : tri par µ décroissant, puis alternance équipe1/équipe2.
    Exemple avec 6 joueurs triés par niveau [A,B,C,D,E,F] :
        Équipe 1 = [A, D, E]
        Équipe 2 = [B, C, F]

    Cela donne des matchs plus équilibrés que l'aléatoire.

    Paramètres
    ----------
    joueurs : list[dict]
    taille  : int

    Retourne
    --------
    tuple (equipe1, equipe2)
    """
    if len(joueurs) < taille * 2:
        return None, None

    # Tri par µ décroissant (du plus fort au plus faible)
    tries = sorted(joueurs, key=lambda j: j['rating'].mu, reverse=True)

    equipe1, equipe2 = [], []
    for i, joueur in enumerate(tries[:taille * 2]):
        if i % 2 == 0:
            equipe1.append(joueur)
        else:
            equipe2.append(joueur)

    return equipe1, equipe2


# =============================================================================
# Calculs de force d'équipe
# =============================================================================

def force_equipe(equipe):
    """
    Calcule la force (µ total) et l'incertitude (σ combiné) d'une équipe.

    Formules :
        µ_equipe    = Σ µᵢ (somme des moyennes individuelles)
        σ_equipe    = √(Σ σᵢ²) (racine de la somme des variances)

    Paramètres
    ----------
    equipe : list[dict]

    Retourne
    --------
    tuple (mu_total, sigma_total)
    """
    mu_total    = sum(j['rating'].mu    for j in equipe)
    sigma_total = math.sqrt(sum(j['rating'].sigma ** 2 for j in equipe))
    return mu_total, sigma_total


# =============================================================================
# Simulation d'un match d'équipe
# =============================================================================

def simuler_match_equipe(equipe1, equipe2, draw_prob=0.15):
    """
    Simule un match entre deux équipes avec possibilité de match nul.

    Étapes :
    1. Calcul des forces de chaque équipe
    2. Décision du résultat (équipe1 gagne / équipe2 gagne / nul)
       basée sur les compétences cachées des joueurs
    3. Mise à jour TrueSkill de TOUS les joueurs des deux équipes

    La draw_probability de TrueSkill détermine combien d'information
    est transmise lors d'un match nul (moins qu'une victoire nette).

    Paramètres
    ----------
    equipe1  : list[dict]  Joueurs de l'équipe 1
    equipe2  : list[dict]  Joueurs de l'équipe 2
    draw_prob : float       Probabilité de match nul (0 à 1)

    Retourne
    --------
    str : 'equipe1', 'equipe2', ou 'nul'
    """
    # ── Compétence cachée totale de chaque équipe ─────────────────────────────
    comp1 = sum(j['competence'] for j in equipe1)
    comp2 = sum(j['competence'] for j in equipe2)

    # Probabilité de victoire de l'équipe 1 (formule logistique)
    diff = comp1 - comp2
    prob_e1 = 1.0 / (1.0 + 10.0 ** (-diff / (10.0 * len(equipe1))))

    # ── Décision du résultat ──────────────────────────────────────────────────
    # La probabilité de nul est plus forte quand les équipes sont proches
    proximite = 1.0 - abs(prob_e1 - 0.5) * 2  # 1 = parfaitement équilibré
    prob_nulle = draw_prob * proximite

    r = random.random()
    if r < prob_nulle:
        resultat = 'nul'
    elif r < prob_nulle + (prob_e1 * (1 - prob_nulle)):
        resultat = 'equipe1'
    else:
        resultat = 'equipe2'

    # ── Mise à jour TrueSkill en mode équipe ──────────────────────────────────
    # Créer un environnement TrueSkill avec la draw_probability configurée
    env = trueskill.TrueSkill(draw_probability=draw_prob)

    # Extraire les ratings sous forme de liste (attendu par trueskill.rate)
    ratings_e1 = [j['rating'] for j in equipe1]
    ratings_e2 = [j['rating'] for j in equipe2]

    # ranks : 0 = 1ère place, 1 = 2ème place, [0,0] = match nul
    if resultat == 'equipe1':
        ranks = [0, 1]
    elif resultat == 'equipe2':
        ranks = [1, 0]
    else:  # nul
        ranks = [0, 0]

    # trueskill.rate retourne [[new_r1a, new_r1b], [new_r2a, new_r2b]]
    nouveaux_ratings = env.rate([ratings_e1, ratings_e2], ranks=ranks)

    # ── Application des nouveaux ratings aux joueurs ──────────────────────────
    for joueur, new_r in zip(equipe1, nouveaux_ratings[0]):
        joueur['rating'] = new_r
        joueur['historique_mu'].append(new_r.mu)
        joueur['historique_sigma'].append(new_r.sigma)

    for joueur, new_r in zip(equipe2, nouveaux_ratings[1]):
        joueur['rating'] = new_r
        joueur['historique_mu'].append(new_r.mu)
        joueur['historique_sigma'].append(new_r.sigma)

    return resultat


# =============================================================================
# Simulation d'un tournoi par équipes
# =============================================================================

def simuler_tournoi_equipes(joueurs, nb_matchs=100, taille=2,
                             draw_prob=0.15, equilibre=True):
    """
    Simule un tournoi complet de matchs par équipes.

    À chaque match :
    - Formation de deux équipes (aléatoires ou équilibrées)
    - Simulation du match avec possible match nul
    - Mise à jour des ratings individuels

    Paramètres
    ----------
    joueurs   : list[dict]   Joueurs (doivent avoir au moins taille*2 joueurs)
    nb_matchs : int          Nombre de matchs à jouer
    taille    : int          Joueurs par équipe (2 = 2v2, 3 = 3v3...)
    draw_prob : float        Probabilité de match nul
    equilibre : bool         True = équipes équilibrées, False = aléatoire

    Retourne
    --------
    tuple (historique, stats) :
        historique : list[dict]  Détail de chaque match
        stats      : dict        Statistiques globales (nb nuls, etc.)
    """
    historique = []
    nb_nuls    = 0
    nb_e1_gagne = 0
    nb_e2_gagne = 0

    for num_match in range(1, nb_matchs + 1):
        # Formation des équipes
        if equilibre:
            e1, e2 = creer_equipes_equilibrees(joueurs, taille=taille)
        else:
            e1, e2 = creer_equipes_aleatoires(joueurs, taille=taille)

        if e1 is None:
            continue  # Pas assez de joueurs, on passe

        # Simulation du match
        resultat = simuler_match_equipe(e1, e2, draw_prob=draw_prob)

        # Statistiques
        if resultat == 'nul':
            nb_nuls += 1
        elif resultat == 'equipe1':
            nb_e1_gagne += 1
        else:
            nb_e2_gagne += 1

        historique.append({
            'match_num':  num_match,
            'equipe1':    [j['nom'] for j in e1],
            'equipe2':    [j['nom'] for j in e2],
            'force_e1':   round(force_equipe(e1)[0], 2),
            'force_e2':   round(force_equipe(e2)[0], 2),
            'resultat':   resultat,
        })

    stats = {
        'nb_matchs':    len(historique),
        'nb_nuls':      nb_nuls,
        'pct_nuls':     round(100 * nb_nuls / max(len(historique), 1), 1),
        'nb_e1_gagne':  nb_e1_gagne,
        'nb_e2_gagne':  nb_e2_gagne,
    }

    return historique, stats
