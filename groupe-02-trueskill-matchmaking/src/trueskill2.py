"""
trueskill2.py - Fonctionnalités inspirées de TrueSkill 2
=========================================================
Niveau EXCELLENT : TrueSkill 2 (Microsoft, 2018)

TrueSkill 2 améliore TrueSkill 1 sur plusieurs points :
1. CONSISTANCE COMPORTEMENTALE : un joueur régulier est mieux classé
   qu'un joueur imprévisible avec la même moyenne µ
2. FACTEUR D'ACTIVITÉ : les joueurs qui jouent souvent ont un score plus fiable
3. SCORE DE CONFIANCE : pondère le rating selon le nombre de matchs joués
4. MULTIPLICATEUR DE PERFORMANCE : modélise les "bonnes/mauvaises journées"

Référence :
    "TrueSkill 2: An improved Bayesian skill rating system"
    Minka et al., Microsoft Research, 2018
    https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/

Note : cette implémentation est une approximation pédagogique,
pas une reproduction exacte du papier complet.
"""

import math
import random
import trueskill


# =============================================================================
# Classe JoueurTS2 : joueur enrichi avec les concepts TrueSkill 2
# =============================================================================

class JoueurTS2:
    """
    Représente un joueur avec les métriques étendues de TrueSkill 2.

    En plus des attributs TrueSkill classiques (µ, σ), ce joueur possède :
    - Un historique de performances (pour calculer la consistance)
    - Un compteur de victoires/défaites
    - Un facteur de confiance basé sur l'activité

    Attributs
    ----------
    nom          : str            Identifiant du joueur
    competence   : float          Vrai niveau caché (0-50)
    rating       : trueskill.Rating   Rating TrueSkill courant
    historique_mu    : list[float]    Évolution de µ
    historique_sigma : list[float]    Évolution de σ
    nb_victoires : int            Nombre de victoires
    nb_defaites  : int            Nombre de défaites
    performances : list[float]    Score de performance de chaque match
    """

    def __init__(self, nom, competence):
        """
        Initialise un joueur TrueSkill 2.

        Paramètres
        ----------
        nom        : str    Nom du joueur
        competence : float  Niveau caché (ex: 35.0)
        """
        self.nom         = nom
        self.competence  = competence

        # Rating TrueSkill de base (µ=25, σ=8.333)
        self.rating = trueskill.Rating()

        # Historiques pour les graphes
        self.historique_mu    = [trueskill.global_env().mu]
        self.historique_sigma = [trueskill.global_env().sigma]

        # Statistiques de jeu
        self.nb_victoires = 0
        self.nb_defaites  = 0
        self.performances = []  # Performance relative par match [-1, +1]

    # ── Propriétés calculées ──────────────────────────────────────────────────

    @property
    def nb_matchs(self):
        """Nombre total de matchs joués."""
        return self.nb_victoires + self.nb_defaites

    @property
    def taux_victoire(self):
        """Pourcentage de victoires (0 à 1)."""
        if self.nb_matchs == 0:
            return 0.0
        return self.nb_victoires / self.nb_matchs

    @property
    def consistance(self):
        """
        Score de consistance comportementale (0 à 1).

        Basé sur le taux de victoire réel du joueur :
        - Un joueur qui gagne souvent = performances cohérentes avec son niveau
        - Un joueur qui perd souvent malgré un µ élevé = incohérent

        Formule : consistance = taux_victoire
        → 1.0 = gagne tous ses matchs (très consistant)
        → 0.5 = gagne la moitié (neutre)
        → 0.0 = perd tous ses matchs (très inconsistant)
        """
        return round(self.taux_victoire, 4)

    @property
    def facteur_activite(self):
        """
        Facteur de confiance lié à l'activité du joueur (0 à 1).

        Un joueur qui a joué peu de matchs = score moins fiable.
        → 0 matchs  : facteur = 0.0
        → 10 matchs : facteur ≈ 0.5
        → 20+ matchs : facteur → 1.0

        Formule : f_activite = 1 - exp(-nb_matchs / 15)
        """
        return round(1.0 - math.exp(-self.nb_matchs / 15.0), 4)

    @property
    def score_ts2(self):
        """
        Score composite TrueSkill 2.

        Combine le score TrueSkill classique (µ - 3σ) avec :
        - La consistance (taux de victoire du joueur)
        - Le facteur d'activité (nombre de matchs joués)

        Formule :
            score_ts2 = (µ - 3σ) × (0.4 + 0.6 × consistance) × (0.4 + 0.6 × activité)

        Les coefficients (0.4 + 0.6×x) gardent le score dans [40%, 100%] du score de base.
        → Évite les scores nuls pour les joueurs peu actifs ou peu consistants.
        → Un joueur à 70% de victoires × activité forte ≈ 0.82 × 0.94 du score TS1.
        → Un joueur à 30% de victoires × activité faible ≈ 0.58 × 0.64 du score TS1.

        Retourne
        --------
        float : score TrueSkill 2
        """
        score_base       = self.rating.mu - 3.0 * self.rating.sigma
        mult_consistance = 0.4 + 0.6 * self.consistance
        mult_activite    = 0.4 + 0.6 * self.facteur_activite
        return round(score_base * mult_consistance * mult_activite, 3)

    @property
    def score_ts1(self):
        """Score TrueSkill 1 classique (µ - 3σ) pour comparaison."""
        return round(self.rating.mu - 3.0 * self.rating.sigma, 3)

    # ── Méthodes utilitaires ──────────────────────────────────────────────────

    def enregistrer_performance(self, a_gagne):
        """
        Enregistre la performance du joueur pour ce match.

        Performance = 1.0 si victoire, 0.0 si défaite.

        Paramètres
        ----------
        a_gagne : bool  True si le joueur a gagné ce match
        """
        self.performances.append(1.0 if a_gagne else 0.0)

    def to_dict(self):
        """
        Convertit le joueur en dictionnaire compatible avec le reste du code.
        Utile pour les fonctions qui attendent des dict classiques.
        """
        return {
            'nom':               self.nom,
            'competence':        self.competence,
            'rating':            self.rating,
            'historique_mu':     self.historique_mu,
            'historique_sigma':  self.historique_sigma,
        }

    def __repr__(self):
        return (f"JoueurTS2({self.nom!r}, "
                f"µ={self.rating.mu:.1f}, σ={self.rating.sigma:.1f}, "
                f"TS2={self.score_ts2:.1f})")


# =============================================================================
# Création des joueurs TrueSkill 2
# =============================================================================

def creer_joueurs_ts2(nb_joueurs=10, mu_min=10, mu_max=50):
    """
    Crée une liste de joueurs TrueSkill 2 avec compétences cachées aléatoires.

    Paramètres
    ----------
    nb_joueurs : int    Nombre de joueurs (défaut : 10)
    mu_min     : float  Compétence minimale cachée
    mu_max     : float  Compétence maximale cachée

    Retourne
    --------
    list[JoueurTS2]
    """
    joueurs = []
    for i in range(nb_joueurs):
        competence = random.uniform(mu_min, mu_max)
        joueur = JoueurTS2(nom=f"Joueur_{i + 1}", competence=competence)
        joueurs.append(joueur)
    return joueurs


def creer_joueurs_ts2_depuis_dicts(joueurs_dicts):
    """
    Crée des joueurs TS2 à partir de la liste de dictionnaires existante.
    Permet de comparer TS1 et TS2 sur les MÊMES compétences cachées.

    Paramètres
    ----------
    joueurs_dicts : list[dict]  Joueurs créés par simulation.creer_joueurs()

    Retourne
    --------
    list[JoueurTS2]
    """
    joueurs_ts2 = []
    for j in joueurs_dicts:
        ts2 = JoueurTS2(nom=j['nom'], competence=j['competence'])
        joueurs_ts2.append(ts2)
    return joueurs_ts2


# =============================================================================
# Simulation d'un tournoi TrueSkill 2
# =============================================================================

def simuler_match_ts2(joueur1, joueur2):
    """
    Simule un match 1v1 avec enregistrement des métriques TS2.

    Paramètres
    ----------
    joueur1 : JoueurTS2
    joueur2 : JoueurTS2

    Retourne
    --------
    str : 'joueur1' ou 'joueur2' (vainqueur)
    """
    # Détermination du vainqueur via les compétences cachées
    diff = joueur1.competence - joueur2.competence
    prob_j1 = 1.0 / (1.0 + 10.0 ** (-diff / 10.0))

    if random.random() < prob_j1:
        vainqueur, perdant = joueur1, joueur2
        resultat = 'joueur1'
    else:
        vainqueur, perdant = joueur2, joueur1
        resultat = 'joueur2'

    # Mise à jour TrueSkill classique
    new_v, new_p = trueskill.rate_1vs1(vainqueur.rating, perdant.rating)
    vainqueur.rating = new_v
    perdant.rating   = new_p

    # Enregistrement des historiques
    joueur1.historique_mu.append(joueur1.rating.mu)
    joueur1.historique_sigma.append(joueur1.rating.sigma)
    joueur2.historique_mu.append(joueur2.rating.mu)
    joueur2.historique_sigma.append(joueur2.rating.sigma)

    # Mise à jour des statistiques TS2
    vainqueur.nb_victoires += 1
    perdant.nb_defaites    += 1
    vainqueur.enregistrer_performance(a_gagne=True)
    perdant.enregistrer_performance(a_gagne=False)

    return resultat


def simuler_tournoi_ts2(joueurs_ts2, nb_matchs=200):
    """
    Simule un tournoi complet avec les joueurs TrueSkill 2.

    Paramètres
    ----------
    joueurs_ts2 : list[JoueurTS2]
    nb_matchs   : int

    Retourne
    --------
    list[dict] : historique [{match, j1, j2, vainqueur}]
    """
    historique = []

    for num_match in range(1, nb_matchs + 1):
        j1, j2 = random.sample(joueurs_ts2, 2)
        resultat = simuler_match_ts2(j1, j2)
        vainqueur_nom = j1.nom if resultat == 'joueur1' else j2.nom

        historique.append({
            'match':     num_match,
            'joueur1':   j1.nom,
            'joueur2':   j2.nom,
            'vainqueur': vainqueur_nom,
        })

    return historique


# =============================================================================
# Classement et comparaison
# =============================================================================

def classement_ts2(joueurs_ts2):
    """
    Classe les joueurs par score TrueSkill 2 décroissant.

    Paramètres
    ----------
    joueurs_ts2 : list[JoueurTS2]

    Retourne
    --------
    list[JoueurTS2] : triés par score_ts2 décroissant
    """
    return sorted(joueurs_ts2, key=lambda j: j.score_ts2, reverse=True)


def comparer_ts1_ts2(joueurs_ts2_classes, joueurs_ts1_classes):
    """
    Compare les classements TrueSkill 1 et TrueSkill 2.

    Montre les joueurs dont le classement change significativement.

    Paramètres
    ----------
    joueurs_ts2_classes : list[JoueurTS2]  Classement TS2 (trié)
    joueurs_ts1_classes : list[dict]       Classement TS1 (trié, dicts)

    Retourne
    --------
    list[dict] : [{nom, rang_ts1, rang_ts2, ecart, raison_changement}]
    """
    rang_ts2 = {j.nom: i + 1 for i, j in enumerate(joueurs_ts2_classes)}
    rang_ts1 = {j['nom']: i + 1 for i, j in enumerate(joueurs_ts1_classes)}

    comparaison = []
    for j in joueurs_ts2_classes:
        r2    = rang_ts2[j.nom]
        r1    = rang_ts1.get(j.nom, '?')
        ecart = (r1 - r2) if isinstance(r1, int) else 0

        # Identifier pourquoi le classement change
        if ecart > 1:
            raison = f"Monté de {ecart} places (consistance={j.consistance:.2f}, activité={j.facteur_activite:.2f})"
        elif ecart < -1:
            raison = f"Descendu de {abs(ecart)} places (consistance={j.consistance:.2f})"
        else:
            raison = "Classement stable"

        comparaison.append({
            'joueur':       j.nom,
            'rang_ts1':     r1,
            'rang_ts2':     r2,
            'ecart':        ecart,
            'score_ts1':    j.score_ts1,
            'score_ts2':    j.score_ts2,
            'consistance':  j.consistance,
            'activite':     j.facteur_activite,
            'taux_victoire': round(j.taux_victoire * 100, 1),
            'raison':       raison,
        })

    return sorted(comparaison, key=lambda x: x['rang_ts2'])


def tableau_stats_ts2(joueurs_ts2_classes):
    """
    Génère un tableau de statistiques complètes TrueSkill 2.

    Paramètres
    ----------
    joueurs_ts2_classes : list[JoueurTS2]  Joueurs classés

    Retourne
    --------
    list[dict] : tableau de stats pour affichage ou export CSV
    """
    tableau = []
    for rang, j in enumerate(joueurs_ts2_classes, start=1):
        tableau.append({
            'rang':          rang,
            'joueur':        j.nom,
            'mu':            round(j.rating.mu, 2),
            'sigma':         round(j.rating.sigma, 2),
            'score_ts1':     j.score_ts1,
            'score_ts2':     j.score_ts2,
            'consistance':   j.consistance,
            'activite':      j.facteur_activite,
            'nb_victoires':  j.nb_victoires,
            'nb_defaites':   j.nb_defaites,
            'taux_victoire': round(j.taux_victoire * 100, 1),
            'competence_reelle': round(j.competence, 2),
        })
    return tableau
