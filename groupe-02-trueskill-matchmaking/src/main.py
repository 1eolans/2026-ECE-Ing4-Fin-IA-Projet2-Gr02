"""
main.py - Point d'entrée principal du projet TrueSkill
========================================================
Projet ECE Paris ING4 — Groupe 02

Lance la simulation complète en mode texte (terminal).
Pour l'interface graphique interactive, utiliser :
    streamlit run src/dashboard.py

Niveaux implémentés :
  🎯 Minimum  : TrueSkill 1v1, convergence µ/σ, classement
  🏆 Bon      : Équipes hétérogènes, draw margin, comparaison ELO
  🚀 Excellent : Dynamique temporelle (saisons), TrueSkill 2

Utilisation :
    cd groupe-02-trueskill-matchmaking/
    python src/main.py
"""

import sys
import os
import random
import pandas as pd

# Ajout du dossier src au path Python pour les imports
sys.path.insert(0, os.path.dirname(__file__))

# ── Imports des modules locaux ────────────────────────────────────────────────
from simulation  import creer_joueurs, simuler_tournoi, classement_final
from matchmaking import probabilite_victoire, qualite_match
from elo         import creer_joueurs_elo, simuler_tournoi_elo, classement_elo, comparer_classements
from equipes     import simuler_tournoi_equipes
from dynamique   import simuler_saisons
from trueskill2  import (creer_joueurs_ts2_depuis_dicts, simuler_tournoi_ts2,
                         classement_ts2, comparer_ts1_ts2, tableau_stats_ts2)
from visualisation import (
    graphe_convergence_mu,
    graphe_convergence_sigma,
    graphe_classement_final,
    graphe_mu_individuel,
)

# =============================================================================
# PARAMÈTRES DE LA SIMULATION
# =============================================================================

NB_JOUEURS       = 10    # Nombre de joueurs
NB_MATCHS        = 200   # Matchs 1v1 (niveau minimum)
GRAINE           = 42    # Graine aléatoire (changer pour des résultats différents)
TAILLE_EQUIPE    = 2     # Joueurs par équipe pour les matchs d'équipe
DRAW_PROB        = 0.15  # Probabilité de match nul
NB_SAISONS       = 3     # Nombre de saisons (niveau excellent)
MATCHS_SAISON    = 70    # Matchs par saison
DECAY_INTER      = 0.15  # Taux de decay inter-saison

DOSSIER_DATA = os.path.join(os.path.dirname(__file__), "..", "data")


# =============================================================================
# Fonctions d'affichage terminal
# =============================================================================

def titre(texte, car="█"):
    """Affiche un titre encadré."""
    bord = car * (len(texte) + 6)
    print(f"\n{bord}")
    print(f"{car}  {texte}  {car}")
    print(f"{bord}")


def section(texte):
    """Affiche un titre de section."""
    print(f"\n{'─'*60}")
    print(f"  {texte}")
    print(f"{'─'*60}")


def afficher_joueurs(joueurs):
    """Affiche les joueurs avec leur compétence cachée."""
    print(f"\n  {'Joueur':<15} {'Compétence cachée':>20}")
    print("  " + "-" * 38)
    for j in joueurs:
        print(f"  {j['nom']:<15} {j['competence']:>20.2f}")


def afficher_classement_ts(joueurs_classes, titre_col="TrueSkill"):
    """Affiche le classement TrueSkill."""
    print(f"\n  {'Rang':<5} {'Joueur':<13} {'µ':>7} {'σ':>7} {'µ-3σ':>9} {'Vrai':>8}")
    print("  " + "-" * 55)
    for rang, j in enumerate(joueurs_classes, 1):
        score = j['rating'].mu - 3 * j['rating'].sigma
        print(
            f"  {rang:<5} {j['nom']:<13} "
            f"{j['rating'].mu:>7.2f} "
            f"{j['rating'].sigma:>7.2f} "
            f"{score:>9.2f} "
            f"{j['competence']:>8.2f}"
        )


def afficher_classement_elo(joueurs_elo_classes):
    """Affiche le classement ELO."""
    print(f"\n  {'Rang':<5} {'Joueur':<13} {'ELO':>8} {'Vrai niveau':>12}")
    print("  " + "-" * 42)
    for rang, j in enumerate(joueurs_elo_classes, 1):
        print(f"  {rang:<5} {j['nom']:<13} {j['elo']:>8.1f} {j['competence']:>12.2f}")


def exporter_csv(joueurs_classes, historique, dossier):
    """Exporte les résultats en CSV dans data/."""
    os.makedirs(dossier, exist_ok=True)

    # CSV 1 : Classement final TrueSkill
    rows = []
    for rang, j in enumerate(joueurs_classes, 1):
        rows.append({
            'rang':             rang,
            'joueur':           j['nom'],
            'mu_final':         round(j['rating'].mu, 4),
            'sigma_final':      round(j['rating'].sigma, 4),
            'score_conservateur': round(j['rating'].mu - 3 * j['rating'].sigma, 4),
            'competence_reelle': round(j['competence'], 4),
        })
    chemin1 = os.path.join(dossier, "classement_final.csv")
    pd.DataFrame(rows).to_csv(chemin1, index=False)
    print(f"  [CSV] Classement TrueSkill : {chemin1}")

    # CSV 2 : Historique des matchs
    chemin2 = os.path.join(dossier, "historique_matchs.csv")
    pd.DataFrame(historique).to_csv(chemin2, index=False)
    print(f"  [CSV] Historique matchs    : {chemin2}")


# =============================================================================
# NIVEAU MINIMUM
# =============================================================================

def niveau_minimum():
    """Simulation TrueSkill 1v1 basique."""
    titre("NIVEAU MINIMUM — TrueSkill 1v1", "█")

    random.seed(GRAINE)

    # Création des joueurs
    section("ÉTAPE 1 : Création des joueurs")
    joueurs = creer_joueurs(nb_joueurs=NB_JOUEURS, mu_min=10, mu_max=50)
    afficher_joueurs(joueurs)

    # Simulation des matchs
    section(f"ÉTAPE 2 : Simulation de {NB_MATCHS} matchs 1v1")
    historique = simuler_tournoi(joueurs, nb_matchs=NB_MATCHS)
    print(f"  [OK] {len(historique)} matchs simulés.")

    print("\n  Aperçu des 5 premiers matchs :")
    print(f"  {'N°':<5} {'J1':<13} {'J2':<13} {'Résultat'}")
    print("  " + "-" * 48)
    for m in historique[:5]:
        gagnant = m['joueur1'] if m['resultat'] == 'joueur1' else m['joueur2']
        print(f"  {m['match_num']:<5} {m['joueur1']:<13} {m['joueur2']:<13} → {gagnant}")

    # Classement final
    section("ÉTAPE 3 : Classement final TrueSkill")
    joueurs_classes = classement_final(joueurs)
    afficher_classement_ts(joueurs_classes)

    # Probabilités de victoire
    top1, bot1 = joueurs_classes[0], joueurs_classes[-1]
    prob = probabilite_victoire(top1, bot1)
    q    = qualite_match(top1, bot1)
    print(f"\n  P({top1['nom']} bat {bot1['nom']}) = {prob:.1%}")
    print(f"  Qualité du match = {q:.3f}")

    # Génération des graphes
    section("ÉTAPE 4 : Génération des graphes (PNG dans data/)")
    graphe_convergence_mu(joueurs,         dossier_sortie=DOSSIER_DATA)
    graphe_convergence_sigma(joueurs,      dossier_sortie=DOSSIER_DATA)
    graphe_classement_final(joueurs_classes, dossier_sortie=DOSSIER_DATA)
    graphe_mu_individuel(joueurs_classes[0], dossier_sortie=DOSSIER_DATA)

    # Export CSV
    section("ÉTAPE 5 : Export CSV")
    exporter_csv(joueurs_classes, historique, dossier=DOSSIER_DATA)

    return joueurs, joueurs_classes, historique


# =============================================================================
# NIVEAU BON
# =============================================================================

def niveau_bon(joueurs, joueurs_classes, historique):
    """Équipes hétérogènes, draw margin, comparaison ELO."""
    titre("NIVEAU BON — Équipes & ELO", "▓")

    # ── Comparaison ELO ────────────────────────────────────────────────────────
    section("A) Comparaison ELO vs TrueSkill (mêmes matchs)")

    random.seed(GRAINE)
    joueurs_elo = creer_joueurs_elo(joueurs)
    simuler_tournoi_elo(joueurs_elo, historique)
    classes_elo = classement_elo(joueurs_elo)

    print("\n  [Classement ELO]")
    afficher_classement_elo(classes_elo)

    comparaison = comparer_classements(joueurs_classes, classes_elo)
    print("\n  [Comparaison des classements]")
    print(f"  {'Joueur':<13} {'Rang TS':>8} {'Rang ELO':>9} {'Écart':>6}")
    print("  " + "-" * 40)
    for c in comparaison:
        print(f"  {c['joueur']:<13} {c['rang_ts']:>8} {c['rang_elo']:>9} {c['ecart']:>6}")

    ecart_moy = sum(c['ecart'] for c in comparaison) / len(comparaison)
    print(f"\n  → Écart moyen de classement ELO/TrueSkill : {ecart_moy:.2f} positions")

    # ── Matchs par équipes ─────────────────────────────────────────────────────
    section(f"B) Matchs {TAILLE_EQUIPE}v{TAILLE_EQUIPE} avec marge de nul ({DRAW_PROB*100:.0f}%)")

    random.seed(GRAINE)
    joueurs_eq = creer_joueurs(nb_joueurs=NB_JOUEURS, mu_min=10, mu_max=50)
    hist_eq, stats_eq = simuler_tournoi_equipes(
        joueurs_eq,
        nb_matchs=NB_MATCHS // 2,
        taille=TAILLE_EQUIPE,
        draw_prob=DRAW_PROB,
        equilibre=True,
    )

    print(f"\n  Matchs joués   : {stats_eq['nb_matchs']}")
    print(f"  Matchs nuls    : {stats_eq['nb_nuls']} ({stats_eq['pct_nuls']}%)")
    print(f"  Victoires E1   : {stats_eq['nb_e1_gagne']}")
    print(f"  Victoires E2   : {stats_eq['nb_e2_gagne']}")

    # Export CSV des équipes
    if hist_eq:
        chemin_eq = os.path.join(DOSSIER_DATA, "historique_equipes.csv")
        os.makedirs(DOSSIER_DATA, exist_ok=True)
        rows = []
        for m in hist_eq:
            rows.append({
                'match_num': m['match_num'],
                'equipe1':   '|'.join(m['equipe1']),
                'equipe2':   '|'.join(m['equipe2']),
                'resultat':  m['resultat'],
            })
        pd.DataFrame(rows).to_csv(chemin_eq, index=False)
        print(f"  [CSV] Matchs équipes : {chemin_eq}")

    return joueurs_eq, stats_eq


# =============================================================================
# NIVEAU EXCELLENT
# =============================================================================

def niveau_excellent(joueurs, joueurs_classes):
    """Dynamique temporelle et TrueSkill 2."""
    titre("NIVEAU EXCELLENT — Saisons & TrueSkill 2", "░")

    # ── Simulation par saisons ─────────────────────────────────────────────────
    section(f"A) {NB_SAISONS} saisons × {MATCHS_SAISON} matchs (decay inter-saison={DECAY_INTER*100:.0f}%)")

    random.seed(GRAINE)
    joueurs_dyn = creer_joueurs(nb_joueurs=NB_JOUEURS, mu_min=10, mu_max=50)
    hist_dyn, info_saisons, _ = simuler_saisons(
        joueurs_dyn,
        nb_saisons=NB_SAISONS,
        matchs_par_saison=MATCHS_SAISON,
        decay_inter_saison=DECAY_INTER,
    )

    for snap in info_saisons:
        print(f"\n  ═══ Classement fin Saison {snap['saison']} ═══")
        print(f"  {'Rang':<4} {'Joueur':<13} {'µ':>7} {'σ':>7} {'Score':>8}")
        print("  " + "-" * 44)
        for i, j in enumerate(snap['joueurs'], 1):
            print(f"  {i:<4} {j['nom']:<13} {j['mu']:>7.2f} {j['sigma']:>7.2f} {j['score_ts']:>8.2f}")

    # Graphes saisons
    graphe_convergence_mu(joueurs_dyn,
                          dossier_sortie=DOSSIER_DATA)
    graphe_convergence_sigma(joueurs_dyn,
                             dossier_sortie=DOSSIER_DATA)

    # ── TrueSkill 2 ────────────────────────────────────────────────────────────
    section("B) TrueSkill 2 — Consistance & Activité")

    random.seed(GRAINE)
    joueurs_ts2 = creer_joueurs_ts2_depuis_dicts(joueurs)
    simuler_tournoi_ts2(joueurs_ts2, nb_matchs=NB_MATCHS)

    classes_ts2  = classement_ts2(joueurs_ts2)
    comp_ts      = comparer_ts1_ts2(classes_ts2, joueurs_classes)
    stats_ts2    = tableau_stats_ts2(classes_ts2)

    print(f"\n  {'Rang':>4} {'Joueur':<13} {'TS1':>8} {'TS2':>8} {'Consist.':>9} {'Activité':>9}")
    print("  " + "-" * 55)
    for r in comp_ts:
        print(
            f"  #{r['rang_ts2']:<3} {r['joueur']:<13} "
            f"{r['score_ts1']:>8.2f} {r['score_ts2']:>8.2f} "
            f"{r['consistance']:>9.3f} {r['activite']:>9.3f}"
        )

    # Export CSV TrueSkill 2
    chemin_ts2 = os.path.join(DOSSIER_DATA, "classement_trueskill2.csv")
    os.makedirs(DOSSIER_DATA, exist_ok=True)
    pd.DataFrame(stats_ts2).to_csv(chemin_ts2, index=False)
    print(f"\n  [CSV] Classement TrueSkill 2 : {chemin_ts2}")

    return classes_ts2, comp_ts


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Fonction principale — orchestre tous les niveaux."""

    titre("PROJET TRUESKILL — ECE Paris ING4 Groupe 02", "█")
    print(f"  Graine : {GRAINE}  |  Joueurs : {NB_JOUEURS}  |  Matchs : {NB_MATCHS}")
    print("\n  Pour l'interface graphique interactive :")
    print("  → streamlit run src/dashboard.py")

    # ── Niveau Minimum ──────────────────────────────────────────────────────
    joueurs, joueurs_classes, historique = niveau_minimum()

    # ── Niveau Bon ──────────────────────────────────────────────────────────
    niveau_bon(joueurs, joueurs_classes, historique)

    # ── Niveau Excellent ────────────────────────────────────────────────────
    niveau_excellent(joueurs, joueurs_classes)

    # ── Conclusion ──────────────────────────────────────────────────────────
    titre("SIMULATION TERMINÉE AVEC SUCCÈS", "═")
    print(f"  Graphes PNG et CSV disponibles dans :")
    print(f"  → {os.path.abspath(DOSSIER_DATA)}")
    print("\n  Pour l'interface graphique :")
    print("  → streamlit run src/dashboard.py\n")


if __name__ == "__main__":
    main()
