"""
visualisation.py - Graphes de convergence µ et σ pour TrueSkill
=================================================================
Ce module génère :
1. Graphe de convergence de µ (estimation du niveau) vers la compétence cachée
2. Graphe de convergence de σ (incertitude) vers 0 (certitude croissante)
3. Graphe du classement final comparé aux vraies compétences
"""

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def graphe_convergence_mu(joueurs, dossier_sortie="../data"):
    """
    Trace l'évolution de µ (estimation TrueSkill du niveau) pour chaque joueur.

    On trace aussi une ligne horizontale pointillée représentant la compétence
    cachée de chaque joueur, pour visualiser la convergence.

    Paramètres
    ----------
    joueurs : list[dict]
    dossier_sortie : str
        Dossier où sauvegarder l'image (défaut : ../data)
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Palette de couleurs pour distinguer les joueurs
    couleurs = cm.tab10(np.linspace(0, 1, len(joueurs)))

    for joueur, couleur in zip(joueurs, couleurs):
        matchs_joues = range(len(joueur['historique_mu']))

        # Courbe de µ au fil des matchs
        ax.plot(
            matchs_joues,
            joueur['historique_mu'],
            color=couleur,
            label=joueur['nom'],
            linewidth=1.5,
            alpha=0.85
        )

        # Ligne horizontale : vraie compétence cachée (valeur à atteindre)
        ax.axhline(
            y=joueur['competence'],
            color=couleur,
            linestyle='--',
            linewidth=0.8,
            alpha=0.5
        )

    # Mise en forme du graphe
    ax.set_xlabel("Nombre de matchs joués", fontsize=13)
    ax.set_ylabel("µ (estimation TrueSkill du niveau)", fontsize=13)
    ax.set_title(
        "Convergence de µ vers la compétence cachée\n"
        "(lignes pleines = µ estimé, lignes pointillées = vraie compétence)",
        fontsize=14
    )
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chemin = os.path.join(dossier_sortie, "convergence_mu.png")
    plt.savefig(chemin, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Graphe µ sauvegardé : {chemin}")


def graphe_convergence_sigma(joueurs, dossier_sortie="../data"):
    """
    Trace l'évolution de σ (incertitude sur le niveau) pour chaque joueur.

    σ doit diminuer au fil des matchs : plus on joue, plus le système est certain
    du niveau du joueur.

    Paramètres
    ----------
    joueurs : list[dict]
    dossier_sortie : str
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    couleurs = cm.tab10(np.linspace(0, 1, len(joueurs)))

    for joueur, couleur in zip(joueurs, couleurs):
        matchs_joues = range(len(joueur['historique_sigma']))

        ax.plot(
            matchs_joues,
            joueur['historique_sigma'],
            color=couleur,
            label=joueur['nom'],
            linewidth=1.5,
            alpha=0.85
        )

    ax.set_xlabel("Nombre de matchs joués", fontsize=13)
    ax.set_ylabel("σ (incertitude sur le niveau)", fontsize=13)
    ax.set_title(
        "Diminution de σ au fil des matchs\n"
        "(σ → 0 signifie que le système est de plus en plus certain du niveau)",
        fontsize=14
    )
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chemin = os.path.join(dossier_sortie, "convergence_sigma.png")
    plt.savefig(chemin, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Graphe σ sauvegardé : {chemin}")


def graphe_classement_final(joueurs_classes, dossier_sortie="../data"):
    """
    Affiche un graphe à barres comparant le classement TrueSkill aux vraies compétences.

    Chaque barre représente le score TrueSkill conservateur (µ - 3σ) d'un joueur.
    Un point rouge indique sa vraie compétence cachée.

    Paramètres
    ----------
    joueurs_classes : list[dict]
        Joueurs triés par score TrueSkill décroissant
    dossier_sortie : str
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    noms = [j['nom'] for j in joueurs_classes]
    scores_ts = [j['rating'].mu - 3 * j['rating'].sigma for j in joueurs_classes]
    mu_values = [j['rating'].mu for j in joueurs_classes]
    competences = [j['competence'] for j in joueurs_classes]
    sigmas = [j['rating'].sigma for j in joueurs_classes]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(noms))

    # Barres : µ avec barre d'erreur de ±σ
    bars = ax.bar(x, mu_values, yerr=sigmas, color='steelblue', alpha=0.7,
                  capsize=5, label='µ ± σ (estimation TrueSkill)')

    # Points : vraies compétences cachées
    ax.scatter(x, competences, color='red', zorder=5, s=80,
               label='Vraie compétence cachée', marker='D')

    # Points : score conservateur (µ - 3σ)
    ax.scatter(x, scores_ts, color='orange', zorder=5, s=60,
               label='Score conservateur (µ - 3σ)', marker='o')

    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Niveau", fontsize=13)
    ax.set_title(
        "Classement final TrueSkill vs vraies compétences\n"
        "(ordre décroissant par score conservateur µ - 3σ)",
        fontsize=14
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    chemin = os.path.join(dossier_sortie, "classement_final.png")
    plt.savefig(chemin, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Graphe classement sauvegardé : {chemin}")


def graphe_mu_individuel(joueur, dossier_sortie="../data"):
    """
    Trace l'évolution de µ et σ pour un seul joueur, avec zone d'incertitude.

    La zone colorée représente l'intervalle [µ - σ, µ + σ] : là où le système
    pense que se trouve le vrai niveau avec ~68% de probabilité.

    Paramètres
    ----------
    joueur : dict
    dossier_sortie : str
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    mus = np.array(joueur['historique_mu'])
    sigmas = np.array(joueur['historique_sigma'])
    matchs = np.arange(len(mus))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Zone d'incertitude µ ± σ
    ax.fill_between(matchs, mus - sigmas, mus + sigmas,
                    alpha=0.2, color='steelblue', label='Intervalle µ ± σ')

    # Courbe µ
    ax.plot(matchs, mus, color='steelblue', linewidth=2, label='µ estimé')

    # Vraie compétence cachée
    ax.axhline(y=joueur['competence'], color='red', linestyle='--',
               linewidth=1.5, label=f"Vraie compétence = {joueur['competence']:.1f}")

    ax.set_xlabel("Matchs joués", fontsize=12)
    ax.set_ylabel("Niveau estimé", fontsize=12)
    ax.set_title(f"Évolution TrueSkill de {joueur['nom']}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chemin = os.path.join(dossier_sortie, f"individuel_{joueur['nom']}.png")
    plt.savefig(chemin, dpi=150, bbox_inches='tight')
    plt.close()
