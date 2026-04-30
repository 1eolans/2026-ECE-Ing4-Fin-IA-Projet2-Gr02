# Sujet I.1 — TrueSkill et Matchmaking Compétitif

**ECE Paris — ING4 — IA Probabiliste, Théorie des Jeux et ML — Groupe 02**

> Implémentation du système de classement TrueSkill (Microsoft Research) basé sur l'inférence bayésienne avec Expectation Propagation, appliqué au matchmaking compétitif.

---

## Membres du groupe

| Nom | GitHub |
|-----|--------|
| Jean EL HAJ | [@github_username_1](https://github.com/jeanhajjj) |
| Theo MELLY | [@github_username_2](https://github.com/TheoMly) |
| Sacha GIRSZYN | [@github_username_3](https://github.com/sachagirszyn) |

---

## Description du projet

Le classement de joueurs dans les jeux en ligne (Xbox Live, League of Legends, Chess.com) est un problème probabiliste complexe. Ce projet implémente le système **TrueSkill** de Microsoft Research, qui modélise le niveau de chaque joueur par une **distribution gaussienne** N(µ, σ²) :

- **µ** (mu) — estimation du niveau moyen du joueur
- **σ** (sigma) — incertitude sur cette estimation

Après chaque match, l'algorithme **Expectation Propagation** met à jour ces paramètres de manière bayésienne.

### Objectifs réalisés

- **Niveau Minimum ✅**
  - 10 joueurs avec compétences cachées aléatoires
  - 200 matchs 1v1 simulés
  - Mise à jour TrueSkill après chaque match
  - Graphe de convergence de µ vers la vraie compétence
  - Graphe de diminution de σ (certitude croissante)
  - Classement final avec score conservateur (µ − 3σ)
  - Export CSV des résultats

- **Niveau Bon** *(à venir)*
  - Équipes hétérogènes
  - Draw margin (matchs nuls)
  - Comparaison avec ELO

- **Niveau Excellent** *(à venir)*
  - Dynamique temporelle
  - TrueSkill 2
  - Données réelles Kaggle

---

## Installation

### Prérequis

- Python 3.9+
- pip

### Installation des dépendances

```bash
# Depuis le dossier groupe-02-trueskill-matchmaking/
pip install -r requirements.txt
```

**Dépendances :**

| Package | Version min | Usage |
|---------|-------------|-------|
| `trueskill` | 0.4.5 | Algorithme TrueSkill (EP) |
| `matplotlib` | 3.5.0 | Graphes de convergence |
| `pandas` | 1.4.0 | Export CSV |
| `numpy` | 1.22.0 | Calculs numériques |
| `scipy` | 1.8.0 | Distributions statistiques |

---

## Utilisation

```bash
# Depuis le dossier groupe-02-trueskill-matchmaking/
python src/main.py
```

### Sortie attendue

```
█████████████████████████████████████████████████████
  PROJET TRUESKILL - ECE Paris ING4 Groupe 02
  Simulation de matchmaking compétitif
█████████████████████████████████████████████████████

[ÉTAPE 1] Création de 10 joueurs...
[ÉTAPE 2] Simulation de 200 matchs 1v1...
[ÉTAPE 3] Calcul du classement final...
[ÉTAPE 4] Génération des graphes...
[ÉTAPE 5] Export des résultats en CSV...

SIMULATION TERMINÉE AVEC SUCCÈS
```

### Fichiers générés (dans `data/`)

| Fichier | Description |
|---------|-------------|
| `convergence_mu.png` | Évolution de µ pour chaque joueur |
| `convergence_sigma.png` | Diminution de σ pour chaque joueur |
| `classement_final.png` | Classement TrueSkill vs vraies compétences |
| `individuel_Joueur_X.png` | Zoom sur le joueur classé 1er |
| `classement_final.csv` | Résultats numériques du classement |
| `historique_matchs.csv` | Log de tous les matchs joués |

---

## Structure du projet

```
groupe-02-trueskill-matchmaking/
├── README.md                  # Ce fichier
├── requirements.txt           # Dépendances Python
├── src/
│   ├── main.py                # Point d'entrée (lancer ici)
│   ├── simulation.py          # Joueurs, matchs 1v1, TrueSkill
│   ├── matchmaking.py         # Sélection adversaires, proba de victoire
│   └── visualisation.py      # Graphes matplotlib
├── data/                      # Graphes et CSV générés
├── docs/
│   └── rapport_technique.md  # Explication bayésienne du modèle
└── slides/                    # Diapositives de soutenance
```

---

## Explication rapide du modèle

### Pourquoi une gaussienne ?

TrueSkill représente le niveau d'un joueur par N(µ, σ²) car :
- On ne connaît jamais le "vrai" niveau avec certitude
- La gaussienne permet de quantifier cette incertitude
- Après chaque match, on fait une mise à jour bayésienne

### Mise à jour bayésienne

```
Postérieur ∝ Vraisemblance × Prieur
N(µ_new, σ_new²) ∝ P(résultat | µ₁, µ₂) × N(µ, σ²)
```

L'algorithme **Expectation Propagation** calcule cette mise à jour de
manière efficace en propageant des messages dans un graphe de facteurs.

### Score de classement

```
score_conservateur = µ - 3σ
```

Ce score représente le niveau minimum estimé avec 99.7% de confiance.
Il évite de sur-classer les joueurs avec peu de matchs (grand σ).

---

## Références

- [TrueSkill — Microsoft Research](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- [TrueSkill 2 — Article technique](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)
- [Model-Based Machine Learning — Chapitre TrueSkill](https://mbmlbook.com/TrueSkill.html)
- [Bibliothèque Python trueskill](https://trueskill.org/)
