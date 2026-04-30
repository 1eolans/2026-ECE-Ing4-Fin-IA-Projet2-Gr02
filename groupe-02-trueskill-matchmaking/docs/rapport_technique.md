# Rapport Technique — TrueSkill et Matchmaking Compétitif

**Projet ECE Paris ING4 — Groupe 02**
**Date :** Mars 2026

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Le modèle bayésien TrueSkill](#2-le-modèle-bayésien-trueskill)
3. [Graphe de facteurs](#3-graphe-de-facteurs)
4. [Algorithme Expectation Propagation](#4-algorithme-expectation-propagation)
5. [Implémentation](#5-implémentation)
6. [Résultats et analyse](#6-résultats-et-analyse)
7. [Comparaison avec ELO](#7-comparaison-avec-elo)
8. [Références](#8-références)

---

## 1. Introduction

Le classement de joueurs dans les jeux compétitifs est un problème fondamental de l'IA probabiliste. Le système **ELO** (1960) fut le premier standard, mais il souffre de limitations importantes :

- Il ne modélise pas l'**incertitude** sur le niveau d'un joueur
- Il n'est pas conçu pour les matchs en **équipes**
- Il converge lentement pour les nouveaux joueurs

En 2006, Microsoft Research publie **TrueSkill** (Herbrich, Minka, Graepel), qui surmonte ces limitations en modélisant chaque joueur par une **distribution gaussienne** sur son niveau.

---

## 2. Le modèle bayésien TrueSkill

### 2.1 Représentation d'un joueur

Chaque joueur `i` est caractérisé par deux paramètres :

- **µᵢ** (mu) : estimation de son niveau moyen
- **σᵢ** (sigma) : incertitude sur cette estimation

La compétence du joueur est modélisée par une gaussienne :

```
compétence_i ~ N(µᵢ, σᵢ²)
```

**Interprétation :**
- Grand µ → joueur perçu comme fort
- Grand σ → le système est incertain (peu de matchs joués)
- Petit σ → le système est confiant dans son estimation

**Valeurs initiales par défaut (TrueSkill) :**
- µ₀ = 25
- σ₀ = 25/3 ≈ 8.33

### 2.2 Modèle de performance

Lors d'un match, la **performance** d'un joueur est tirée de :

```
performance_i ~ N(µᵢ, σᵢ² + β²)
```

où β est l'incertitude de performance par match (β = σ₀/2 par défaut).

β représente le fait que même un joueur de niveau connu peut avoir
un "bon" ou "mauvais" jour.

### 2.3 Probabilité de victoire

Le joueur 1 bat le joueur 2 si `performance_1 > performance_2`.

```
P(j1 gagne) = Φ( (µ₁ - µ₂) / √(σ₁² + σ₂² + 2β²) )
```

où Φ est la **fonction de répartition de la loi normale standard** (CDF).

---

## 3. Graphe de facteurs

TrueSkill utilise un **graphe de facteurs** pour représenter le modèle probabiliste.

```
         compétence_1          compétence_2
              |                      |
         N(µ₁,σ₁²)             N(µ₂,σ₂²)
              |                      |
    [facteur de performance]  [facteur de performance]
              |                      |
         perf_1                 perf_2
              \                    /
               [facteur de résultat]
                        |
                   résultat observé
                   (1 gagne / 2 gagne)
```

### Nœuds du graphe

| Nœud | Type | Description |
|------|------|-------------|
| `compétence_i` | Variable | Niveau latent (caché) du joueur |
| `performance_i` | Variable | Performance lors d'un match spécifique |
| `facteur_perf` | Facteur | Lie compétence à performance via N(µ, β²) |
| `facteur_résultat` | Facteur | Encode le résultat observé du match |

---

## 4. Algorithme Expectation Propagation

### 4.1 Principe

TrueSkill utilise **Expectation Propagation (EP)** pour mettre à jour les croyances sur les compétences après chaque match.

EP est un algorithme d'**inférence approchée** qui :
1. Propage des messages (distributions gaussiennes) dans le graphe
2. Approxime la distribution postérieure par une gaussienne
3. Converge itérativement vers la vraie postérieure

### 4.2 Mise à jour après un match (1v1)

Après que le joueur 1 bat le joueur 2, les mises à jour sont :

**Pour le vainqueur (joueur 1) :**
```
µ₁_new = µ₁ + σ₁² / c × v(t/c)
σ₁_new² = σ₁² × [1 - σ₁²/c² × w(t/c)]
```

**Pour le perdant (joueur 2) :**
```
µ₂_new = µ₂ - σ₂² / c × v(t/c)
σ₂_new² = σ₂² × [1 - σ₂²/c² × w(t/c)]
```

**Où :**
- `c² = 2β² + σ₁² + σ₂²` (variance totale du match)
- `t = (µ₁ - µ₂) / c` (z-score du résultat)
- `v(t) = φ(t) / Φ(t)` (**fonction de mills**, dérivée log de la CDF)
- `w(t) = v(t) × (v(t) + t)` (facteur de réduction de variance)
- φ = densité de la loi normale, Φ = CDF de la loi normale

### 4.3 Intuition des formules

- **v(t)** représente "à quel point le résultat était surprenant"
  - Si µ₁ >> µ₂ (victoire attendue), v est petit → mise à jour faible
  - Si µ₁ ≈ µ₂ (match serré), v est grand → mise à jour forte
- **w(t)** contrôle la réduction de σ
  - σ diminue toujours après un match (on en apprend plus)
  - La réduction est plus forte si le match était informatif

---

## 5. Implémentation

### 5.1 Structure du code

```
src/
├── simulation.py    # Joueurs, matchs 1v1, mise à jour TrueSkill
├── matchmaking.py   # Sélection d'adversaires, probabilités de victoire
├── visualisation.py # Graphes de convergence µ et σ
└── main.py          # Point d'entrée principal
```

### 5.2 Bibliothèque utilisée

Nous utilisons la bibliothèque Python `trueskill` (v0.4.5) qui implémente
l'algorithme EP complet, incluant la gestion des draws (matchs nuls).

```python
import trueskill

# Créer un rating initial
rating = trueskill.Rating()  # µ=25, σ=8.333

# Mettre à jour après un match (joueur1 gagne)
r1_new, r2_new = trueskill.rate_1vs1(rating1, rating2)
```

### 5.3 Score de classement conservateur

Pour le classement, on utilise le **score conservateur** :

```
score = µ - 3σ
```

Cela représente la borne inférieure de l'intervalle de confiance à 99.7%.
Ce score garantit que même un joueur incertain (grand σ) ne sera pas
sur-classé par rapport à ses vraies capacités.

---

## 6. Résultats et analyse

### 6.1 Convergence de µ

Après 200 matchs, les estimations µ convergent vers les vraies compétences cachées.
La convergence est plus rapide pour les joueurs dont le niveau est très différent
des autres (ils gagnent/perdent de manière plus consistante).

### 6.2 Diminution de σ

σ commence à 8.33 et diminue au fil des matchs. Typiquement après 200 matchs
répartis entre 10 joueurs (≈20 matchs par joueur), σ ≈ 3-5.

La valeur asymptotique de σ dépend de β : σ ne peut pas descendre en dessous
d'un plancher lié à l'incertitude intrinsèque de performance.

### 6.3 Qualité du classement

Le rang TrueSkill final est fortement corrélé avec le vrai niveau caché.
Des erreurs de classement subsistent pour les joueurs de niveau proche.

---

## 7. Comparaison avec ELO

| Critère | ELO | TrueSkill |
|---------|-----|-----------|
| Modèle | Différence de score | Distribution gaussienne |
| Incertitude | Non modélisée | σ explicite |
| Nouveaux joueurs | Convergence lente | σ élevé → ajustements rapides |
| Équipes | Extension ad hoc | Natif |
| Matchs nuls | Extension | Paramètre ε (draw margin) |
| Fondement théorique | Empirique | Inférence bayésienne |

---

## 8. Références

1. **Herbrich R., Minka T., Graepel T.** (2006). *TrueSkill™: A Bayesian Skill Rating System*. NeurIPS 2006.
   https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/

2. **Minka T. et al.** (2018). *TrueSkill 2: An improved Bayesian skill rating system*.
   https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/

3. **Bishop C.M., Winn J.** (2023). *Model-Based Machine Learning*.
   https://mbmlbook.com/TrueSkill.html

4. **Minka T.** (2001). *Expectation Propagation for approximate Bayesian inference*. UAI 2001.
