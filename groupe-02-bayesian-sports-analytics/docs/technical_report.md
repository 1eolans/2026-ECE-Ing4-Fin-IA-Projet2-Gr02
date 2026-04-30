# Rapport Technique — Bayesian Sports Analytics

**Groupe 02** | ECE Paris — IA Probabiliste 2026
**Membres** : Cian Higgins (cian04) · Jules Dantin (julesd13) · Hugo Ferré (hugo-frr)
**Sujet** : A.2 — Bayesian Sports Analytics
**Championnat** : Premier League (2022–2025)

---

## 1. Introduction

Ce projet applique l'inférence bayésienne à la prédiction des résultats de football. L'objectif est de construire un modèle probabiliste capable de prédire les scores de Premier League et d'identifier des **value bets** en comparant nos probabilités aux cotes de bookmakers (Bet365).

La question centrale : **peut-on battre les bookmakers avec un modèle bayésien hiérarchique ?**

---

## 2. Données

**Source** : [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — CSV gratuits, sans API key.

| Saison | Matchs | Équipes |
|--------|--------|---------|
| 2022-23 | 380 | 20 |
| 2023-24 | 380 | 20 |
| 2024-25 | ~300 | 20 |

**Variables utilisées** :
- `FTHG`, `FTAG` : buts domicile/extérieur à temps plein
- `FTR` : résultat (H/D/A)
- `B365H`, `B365D`, `B365A` : cotes Bet365 (home/draw/away)

**Split train/test** : les 5 dernières matchweeks de 2024-25 constituent le set de test (simulation de prédiction live).

---

## 3. Modèle Bayésien Hiérarchique Statique

### 3.1 Spécification

Basé sur Baio & Blangiardo (2010) — *Bayesian hierarchical model for the prediction of football results*.

**Vraisemblance** :
```
goals_home ~ Poisson(θ_home)
goals_away ~ Poisson(θ_away)

log(θ_home) = μ + δ + α_h - β_a
log(θ_away) = μ     + α_a - β_h
```

**Paramètres** :
- `μ` : intercepte (log-nombre de buts moyen)
- `δ` : avantage à domicile (_home advantage_)
- `α_k` : force d'attaque de l'équipe k
- `β_k` : force défensive de l'équipe k

**Priors hiérarchiques** :
```
μ_α ~ N(0, 0.5),  σ_α ~ HalfN(0.5)
μ_β ~ N(0, 0.5),  σ_β ~ HalfN(0.5)
α_k ~ N(μ_α, σ_α)
β_k ~ N(μ_β, σ_β)
δ   ~ N(0.1, 0.2)
μ   ~ N(0, 1)
```

**Contrainte d'identifiabilité** : somme à zéro sur les paramètres d'attaque (`α_k - mean(α_k)`).

### 3.2 Inférence

- Algorithme : **NUTS** (No-U-Turn Sampler) via PyMC 5
- 4 chaînes × 2000 draws (1000 tune)
- `target_accept = 0.95`
- Convergence : R-hat < 1.01, ESS > 400 pour tous les paramètres

### 3.3 Résultats clés

| Paramètre | Moyenne | HDI 94% |
|-----------|---------|---------|
| δ (home advantage) | ~0.14 | [0.08, 0.20] |
| μ (intercept) | ~0.05 | [-0.15, 0.24] |
| σ_att | ~0.22 | [0.16, 0.29] |
| σ_def | ~0.18 | [0.12, 0.25] |

**P(δ > 0) ≈ 99.5%** — l'avantage domicile est statistiquement robuste.

---

## 4. Modèle Dynamique (Gaussian Random Walk)

Pour capturer l'évolution de la forme au cours de la saison :

```
α_{k,t} = α_{k,t-1} + ε_t,  ε_t ~ N(0, σ_walk)
β_{k,t} = β_{k,t-1} + η_t,  η_t ~ N(0, σ_walk)
```

- `σ_walk_att ~ HalfN(0.1)` — volatilité de l'attaque
- `σ_walk_def ~ HalfN(0.1)` — volatilité de la défense

**Avantages** :
- Capture les blessures, changements de forme, mercato
- Prédictions plus précises à court terme (dernières semaines)

**Compromis** :
- Plus coûteux computationnellement (O(T × K) paramètres)
- Nécessite plus de données pour l'identification

---

## 5. Value Bets

### 5.1 Théorie

Un **value bet** existe quand :
```
EV = p_model × odds_decimal - 1 > 0
```

Autrement dit, notre modèle estime que la probabilité réelle est supérieure à celle implicite dans les cotes.

### 5.2 Critère de Kelly

Fraction optimale du bankroll à miser :
```
f* = (p × odds - 1) / (odds - 1)
```
On applique un cap à 20% pour éviter la ruine.

### 5.3 Processus

1. Calculer `p_model` via simulation Monte Carlo depuis le posterior
2. Calculer `p_implied = 1/odds` ajustée de la marge bookmaker
3. Identifier les matchs où `EV > 3%`
4. Backtest sur l'ensemble de test

---

## 6. Métriques d'Évaluation

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **Accuracy** | % de résultats correctement prédits | Baseline naïf ~33%, bookmakers ~53% |
| **RPS** | Ranked Probability Score | Score propre ordinal, plus bas = mieux |
| **Brier score** | MSE des probabilités | Mesure la calibration globale |
| **ROI** | Return on Investment du backtest | Profit / mise totale |

---

## 7. Architecture du Code

```
src/
├── data/
│   └── fetch_data.py        # Téléchargement & préparation données
├── models/
│   ├── bayesian_model.py    # Modèle hiérarchique statique
│   └── dynamic_model.py     # Modèle dynamique (GRW)
└── analysis/
    ├── predictions.py       # Métriques d'évaluation
    └── value_bets.py        # Identification & backtest value bets
```

---

## 8. Références

1. **Baio & Blangiardo (2010)** — *Bayesian hierarchical model for the prediction of football results*. Journal of Applied Statistics.
2. **Karlis & Ntzoufras (2003)** — *Analysis of sports data by using bivariate Poisson models*. The Statistician.
3. **Dixon & Coles (1997)** — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*.
4. **PyMC Documentation** — https://www.pymc.io/
5. **ArviZ Documentation** — https://python.arviz.org/
6. **football-data.co.uk** — données gratuites Premier League

---

## 9. Instructions d'installation

```bash
# 1. Cloner le repo et accéder au dossier
cd groupe-02-bayesian-sports-analytics

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook notebooks/
```

**Ordre recommandé** :
1. `01_exploratory_analysis.ipynb` — EDA (~2 min)
2. `02_bayesian_model.ipynb` — Modèle statique (~10 min de sampling)
3. `03_value_bets.ipynb` — Comparaison bookmakers
4. `04_dynamic_model.ipynb` — Modèle dynamique (~20 min de sampling)

---

## 10. Perspectives

- **Dixon-Coles correction** : dépendance entre scores proches (0-0, 1-0, 0-1, 1-1)
- **Features contextuels** : blessures, historique head-to-head, fatigue (matchs européens)
- **Modèle temps réel** : mise à jour incrémentale du posterior (Kalman filter bayésien)
- **Multi-championnats** : transfert d'apprentissage entre ligues (partage de priors)
