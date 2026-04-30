# Groupe 02 — Bayesian Sports Analytics

**ECE Paris — IA Probabiliste, Théorie des Jeux et Machine Learning 2026**
**Sujet A.2** — Prédiction de résultats sportifs par inférence bayésienne

**Membres** :
| Nom | GitHub |
|-----|--------|
| Cian Higgins | [@cian04](https://github.com/cian04) |
| Jules Dantin | [@julesd13](https://github.com/julesd13) |
| Hugo Ferré | [@hugo-frr](https://github.com/hugo-frr) |

---

## Présentation

Ce projet prédit les résultats de **Premier League** (2022–2025) à l'aide d'un **modèle de Poisson bayésien hiérarchique** qui modélise la force d'attaque et de défense de chaque équipe. Nous comparons ensuite nos probabilités aux cotes de Bet365 pour identifier des **value bets**.

### Ce que fait le projet

- **Modèle statique** : chaque équipe possède des paramètres d'attaque et de défense appris sur 3 saisons via NUTS (PyMC 5)
- **Avantage à domicile** : paramètre dédié avec prior informé par la littérature
- **Modèle dynamique** : les forces évoluent dans le temps via une marche aléatoire gaussienne
- **Value bets** : comparaison avec les cotes Bet365, identification des paris à EV positif, backtest avec critère de Kelly
- **Évaluation rigoureuse** : Ranked Probability Score, calibration, matrice de confusion

### Niveau

**Excellent** (tous les objectifs du sujet couverts) :
- ✅ Modèle Poisson bayésien (attaque/défense par équipe)
- ✅ Modèle hiérarchique avec avantage terrain et intervalles de crédibilité
- ✅ Comparaison avec cotes bookmakers
- ✅ Analyse de value bets
- ✅ Modèle dynamique (force évoluant dans le temps)

---

## Installation

### Prérequis

- Python 3.10+
- pip

### Étapes

```bash
# 1. Accéder au dossier du projet
cd groupe-02-bayesian-sports-analytics

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook notebooks/
```

### Dépendances principales

| Package | Rôle |
|---------|------|
| `pymc >= 5.0` | Inférence bayésienne (NUTS) |
| `arviz >= 0.17` | Diagnostics MCMC et visualisation |
| `pandas / numpy` | Manipulation des données |
| `matplotlib / seaborn` | Visualisation |
| `scipy` | Tests statistiques |
| `requests` | Téléchargement des données |

---

## Utilisation

### Ordre recommandé des notebooks

| Notebook | Description | Durée estimée |
|----------|-------------|---------------|
| [01_exploratory_analysis.ipynb](notebooks/01_exploratory_analysis.ipynb) | EDA : distribution des buts, home advantage, cotes | ~2 min |
| [02_bayesian_model.ipynb](notebooks/02_bayesian_model.ipynb) | Modèle hiérarchique statique, diagnostics MCMC, prédictions | ~10-15 min |
| [03_value_bets.ipynb](notebooks/03_value_bets.ipynb) | Comparaison bookmakers, value bets, backtest Kelly | ~10 min |
| [04_dynamic_model.ipynb](notebooks/04_dynamic_model.ipynb) | Modèle dynamique GRW, trajectoires de forme | ~20-30 min |

> **Note** : Le sampling MCMC prend du temps. Pour les notebooks 02 et 03, le modèle est sauvegardé automatiquement en `docs/idata_static.nc` après le premier run, et peut être rechargé avec `az.from_netcdf('../docs/idata_static.nc')`.

### Utilisation des modules Python

```python
from src.data.fetch_data import fetch_all_seasons, prepare_dataset
from src.models.bayesian_model import build_model, sample_model, predict_match
from src.analysis.value_bets import find_value_bets, backtest_strategy

# Charger et préparer les données
raw = fetch_all_seasons(['2023-24', '2024-25'])
df, teams, team_to_idx = prepare_dataset(raw)

# Construire et sampler le modèle
model = build_model(df, len(teams))
idata = sample_model(model)

# Prédire un match
pred = predict_match(idata, team_to_idx['Arsenal'], team_to_idx['Chelsea'])
print(f"P(Arsenal win) = {pred['p_home']:.1%}")
```

---

## Structure du projet

```
groupe-02-bayesian-sports-analytics/
├── README.md                      # Ce fichier
├── requirements.txt               # Dépendances Python
├── src/
│   ├── data/
│   │   └── fetch_data.py         # Téléchargement & préparation (football-data.co.uk)
│   ├── models/
│   │   ├── bayesian_model.py     # Modèle hiérarchique statique (PyMC)
│   │   └── dynamic_model.py      # Modèle dynamique (GRW)
│   └── analysis/
│       ├── predictions.py        # Métriques : RPS, Brier, calibration
│       └── value_bets.py         # Value bets, Kelly criterion, backtest
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_bayesian_model.ipynb
│   ├── 03_value_bets.ipynb
│   └── 04_dynamic_model.ipynb
├── docs/
│   ├── technical_report.md       # Rapport technique détaillé
│   └── figures/                  # Graphiques générés par les notebooks
└── slides/                       # Support de présentation
```

---

## Données

**Source** : [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — accès libre, sans API key.

Les données sont téléchargées automatiquement à l'exécution des notebooks. Pas de téléchargement manuel nécessaire.

**Saisons utilisées** : 2022-23, 2023-24, 2024-25 (≈ 1000 matchs)

**Variables clés** :
- `FTHG` / `FTAG` : buts domicile / extérieur (full-time)
- `FTR` : résultat (H=home win, D=draw, A=away win)
- `B365H/D/A` : cotes Bet365

---

## Résultats

### Avantage à domicile

Le modèle estime δ ≈ 0.14 [HDI 94% : 0.08, 0.20] avec **P(δ > 0) > 99%**, confirmant statistiquement l'avantage à domicile en Premier League.

### Performance prédictive

| Métrique | Valeur |
|----------|--------|
| Accuracy (résultat le plus probable) | ~49-52% |
| Ranked Probability Score | ~0.20 |
| Brier score | ~0.22 |

_Référence : la précision des cotes Bet365 est d'environ 53%._

### Value bets

Sur les 5 dernières matchweeks, le modèle identifie des paris avec EV > 3% qui, backtestés avec la stratégie Kelly fractionnaire, montrent un ROI positif (voir notebook 03 pour les chiffres exacts).

---

## Références

1. Baio & Blangiardo (2010) — *Bayesian hierarchical model for the prediction of football results*
2. Karlis & Ntzoufras (2003) — *Analysis of sports data by using bivariate Poisson models*
3. Dixon & Coles (1997) — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*
4. [PyMC Documentation](https://www.pymc.io/)
5. [ArviZ Documentation](https://python.arviz.org/)

---

## Support de présentation

Les slides seront disponibles dans le dossier `slides/` avant la soutenance du 31 mars 2026.
