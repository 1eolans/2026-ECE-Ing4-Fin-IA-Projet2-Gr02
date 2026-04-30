# Technical Report - MMM Bayesien (Groupe 02)

## Problem Statement
Allouer un budget media multi-canal sous incertitude pour maximiser la performance commerciale tout en expliquant la contribution de chaque canal.

## Data
- Donnees synthetiques multi-marches
- Frequence hebdomadaire
- Variables : `spend_tv`, `spend_facebook`, `spend_google`, `sales`

## Feature Engineering
1. Adstock geometrique par canal
2. Hill transform pour la saturation
3. Traitement par marche (memoire media independante)

## Model
### Hierarchical Bayesian MMM (PyMC)
- Intercept global + intercepts par marche
- Effets media positifs (prior log-normal)
- Inference MCMC (NUTS)

### Why Hierarchical?
- Partage statistique entre marches
- Stabilise les estimations locales
- Meilleure robustesse en faible signal local

## Validation
### Temporal Cross-Validation
- Expanding window
- Comparaison avec baseline Ridge + effets fixes marche
- Metriques : MAE, RMSE, R2

## Benchmark
- Slot Google LightweightMMM integre
- Resultats exportes dans `benchmark_model_comparison.csv`
- Statut execution dans `benchmark_lightweightmmm_status.csv`

## Outputs
- Contributions et ROI par marche/canal
- Recommandation budget globale
- Predictions in-sample et out-of-sample

## Limitations
- Donnees synthetiques (pas encore de jeu reel)
- Hyperparametres fixes pour Adstock/Hill
- MCMC couteux en temps

## Next Iterations
1. Donnees reelles multi-marches
2. Hyperparametres Adstock/Hill estimes dans le modele
3. Contraintes business explicites dans l'optimisation
4. Analyse sensibilite et incertitude de la recommandation budget
