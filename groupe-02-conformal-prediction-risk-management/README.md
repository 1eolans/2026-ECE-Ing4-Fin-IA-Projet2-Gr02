Sujet : Conformal Prediction for Risk Management
Groupe: 
EL BAKKALI Badr
ID- EL OUALI Kawthar
EL YOUSSOUFI Zakaria

Ce projet implémente un système de gestion des risques financiers basé sur la Prédiction Conforme (Conformal Prediction). Il intègre des mécanismes avancés pour garantir une gestion des risques robuste, particulièrement lors de crises aiguës sur les marchés financiers (ex: pandémie de COVID-19, inflation de 2022). Le système utilise la prédiction conforme en ligne (Online CP), le calibrage piloté par le VIX (Multi-Period Conformal), et l'évaluation asymétrique de la Value at Risk (VaR).

## Procédure d'installation

Il est recommandé d'utiliser Python 3.8 ou une version plus récente. 

1. **Ouvrez votre terminal et naviguez vers le dossier du projet :**
   ```bash
   cd "Conformal Prediction for Risk Management"
   ```

2. **Création d'un environnement virtuel (recommandé) :**
   ```bash
   python -m venv venv
   ```
   *Activation sous Windows :*
   ```bash
   venv\Scripts\activate
   ```
   *Activation sous Linux/Mac :*
   ```bash
   source venv/bin/activate
   ```

3. **Installation des dépendances principales :**
   L'application s'appuie sur de nombreuses bibliothèques d'analyse de données et de machine learning. Exécutez la commande suivante pour les installer :
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn lightgbm torch yfinance
   ```
   *Note : Le package `xgboost` est optionnel/auto-installé. Le script détectera s'il manque et tentera de l'installer automatiquement.*

## Exécution et Tests

Le fichier `level3_cpps_v4.py` constitue le moteur principal ainsi que le pipeline complet de test et de backtesting. 

Pour lancer les tests complets des modèles, générer les prédictions et évaluer les performances de couverture :

```bash
python level3_cpps_v4.py
```

### Déroulement du test et de l'évaluation :

1. **Acquisition des données** : Le système télécharge l'historique de marché via Yahoo Finance (couvrant une période de formation, de calibrage et une période de test allant de 2020 à 2022).
2. **Entraînement des modèles (Train)** : Apprentissage des régressions quantiles avec Random Forest, XGBoost, LSTM, etc.
3. **Calibrage de l'incertitude (Calibrate)** : Les intervalles de confiance sont ajustés de manière dynamique et par régime de volatilité (basé sur le VIX).
4. **Évaluation (Test)** : Les performances de la couverture marginale et conditionnelle (notamment le test de Kupiec) sont mesurées sur la période de test. 
5. **Résultats** : L'ensemble des métriques d'évaluation, les rendements du portefeuille de test et les éventuels graphiques générés sont sauvegardés dans le sous-dossier `results_v4` qui sera créé automatiquement au premier lancement.