# Recommandation Bayésienne d'Actifs Financiers

Système de recommandation probabiliste inspiré d'Infer.NET (implémenté avec PyMC).

Fonctionnalités:
- Inférence bayésienne des rendements (mu, sigma) avec quantification d'incertitude
- Score = E[r] - λ * σ (profil de risque)
- Clustering par mélange gaussien
- VaR & CVaR
- Backtesting roulant
- Interface Streamlit

Lancer l'app:
streamlit run app/streamlit_app.py

Test console:
python main.py