# PR Description Template - Groupe 02

## Sujet
Marketing Mix Modeling bayesien multi-marche (Adstock + Hill + optimisation budget)

## Membres
- Lans Leo (@github_username)
- Esnault Wandrille (@github_username)
- Jezequel Martin (@github_username)

## Ce que cette PR apporte
- Modele bayesien hierarchique multi-marche
- Validation croisee temporelle
- Benchmark vs Ridge + integration LightweightMMM
- Dashboard Streamlit
- Documentation technique et scripts de reproductibilite

## Commandes de reproduction
```bash
pip install -r requirements.txt
python src/main.py
python scripts/smoke_test.py
streamlit run src/dashboard.py
```

## Artefacts generes
- `docs/*.csv` (predictions, ROI, budget, CV, benchmark)
- `docs/TECHNICAL_REPORT.md`

## Checklist
- [ ] Le pipeline s'execute sans erreur
- [ ] Le smoke test passe
- [ ] Les membres du groupe sont mentionnes
- [ ] Captures dashboard ajoutees
