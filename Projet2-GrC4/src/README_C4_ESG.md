# 🌿 Classification de Risque ESG Multi-label — Projet C4

Système de classification multi-label de documents financiers selon les critères
ESG (Environnemental, Social, Gouvernance), fonctionnant entièrement en local
via **Ollama + Mistral** (aucun envoi de données vers le cloud).

---

## 📁 Structure du projet

```
projet-esg-c4/
├── run.py                   ← 🚀 LANCEUR PRINCIPAL (commencez ici)
├── esg_classifier.py        ← Classifieur ESG (logique principale)
├── utils.py                 ← Dataset synthétique + métriques d'évaluation
├── esg_classification.ipynb ← Notebook interactif avec visualisations
├── verify_setup.py          ← Script de vérification de l'environnement
├── requirements.txt         ← Liste des dépendances Python
└── README_C4_ESG.md         ← Ce fichier
```

---

## ⚡ Démarrage en 3 étapes (aucun venv requis)

### Étape 1 — Installer Python 3.8+

Téléchargez Python depuis [python.org](https://www.python.org/downloads/) et
**cochez "Add Python to PATH"** lors de l'installation (Windows).

Vérifiez dans un terminal :
```bash
python --version
# doit afficher Python 3.8.x ou supérieur
```

---

### Étape 2 — Installer et démarrer Ollama

1. Téléchargez Ollama depuis [ollama.com](https://ollama.com/) et installez-le.
2. **Dans un terminal séparé**, démarrez le serveur Ollama :
   ```bash
   ollama serve
   ```
   > ⚠️ Laissez ce terminal ouvert pendant toute votre session de travail.

3. Téléchargez le modèle Mistral (une seule fois, ~4 Go) :
   ```bash
   ollama pull mistral
   ```

---

### Étape 3 — Lancer le projet

Ouvrez un **nouveau terminal** dans le dossier du projet, puis :

```bash
# Installer les dépendances Python (une seule fois)
pip install -r requirements.txt

# Vérifier que tout est correct
python verify_setup.py

# Lancer le menu principal
python run.py
```

Le menu interactif propose :
```
╔══════════════════════════════════════════════════╗
║   🌿  Projet ESG C4 — ECE Paris 2026             ║
╠══════════════════════════════════════════════════╣
║   1. 📦 Installer les dépendances                ║
║   2. 🔍 Vérifier l'environnement                 ║
║   3. ⚡ Démo rapide (3 documents)                 ║
║   4. 📊 Évaluer les métriques (10 docs)          ║
║   5. 📓 Ouvrir le notebook Jupyter               ║
║   0. 🚪 Quitter                                  ║
╚══════════════════════════════════════════════════╝
```

> **Commandes directes** (sans menu) :
> ```bash
> python run.py install    # installe les dépendances
> python run.py verify     # vérifie l'environnement
> python run.py demo       # démo rapide
> python run.py metrics    # calcule les métriques
> python run.py notebook   # ouvre Jupyter
> ```

---

## 💻 Utilisation dans votre code Python

```python
from esg_classifier import ESGClassifier

# Initialisation (utilise Mistral via Ollama)
classifier = ESGClassifier(model="mistral")

# ── Classer un document ──────────────────────────────────────────────
document = """
Tesla a réduit ses émissions de CO2 de 25% en 2024 grâce à son programme
d'énergie renouvelable. Cependant, des accidents du travail ont été signalés
dans l'usine de Berlin, et la rémunération du PDG a augmenté de 40%.
"""

result = classifier.classify_document(document, document_id="tesla_2024")

print("Labels détectés :", result.labels)
# {'E': True, 'S': True, 'G': True}

print("Scores de risque :", result.risk_scores)
# {'E': 0.18, 'S': 0.64, 'G': 0.72}

print("Nombre de mentions ESG :", len(result.mentions))
print("Résumé :", result.summary)

# ── Détail des mentions ──────────────────────────────────────────────
for mention in result.mentions:
    print(f"  [{mention.category}] {mention.subcategory} "
          f"— risque : {mention.risk_level} "
          f"(confiance : {mention.confidence:.0%})")

# ── Analyser une entreprise (plusieurs documents) ────────────────────
from esg_classifier import ESGClassifier

profile = classifier.classify_company(
    "Tesla Inc.",
    documents=[
        {"id": "rapport_env",  "text": "Tesla a réduit ses émissions…"},
        {"id": "rapport_rh",   "text": "Des accidents du travail ont été signalés…"},
        {"id": "rapport_gouv", "text": "La rémunération du PDG a augmenté de 40%…"},
    ]
)

print(f"Rating ESG : {profile.esg_rating}")          # ex: BB
print(f"Risque global : {profile.overall_risk}")      # ex: modere
print(f"Scores agrégés : {profile.aggregated_scores}")
```

---

## 📊 Dataset synthétique et métriques

Le fichier `utils.py` fournit deux fonctionnalités :

### Dataset synthétique

```python
from utils import generate_synthetic_dataset, get_all_companies, get_company_documents

# Générer 20 documents annotés pour l'entraînement/évaluation
dataset = generate_synthetic_dataset(n_samples=20, seed=42)
print(dataset[0])
# {'id': 'doc_000', 'text': '...', 'labels': {'E': True, 'S': False, 'G': False},
#  'risk_scores': {'E': 0.85, 'S': 0.0, 'G': 0.0}}

# Récupérer les profils d'entreprises réelles simulées
companies = get_all_companies()
for company in companies:
    docs = get_company_documents(company)
    print(f"{company['name']} → {len(docs)} documents")
```

### Calcul des métriques multi-label

```python
import numpy as np
from utils import compute_all_metrics, print_metrics_report

# y_true, y_pred : tableaux (n_samples, 3) avec 0/1 pour E, S, G
y_true   = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1]])
y_pred   = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0]])
y_scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.4, 0.8], [0.8, 0.7, 0.3]])

metrics = compute_all_metrics(y_true, y_pred, y_scores, label_names=["E", "S", "G"])
print_metrics_report(metrics, ["E", "S", "G"])
```

---

## 🆘 Dépannage

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError` | Package manquant | `pip install -r requirements.txt` |
| `Connection refused` | Ollama non démarré | `ollama serve` (terminal séparé) |
| `Model 'mistral' not found` | Modèle non téléchargé | `ollama pull mistral` |
| `python: command not found` | Python absent du PATH | Réinstaller Python en cochant "Add to PATH" |
| `Permission denied` (Linux/macOS) | Droits insuffisants | `pip install --user -r requirements.txt` |

---

## 🏗️ Architecture technique

```
Document texte
      │
      ▼
  Chunking (2000 mots, overlap 20%)
      │
      ▼
  Prompt ESG → Ollama (Mistral local)
      │
      ▼
  JSON structuré (mentions, risques, confiance)
      │
      ├─→ Labels binaires E/S/G
      ├─→ Scores de risque agrégés (0.0–1.0)
      └─→ ESGDocumentResult
              │
              ▼ (plusieurs documents)
          CompanyESGProfile
              ├─→ Rating (A / BBB / BB / B / CCC)
              └─→ Risque global (negligeable→critique)
```

---

## 📦 Dépendances

| Package | Rôle |
|---------|------|
| `ollama` | Interface avec le LLM local (Mistral) |
| `numpy` | Calculs matriciels (métriques) |
| `pandas` | Manipulation de données |
| `scikit-learn` | Métriques ML de référence |
| `matplotlib` / `seaborn` | Visualisations statiques |
| `plotly` | Graphiques interactifs (notebook) |
| `jupyter` | Exécution du notebook |

Installez tout en une commande :
```bash
pip install -r requirements.txt
```
