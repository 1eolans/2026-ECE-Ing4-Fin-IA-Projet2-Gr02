#!/usr/bin/env python3
"""
run.py — Lanceur principal du projet ESG C4
============================================
Usage :  python run.py            → menu interactif
         python run.py install    → installe les dépendances
         python run.py verify     → vérifie l'environnement
         python run.py demo       → lance la démo rapide
         python run.py notebook   → ouvre le notebook Jupyter
         python run.py metrics    → affiche les métriques sur le dataset synthétique
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
REQUIREMENTS = PROJECT_DIR / "requirements.txt"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list, **kwargs):
    """Lance une commande et laisse stdout/stderr s'afficher normalement."""
    return subprocess.run(cmd, **kwargs)


def _pip_install():
    print("\n📦 Installation des dépendances depuis requirements.txt …")
    result = _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    if result.returncode == 0:
        print("✅ Dépendances installées avec succès.")
    else:
        print("❌ Erreur lors de l'installation. Vérifiez le message ci-dessus.")
    return result.returncode == 0


def _check_package(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _ensure_dependencies() -> bool:
    """
    Vérifie les packages critiques ; propose l'installation automatique si manquants.
    Retourne True si tout est OK.
    """
    missing = [p for p in ("ollama", "numpy", "pandas", "sklearn") if not _check_package(p)]
    if not missing:
        return True

    print(f"\n⚠️  Packages manquants : {', '.join(missing)}")
    answer = input("   Installer automatiquement ? [O/n] : ").strip().lower()
    if answer in ("", "o", "oui", "y", "yes"):
        return _pip_install()
    else:
        print("   → Installez manuellement :  pip install -r requirements.txt")
        return False


# ─── Actions ──────────────────────────────────────────────────────────────────

def action_install():
    """Installe toutes les dépendances."""
    _pip_install()


def action_verify():
    """Lance le script de vérification de l'environnement."""
    script = PROJECT_DIR / "verify_setup.py"
    _run([sys.executable, str(script)])


def action_demo():
    """Démonstration rapide : classifie 3 extraits ESG avec le dataset synthétique."""
    if not _ensure_dependencies():
        return

    try:
        import ollama as _ollama
        _ollama.list()
    except Exception:
        print(
            "\n❌ Ollama n'est pas accessible.\n"
            "   Lancez-le dans un terminal séparé :  ollama serve\n"
            "   Puis relancez :  python run.py demo"
        )
        return

    from esg_classifier import ESGClassifier
    from utils import generate_synthetic_dataset

    print("\n" + "=" * 60)
    print("  DÉMO ESG — Classification de 3 documents synthétiques")
    print("=" * 60)

    dataset = generate_synthetic_dataset(n_samples=3, seed=0)
    classifier = ESGClassifier(model="mistral")

    for item in dataset:
        doc_id = item["id"]
        text = item["text"].strip()
        print(f"\n📄 Document : {doc_id}")
        print(f"   Texte    : {text[:120]}…")
        print("   Classification en cours…")
        result = classifier.classify_document(text, document_id=doc_id)
        print(f"   Labels prédits   : {result.labels}")
        print(f"   Labels attendus  : {item['labels']}")
        print(f"   Scores de risque : {result.risk_scores}")
        print(f"   Mentions trouvées: {len(result.mentions)}")
        print(f"   Résumé           : {result.summary[:150]}")

    print("\n✅ Démo terminée.")


def action_metrics():
    """Évalue le classifieur sur le dataset synthétique et affiche les métriques."""
    if not _ensure_dependencies():
        return

    try:
        import ollama as _ollama
        _ollama.list()
    except Exception:
        print(
            "\n❌ Ollama n'est pas accessible.\n"
            "   Lancez-le dans un terminal séparé :  ollama serve"
        )
        return

    import numpy as np
    from esg_classifier import ESGClassifier
    from utils import generate_synthetic_dataset, compute_all_metrics, print_metrics_report

    LABEL_NAMES = ["E", "S", "G"]
    dataset = generate_synthetic_dataset(n_samples=10, seed=42)
    classifier = ESGClassifier(model="mistral")

    y_true_list, y_pred_list, y_scores_list = [], [], []

    print(f"\n🔬 Évaluation sur {len(dataset)} documents…\n")
    for item in dataset:
        result = classifier.classify_document(item["text"], document_id=item["id"])
        true_row = [int(item["labels"].get(l, False)) for l in LABEL_NAMES]
        pred_row = [int(result.labels.get(l, False)) for l in LABEL_NAMES]
        score_row = [result.risk_scores.get(l, 0.0) for l in LABEL_NAMES]
        y_true_list.append(true_row)
        y_pred_list.append(pred_row)
        y_scores_list.append(score_row)
        print(f"  {item['id']} → prédit {pred_row}, attendu {true_row}")

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_scores = np.array(y_scores_list)

    metrics = compute_all_metrics(y_true, y_pred, y_scores, LABEL_NAMES)
    print_metrics_report(metrics, LABEL_NAMES)


def action_notebook():
    """Ouvre le notebook Jupyter dans le navigateur par défaut."""
    notebook = PROJECT_DIR / "esg_classification.ipynb"
    if not notebook.exists():
        print(f"❌ Notebook introuvable : {notebook}")
        return
    print("\n🚀 Lancement de Jupyter Notebook…")
    print("   (Fermez cette fenêtre de terminal pour arrêter Jupyter)")
    _run([sys.executable, "-m", "jupyter", "notebook", str(notebook)])


# ─── Menu interactif ──────────────────────────────────────────────────────────

MENU = {
    "1": ("📦 Installer les dépendances",        action_install),
    "2": ("🔍 Vérifier l'environnement",          action_verify),
    "3": ("⚡ Démo rapide (3 documents)",          action_demo),
    "4": ("📊 Évaluer les métriques (10 docs)",   action_metrics),
    "5": ("📓 Ouvrir le notebook Jupyter",        action_notebook),
    "0": ("🚪 Quitter",                            None),
}


def show_menu():
    print("\n" + "╔" + "═" * 50 + "╗")
    print("║   🌿  Projet ESG C4 — ECE Paris 2026" + " " * 13 + "║")
    print("╠" + "═" * 50 + "╣")
    for key, (label, _) in MENU.items():
        print(f"║   {key}. {label:<44}║")
    print("╚" + "═" * 50 + "╝")
    return input("\nChoix : ").strip()


def main():
    args = sys.argv[1:]

    # Mode ligne de commande directe
    if args:
        cmd = args[0].lower()
        dispatch = {
            "install":  action_install,
            "verify":   action_verify,
            "demo":     action_demo,
            "metrics":  action_metrics,
            "notebook": action_notebook,
        }
        fn = dispatch.get(cmd)
        if fn:
            fn()
        else:
            print(f"Commande inconnue : '{cmd}'")
            print("Commandes disponibles : install, verify, demo, metrics, notebook")
            sys.exit(1)
        return

    # Mode menu interactif
    while True:
        choice = show_menu()
        if choice == "0":
            print("\nAu revoir ! 🌍\n")
            break
        entry = MENU.get(choice)
        if entry is None:
            print("⚠️  Choix invalide, réessayez.")
            continue
        _, fn = entry
        if fn:
            fn()
        input("\nAppuyez sur Entrée pour revenir au menu…")


if __name__ == "__main__":
    main()
