#!/usr/bin/env python3
"""
Script de vérification de l'environnement - Projet ESG C4
Vérifie que Python, les packages et Ollama sont correctement configurés.
Aucun environnement virtuel requis.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_python_version() -> bool:
    """Vérifie que Python 3.8+ est disponible."""
    print("🔍 Version Python...")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print(f"  ❌ Python {v.major}.{v.minor} trop ancien — Python 3.8+ requis.")
        return False
    print(f"  ✅ Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_required_packages() -> bool:
    """Vérifie que tous les packages requis sont installés."""
    print("\n🔍 Packages requis...")
    packages = {
        "ollama": "ollama",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "plotly": "plotly",
        "jupyter": "jupyter",
    }
    all_ok = True
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} — installez avec : pip install {package_name}")
            all_ok = False
    return all_ok


def check_project_files() -> bool:
    """Vérifie que les fichiers du projet sont présents."""
    print("\n🔍 Fichiers du projet...")
    required_files = [
        "esg_classifier.py",
        "utils.py",
        "esg_classification.ipynb",
        "requirements.txt",
        "run.py",
    ]
    project_dir = Path(__file__).parent
    all_ok = True
    for filename in required_files:
        path = project_dir / filename
        if path.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} introuvable")
            all_ok = False
    return all_ok


def check_imports() -> bool:
    """Vérifie que les imports principaux fonctionnent."""
    print("\n🔍 Imports principaux...")
    ok = True
    try:
        from esg_classifier import ESGClassifier  # noqa: F401
        print("  ✅ ESGClassifier (esg_classifier.py)")
    except Exception as e:
        print(f"  ❌ ESGClassifier : {e}")
        ok = False
    try:
        from utils import generate_synthetic_dataset, compute_all_metrics  # noqa: F401
        print("  ✅ utils (dataset + métriques)")
    except Exception as e:
        print(f"  ❌ utils : {e}")
        ok = False
    return ok


def check_ollama_service() -> bool:
    """Vérifie que le service Ollama répond."""
    print("\n🔍 Service Ollama...")
    try:
        import ollama
        ollama.list()
        print("  ✅ Ollama est en cours d'exécution")
        return True
    except ImportError:
        print("  ❌ Package 'ollama' non installé")
        return False
    except Exception as e:
        print(f"  ⚠️  Ollama ne répond pas : {e}")
        print("     → Lancez Ollama dans un autre terminal : ollama serve")
        return False


def check_mistral_model() -> bool:
    """Vérifie que le modèle Mistral est téléchargé."""
    print("\n🔍 Modèle Mistral...")
    try:
        import ollama
        models = ollama.list()
        for model in models.get("models", []):
            name = model.get("name", "") or model.get("model", "")
            if "mistral" in name.lower():
                print(f"  ✅ Trouvé : {name}")
                return True
        print("  ❌ Modèle Mistral introuvable")
        print("     → Téléchargez-le avec : ollama pull mistral")
        return False
    except Exception as e:
        print(f"  ⚠️  Impossible de vérifier les modèles : {e}")
        return False


def main() -> int:
    print("\n" + "=" * 55)
    print("   ESG Classifier C4 — Vérification de l'environnement")
    print("=" * 55)

    checks = [
        ("Python 3.8+",       check_python_version),
        ("Packages requis",   check_required_packages),
        ("Fichiers projet",   check_project_files),
        ("Imports Python",    check_imports),
        ("Service Ollama",    check_ollama_service),
        ("Modèle Mistral",    check_mistral_model),
    ]

    results: dict = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n❌ Erreur inattendue lors de '{name}' : {exc}")
            results[name] = False

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSUMÉ")
    print("=" * 55)

    critical_ok = all(
        results.get(k, False)
        for k in ("Python 3.8+", "Packages requis", "Fichiers projet", "Imports Python")
    )

    if not critical_ok:
        print("\n❌ Des vérifications critiques ont échoué.")
        print("\nSteps de correction :")
        print("  1. Assurez-vous d'avoir Python 3.8+  →  python.org")
        print("  2. Installez les dépendances         →  pip install -r requirements.txt")
        print("  3. Relancez ce script                →  python verify_setup.py")
        return 1

    print("\n✅ Vérifications critiques : OK")

    if not results.get("Service Ollama"):
        print("\n⚠️  Ollama n'est pas démarré.")
        print("   Dans un terminal séparé, lancez :  ollama serve")

    if not results.get("Modèle Mistral"):
        print("\n⚠️  Modèle Mistral absent.")
        print("   Téléchargez-le avec :  ollama pull mistral")

    if all(results.values()):
        print("\n🎉 Tout est prêt ! Lancez le projet avec :  python run.py")
    else:
        print("\n⚠️  Quelques étapes optionnelles restent à faire (voir ci-dessus).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
