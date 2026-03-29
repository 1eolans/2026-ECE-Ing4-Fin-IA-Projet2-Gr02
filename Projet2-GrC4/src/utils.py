"""
Utilitaires pour la Classification ESG Multi-label.
Regroupe la génération de datasets synthétiques et le calcul des métriques d'évaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# 1. PARTIE DATASET (Génération de données)
# =============================================================================

# ─── Templates de documents ESG ───────────────────────────────────────────────

ENVIRONMENTAL_PASSAGES = [
    {
        "text": """L'entreprise a émis 2,3 millions de tonnes de CO2 équivalent en 2024,
        soit une augmentation de 15% par rapport à l'année précédente. Aucun plan de
        réduction n'a été présenté aux actionnaires. Les installations de production
        continuent d'utiliser principalement des combustibles fossiles (87% du mix
        énergétique), sans calendrier de transition vers les énergies renouvelables.""",
        "labels": {"E": True, "S": False, "G": False},
        "risk_scores": {"E": 0.85, "S": 0.0, "G": 0.0},
    },
    {
        "text": """La société a investi 45 millions d'euros dans des panneaux solaires
        couvrant désormais 23% de ses besoins énergétiques. L'objectif est d'atteindre
        50% d'énergie renouvelable d'ici 2030. Des émissions de méthane ont toutefois
        été détectées sur le site de Toulouse, faisant l'objet d'une enquête interne.""",
        "labels": {"E": True, "S": False, "G": False},
        "risk_scores": {"E": 0.5, "S": 0.0, "G": 0.0},
    },
    {
        "text": """Notre programme de compensation carbone certifié Gold Standard
        nous a permis d'atteindre la neutralité carbone en 2023. Le ratio d'intensité
        hydrique a baissé de 30% grâce au recyclage des eaux de process.
        L'ensemble de notre chaîne logistique est désormais alimenté par des flottes
        électriques.""",
        "labels": {"E": True, "S": False, "G": False},
        "risk_scores": {"E": 0.15, "S": 0.0, "G": 0.0},
    },
]

SOCIAL_PASSAGES = [
    {
        "text": """Suite à l'audit indépendant commandé par l'ONG Transparence Travail,
        des conditions de travail dangereuses ont été documentées dans trois usines
        au Bangladesh. Des enfants de moins de 15 ans ont été identifiés parmi les
        employés. L'entreprise a contesté ces conclusions mais n'a pas fourni de
        données de conformité.""",
        "labels": {"E": False, "S": True, "G": False},
        "risk_scores": {"E": 0.0, "S": 0.95, "G": 0.0},
    },
    {
        "text": """Le taux d'accidents du travail est de 3,2 pour 1000 employés,
        supérieur à la moyenne sectorielle de 2,1. Un programme de réduction des
        accidents a été lancé en Q3 2024. La parité femme-homme dans les postes
        d'encadrement reste à 28%, en-deçà de l'objectif de 40% fixé pour 2025.""",
        "labels": {"E": False, "S": True, "G": False},
        "risk_scores": {"E": 0.0, "S": 0.55, "G": 0.0},
    },
    {
        "text": """Notre indice d'engagement employé atteint 78/100, en hausse de
        5 points. Les femmes représentent 45% du comité exécutif. Nous avons lancé
        un programme de formation professionnelle bénéficiant à 12 000 employés
        dans 15 pays.""",
        "labels": {"E": False, "S": True, "G": False},
        "risk_scores": {"E": 0.0, "S": 0.2, "G": 0.0},
    },
]

GOVERNANCE_PASSAGES = [
    {
        "text": """Le PDG cumule les fonctions de Président du Conseil d'Administration
        depuis 12 ans, en violation des recommandations du Code AFEP-MEDEF. Sa
        rémunération a augmenté de 43% alors que les bénéfices chutaient de 18%.
        Deux administrateurs indépendants ont démissionné en protestant contre le
        manque de transparence sur les transactions avec des parties liées.""",
        "labels": {"E": False, "S": False, "G": True},
        "risk_scores": {"E": 0.0, "S": 0.0, "G": 0.82},
    },
    {
        "text": """Le conseil d'administration compte 8 membres dont 3 indépendants
        (37,5%), en-deçà du seuil recommandé de 50%. La politique de rémunération
        variable est liée à des objectifs financiers à court terme uniquement, sans
        critères ESG. Un programme anti-corruption a été déployé mais sans mécanisme
        de dénonciation anonyme.""",
        "labels": {"E": False, "S": False, "G": True},
        "risk_scores": {"E": 0.0, "S": 0.0, "G": 0.52},
    },
    {
        "text": """La structure de gouvernance a été renforcée : 58% d'administrateurs
        indépendants, un comité d'audit entièrement indépendant, et une politique
        Say-on-Pay approuvée à 94%. Les objectifs ESG représentent désormais 30%
        de la rémunération variable des dirigeants.""",
        "labels": {"E": False, "S": False, "G": True},
        "risk_scores": {"E": 0.0, "S": 0.0, "G": 0.18},
    },
]

MULTI_LABEL_PASSAGES = [
    {
        "text": """La fermeture de l'usine de Dunkerque libère 120 000 tonnes de CO2
        annuellement mais entraîne la suppression de 450 emplois locaux. L'entreprise
        propose un plan de reconversion de 18 mois mais les syndicats contestent
        l'insuffisance des indemnités. Parallèlement, les rejets de métaux lourds
        dans la rivière voisine ont dépassé les seuils légaux en Q2 2024.""",
        "labels": {"E": True, "S": True, "G": False},
        "risk_scores": {"E": 0.72, "S": 0.68, "G": 0.0},
    },
    {
        "text": """Trois plaintes pour discrimination à l'embauche ont été déposées
        contre la filiale américaine. En parallèle, une enquête du régulateur SEC
        porte sur des irrégularités comptables dans les rapports de 2022 et 2023.
        Le directeur financier a exercé des stock-options 3 jours avant l'annonce
        de résultats records, faisant l'objet d'une enquête pour délit d'initié.""",
        "labels": {"E": False, "S": True, "G": True},
        "risk_scores": {"E": 0.0, "S": 0.75, "G": 0.88},
    },
    {
        "text": """L'entreprise n'a pas divulgué ses données d'émissions carbone
        dans son rapport annuel malgré les obligations CSRD. Le commissaire aux
        comptes a émis une réserve sur l'évaluation des actifs environnementaux.
        Le conseil d'administration n'inclut aucun expert en matière climatique.""",
        "labels": {"E": True, "S": False, "G": True},
        "risk_scores": {"E": 0.65, "S": 0.0, "G": 0.70},
    },
    {
        "text": """Le rapport de durabilité 2024 révèle des insuffisances majeures :
        aucun objectif carbone aligné avec l'Accord de Paris, un taux de fréquence
        d'accidents 4x la moyenne sectorielle, et un conseil d'administration sans
        administrateur indépendant. Les actionnaires minoritaires ont été écartés
        du vote sur la politique de rémunération lors de la dernière AGO.""",
        "labels": {"E": True, "S": True, "G": True},
        "risk_scores": {"E": 0.78, "S": 0.82, "G": 0.80},
    },
    {
        "text": """La société annonce un chiffre d'affaires record de 4,2 milliards
        d'euros pour l'exercice 2024, en hausse de 12% par rapport à l'an dernier.
        L'EBITDA s'établit à 840 millions d'euros. Le carnet de commandes est
        plein jusqu'à fin 2026. Un dividende de 2,50€ par action sera proposé
        à l'assemblée générale.""",
        "labels": {"E": False, "S": False, "G": False},
        "risk_scores": {"E": 0.0, "S": 0.0, "G": 0.0},
    },
]

COMPANY_PROFILES = [
    {
        "name": "TotalEnergies SA",
        "sector": "Énergie",
        "expected_risk": "eleve",
        "passages": [ENVIRONMENTAL_PASSAGES[0], SOCIAL_PASSAGES[1], GOVERNANCE_PASSAGES[1]],
    },
    {
        "name": "Schneider Electric SE",
        "sector": "Industrie",
        "expected_risk": "faible",
        "passages": [ENVIRONMENTAL_PASSAGES[2], SOCIAL_PASSAGES[2], GOVERNANCE_PASSAGES[2]],
    },
    {
        "name": "Danone SA",
        "sector": "Agroalimentaire",
        "expected_risk": "modere",
        "passages": [ENVIRONMENTAL_PASSAGES[1], SOCIAL_PASSAGES[0], GOVERNANCE_PASSAGES[1]],
    },
    {
        "name": "BNP Paribas SA",
        "sector": "Finance",
        "expected_risk": "modere",
        "passages": [MULTI_LABEL_PASSAGES[2], SOCIAL_PASSAGES[1], GOVERNANCE_PASSAGES[1]],
    },
    {
        "name": "Kering SA",
        "sector": "Luxe",
        "expected_risk": "critique",
        "passages": [MULTI_LABEL_PASSAGES[1], ENVIRONMENTAL_PASSAGES[0], GOVERNANCE_PASSAGES[0]],
    },
]


def generate_synthetic_dataset(n_samples: int = 20, seed: int = 42) -> List[Dict]:
    """
    Génère un dataset synthétique de documents ESG annotés.

    Args:
        n_samples: Nombre de documents à générer.
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Liste de dicts avec 'id', 'text', 'labels', 'risk_scores'.
    """
    random.seed(seed)
    all_passages = (
        ENVIRONMENTAL_PASSAGES
        + SOCIAL_PASSAGES
        + GOVERNANCE_PASSAGES
        + MULTI_LABEL_PASSAGES
    )
    dataset = []
    for i in range(n_samples):
        passage = random.choice(all_passages)
        dataset.append(
            {
                "id": f"doc_{i:03d}",
                "text": passage["text"],
                "labels": passage["labels"].copy(),
                "risk_scores": passage["risk_scores"].copy(),
            }
        )
    return dataset


def get_company_documents(company_profile: Dict) -> List[Dict]:
    """Retourne les documents structurés pour une entreprise."""
    docs = []
    for j, passage in enumerate(company_profile["passages"]):
        docs.append(
            {
                "id": f"{company_profile['name'].split()[0]}_doc_{j + 1}",
                "text": passage["text"],
            }
        )
    return docs


def get_all_companies() -> List[Dict]:
    """Retourne tous les profils d'entreprises de test."""
    return COMPANY_PROFILES


# =============================================================================
# 2. PARTIE MÉTRIQUES (Évaluation multi-label)
# =============================================================================

@dataclass
class MultiLabelMetrics:
    """Résultats d'évaluation multi-label."""
    hamming_loss: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_micro: float
    precision_at_1: float
    precision_at_2: float
    recall_at_1: float
    recall_at_2: float
    subset_accuracy: float
    per_label_f1: Dict
    coverage_error: float
    label_ranking_avg_precision: float


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true et y_pred doivent avoir la même forme.")
    return float(np.mean(y_true != y_pred))


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    n_samples, n_labels = y_scores.shape
    k = min(k, n_labels)
    precisions = []
    for i in range(n_samples):
        top_k_idx = np.argsort(y_scores[i])[::-1][:k]
        hits = sum(y_true[i, j] for j in top_k_idx)
        precisions.append(hits / k)
    return float(np.mean(precisions))


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    n_samples, _ = y_scores.shape
    k = min(k, y_scores.shape[1])
    recalls = []
    for i in range(n_samples):
        total_pos = y_true[i].sum()
        if total_pos == 0:
            recalls.append(1.0)
            continue
        top_k_idx = np.argsort(y_scores[i])[::-1][:k]
        hits = sum(y_true[i, j] for j in top_k_idx)
        recalls.append(hits / total_pos)
    return float(np.mean(recalls))


def per_label_f1(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]
) -> Dict:
    results = {}
    for idx, name in enumerate(label_names):
        tp = np.sum((y_pred[:, idx] == 1) & (y_true[:, idx] == 1))
        fp = np.sum((y_pred[:, idx] == 1) & (y_true[:, idx] == 0))
        fn = np.sum((y_pred[:, idx] == 0) & (y_true[:, idx] == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(y_true[:, idx].sum()),
        }
    return results


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def coverage_error(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    errors = []
    for i in range(y_scores.shape[0]):
        if y_true[i].sum() == 0:
            continue
        ranked = np.argsort(y_scores[i])[::-1]
        max_rank = 0
        for rank, idx in enumerate(ranked):
            if y_true[i, idx] == 1:
                max_rank = rank + 1
        errors.append(max_rank)
    return float(np.mean(errors)) if errors else 0.0


def label_ranking_avg_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    lraps = []
    for i in range(y_scores.shape[0]):
        if y_true[i].sum() == 0:
            continue
        ranked = np.argsort(y_scores[i])[::-1]
        precisions = []
        n_pos_found = 0
        for rank, idx in enumerate(ranked):
            if y_true[i, idx] == 1:
                n_pos_found += 1
                precisions.append(n_pos_found / (rank + 1))
        if precisions:
            lraps.append(np.mean(precisions))
    return float(np.mean(lraps)) if lraps else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    label_names: List[str],
) -> MultiLabelMetrics:
    label_f1 = per_label_f1(y_true, y_pred, label_names)
    f1_per_label = [v["f1"] for v in label_f1.values()]
    f1_macro = float(np.mean(f1_per_label))

    prec_per_label = [v["precision"] for v in label_f1.values()]
    rec_per_label = [v["recall"] for v in label_f1.values()]
    prec_macro = float(np.mean(prec_per_label))
    rec_macro = float(np.mean(rec_per_label))

    tp_total = np.sum((y_pred == 1) & (y_true == 1))
    fp_total = np.sum((y_pred == 1) & (y_true == 0))
    fn_total = np.sum((y_pred == 0) & (y_true == 1))
    prec_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_micro = (
        2 * prec_micro * rec_micro / (prec_micro + rec_micro)
        if (prec_micro + rec_micro) > 0
        else 0.0
    )

    return MultiLabelMetrics(
        hamming_loss=hamming_loss(y_true, y_pred),
        precision_macro=round(prec_macro, 4),
        recall_macro=round(rec_macro, 4),
        f1_macro=round(f1_macro, 4),
        f1_micro=round(float(f1_micro), 4),
        precision_at_1=precision_at_k(y_true, y_scores, k=1),
        precision_at_2=precision_at_k(y_true, y_scores, k=2),
        recall_at_1=recall_at_k(y_true, y_scores, k=1),
        recall_at_2=recall_at_k(y_true, y_scores, k=2),
        subset_accuracy=subset_accuracy(y_true, y_pred),
        per_label_f1=label_f1,
        coverage_error=coverage_error(y_true, y_scores),
        label_ranking_avg_precision=label_ranking_avg_precision(y_true, y_scores),
    )


def print_metrics_report(metrics: MultiLabelMetrics, label_names: List[str]) -> None:
    print("\n" + "=" * 60)
    print("  RAPPORT D'ÉVALUATION ESG MULTI-LABEL")
    print("=" * 60)
    print(f"\n{'Métrique':<35} {'Valeur':>10}")
    print("-" * 46)
    print(f"{'Hamming Loss (↓ = meilleur)':<35} {metrics.hamming_loss:>10.4f}")
    print(f"{'Subset Accuracy (exact match)':<35} {metrics.subset_accuracy:>10.4f}")
    print(f"{'F1 Macro':<35} {metrics.f1_macro:>10.4f}")
    print(f"{'F1 Micro':<35} {metrics.f1_micro:>10.4f}")
    print(f"{'Precision Macro':<35} {metrics.precision_macro:>10.4f}")
    print(f"{'Recall Macro':<35} {metrics.recall_macro:>10.4f}")
    print(f"{'Precision@1':<35} {metrics.precision_at_1:>10.4f}")
    print(f"{'Precision@2':<35} {metrics.precision_at_2:>10.4f}")
    print(f"{'Recall@1':<35} {metrics.recall_at_1:>10.4f}")
    print(f"{'Recall@2':<35} {metrics.recall_at_2:>10.4f}")
    print(f"{'Coverage Error (↓ = meilleur)':<35} {metrics.coverage_error:>10.4f}")
    print(f"{'LRAP (↑ = meilleur)':<35} {metrics.label_ranking_avg_precision:>10.4f}")

    print(f"\n{'─' * 46}")
    print("  Métriques par label (E / S / G)")
    print(f"{'─' * 46}")
    for name in label_names:
        m = metrics.per_label_f1.get(name, {})
        print(f"\n  [{name}]")
        print(f"    Precision : {m.get('precision', 0):.4f}")
        print(f"    Recall    : {m.get('recall', 0):.4f}")
        print(f"    F1        : {m.get('f1', 0):.4f}")
        print(f"    Support   : {m.get('support', 0)} exemples")
    print("=" * 60 + "\n")
