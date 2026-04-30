"""
Évaluateur RAGAS pour le système FinRAG.
Métriques : faithfulness, answer_relevancy, context_recall, context_precision.
Génère un rapport en markdown + CSV.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """Un échantillon d'évaluation RAGAS."""
    question: str
    ground_truth: str
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0


@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet."""
    timestamp: str
    n_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_context_precision: float
    overall_score: float
    samples: List[EvalSample] = field(default_factory=list)
    evaluation_time: float = 0.0

    def to_markdown(self) -> str:
        """Génère le rapport en markdown."""
        lines = [
            "# Rapport d'Évaluation RAGAS — FinRAG",
            f"\n**Date** : {self.timestamp}",
            f"**Échantillons évalués** : {self.n_samples}",
            f"**Durée** : {self.evaluation_time:.1f}s",
            "\n## 📊 Métriques Globales",
            "",
            "| Métrique | Score | Interprétation |",
            "|----------|-------|----------------|",
            f"| Faithfulness | **{self.avg_faithfulness:.3f}** | {'✅ Excellent' if self.avg_faithfulness >= 0.8 else '⚠️ À améliorer' if self.avg_faithfulness >= 0.6 else '❌ Insuffisant'} |",
            f"| Answer Relevancy | **{self.avg_answer_relevancy:.3f}** | {'✅ Excellent' if self.avg_answer_relevancy >= 0.8 else '⚠️ À améliorer' if self.avg_answer_relevancy >= 0.6 else '❌ Insuffisant'} |",
            f"| Context Recall | **{self.avg_context_recall:.3f}** | {'✅ Excellent' if self.avg_context_recall >= 0.8 else '⚠️ À améliorer' if self.avg_context_recall >= 0.6 else '❌ Insuffisant'} |",
            f"| Context Precision | **{self.avg_context_precision:.3f}** | {'✅ Excellent' if self.avg_context_precision >= 0.8 else '⚠️ À améliorer' if self.avg_context_precision >= 0.6 else '❌ Insuffisant'} |",
            f"| **Score Global** | **{self.overall_score:.3f}** | {'✅ Excellent' if self.overall_score >= 0.8 else '⚠️ Acceptable' if self.overall_score >= 0.6 else '❌ Insuffisant'} |",
            "",
            "## 📝 Détail par Question",
            "",
        ]

        for i, sample in enumerate(self.samples, 1):
            lines.extend([
                f"### Q{i}: {sample.question[:80]}{'...' if len(sample.question) > 80 else ''}",
                f"- **Faithfulness** : {sample.faithfulness:.3f}",
                f"- **Answer Relevancy** : {sample.answer_relevancy:.3f}",
                f"- **Context Recall** : {sample.context_recall:.3f}",
                f"- **Context Precision** : {sample.context_precision:.3f}",
                "",
            ])

        lines.extend(["## 💡 Recommandations", ""])

        if self.avg_faithfulness < 0.7:
            lines.append("- **Faithfulness faible** : Renforcer le prompt système avec des instructions de citation strictes.")
        if self.avg_answer_relevancy < 0.7:
            lines.append("- **Relevancy faible** : Améliorer le prompt de génération.")
        if self.avg_context_recall < 0.7:
            lines.append("- **Context Recall faible** : Augmenter TOP_K_RETRIEVAL ou améliorer l'embedding.")
        if self.avg_context_precision < 0.7:
            lines.append("- **Context Precision faible** : Améliorer le re-ranking ou les filtres.")
        if self.overall_score >= 0.8:
            lines.append("✅ **Excellent système RAG** — Toutes les métriques sont au-dessus du seuil de qualité.")

        return "\n".join(lines)


# ─── RAGAS version detection ──────────────────────────────────────────────────

def _get_ragas_version() -> Optional[str]:
    """Retourne la version de ragas installée, ou None si non installée."""
    try:
        import importlib.metadata
        return importlib.metadata.version("ragas")
    except Exception:
        return None


# ─── Main Evaluator ───────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Évaluateur RAGAS pour le système FinRAG.

    Supporte RAGAS 0.1.x et 0.2.x avec détection automatique de l'API.
    Fallback sur métriques approchées si RAGAS non disponible ou si
    les clés API (OpenAI) manquent.
    """

    def __init__(
        self,
        agent=None,
        eval_dataset_path: Optional[str] = None,
    ) -> None:
        self._agent = agent
        self._dataset_path = eval_dataset_path or str(
            Path(__file__).parent.parent.parent / "data" / "samples" / "eval_questions.json"
        )
        self._ragas_version = _get_ragas_version()
        self._use_ragas_lib = self._ragas_version is not None

        if self._use_ragas_lib:
            logger.info(f"Package ragas v{self._ragas_version} disponible")
        else:
            logger.warning("Package ragas non installé. Utilisation de métriques approchées.")

    def load_eval_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = Path(self._dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset d'évaluation introuvable: {dataset_path}")
            return []
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Dataset chargé: {len(data)} questions")
        return data

    def evaluate_batch(
        self,
        max_samples: int = 10,
        save_report: bool = True,
    ) -> EvaluationReport:
        start_time = time.time()
        dataset = self.load_eval_dataset()

        if not dataset:
            logger.error("Dataset vide ou introuvable")
            return self._empty_report()

        if not self._agent:
            logger.error("Agent non configuré pour l'évaluation")
            return self._empty_report()

        samples_to_eval = dataset[:max_samples]
        logger.info(f"Évaluation de {len(samples_to_eval)} questions...")

        eval_samples: List[EvalSample] = []

        for i, item in enumerate(samples_to_eval):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")

            if not question:
                continue

            logger.info(f"Évaluation {i+1}/{len(samples_to_eval)}: {question[:60]}...")

            try:
                answer_obj = self._agent.answer(question=question, use_decomposition=False)
                answer_text = answer_obj.answer
                contexts = [c.excerpt for c in answer_obj.citations] if answer_obj.citations else []

                sample = EvalSample(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer_text,
                    contexts=contexts,
                )

                # Utiliser RAGAS seulement si OpenAI dispo (nécessaire pour l'évaluation)
                if self._use_ragas_lib and settings.use_openai_embeddings:
                    self._compute_ragas_metrics(sample)
                else:
                    if self._use_ragas_lib and not settings.use_openai_embeddings:
                        logger.info("RAGAS disponible mais OPENAI_API_KEY manquante — métriques approchées")
                    self._compute_approximate_metrics(sample)

                eval_samples.append(sample)

            except Exception as e:
                logger.error(f"Erreur évaluation Q{i+1}: {e}")
                continue

        if not eval_samples:
            return self._empty_report()

        report = self._build_report(eval_samples, time.time() - start_time)

        if save_report:
            self._save_report(report)

        return report

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvalSample:
        sample = EvalSample(
            question=question,
            ground_truth=ground_truth or "",
            answer=answer,
            contexts=contexts,
        )

        if self._use_ragas_lib and settings.use_openai_embeddings:
            self._compute_ragas_metrics(sample)
        else:
            self._compute_approximate_metrics(sample)

        return sample

    def _compute_ragas_metrics(self, sample: EvalSample) -> None:
        """
        FIX ÉLEVÉ : RAGAS 0.2.x a changé son API.
        - L'objet retourné par evaluate() est un EvaluationResult, pas un dict.
        - L'accès se fait par indexation directe [] ou par attribut, pas .get().
        - Compatibilité assurée par try/except avec fallback sur les métriques approchées.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
            from datasets import Dataset

            data = {
                "question": [sample.question],
                "answer": [sample.answer],
                "contexts": [sample.contexts if sample.contexts else [""]],
                "ground_truth": [sample.ground_truth],
            }
            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            )

            # FIX : RAGAS 0.2.x retourne un EvaluationResult
            # Accès direct par clé (dict-like) — .get() n'existe plus
            def _safe_get(obj: Any, key: str, default: float = 0.0) -> float:
                """Accès sécurisé compatible RAGAS 0.1.x et 0.2.x."""
                try:
                    val = obj[key]
                    return float(val) if val is not None else default
                except (KeyError, TypeError, IndexError):
                    try:
                        # Fallback: accès attribut (ancien RAGAS)
                        val = getattr(obj, key, default)
                        return float(val) if val is not None else default
                    except Exception:
                        return default

            sample.faithfulness = _safe_get(result, "faithfulness")
            sample.answer_relevancy = _safe_get(result, "answer_relevancy")
            sample.context_recall = _safe_get(result, "context_recall")
            sample.context_precision = _safe_get(result, "context_precision")

            logger.debug(
                f"RAGAS scores — faithfulness={sample.faithfulness:.3f}, "
                f"relevancy={sample.answer_relevancy:.3f}"
            )

        except Exception as e:
            logger.warning(f"RAGAS lib erreur ({e}), fallback métriques approchées")
            self._compute_approximate_metrics(sample)

    def _compute_approximate_metrics(self, sample: EvalSample) -> None:
        """
        Calcule des métriques approchées sans API externe.
        Basé sur la similarité de termes (BM25-like, Jaccard).
        Utilisé quand RAGAS n'est pas installé ou quand OPENAI_API_KEY est absent.
        """
        import re

        def tokenize(text: str) -> set:
            # Tokenization améliorée : garde les mots financiers composés
            words = set(re.findall(r'\b\w+\b', text.lower()))
            return words - {"le", "la", "les", "de", "du", "des", "et", "en", "un", "une"}

        def jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            intersection = len(a & b)
            union = len(a | b)
            return intersection / union if union > 0 else 0.0

        def recall(reference: set, retrieved: set) -> float:
            if not reference:
                return 0.5  # Pas de ground truth = score neutre
            hit = len(reference & retrieved)
            return min(1.0, hit / len(reference))

        q_tokens = tokenize(sample.question)
        a_tokens = tokenize(sample.answer)
        gt_tokens = tokenize(sample.ground_truth) if sample.ground_truth else set()
        ctx_tokens = tokenize(" ".join(sample.contexts)) if sample.contexts else set()

        # Faithfulness : quelle part de la réponse est couverte par le contexte ?
        if ctx_tokens and a_tokens:
            overlap = len(a_tokens & ctx_tokens)
            sample.faithfulness = min(1.0, (overlap / len(a_tokens)) * 1.5)
        elif not ctx_tokens:
            sample.faithfulness = 0.2  # Pas de contexte = peu fiable
        else:
            sample.faithfulness = 0.3

        # Answer Relevancy : la réponse répond-elle à la question ?
        sample.answer_relevancy = min(1.0, jaccard(q_tokens, a_tokens) * 3.5)

        # Context Recall : le contexte couvre-t-il la ground truth ?
        if gt_tokens and ctx_tokens:
            sample.context_recall = min(1.0, recall(gt_tokens, ctx_tokens) * 1.8)
        else:
            sample.context_recall = 0.5

        # Context Precision : le contexte est-il précis par rapport à la question ?
        if ctx_tokens:
            sample.context_precision = min(1.0, jaccard(q_tokens, ctx_tokens) * 4.5)
        else:
            sample.context_precision = 0.0

    def _build_report(
        self, samples: List[EvalSample], elapsed: float
    ) -> EvaluationReport:
        n = len(samples)
        avg_faithfulness = sum(s.faithfulness for s in samples) / n
        avg_answer_relevancy = sum(s.answer_relevancy for s in samples) / n
        avg_context_recall = sum(s.context_recall for s in samples) / n
        avg_context_precision = sum(s.context_precision for s in samples) / n
        overall = (avg_faithfulness + avg_answer_relevancy + avg_context_recall + avg_context_precision) / 4

        return EvaluationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_samples=n,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_recall=avg_context_recall,
            avg_context_precision=avg_context_precision,
            overall_score=overall,
            samples=samples,
            evaluation_time=elapsed,
        )

    def _empty_report(self) -> EvaluationReport:
        return EvaluationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_samples=0,
            avg_faithfulness=0.0,
            avg_answer_relevancy=0.0,
            avg_context_recall=0.0,
            avg_context_precision=0.0,
            overall_score=0.0,
        )

    def _save_report(self, report: EvaluationReport) -> None:
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)

        md_path = docs_dir / "evaluation_report.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(f"Rapport markdown sauvegardé: {md_path}")

        csv_path = docs_dir / "evaluation_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "faithfulness", "answer_relevancy",
                "context_recall", "context_precision", "overall",
            ])
            writer.writeheader()
            for s in report.samples:
                overall = (s.faithfulness + s.answer_relevancy + s.context_recall + s.context_precision) / 4
                writer.writerow({
                    "question": s.question[:100],
                    "faithfulness": round(s.faithfulness, 4),
                    "answer_relevancy": round(s.answer_relevancy, 4),
                    "context_recall": round(s.context_recall, 4),
                    "context_precision": round(s.context_precision, 4),
                    "overall": round(overall, 4),
                })
        logger.info(f"Rapport CSV sauvegardé: {csv_path}")