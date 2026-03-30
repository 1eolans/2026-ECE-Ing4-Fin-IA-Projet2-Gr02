"""
Agent RAG financier avec décomposition de requêtes et exécution parallèle.
Orchestre le retrieval, re-ranking et la génération de réponses.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.agents.query_decomposer import FinancialQueryDecomposer, SubQuery
from src.generation.generator import FinancialAnswer, FinancialAnswerGenerator, Citation


# ─── Tool Functions ───────────────────────────────────────────────────────────

class AgentTools:
    """Outils disponibles pour l'agent RAG financier."""

    def __init__(self, retriever, reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    def search_financial_reports(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        ticker: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Recherche dans les rapports financiers."""
        return self._retriever.retrieve(
            query=query,
            date_range=date_range,
            document_type=["annual_report", "quarterly_report", "market_overview"],
            ticker=ticker,
        )

    def search_news(
        self,
        query: str,
        ticker: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Recherche dans les articles de news."""
        return self._retriever.retrieve(
            query=query,
            date_range=date_range,
            document_type=["news_article"],
            ticker=ticker,
        )

    def search_tables(
        self,
        query: str,
        metric_type: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Recherche dans les tableaux financiers extraits."""
        search_query = f"{query} {metric_type}" if metric_type else query
        return self._retriever.retrieve(
            query=search_query,
            document_type=["financial_table"],
        )

    def search_all(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        ticker: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Recherche dans toutes les sources."""
        return self._retriever.retrieve(
            query=query,
            date_range=date_range,
            ticker=ticker,
        )

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """Re-classe les documents par pertinence."""
        return self._reranker.rerank(query=query, documents=documents)


# ─── Deduplication ────────────────────────────────────────────────────────────

def _deduplicate_results(
    results: List[Tuple[Document, float]],
    similarity_threshold: float = 0.9,
) -> List[Tuple[Document, float]]:
    """
    Déduplique les résultats en supprimant les chunks quasi-identiques.

    Args:
        results: Liste de (Document, score).
        similarity_threshold: Seuil de similarité pour considérer deux chunks identiques.

    Returns:
        Liste dédupliquée.
    """
    seen_texts: List[str] = []
    deduplicated: List[Tuple[Document, float]] = []

    for doc, score in results:
        text = doc.page_content.strip()[:200]

        # Check if similar to existing
        is_duplicate = False
        for seen in seen_texts:
            # Simple character-level similarity
            common = sum(1 for a, b in zip(text, seen) if a == b)
            similarity = common / max(len(text), len(seen), 1)
            if similarity > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append((doc, score))
            seen_texts.append(text)

    return deduplicated


# ─── Main Agent ───────────────────────────────────────────────────────────────

class FinancialRAGAgent:
    """
    Agent RAG financier avec décomposition de requêtes.

    Workflow :
    1. Analyse la question → QueryDecomposer
    2. Execute les sous-requêtes (séquentiel ou parallèle via asyncio)
    3. Agrège les résultats avec déduplication
    4. Re-rank global
    5. Génère la réponse finale avec citations

    Gestion des erreurs : fallback sur RAG simple si décomposition échoue.
    """

    def __init__(
        self,
        vector_store,           # FinancialVectorStore
        retriever,              # HybridFinancialRetriever
        reranker,               # CrossEncoderReRanker
        generator: Optional[FinancialAnswerGenerator] = None,
    ) -> None:
        """
        Args:
            vector_store: Instance FinancialVectorStore.
            retriever: Instance HybridFinancialRetriever.
            reranker: Instance CrossEncoderReRanker.
            generator: Instance FinancialAnswerGenerator (créé si None).
        """
        self._vector_store = vector_store
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator or FinancialAnswerGenerator()
        self._decomposer = FinancialQueryDecomposer()
        self._tools = AgentTools(retriever=retriever, reranker=reranker)

        logger.info("FinancialRAGAgent initialisé")

    def answer(
        self,
        question: str,
        use_decomposition: bool = True,
        date_range: Optional[Tuple[str, str]] = None,
        document_type: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        max_context_docs: int = 10,
    ) -> FinancialAnswer:
        """
        Répond à une question financière via le pipeline RAG complet.

        Args:
            question: Question de l'utilisateur.
            use_decomposition: Active la décomposition de requêtes.
            date_range: Filtre temporel optionnel.
            document_type: Types de documents à interroger.
            ticker: Filtre ticker boursier.
            max_context_docs: Nombre max de documents de contexte.

        Returns:
            FinancialAnswer avec réponse, citations et métriques.
        """
        start_time = time.time()
        sub_query_texts: List[str] = []

        try:
            # 1. Decide whether to decompose
            if use_decomposition and self._decomposer.needs_decomposition(question):
                logger.info("Décomposition de la requête activée")
                results, sub_query_texts = self._execute_with_decomposition(
                    question=question,
                    date_range=date_range,
                    ticker=ticker,
                    max_docs=max_context_docs,
                )
            else:
                logger.info("Requête simple, RAG direct")
                results = self._execute_simple_rag(
                    query=question,
                    date_range=date_range,
                    document_type=document_type,
                    ticker=ticker,
                )

            # 2. Extract docs and scores
            if results:
                context_docs = [doc for doc, _ in results[:max_context_docs]]
                context_scores = [score for _, score in results[:max_context_docs]]
            else:
                context_docs = []
                context_scores = []

            # 3. Generate answer
            answer = self._generator.generate(
                question=question,
                context_documents=context_docs,
                context_scores=context_scores,
                sub_queries=sub_query_texts,
            )
            answer.processing_time = time.time() - start_time
            return answer

        except Exception as e:
            logger.error(f"Erreur agent RAG: {e}")
            # Fallback: simple RAG
            try:
                results = self._execute_simple_rag(question)
                context_docs = [doc for doc, _ in results[:max_context_docs]]
                context_scores = [score for _, score in results[:max_context_docs]]
                answer = self._generator.generate(
                    question=question,
                    context_documents=context_docs,
                    context_scores=context_scores,
                )
                answer.processing_time = time.time() - start_time
                return answer
            except Exception as e2:
                logger.error(f"Fallback RAG aussi échoué: {e2}")
                return FinancialAnswer(
                    question=question,
                    answer=f"⚠️ Erreur lors du traitement de votre question: {e2}",
                    processing_time=time.time() - start_time,
                )

    def _execute_with_decomposition(
        self,
        question: str,
        date_range: Optional[Tuple[str, str]],
        ticker: Optional[str],
        max_docs: int,
    ) -> Tuple[List[Tuple[Document, float]], List[str]]:
        """Exécute le RAG avec décomposition de requêtes."""
        sub_queries: List[SubQuery] = self._decomposer.decompose(question)
        sub_query_texts = [sq.query for sq in sub_queries]

        if not sub_queries:
            return self._execute_simple_rag(question, date_range=date_range, ticker=ticker), []

        # Execute sub-queries
        all_results: List[Tuple[Document, float]] = []

        for sq in sub_queries:
            try:
                sq_results = self._retriever.retrieve(
                    query=sq.query,
                    date_range=(sq.time_filter, sq.time_filter) if sq.time_filter else date_range,
                    ticker=ticker or (sq.entities[0] if sq.entities else None),
                )
                all_results.extend(sq_results)
            except Exception as e:
                logger.warning(f"Sous-requête '{sq.query[:50]}' échouée: {e}")
                continue

        if not all_results:
            return self._execute_simple_rag(question), []

        # Deduplicate
        deduped = _deduplicate_results(all_results)

        # Global re-rank
        reranked = self._reranker.rerank(
            query=question,
            documents=deduped,
            top_k=max_docs,
        )

        return reranked, sub_query_texts

    def _execute_simple_rag(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        document_type: Optional[List[str]] = None,
        ticker: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Exécute un RAG simple sans décomposition."""
        results = self._retriever.retrieve(
            query=query,
            date_range=date_range,
            document_type=document_type,
            ticker=ticker,
        )

        if results:
            results = self._reranker.rerank(
                query=query,
                documents=results,
                top_k=settings.TOP_K_RERANK,
            )

        return results

    def get_relevant_sources(self, question: str) -> List[Dict[str, Any]]:
        """
        Retourne les sources pertinentes sans générer de réponse.
        Utile pour le panel de sources dans l'UI.
        """
        results = self._execute_simple_rag(question)
        sources = []
        for doc, score in results:
            meta = doc.metadata
            sources.append({
                "filename": meta.get("filename", "unknown"),
                "page": meta.get("page_number"),
                "date": meta.get("date", ""),
                "doc_type": meta.get("document_type", ""),
                "score": round(float(score), 3),
                "excerpt": doc.page_content[:150].replace("\n", " "),
            })
        return sources
