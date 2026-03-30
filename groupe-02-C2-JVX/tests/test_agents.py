"""Tests des agents FinRAG (décomposition, pipeline complet)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.query_decomposer import (
    FinancialQueryDecomposer,
    SubQuery,
    _rule_based_decompose,
    _extract_entities,
)


class TestFinancialQueryDecomposer:
    """Tests pour FinancialQueryDecomposer."""

    def test_decomposer_initializes(self):
        """Test initialisation du décomposeur."""
        decomposer = FinancialQueryDecomposer()
        assert decomposer is not None

    def test_simple_query_no_decomposition(self):
        """Test qu'une question simple ne nécessite pas de décomposition."""
        decomposer = FinancialQueryDecomposer()
        needs = decomposer.needs_decomposition("Quel est le CA d'Apple ?")
        assert isinstance(needs, bool)

    def test_complex_query_needs_decomposition(self):
        """Test qu'une question complexe nécessite une décomposition."""
        decomposer = FinancialQueryDecomposer()
        complex_q = "Compare la croissance du CA d'Apple et Microsoft sur les 3 dernières années en tenant compte de l'impact de l'IA"
        needs = decomposer.needs_decomposition(complex_q)
        assert needs is True

    def test_rule_based_decompose_simple(self):
        """Test décomposition par règles pour une question simple."""
        sub_queries = _rule_based_decompose("Quel est le CA d'Apple en 2023 ?")
        assert len(sub_queries) >= 1
        assert all(isinstance(sq, SubQuery) for sq in sub_queries)

    def test_rule_based_decompose_comparison(self):
        """Test décomposition par règles pour une comparaison."""
        sub_queries = _rule_based_decompose("Comparez Apple et Microsoft sur leurs résultats 2023")
        assert len(sub_queries) >= 2
        entities_found = set()
        for sq in sub_queries:
            entities_found.update(sq.entities)
        assert len(entities_found) > 0

    def test_extract_entities(self):
        """Test extraction des entités."""
        entities = _extract_entities("Apple et Microsoft ont publié leurs résultats. AAPL et MSFT.")
        assert "AAPL" in entities
        assert "MSFT" in entities

    def test_decompose_returns_sorted_by_priority(self):
        """Test que les sous-requêtes sont triées par priorité."""
        decomposer = FinancialQueryDecomposer()
        sub_queries = decomposer.decompose("Comparez Apple et Microsoft")
        if len(sub_queries) > 1:
            priorities = [sq.priority for sq in sub_queries]
            assert priorities == sorted(priorities)

    def test_decompose_empty_query(self):
        """Test décomposition d'une requête vide."""
        decomposer = FinancialQueryDecomposer()
        result = decomposer.decompose("")
        assert result == []


class TestFinancialRAGAgent:
    """Tests pour FinancialRAGAgent."""

    def test_agent_initializes(self, vector_store, retriever, reranker):
        """Test initialisation de l'agent."""
        from src.agents.rag_agent import FinancialRAGAgent

        agent = FinancialRAGAgent(
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
        )
        assert agent is not None

    def test_agent_answers_simple_query(self, vector_store, retriever, reranker):
        """Test réponse à une question simple."""
        from src.agents.rag_agent import FinancialRAGAgent
        from src.generation.generator import FinancialAnswer

        agent = FinancialRAGAgent(
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
        )

        answer = agent.answer(
            question="Quel est le CA d'Apple en FY2023 ?",
            use_decomposition=False,
        )

        assert isinstance(answer, FinancialAnswer)
        assert answer.question == "Quel est le CA d'Apple en FY2023 ?"
        assert len(answer.answer) > 0
        assert answer.processing_time > 0

    def test_agent_get_relevant_sources(self, vector_store, retriever, reranker):
        """Test récupération des sources pertinentes."""
        from src.agents.rag_agent import FinancialRAGAgent

        agent = FinancialRAGAgent(
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
        )

        sources = agent.get_relevant_sources("Apple résultats financiers")
        assert isinstance(sources, list)
        for src in sources:
            assert "filename" in src
            assert "score" in src

    # FIX CRITIQUE : ajout du paramètre self manquant
    def test_deduplication(self):
        """Test de la déduplication des résultats."""
        from src.agents.rag_agent import _deduplicate_results

        doc = Document(
            page_content="Apple CA 383 milliards FY2023",
            metadata={"source": "a.pdf", "chunk_index": 0},
        )
        doc_dup = Document(
            page_content="Apple CA 383 milliards FY2023",
            metadata={"source": "a.pdf", "chunk_index": 0},
        )
        doc_other = Document(
            page_content="Microsoft CA 245 milliards FY2024",
            metadata={"source": "b.pdf", "chunk_index": 0},
        )

        results = [(doc, 0.9), (doc_dup, 0.85), (doc_other, 0.7)]
        deduped = _deduplicate_results(results)

        # Should have removed the duplicate
        assert len(deduped) <= 2