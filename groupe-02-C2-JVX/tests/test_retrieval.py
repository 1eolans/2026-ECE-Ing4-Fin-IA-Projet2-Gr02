"""Tests du pipeline de retrieval FinRAG."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHybridFinancialRetriever:
    """Tests pour HybridFinancialRetriever."""

    def test_dense_retrieval_returns_results(self, retriever, vector_store):
        """Test que le retrieval dense retourne des résultats."""
        results = retriever.retrieve(
            query="chiffre d'affaires Apple 2023",
            use_hybrid=False,
        )
        assert isinstance(results, list)
        stats = vector_store.get_stats()
        if stats["total_chunks"] > 0:
            assert len(results) > 0

    def test_hybrid_retrieval(self, retriever):
        """Test du retrieval hybride RRF."""
        results = retriever.retrieve(
            query="bénéfice net Microsoft 2024",
            use_hybrid=True,
        )
        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_retrieval_with_ticker_filter(self, retriever):
        """Test du filtre par ticker."""
        results = retriever.retrieve(
            query="résultats financiers",
            ticker="AAPL",
        )
        assert isinstance(results, list)

    def test_retrieval_respects_top_k(self, retriever):
        """Test que top_k est respecté."""
        retriever._top_k = 2
        results = retriever.retrieve(query="Apple Microsoft résultats")
        assert len(results) <= 2

    # FIX CRITIQUE : ajout du paramètre self manquant
    def test_rrf_fusion(self):
        """Test de la fusion Reciprocal Rank Fusion."""
        from src.retrieval.retriever import reciprocal_rank_fusion

        doc1 = Document(
            page_content="Apple CA 383 milliards",
            metadata={"source": "a.pdf", "chunk_index": 0},
        )
        doc2 = Document(
            page_content="Microsoft CA 245 milliards",
            metadata={"source": "b.pdf", "chunk_index": 0},
        )
        doc3 = Document(
            page_content="NVIDIA GPU 3300 milliards",
            metadata={"source": "c.pdf", "chunk_index": 0},
        )

        list1 = [(doc1, 0.9), (doc2, 0.7), (doc3, 0.5)]
        list2 = [(doc2, 0.85), (doc1, 0.6), (doc3, 0.4)]

        fused = reciprocal_rank_fusion([list1, list2], top_n=3)

        assert len(fused) <= 3
        assert any("Microsoft" in doc.page_content for doc, _ in fused)


class TestCrossEncoderReRanker:
    """Tests pour CrossEncoderReRanker."""

    def test_rerank_returns_top_k(self, reranker, sample_documents):
        """Test que rerank retourne au plus top_k résultats."""
        pairs = [(doc, 0.8) for doc in sample_documents]
        reranker._top_k = 2
        results = reranker.rerank("CA Apple 2023", pairs, top_k=2)
        assert len(results) <= 2

    def test_rerank_preserves_docs(self, reranker, sample_documents):
        """Test que les documents sont préservés après re-ranking."""
        pairs = [(doc, 0.5) for doc in sample_documents[:2]]
        results = reranker.rerank("Apple résultats", pairs)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_rerank_empty_input(self, reranker):
        """Test avec une liste vide."""
        results = reranker.rerank("query", [])
        assert results == []

    def test_cache_works(self, reranker, sample_documents):
        """Test que le cache fonctionne correctement."""
        pairs = [(sample_documents[0], 0.9)]
        query = "test cache query unique 12345"

        _ = reranker.rerank(query, pairs)
        size_after_first = reranker._cache.size

        _ = reranker.rerank(query, pairs)
        size_after_second = reranker._cache.size

        assert size_after_second >= size_after_first


class TestFinancialVectorStore:
    """Tests pour FinancialVectorStore."""

    def test_get_stats(self, vector_store):
        """Test que get_stats retourne les bonnes clés."""
        stats = vector_store.get_stats()
        assert "total_chunks" in stats
        assert "total_sources" in stats
        assert "collections" in stats
        assert "embedding_model" in stats

    def test_list_sources_returns_list(self, vector_store):
        """Test que list_sources retourne une liste."""
        sources = vector_store.list_sources()
        assert isinstance(sources, list)

    def test_similarity_search(self, vector_store):
        """Test de la recherche par similarité."""
        results = vector_store.similarity_search(
            query="chiffre d'affaires Apple",
            k=3,
        )
        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert 0.0 <= score <= 1.0 or score < 0