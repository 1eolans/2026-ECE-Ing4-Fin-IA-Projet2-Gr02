"""Tests de la génération de réponses FinRAG."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.generator import (
    FinancialAnswerGenerator,
    FinancialAnswer,
    Citation,
    _build_context,
    _extract_citations,
    _estimate_confidence,
)


class TestFinancialAnswerGenerator:
    """Tests pour FinancialAnswerGenerator."""

    def test_generator_initializes(self):
        """Test que le générateur s'initialise sans erreur."""
        gen = FinancialAnswerGenerator()
        assert gen is not None

    def test_generate_with_empty_context(self):
        """Test génération sans documents."""
        gen = FinancialAnswerGenerator()
        answer = gen.generate(
            question="Quel est le CA d'Apple ?",
            context_documents=[],
        )
        assert isinstance(answer, FinancialAnswer)
        assert answer.question == "Quel est le CA d'Apple ?"
        assert len(answer.answer) > 0
        assert answer.confidence_score == 0.0

    def test_generate_with_context(self, sample_documents):
        """Test génération avec des documents de contexte."""
        gen = FinancialAnswerGenerator()
        answer = gen.generate(
            question="Quel est le CA d'Apple en FY2023 ?",
            context_documents=sample_documents[:2],
            context_scores=[0.9, 0.7],
        )

        assert isinstance(answer, FinancialAnswer)
        assert answer.question == "Quel est le CA d'Apple en FY2023 ?"
        assert len(answer.answer) > 0
        assert isinstance(answer.citations, list)
        assert answer.processing_time > 0

    def test_financial_answer_to_dict(self, sample_documents):
        """Test sérialisation FinancialAnswer."""
        gen = FinancialAnswerGenerator()
        answer = gen.generate(
            question="Test question",
            context_documents=sample_documents[:1],
        )

        d = answer.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "citations" in d
        assert "confidence_score" in d
        assert "processing_time" in d


class TestCitation:
    """Tests pour la classe Citation."""

    def test_citation_to_markdown(self):
        """Test formatage markdown d'une citation."""
        cit = Citation(
            chunk_id="test::0",
            source_file="apple_annual_report_2023.pdf",
            page_number=5,
            date="2023",
            excerpt="Le CA d'Apple est de 383,3 Md$",
            relevance_score=0.9,
        )
        md = cit.to_markdown()
        assert "apple_annual_report_2023.pdf" in md
        assert "p.5" in md
        assert "2023" in md

    def test_citation_without_page(self):
        """Test citation sans numéro de page."""
        cit = Citation(
            chunk_id="test::0",
            source_file="article.json",
            page_number=None,
            date="2024",
            excerpt="Test excerpt",
            relevance_score=0.5,
        )
        md = cit.to_markdown()
        assert "article.json" in md
        assert "p." not in md


class TestHelperFunctions:
    """Tests des fonctions utilitaires."""

    def test_build_context(self, sample_documents):
        """Test construction du contexte."""
        context = _build_context(sample_documents[:2])
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Document" in context

    def test_build_context_max_chars(self, sample_documents):
        """Test limitation du contexte."""
        context = _build_context(sample_documents, max_context_chars=100)
        assert len(context) <= 200  # Some flexibility for the headers

    def test_extract_citations(self, sample_documents):
        """Test extraction des citations."""
        citations = _extract_citations(sample_documents[:2], scores=[0.9, 0.7])
        assert len(citations) == 2
        for cit in citations:
            assert isinstance(cit, Citation)
            assert cit.source_file != ""

    def test_estimate_confidence(self):
        """Test estimation de la confiance."""
        # Good answer with data and citations
        mock_citations = [
            Citation("a::0", "test.pdf", 1, "2023", "excerpt", 0.9)
        ]
        score = _estimate_confidence(
            "Apple a réalisé 383,3 Md$ de CA en FY2023.",
            mock_citations,
            n_docs=3,
        )
        assert 0.0 <= score <= 1.0

        # Empty answer
        score_empty = _estimate_confidence("", [], 0)
        assert score_empty < 0.5
