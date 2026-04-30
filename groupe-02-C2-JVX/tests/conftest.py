"""
Fixtures communes pour les tests FinRAG.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ─── Sample Documents ────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(
        page_content=(
            "Apple Inc. a réalisé un chiffre d'affaires de 383,3 milliards de dollars en FY2023, "
            "en baisse de 2,8% par rapport à FY2022. Le bénéfice net s'est établi à 97,0 milliards. "
            "La marge brute a progressé à 44,1%. L'iPhone représente 52,3% du CA total."
        ),
        metadata={
            "source": "apple_annual_report_2023.pdf",
            "filename": "apple_annual_report_2023.pdf",
            "document_type": "annual_report",
            "page_number": 1,
            "date": "2023",
            "ticker_symbols": '["AAPL"]',
            "chunk_index": 0,
            "contains_table": False,
            "contains_numbers": True,
            "time_period": "FY2023",
            "financial_entities": '["Apple"]',
        },
    ),
    Document(
        page_content=(
            "Microsoft a publié un chiffre d'affaires de 64,7 milliards au T4 FY2024, "
            "en hausse de 15,2% YoY. Azure a crû de 29%. La marge opérationnelle est à 43,1%. "
            "Copilot compte 1,8 million de sièges entreprise déployés."
        ),
        metadata={
            "source": "microsoft_q4_2024.pdf",
            "filename": "microsoft_q4_2024.pdf",
            "document_type": "quarterly_report",
            "page_number": 2,
            "date": "2024",
            "ticker_symbols": '["MSFT"]',
            "chunk_index": 0,
            "contains_table": False,
            "contains_numbers": True,
            "time_period": "Q4 2024",
            "financial_entities": '["Microsoft"]',
        },
    ),
    Document(
        page_content=(
            "| Segment | FY2023 (Md$) | FY2022 (Md$) | Croissance |\n"
            "|---------|-------------|-------------|----------|\n"
            "| iPhone  | 200,6       | 205,5       | -2,4%    |\n"
            "| Services| 85,2        | 73,4        | +16,1%   |\n"
            "| Mac     | 29,4        | 40,2        | -26,9%   |\n"
        ),
        metadata={
            "source": "apple_annual_report_2023.pdf",
            "filename": "apple_annual_report_2023.pdf",
            "document_type": "financial_table",
            "page_number": 3,
            "date": "2023",
            "ticker_symbols": '["AAPL"]',
            "chunk_index": 1,
            "contains_table": True,
            "contains_numbers": True,
            "time_period": "FY2023",
            "financial_entities": '["Apple", "iPhone"]',
        },
    ),
    Document(
        page_content=(
            "NVIDIA devient la première capitalisation mondiale avec 3 300 milliards de dollars. "
            "Les GPU H100 sont au cœur de l'infrastructure IA. La demande dépasse l'offre. "
            "NVIDIA a réalisé 26,0 Md$ de CA au T1 FY2025, en hausse de 262% YoY."
        ),
        metadata={
            "source": "article_003.json",
            "filename": "article_003.json",
            "document_type": "news_article",
            "page_number": None,
            "date": "2024",
            "ticker_symbols": '["NVDA"]',
            "chunk_index": 0,
            "contains_table": False,
            "contains_numbers": True,
            "time_period": "2024",
            "financial_entities": '["NVIDIA"]',
        },
    ),
]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_documents() -> List[Document]:
    """Retourne une liste de documents d'exemple."""
    return SAMPLE_DOCS


@pytest.fixture(scope="session")
def tmp_chroma_dir(tmp_path_factory):
    """Crée un répertoire temporaire pour ChromaDB (session-scoped)."""
    return str(tmp_path_factory.mktemp("chroma_test"))


@pytest.fixture(scope="session")
def vector_store(tmp_chroma_dir, sample_documents):
    """Crée un FinancialVectorStore de test avec des documents d'exemple."""
    os.environ.setdefault("OPENAI_API_KEY", "")  # Force local embeddings

    from src.retrieval.vector_store import FinancialVectorStore

    vs = FinancialVectorStore(persist_dir=tmp_chroma_dir)
    vs.add_documents(sample_documents, show_progress=False)
    return vs


@pytest.fixture(scope="session")
def retriever(vector_store):
    """Crée un HybridFinancialRetriever de test."""
    from src.retrieval.retriever import HybridFinancialRetriever
    return HybridFinancialRetriever(
        vector_store=vector_store,
        top_k=3,
        dense_k=4,
        sparse_k=4,
    )


@pytest.fixture(scope="session")
def reranker():
    """Crée un CrossEncoderReRanker de test."""
    from src.retrieval.reranker import CrossEncoderReRanker
    return CrossEncoderReRanker(top_k=3)


@pytest.fixture
def sample_pdf_content() -> str:
    """Contenu textuel simulant un PDF financier."""
    return """
    Apple Inc. — Rapport Annuel FY2023

    Résultats financiers :
    - Chiffre d'affaires : 383,3 milliards de dollars
    - Bénéfice net : 97,0 milliards de dollars
    - Marge brute : 44,1%
    - EPS dilué : 6,13 dollars

    Segment iPhone : 200,6 Md$ (+52,3% du CA)
    Segment Services : 85,2 Md$ (+16,1% YoY)
    """


@pytest.fixture
def eval_questions() -> List[dict]:
    """Questions d'évaluation de test."""
    return [
        {
            "question": "Quel est le CA d'Apple en FY2023 ?",
            "ground_truth": "383,3 milliards de dollars",
            "document": "apple_annual_report_2023.pdf",
        },
        {
            "question": "Quelle est la croissance d'Azure au T4 FY2024 ?",
            "ground_truth": "29% de croissance",
            "document": "microsoft_q4_2024.pdf",
        },
    ]
