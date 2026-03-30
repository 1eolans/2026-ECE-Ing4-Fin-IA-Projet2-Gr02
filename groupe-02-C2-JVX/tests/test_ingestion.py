"""Tests du pipeline d'ingestion FinRAG."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import FinancialDocumentLoader, _extract_tickers, _detect_document_type
from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy
from langchain_core.documents import Document


class TestFinancialDocumentLoader:
    """Tests pour FinancialDocumentLoader."""

    def test_load_csv(self, tmp_path):
        """Test chargement d'un fichier CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("date,ticker,price,volume\n2023-01-01,AAPL,150.0,1000000\n2023-01-02,MSFT,300.0,500000\n")

        loader = FinancialDocumentLoader()
        docs = loader.load(csv_file)

        assert len(docs) == 1
        assert "AAPL" in docs[0].page_content or "ticker" in docs[0].page_content
        assert docs[0].metadata["filename"] == "test.csv"

    def test_load_txt(self, tmp_path):
        """Test chargement d'un fichier texte."""
        txt_file = tmp_path / "test.txt"
        content = "Apple a réalisé un CA de 383,3 milliards en FY2023. Ticker: AAPL."
        txt_file.write_text(content, encoding="utf-8")

        loader = FinancialDocumentLoader()
        docs = loader.load(txt_file)

        assert len(docs) == 1
        assert content == docs[0].page_content
        assert docs[0].metadata["filename"] == "test.txt"

    def test_load_json_news(self, tmp_path):
        """Test chargement d'un article JSON."""
        article = {
            "title": "Apple dépasse les attentes",
            "content": "Le CA d'Apple a atteint 89,5 Md$ au T4 2023.",
            "source": "Reuters",
            "date": "2023-11-02",
            "ticker": "AAPL",
        }
        json_file = tmp_path / "article.json"
        json_file.write_text(json.dumps(article), encoding="utf-8")

        loader = FinancialDocumentLoader()
        docs = loader.load(json_file)

        assert len(docs) == 1
        assert "Apple" in docs[0].page_content
        assert docs[0].metadata["document_type"] == "news_article"
        assert docs[0].metadata["date"] == "2023-11-02"

    def test_load_directory(self, tmp_path):
        """Test chargement d'un répertoire."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Document financier Apple AAPL 2023", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("Document financier Microsoft MSFT 2024", encoding="utf-8")
        (tmp_path / "ignored.xyz").write_text("Fichier ignoré", encoding="utf-8")

        loader = FinancialDocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 2

    def test_unsupported_extension(self, tmp_path):
        """Test avec une extension non supportée."""
        file = tmp_path / "test.xyz"
        file.write_text("test")

        loader = FinancialDocumentLoader()
        docs = loader.load(file)

        assert docs == []

    def test_metadata_extraction(self):
        """Test extraction des métadonnées."""
        tickers = _extract_tickers("Apple (AAPL) et Microsoft (MSFT) ont publié leurs résultats")
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_document_type_detection(self):
        """Test détection du type de document."""
        doc_type = _detect_document_type("annual report 10-K 2023", "apple_annual_report_2023.pdf")
        assert doc_type == "annual_report"

        news_type = _detect_document_type("article news press release", "article_001.json")
        assert news_type == "news_article"


class TestIntelligentFinancialChunker:
    """Tests pour IntelligentFinancialChunker."""

    def test_semantic_chunking(self):
        """Test chunking sémantique."""
        text = " ".join([f"Phrase {i} avec du contenu financier Apple AAPL." for i in range(50)])
        doc = Document(page_content=text, metadata={"source": "test.txt", "filename": "test.txt"})

        chunker = IntelligentFinancialChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=200,
            chunk_overlap=20,
            min_chunk_length=10,
        )
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) >= 10
            assert "chunk_index" in chunk.metadata
            assert "contains_table" in chunk.metadata
            assert "contains_numbers" in chunk.metadata

    def test_table_aware_chunking(self):
        """Test chunking préservant les tableaux."""
        table_text = (
            "Contexte avant le tableau.\n\n"
            "| Colonne A | Colonne B | Colonne C |\n"
            "|-----------|-----------|----------|\n"
            "| Apple     | 383,3 Md$ | +16,1%   |\n"
            "| Microsoft | 245,1 Md$ | +15,7%   |\n\n"
            "Contexte après le tableau."
        )
        doc = Document(page_content=table_text, metadata={"source": "test.pdf", "filename": "test.pdf"})

        chunker = IntelligentFinancialChunker(
            strategy=ChunkingStrategy.TABLE_AWARE,
            min_chunk_length=10,
        )
        chunks = chunker.chunk_documents([doc])

        # Should have preserved the table as one chunk
        table_chunks = [c for c in chunks if c.metadata.get("contains_table")]
        assert len(table_chunks) >= 1
        table_chunk = table_chunks[0]
        assert "| Apple" in table_chunk.page_content or "Colonne" in table_chunk.page_content

    def test_hybrid_strategy(self):
        """Test stratégie hybride (auto-detection)."""
        narrative_text = "Apple a réalisé un excellent exercice en 2023. " * 20
        doc = Document(
            page_content=narrative_text,
            metadata={"source": "test.txt", "filename": "test.txt"},
        )

        chunker = IntelligentFinancialChunker(
            strategy=ChunkingStrategy.HYBRID,
            chunk_size=100,
            min_chunk_length=10,
        )
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 1
        # No table chunks expected for pure narrative
        table_chunks = [c for c in chunks if c.metadata.get("contains_table")]
        assert len(table_chunks) == 0

    def test_filters_short_chunks(self):
        """Test filtrage des chunks trop courts."""
        docs = [Document(page_content="OK " * 100, metadata={"source": "t.txt", "filename": "t.txt"})]
        chunker = IntelligentFinancialChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=50,
            min_chunk_length=100,  # High threshold
        )
        chunks = chunker.chunk_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) >= 100

    def test_chunk_metadata_tags(self):
        """Test que les tags de métadonnées sont correctement assignés."""
        text = "Apple a généré 383,3 milliards de dollars de CA en FY2023."
        doc = Document(page_content=text, metadata={"source": "test.txt", "filename": "test.txt"})

        chunker = IntelligentFinancialChunker(strategy=ChunkingStrategy.SEMANTIC, min_chunk_length=5)
        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 0
        chunk = chunks[0]
        assert isinstance(chunk.metadata.get("contains_numbers"), bool)
        assert isinstance(chunk.metadata.get("contains_table"), bool)
        assert isinstance(chunk.metadata.get("financial_entities"), list)
