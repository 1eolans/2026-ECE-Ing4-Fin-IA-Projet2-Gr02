# Documentation API — FinRAG

Documentation de toutes les classes publiques du système FinRAG.

---

## Ingestion

### `FinancialDocumentLoader`

```python
from src.ingestion.document_loader import FinancialDocumentLoader

loader = FinancialDocumentLoader()
```

#### Méthodes

**`load(source: str | Path) → List[Document]`**

Charge un document depuis un chemin fichier.

```python
# PDF
docs = loader.load("data/samples/apple_annual_report_2023.pdf")

# CSV
docs = loader.load("data/samples/portfolio.csv")

# JSON news
docs = loader.load("data/samples/news/article_001.json")
```

Retourne une liste de `langchain.schema.Document` avec métadonnées enrichies.

---

**`load_directory(directory: str | Path) → List[Document]`**

Charge tous les documents supportés depuis un répertoire (récursif).

```python
docs = loader.load_directory("data/samples/")
# Retourne tous les PDFs, CSVs, JSONs et TXTs
```

---

### `FinancialTableExtractor`

```python
from src.ingestion.table_extractor import FinancialTableExtractor

extractor = FinancialTableExtractor(
    min_rows=2,               # Nombre minimum de lignes
    min_cols=2,               # Nombre minimum de colonnes
    accuracy_threshold=80.0,  # Seuil de précision camelot (0-100)
)
```

#### Méthodes

**`extract_tables_from_pdf(pdf_path: str | Path) → List[Document]`**

Extrait tous les tableaux d'un PDF. Retourne des Documents LangChain avec le tableau en markdown.

```python
table_docs = extractor.extract_tables_from_pdf("rapport.pdf")
for doc in table_docs:
    print(doc.metadata["page_number"])
    print(doc.page_content)  # Tableau markdown
```

---

**`tables_to_context(docs: List[Document]) → str`**

Agrège plusieurs tableaux en un seul contexte textuel formaté.

```python
context = extractor.tables_to_context(table_docs)
# Retourne: "**Tableaux financiers extraits (3 tableaux)**\n..."
```

---

### `IntelligentFinancialChunker`

```python
from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

chunker = IntelligentFinancialChunker(
    strategy=ChunkingStrategy.HYBRID,  # SEMANTIC | TABLE_AWARE | SENTENCE | HYBRID
    chunk_size=512,                    # Taille cible en caractères
    chunk_overlap=51,                  # Overlap (10%)
    min_chunk_length=50,               # Longueur minimum
)
```

#### Méthodes

**`chunk_documents(documents: List[Document]) → List[Document]`**

Découpe une liste de documents en chunks avec métadonnées enrichies.

```python
chunks = chunker.chunk_documents(docs)
# Chaque chunk a: chunk_index, chunk_strategy, contains_table,
#                 contains_numbers, time_period, financial_entities
```

---

## Retrieval

### `FinancialVectorStore`

```python
from src.retrieval.vector_store import FinancialVectorStore

vs = FinancialVectorStore(
    persist_dir="./data/chroma_db"  # Optionnel, utilise config par défaut
)
```

#### Méthodes

**`add_documents(documents: List[Document], batch_size: int = 100, show_progress: bool = True) → int`**

Indexe des documents dans ChromaDB. Retourne le nombre de chunks ajoutés.

```python
added = vs.add_documents(chunks, show_progress=True)
print(f"{added} chunks indexés")
```

---

**`similarity_search(query: str, k: int = 10, collection_types: Optional[List[str]] = None, where: Optional[Dict] = None) → List[Tuple[Document, float]]`**

Recherche dense par similarité cosine.

```python
results = vs.similarity_search(
    query="CA Apple FY2023",
    k=5,
    collection_types=["reports"],  # None = toutes les collections
)
for doc, score in results:
    print(f"Score: {score:.3f} | {doc.metadata['filename']}")
```

---

**`delete_by_source(source_filename: str) → int`**

Supprime tous les chunks d'une source. Retourne le nombre de chunks supprimés.

```python
deleted = vs.delete_by_source("apple_annual_report_2023.pdf")
```

---

**`list_sources() → List[Dict[str, Any]]`**

Liste toutes les sources indexées.

```python
sources = vs.list_sources()
# [{"filename": "...", "document_type": "...", "chunk_count": 42, ...}]
```

---

**`get_stats() → Dict[str, Any]`**

Retourne les statistiques globales.

```python
stats = vs.get_stats()
# {"total_chunks": 500, "total_sources": 12, "collections": {...}, "embedding_model": "..."}
```

---

### `HybridFinancialRetriever`

```python
from src.retrieval.retriever import HybridFinancialRetriever

retriever = HybridFinancialRetriever(
    vector_store=vs,
    top_k=10,               # Résultats finaux après RRF
    dense_k=20,             # Top-k dense avant RRF
    sparse_k=20,            # Top-k BM25 avant RRF
    time_decay_factor=0.1,  # Boost temporel
)
```

#### Méthodes

**`retrieve(query, date_range=None, document_type=None, ticker=None, min_relevance_score=0.0, use_hybrid=True) → List[Tuple[Document, float]]`**

Retrieval hybride avec RRF et filtres.

```python
results = retriever.retrieve(
    query="croissance Azure cloud Microsoft 2024",
    date_range=("2023", "2024"),
    document_type=["quarterly_report"],
    ticker="MSFT",
    min_relevance_score=0.1,
)
```

---

**`invalidate_bm25_cache() → None`**

Invalide le cache BM25 (à appeler après l'ajout de nouveaux documents).

```python
vs.add_documents(new_chunks)
retriever.invalidate_bm25_cache()
```

---

### `CrossEncoderReRanker`

```python
from src.retrieval.reranker import CrossEncoderReRanker

reranker = CrossEncoderReRanker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=5,
    cache_size=1000,
)
```

#### Méthodes

**`rerank(query: str, documents: List[Tuple[Document, float]], top_k: Optional[int] = None) → List[Tuple[Document, float]]`**

Re-classe les documents par pertinence CrossEncoder.

```python
reranked = reranker.rerank(
    query="marge brute Apple",
    documents=retrieval_results,
    top_k=5,
)
```

---

## Génération

### `FinancialAnswerGenerator`

```python
from src.generation.generator import FinancialAnswerGenerator

generator = FinancialAnswerGenerator()
```

#### Méthodes

**`generate(question, context_documents, context_scores=None, sub_queries=None, max_tokens=2000) → FinancialAnswer`**

Génère une réponse financière.

```python
answer = generator.generate(
    question="Quel est le CA d'Apple en FY2023 ?",
    context_documents=reranked_docs,
    context_scores=[0.9, 0.7, 0.6],
    sub_queries=["CA Apple 2023"],
)

print(answer.answer)          # Réponse markdown
print(answer.confidence_score) # Float 0.0-1.0
print(answer.citations)       # List[Citation]
print(answer.tokens_used)     # int
```

---

**`generate_stream(question, context_documents, context_scores=None, max_tokens=2000) → Iterator[str]`**

Génère en streaming (token par token).

```python
for token in generator.generate_stream(question, docs):
    print(token, end="", flush=True)
```

---

### `FinancialAnswer`

```python
@dataclass
class FinancialAnswer:
    question: str
    answer: str                  # Réponse en markdown
    citations: List[Citation]    # Citations vérifiables
    sub_queries: List[str]       # Sous-requêtes si décomposé
    confidence_score: float      # 0.0 à 1.0
    processing_time: float       # Secondes
    tokens_used: int
    context_docs_count: int
```

**`to_dict() → dict`** : Sérialise en dictionnaire JSON-compatible.

---

### `Citation`

```python
@dataclass
class Citation:
    chunk_id: str           # "filename::chunk_index"
    source_file: str        # Nom du fichier source
    page_number: Optional[int]
    date: Optional[str]     # "FY2023", "2024-06"
    excerpt: str            # 100 chars max du chunk
    relevance_score: float
```

**`to_markdown() → str`** : `[Source: filename, p.5, 2023]`

---

## Agents

### `FinancialQueryDecomposer`

```python
from src.agents.query_decomposer import FinancialQueryDecomposer

decomposer = FinancialQueryDecomposer()
```

#### Méthodes

**`decompose(question: str) → List[SubQuery]`**

Décompose une question complexe.

```python
sub_queries = decomposer.decompose(
    "Compare la croissance du CA d'Apple et Microsoft sur 3 ans"
)
for sq in sub_queries:
    print(f"[{sq.type}] {sq.query} (entities={sq.entities}, time={sq.time_filter})")
```

---

**`needs_decomposition(question: str) → bool`**

Détermine si la question nécessite une décomposition.

```python
if decomposer.needs_decomposition(question):
    sub_queries = decomposer.decompose(question)
```

---

### `FinancialRAGAgent`

```python
from src.agents.rag_agent import FinancialRAGAgent

agent = FinancialRAGAgent(
    vector_store=vs,
    retriever=retriever,
    reranker=reranker,
    generator=generator,  # Optionnel, créé automatiquement si None
)
```

#### Méthodes

**`answer(question, use_decomposition=True, date_range=None, document_type=None, ticker=None, max_context_docs=10) → FinancialAnswer`**

Pipeline RAG complet.

```python
answer = agent.answer(
    question="Comparez Apple et Microsoft en 2023",
    use_decomposition=True,
    date_range=("2022", "2024"),
    ticker=None,
)
```

---

**`get_relevant_sources(question: str) → List[Dict[str, Any]]`**

Retourne les sources pertinentes sans générer de réponse.

```python
sources = agent.get_relevant_sources("CA Apple")
# [{"filename": "...", "page": 1, "date": "2023", "score": 0.87, "excerpt": "..."}]
```

---

## Évaluation

### `RAGASEvaluator`

```python
from src.evaluation.evaluator import RAGASEvaluator

evaluator = RAGASEvaluator(
    agent=agent,
    eval_dataset_path="data/samples/eval_questions.json",
)
```

#### Méthodes

**`evaluate_batch(max_samples=10, save_report=True) → EvaluationReport`**

Évaluation batch sur le dataset.

```python
report = evaluator.evaluate_batch(max_samples=15, save_report=True)
print(f"Faithfulness: {report.avg_faithfulness:.3f}")
print(f"Score Global: {report.overall_score:.3f}")
# → Génère docs/evaluation_report.md et docs/evaluation_results.csv
```

---

**`evaluate_single(question, answer, contexts, ground_truth=None) → EvalSample`**

Évalue une seule réponse.

```python
sample = evaluator.evaluate_single(
    question="Quel est le CA d'Apple ?",
    answer="Le CA d'Apple est de 383,3 Md$.",
    contexts=["Apple CA 383 milliards FY2023..."],
    ground_truth="383,3 milliards de dollars",
)
print(f"Faithfulness: {sample.faithfulness:.3f}")
```

---

## Configuration

### `Settings` (`src/config.py`)

```python
from src.config import settings

# Accès aux paramètres
settings.LLM_MODEL           # "claude-sonnet-4-20250514"
settings.CHUNK_SIZE          # 512
settings.TOP_K_RETRIEVAL     # 10
settings.use_openai_embeddings  # True si OPENAI_API_KEY défini
settings.use_anthropic          # True si ANTHROPIC_API_KEY défini
settings.chroma_persist_path    # Path object vers ./data/chroma_db
```

Tous les paramètres sont surchargeables via variables d'environnement ou `.env`.

---

## Guide d'intégration rapide

```python
"""Exemple complet d'intégration FinRAG."""

from pathlib import Path
from src.retrieval.vector_store import FinancialVectorStore
from src.retrieval.retriever import HybridFinancialRetriever
from src.retrieval.reranker import CrossEncoderReRanker
from src.generation.generator import FinancialAnswerGenerator
from src.agents.rag_agent import FinancialRAGAgent
from src.ingestion.document_loader import FinancialDocumentLoader
from src.ingestion.chunker import IntelligentFinancialChunker

# 1. Initialiser les composants
vs = FinancialVectorStore()
retriever = HybridFinancialRetriever(vector_store=vs)
reranker = CrossEncoderReRanker()
generator = FinancialAnswerGenerator()
agent = FinancialRAGAgent(vs, retriever, reranker, generator)

# 2. Indexer un document
loader = FinancialDocumentLoader()
chunker = IntelligentFinancialChunker()

docs = loader.load("rapport_annuel.pdf")
chunks = chunker.chunk_documents(docs)
vs.add_documents(chunks)
retriever.invalidate_bm25_cache()

# 3. Interroger
answer = agent.answer("Quelle est la marge brute de l'entreprise ?")
print(answer.answer)
for citation in answer.citations:
    print(f"  [{citation.source_file}] {citation.excerpt}")

# 4. Évaluer
from src.evaluation.evaluator import RAGASEvaluator
evaluator = RAGASEvaluator(agent=agent)
report = evaluator.evaluate_batch(max_samples=5)
print(f"Score global: {report.overall_score:.3f}")
```
