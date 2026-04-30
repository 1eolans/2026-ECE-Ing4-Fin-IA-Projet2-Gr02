# FinRAG — Système RAG pour l'Analyse Financière

> Système de Retrieval-Augmented Generation de niveau production pour l'analyse de documents financiers complexes. Développé dans le cadre du projet ECE Ing4 IA Finance — Groupe 02-C2-JVX.

## Diapositive 

https://docs.google.com/presentation/d/1ZHgFnpXF54FMbHQaADr3LEy9rIQ4NFkfQzFusvhMNZk/edit?usp=sharing

---

## 🚀 Démarrage rapide (5 minutes)

```bash
# 1. Cloner et se placer dans le projet
cd groupe-02-C2-JVX/

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer les clés API
cp .env.example .env
# Éditer .env et ajouter vos clés OPENAI_API_KEY et ANTHROPIC_API_KEY

# 4. Générer les données d'exemple
python data/generate_samples.py

# 5. Indexer les documents
python src/main.py ingest --source data/samples/

# 6. Lancer l'interface
streamlit run src/ui/app.py
# → Ouvre http://localhost:8501
```

**Ou avec Make :**
```bash
make install && make generate-data && make ingest && make ui
```

> ⚡ **Sans clés API** : le système fonctionne en mode dégradé avec des embeddings locaux (all-MiniLM-L6-v2). Les réponses générées afficheront le contexte brut sans analyse LLM.

---

## 📐 Architecture

```
[Documents Financiers]
  PDF / CSV / News JSON / Excel
        │
        ▼
[Pipeline d'Ingestion]
  ├── FinancialDocumentLoader  (pdfplumber + PyMuPDF)
  ├── FinancialTableExtractor  (camelot + pdfplumber fallback)
  └── IntelligentFinancialChunker  (SEMANTIC | TABLE_AWARE | HYBRID)
        │
        ▼
[ChromaDB — Vector Store Persistant]
  ├── Collection: finrag_reports
  ├── Collection: finrag_news
  ├── Collection: finrag_tables
  └── Collection: finrag_csv
        │
        ▼
[Hybrid Retrieval — Reciprocal Rank Fusion]
  ├── Dense: text-embedding-3-small (OpenAI) ou all-MiniLM-L6-v2 (local)
  ├── Sparse: BM25 (rank_bm25)
  └── Time-aware boost (documents récents favorisés)
        │
        ▼
[Agentic Layer]
  ├── FinancialQueryDecomposer  (Claude — sous-requêtes atomiques)
  └── FinancialRAGAgent  (orchestration parallèle + déduplication)
        │
        ▼
[CrossEncoder Re-ranking]
  └── cross-encoder/ms-marco-MiniLM-L-6-v2
        │
        ▼
[LLM Generation — Claude claude-sonnet-4-20250514]
  ├── Prompt système financier expert
  ├── Citations vérifiables (source + chunk_id + page + date)
  └── Format structuré markdown + JSON
        │
        ▼
[Évaluation RAGAS]
  ├── Faithfulness | Answer Relevancy
  └── Context Recall | Context Precision
        │
        ▼
[Interface Streamlit]
  ├── Chat premium dark theme
  ├── Query Intelligence Panel
  ├── Documents Manager
  └── Analytics Dashboard
```

---

## ✨ Fonctionnalités

### Core RAG
- ✅ **Ingestion multi-format** : PDF (texte + tableaux), CSV, Excel, JSON news, TXT
- ✅ **Chunking intelligent** : adaptatif selon le type de contenu (narratif/tabulaire)
- ✅ **Vector Store ChromaDB** : persistant, 4 collections séparées par type
- ✅ **Embeddings** : OpenAI `text-embedding-3-small` avec fallback local `all-MiniLM-L6-v2`
- ✅ **Retrieval hybride** : Dense (cosine) + Sparse (BM25) via Reciprocal Rank Fusion
- ✅ **Filtres temporels** : boost des documents récents configurable
- ✅ **CrossEncoder Re-ranking** : `ms-marco-MiniLM-L-6-v2` avec cache LRU
- ✅ **Décomposition agentique** : sous-requêtes atomiques pour questions complexes
- ✅ **Génération Claude** : citations vérifiables, format structuré, streaming

### Interface
- ✅ **Streamlit premium** : dark theme navy/or, glassmorphism
- ✅ **Chat streaming** : affichage token par token
- ✅ **Citations cliquables** : expandables avec extrait du chunk source
- ✅ **Upload temps réel** : drag-and-drop avec indexation immédiate
- ✅ **Query Intelligence Panel** : sous-requêtes, scores par source
- ✅ **Analytics Dashboard** : métriques RAGAS, historique, statistiques

### Qualité
- ✅ **Évaluation RAGAS** : 4 métriques, export markdown + CSV
- ✅ **Tests unitaires** : ingestion, retrieval, génération, agents
- ✅ **Fallbacks robustes** : dégradation gracieuse sans clés API
- ✅ **Logging structuré** : loguru avec rotation

---

## 📦 Installation

### Prérequis
- Python 3.11+
- Java (pour tabula-py, extraction tableaux PDF)
- Ghostscript (pour camelot, optionnel)

### Installation standard
```bash
pip install -r requirements.txt
```

### Installation développement
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🔧 Configuration

### Variables d'environnement (`.env`)

```bash
# Obligatoire pour la génération LLM
ANTHROPIC_API_KEY=sk-ant-...

# Obligatoire pour les embeddings OpenAI (sinon fallback local)
OPENAI_API_KEY=sk-...

# Optionnel — valeurs par défaut
LLM_MODEL=claude-sonnet-4-20250514
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=512
CHUNK_OVERLAP=51
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
HYBRID_ALPHA=0.7
CHROMA_PERSIST_DIR=./data/chroma_db
```

### Paramètres RAG (`src/config.py`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `CHUNK_SIZE` | 512 | Taille des chunks en tokens |
| `CHUNK_OVERLAP` | 51 | Overlap entre chunks (10%) |
| `TOP_K_RETRIEVAL` | 10 | Résultats finaux après RRF |
| `TOP_K_RERANK` | 5 | Résultats après CrossEncoder |
| `HYBRID_ALPHA` | 0.7 | Poids dense vs sparse |
| `TIME_DECAY_FACTOR` | 0.1 | Boost temporel (0 = désactivé) |

---

## 💬 Utilisation

### Interface Web
```bash
streamlit run src/ui/app.py
# ou
make ui
```

### CLI
```bash
# Indexer des documents
python src/main.py ingest --source data/samples/
python src/main.py ingest --source mon_rapport.pdf

# Poser une question
python src/main.py query "Quel est le CA d'Apple en FY2023 ?"
python src/main.py query "Comparez Apple et Microsoft" --no-decompose

# Statistiques du vector store
python src/main.py stats

# Interface Streamlit
python src/main.py ui
```

### API Python
```python
from src.retrieval.vector_store import FinancialVectorStore
from src.retrieval.retriever import HybridFinancialRetriever
from src.retrieval.reranker import CrossEncoderReRanker
from src.generation.generator import FinancialAnswerGenerator
from src.agents.rag_agent import FinancialRAGAgent

# Initialiser le système
vs = FinancialVectorStore()
retriever = HybridFinancialRetriever(vector_store=vs)
reranker = CrossEncoderReRanker()
generator = FinancialAnswerGenerator()
agent = FinancialRAGAgent(
    vector_store=vs,
    retriever=retriever,
    reranker=reranker,
    generator=generator,
)

# Indexer des documents
from src.ingestion.document_loader import FinancialDocumentLoader
from src.ingestion.chunker import IntelligentFinancialChunker

loader = FinancialDocumentLoader()
chunker = IntelligentFinancialChunker()
docs = loader.load("mon_rapport.pdf")
chunks = chunker.chunk_documents(docs)
vs.add_documents(chunks)

# Poser une question
answer = agent.answer("Quel est le CA d'Apple en FY2023 ?")
print(answer.answer)
print(f"Confiance: {answer.confidence_score:.0%}")
for citation in answer.citations:
    print(f"  → {citation.source_file}, p.{citation.page_number}")
```

---

## 📊 Évaluation

### Lancer l'évaluation RAGAS
```bash
python src/main.py evaluate
# ou
make evaluate
```

### Métriques calculées

| Métrique | Description | Cible |
|----------|-------------|-------|
| **Faithfulness** | Les réponses sont-elles fidèles aux sources ? | ≥ 0.80 |
| **Answer Relevancy** | Les réponses répondent-elles aux questions ? | ≥ 0.80 |
| **Context Recall** | Le contexte contient-il les infos nécessaires ? | ≥ 0.75 |
| **Context Precision** | Le contexte est-il précis et pertinent ? | ≥ 0.75 |

### Dataset d'évaluation
15 questions financières avec ground truth dans `data/samples/eval_questions.json`.

---

## 🧪 Tests

```bash
# Tous les tests
pytest tests/ -v

# Par module
pytest tests/test_ingestion.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v
pytest tests/test_agents.py -v

# Avec coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Structure du projet

```
groupe-02-C2-JVX/
├── README.md                      # Ce fichier
├── Makefile                       # Commandes make
├── requirements.txt               # Dépendances Python
├── .env.example                   # Template configuration
│
├── src/                           # Code source principal
│   ├── config.py                  # Configuration centralisée (pydantic-settings)
│   ├── main.py                    # CLI point d'entrée
│   │
│   ├── ingestion/                 # Pipeline d'ingestion
│   │   ├── document_loader.py     # Chargeur multi-format (PDF/CSV/JSON/XLSX)
│   │   ├── table_extractor.py     # Extracteur tableaux PDF (camelot)
│   │   └── chunker.py             # Chunker intelligent (SEMANTIC/TABLE_AWARE/HYBRID)
│   │
│   ├── retrieval/                 # Pipeline de retrieval
│   │   ├── vector_store.py        # ChromaDB persistant (4 collections)
│   │   ├── retriever.py           # Hybrid RRF (Dense + BM25)
│   │   └── reranker.py            # CrossEncoder re-ranking
│   │
│   ├── generation/                # Génération LLM
│   │   └── generator.py           # Claude generator + citations
│   │
│   ├── agents/                    # Couche agentique
│   │   ├── query_decomposer.py    # Décomposition requêtes complexes
│   │   └── rag_agent.py           # Agent orchestrateur principal
│   │
│   ├── evaluation/                # Évaluation RAGAS
│   │   └── evaluator.py           # RAGAS evaluator + rapport
│   │
│   └── ui/                        # Interface Streamlit
│       └── app.py                 # Application complète dark theme
│
├── data/                          # Données
│   ├── generate_samples.py        # Générateur données d'exemple
│   ├── chroma_db/                 # Vector store persistant (auto-créé)
│   └── samples/                   # Données d'exemple
│       ├── apple_annual_report_2023.pdf
│       ├── microsoft_q4_2024.pdf
│       ├── market_overview_2024.pdf
│       ├── portfolio.csv
│       ├── eval_questions.json    # 15 QA pour RAGAS
│       └── news/                  # 8 articles JSON
│
├── tests/                         # Tests unitaires
│   ├── conftest.py                # Fixtures communes
│   ├── test_ingestion.py          # Tests ingestion
│   ├── test_retrieval.py          # Tests retrieval
│   ├── test_generation.py         # Tests génération
│   └── test_agents.py             # Tests agents
│
├── notebooks/                     # Jupyter notebooks démonstratifs
│   ├── 01_demo_ingestion.ipynb
│   ├── 02_demo_retrieval.ipynb
│   ├── 03_demo_rag_pipeline.ipynb
│   ├── 04_evaluation_ragas.ipynb
│   └── 05_agentic_rag.ipynb
│
└── docs/                          # Documentation technique
    ├── architecture.md            # Architecture détaillée
    └── api.md                     # Documentation API
```

---

## 👥 Équipe

**Groupe 02-C2-JVX** — ECE Paris, ING4, IA Finance 2025-2026

Projet réalisé dans le cadre du module IA Finance, Niveau Difficulté 3/5.

---

## 📚 Références

- [LangChain Documentation](https://docs.langchain.com)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [RAGAS Framework](https://docs.ragas.io)
- [Anthropic Claude API](https://docs.anthropic.com)
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Sentence Transformers CrossEncoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
