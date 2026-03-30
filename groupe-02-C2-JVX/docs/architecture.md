# Architecture Technique — FinRAG

## Vue d'ensemble

FinRAG est un système RAG (Retrieval-Augmented Generation) de niveau production conçu pour l'analyse de documents financiers complexes. L'architecture suit le pattern **Agentic RAG** avec décomposition de requêtes, retrieval hybride et re-ranking CrossEncoder.

---

## 1. Pipeline d'Ingestion

### FinancialDocumentLoader (`src/ingestion/document_loader.py`)

**Responsabilité** : Charger des documents de formats variés et extraire les métadonnées financières.

**Formats supportés** :
| Format | Bibliothèque | Cas d'usage |
|--------|-------------|-------------|
| PDF | pdfplumber (primaire), PyMuPDF (fallback) | Rapports annuels, rapports trimestriels |
| CSV | pandas | Données de portefeuille, séries temporelles |
| XLSX | openpyxl | Tableaux financiers Excel |
| JSON | json stdlib | Articles de news financière |
| TXT | io stdlib | Documents textuels |

**Métadonnées extraites** :
```python
{
    "source": str,           # Chemin/URL source
    "filename": str,         # Nom du fichier
    "date": str,             # Période détectée (FY2023, Q4 2024...)
    "document_type": str,    # annual_report | news_article | csv_data | ...
    "ticker_symbols": list,  # ["AAPL", "MSFT"]
    "page_count": int,       # Nombre de pages (PDFs)
    "ingestion_timestamp": str,
}
```

### FinancialTableExtractor (`src/ingestion/table_extractor.py`)

**Stratégie** :
1. **camelot lattice** : Tableaux avec bordures nettes (accuracy ≥ 80%)
2. **camelot stream** : Tableaux sans bordures visibles
3. **pdfplumber fallback** : Extraction basique si camelot échoue

**Sortie** : Tableaux convertis en markdown structuré :
```markdown
### Compte de Résultat (p.3)

| Poste | FY2023 | FY2022 | Variation |
|-------|--------|--------|-----------|
| CA    | 383,3  | 394,3  | -2,8%    |
| EBIT  | 114,3  | 119,4  | -4,3%    |
```

### IntelligentFinancialChunker (`src/ingestion/chunker.py`)

**Stratégies de chunking** :

```
HYBRID (auto-détection)
├── contient tableau markdown → TABLE_AWARE
│   └── Préserve les tableaux entiers comme chunks atomiques
│       + Découpe le texte autour sémantiquement
└── texte narratif → SEMANTIC
    └── RecursiveCharacterTextSplitter
        chunk_size=512 tokens, overlap=51 (10%)
```

**Tags automatiques par chunk** :
```python
{
    "contains_table": bool,
    "contains_numbers": bool,
    "time_period": str,          # "FY2023", "Q4 2024"
    "financial_entities": list,  # ["Apple", "EPS", "EBITDA"]
    "chunk_length": int,
    "chunk_strategy": str,
}
```

---

## 2. Vector Store

### FinancialVectorStore (`src/retrieval/vector_store.py`)

**Backend** : ChromaDB persistant (`./data/chroma_db`)

**Collections** :
| Collection | Contenu | Doc types |
|------------|---------|-----------|
| `finrag_reports` | Rapports annuels/trimestriels | annual_report, quarterly_report, market_overview |
| `finrag_news` | Articles de presse | news_article |
| `finrag_tables` | Tableaux financiers | financial_table |
| `finrag_csv` | Données tabulaires | csv_data, excel_data |

**Embeddings** :
- **Primaire** : OpenAI `text-embedding-3-small` (1536 dimensions)
- **Fallback** : `all-MiniLM-L6-v2` (384 dimensions, local, gratuit)

**Distance** : Cosine similarity (configurée au niveau de la collection)

**Gestion des doublons** : ID déterministe SHA-256 basé sur (source, page, chunk_index, contenu[:100]) → upsert idempotent.

---

## 3. Retrieval Hybride

### HybridFinancialRetriever (`src/retrieval/retriever.py`)

**Architecture RRF (Reciprocal Rank Fusion)** :

```
Query → Dense Retrieval (ChromaDB, cosine, top-20)
      + Sparse Retrieval (BM25, rank_bm25, top-20)
      ↓
    RRF Fusion
    score_rrf(d) = Σ 1/(k + rank_i(d))    [k=60]
      ↓
    Top-10 résultats fusionnés
      ↓
    Time-aware boost
    score_final = score_rrf * 1/(1 + decay * years_ago)
```

**Filtres disponibles** :
```python
retriever.retrieve(
    query="CA Apple 2023",
    date_range=("2022", "2024"),    # Filtre temporel
    document_type=["annual_report"], # Type de document
    ticker="AAPL",                  # Filtre ticker
    min_relevance_score=0.1,        # Score minimum
)
```

**Initialisation BM25** : Lazy loading — BM25 est initialisé lors de la première requête hybride, puis mis en cache. Invalidé automatiquement après l'ajout de nouveaux documents.

### CrossEncoderReRanker (`src/retrieval/reranker.py`)

**Modèle** : `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Fonctionnement** :
```
[query, chunk_1] → CrossEncoder → score_1
[query, chunk_2] → CrossEncoder → score_2
...
→ Tri par score décroissant → Top-5 retenus
```

**Cache LRU** : Score mis en cache par (MD5(query[:200] + text[:200])) pour éviter les recalculs. Éviction par demi-suppression quand la limite est atteinte.

**Lazy loading** : Le modèle n'est chargé en mémoire que lors du premier appel.

---

## 4. Couche Agentique

### FinancialQueryDecomposer (`src/agents/query_decomposer.py`)

**Objectif** : Décomposer les questions complexes en sous-requêtes atomiques recherchables dans un seul document.

**Exemple** :
```
Input:  "Compare la croissance du CA d'Apple et Microsoft sur les 3 dernières années"

Output: [
  SubQuery(query="CA Apple FY2022", type="metric", entities=["AAPL"], priority=1),
  SubQuery(query="CA Apple FY2023", type="metric", entities=["AAPL"], priority=2),
  SubQuery(query="CA Apple FY2024", type="metric", entities=["AAPL"], priority=3),
  SubQuery(query="CA Microsoft FY2022", type="metric", entities=["MSFT"], priority=4),
  SubQuery(query="CA Microsoft FY2023", type="metric", entities=["MSFT"], priority=5),
  SubQuery(query="Comparaison Apple vs Microsoft", type="comparison", priority=6),
]
```

**Stratégie** :
1. LLM (Claude) → JSON structuré
2. Fallback règles heuristiques (patterns regex) si API indisponible

### FinancialRAGAgent (`src/agents/rag_agent.py`)

**Workflow** :
```python
def answer(question):
    # 1. Décomposition si nécessaire
    if needs_decomposition(question):
        sub_queries = decomposer.decompose(question)
        results = [retriever.retrieve(sq.query) for sq in sub_queries]
        all_results = flatten(results)
    else:
        all_results = retriever.retrieve(question)

    # 2. Déduplication (character similarity)
    deduped = deduplicate(all_results)

    # 3. Re-ranking global
    reranked = reranker.rerank(question, deduped)

    # 4. Génération
    return generator.generate(question, reranked[:TOP_K_RERANK])
```

**Déduplication** : Similarité caractère-à-caractère sur les 200 premiers caractères. Seuil : 90% (configurable).

---

## 5. Génération

### FinancialAnswerGenerator (`src/generation/generator.py`)

**Modèle** : Claude `claude-sonnet-4-20250514` (Anthropic API)

**Prompt système** (extrait) :
```
Tu es un analyste financier expert. Tu réponds UNIQUEMENT en te basant sur
les documents fournis dans le contexte. Chaque affirmation chiffrée DOIT
être suivie d'une citation [Source: {filename}, p.{page}, {date}].
...
Ne jamais inventer de chiffres.
```

**Format de réponse** :
```python
@dataclass
class FinancialAnswer:
    question: str
    answer: str           # Markdown formaté
    citations: List[Citation]  # [chunk_id, source_file, page, date, excerpt, score]
    sub_queries: List[str]
    confidence_score: float  # 0.0 à 1.0 (heuristique)
    processing_time: float
    tokens_used: int
    context_docs_count: int
```

**Estimation de confiance** (heuristique) :
- Présence de citations (+0.3 à +0.5)
- Présence de données chiffrées (+0.2)
- Nombre de sources (≥3 → +0.2)
- Pas de mention "non disponible" (+0.1)

---

## 6. Évaluation RAGAS

### RAGASEvaluator (`src/evaluation/evaluator.py`)

**Métriques** :

| Métrique | Description | Calcul (approché si pas d'API) |
|----------|-------------|-------------------------------|
| Faithfulness | % des affirmations supportées par le contexte | `len(answer ∩ context) / len(answer)` |
| Answer Relevancy | Pertinence de la réponse par rapport à la question | Jaccard(question, answer) |
| Context Recall | % de la ground truth couverte par le contexte | `len(ground_truth ∩ context) / len(ground_truth)` |
| Context Precision | Précision du contexte par rapport à la question | Jaccard(question, context) |

---

## 7. Interface Utilisateur

### app.py (`src/ui/app.py`)

**Layout** :
```
┌─────────────────────────────────────────────────────────┐
│  SIDEBAR                │  MAIN CONTENT (TABS)           │
│  ─────────              │  ─────────────────             │
│  Logo + Statut          │  [💬 Assistant] [📚 Documents] [📊 Analytics]
│  Upload documents       │                                │
│  Documents indexés      │  ┌─────────────┬────────────┐  │
│  Filtres recherche      │  │ Chat 75%    │ Intel. 25% │  │
│  Options RAG            │  │ ─────────── │ ────────── │  │
│  Données d'exemple      │  │ Messages    │ Sub-queries│  │
│                         │  │ Citations   │ Sources    │  │
│                         │  │ Métriques   │ Stats      │  │
│                         │  └─────────────┴────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Thème** :
- Fond : `#0A0E1A` (bleu marine très sombre)
- Cartes : `#1A2236` avec glassmorphism (`backdrop-filter: blur(12px)`)
- Accent : `#F0B429` (or financier)
- Texte : `#F9FAFB` (blanc cassé)
- Typographie : DM Serif Display (titres) + IBM Plex Sans (corps)

---

## 8. Choix Techniques Justifiés

### Pourquoi ChromaDB ?
- Persistance native (pas de serveur séparé)
- HNSW (cosine) intégré
- Python-first, facile à intégrer avec LangChain
- Collections séparées par type = isolation et filtrabilité

### Pourquoi BM25 + Dense (Hybrid) ?
- BM25 excelle sur les termes rares/spécifiques (tickers, "EBITDA", "Q4 FY2024")
- Dense excelle sur la similarité sémantique
- RRF offre le meilleur des deux sans tuning complexe

### Pourquoi CrossEncoder en re-ranking ?
- Les bi-encodeurs (embeddings) approximent la similarité
- Le CrossEncoder traite (query, doc) ensemble → meilleure pertinence
- Compromis : calcul plus lent → top-5 seulement (acceptable en pratique)

### Pourquoi Claude pour la décomposition ET la génération ?
- Claude excelle en raisonnement structuré (JSON, plans)
- `claude-sonnet-4-20250514` : rapport performance/coût optimal
- Cohérence entre décomposition et génération (même modèle)

### Pourquoi Streamlit ?
- Développement rapide d'UI data science
- Hot reload natif
- Gestion de session state intégrée
- `@st.cache_resource` pour éviter de recréer le vector store à chaque refresh
