SPECIFICATIONS — Système RAG pour Questions Financières Complexes

Projet : groupe-02-C2-JVX | Difficulté : 3/5 | Domaine : ML, NLP, Agentic AI


🎯 MISSION POUR CLAUDE CODE
Tu dois construire un système RAG (Retrieval-Augmented Generation) de niveau production pour l'analyse de documents financiers complexes. Ce projet doit atteindre le niveau Excellent selon les critères définis. Il doit être entièrement fonctionnel, visuellement remarquable, et prêt à être présenté devant un jury technique.
Tu travailles dans le dépôt /groupe-02-C2-JVX/. Respecte impérativement la structure imposée :
/groupe-02-C2-JVX/
├── README.md
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   ├── agents/
│   └── ui/
├── docs/
│   ├── architecture.md
│   └── api.md
├── data/
│   └── samples/
├── tests/
├── notebooks/
└── ...
NE JAMAIS mettre de fichiers à la racine du dépôt (sauf README.md).

🏗️ ARCHITECTURE CIBLE (Niveau Excellent)
Vue d'ensemble
[Documents Financiers]
  PDF / CSV / News Articles / Excel
        │
        ▼
[Pipeline d'Ingestion]
  ├── PDF Parser (pdfplumber + PyMuPDF)
  ├── Table Extractor (camelot / tabula)
  ├── News Scraper (feedparser / newsapi)
  └── Metadata Extractor (dates, sources, entités)
        │
        ▼
[Chunking Intelligent]
  ├── Chunking sémantique (texte narratif)
  ├── Chunking structurel (tableaux financiers)
  ├── Enrichissement temporel (date-aware)
  └── Metadata tagging (ticker, secteur, date)
        │
        ▼
[Vector Store — ChromaDB]
  ├── Dense embeddings (text-embedding-3-small OpenAI)
  ├── Sparse embeddings (BM25 via rank_bm25)
  └── Hybrid retrieval (RRF — Reciprocal Rank Fusion)
        │
        ▼
[Agentic RAG Layer]
  ├── Query Decomposer (LangChain / LLM)
  ├── Sub-query Router
  ├── Time-aware Filter
  └── Re-ranker (CrossEncoder)
        │
        ▼
[LLM Generation]
  ├── Claude claude-sonnet-4-20250514 (via API Anthropic)
  ├── Prompt engineering financier
  ├── Citation verifiable (source + chunk_id + date)
  └── Réponse structurée (markdown + JSON)
        │
        ▼
[Evaluation RAGAS]
  ├── Faithfulness
  ├── Answer Relevancy
  ├── Context Recall
  └── Context Precision
        │
        ▼
[Interface Chat UI]
  Application web Streamlit / Gradio ou FastAPI + React

📦 STACK TECHNIQUE OBLIGATOIRE
Backend / Core RAG
python >= 3.11
langchain >= 0.3.0
langchain-community
langchain-openai
langchain-anthropic
llama-index >= 0.11.0         # pour documents structurés
chromadb >= 0.5.0             # vector store principal
sentence-transformers          # embeddings locaux alternatifs
rank_bm25                     # sparse retrieval
ragas >= 0.2.0                # évaluation RAG
Ingestion de documents
pdfplumber                    # extraction PDF texte
pymupdf (fitz)                # extraction PDF avancée
camelot-py[cv]                # extraction tableaux PDF
tabula-py                     # alternative tableaux
pandas                        # manipulation données tabulaires
openpyxl                      # fichiers Excel
feedparser                    # flux RSS / news
beautifulsoup4                # scraping HTML
LLM et Embeddings
openai >= 1.0.0               # embeddings text-embedding-3-small
anthropic >= 0.40.0           # claude-sonnet-4-20250514 pour génération
tiktoken                      # comptage tokens
Re-ranking et scoring
sentence-transformers          # CrossEncoder re-ranking
flashrank                     # re-ranker léger alternatif
Interface utilisateur
streamlit >= 1.40.0           # interface principale
plotly                        # visualisations financières
streamlit-chat                # composants chat
Évaluation et monitoring
ragas                         # framework évaluation RAG
datasets                      # jeux de données HuggingFace
tqdm                          # progress bars
loguru                        # logging structuré
Dev & Tests
pytest
pytest-asyncio
python-dotenv
pydantic >= 2.0               # validation des modèles de données
fastapi                       # API REST optionnelle
uvicorn

🔧 IMPLÉMENTATION DÉTAILLÉE
1. Pipeline d'Ingestion (src/ingestion/)
src/ingestion/document_loader.py
Crée une classe FinancialDocumentLoader qui :

Supporte : PDF, CSV, XLSX, TXT, JSON, URLs (news)
Extrait les métadonnées : {source, filename, date, document_type, ticker_symbols, page_count}
Détecte automatiquement le type de contenu (rapport annuel, article, tableau de bord)
Retourne des objets langchain.schema.Document enrichis

src/ingestion/table_extractor.py
Crée FinancialTableExtractor qui :

Détecte les tableaux dans les PDFs (camelot en mode lattice + stream)
Convertit les tableaux en markdown structuré lisible par LLM
Préserve les en-têtes de colonnes, unités (M€, %, etc.)
Ajoute des métadonnées de position (page, titre du tableau si détectable)
Gère les tableaux multi-pages

src/ingestion/chunker.py
Crée IntelligentFinancialChunker avec plusieurs stratégies :
pythonclass ChunkingStrategy(Enum):
    SEMANTIC = "semantic"        # RecursiveCharacterTextSplitter avec overlap
    TABLE_AWARE = "table_aware"  # Préserve les tableaux entiers
    SENTENCE = "sentence"        # SentenceTransformers semantic chunking
    HYBRID = "hybrid"            # Auto-détection selon le contenu

class IntelligentFinancialChunker:
    """
    - chunk_size adaptatif (texte narratif: 512 tokens, tableaux: chunk entier)
    - overlap de 10% sur les chunks narratifs
    - Tagging automatique: {"contains_table": bool, "contains_numbers": bool, 
                             "time_period": str, "financial_entities": list}
    - Filtrage des chunks trop courts ou non informatifs
    """
2. Vector Store et Retrieval (src/retrieval/)
src/retrieval/vector_store.py
Crée FinancialVectorStore :

Backend : ChromaDB persistant (path configurable)
Collection séparée par type de document (reports, news, tables)
Embeddings : text-embedding-3-small (OpenAI) — avec fallback all-MiniLM-L6-v2 (local)
CRUD complet : add_documents, delete_by_source, list_sources, get_stats
Sauvegarde des métadonnées enrichies dans ChromaDB

src/retrieval/retriever.py
Crée HybridFinancialRetriever :
pythonclass HybridFinancialRetriever:
    """
    Implémente le Reciprocal Rank Fusion (RRF) entre :
    - Dense retrieval (ChromaDB cosine similarity, top-k=20)
    - Sparse retrieval (BM25 sur le corpus complet, top-k=20)
    Résultat fusionné : top-k=10 après RRF
    
    Filtres disponibles :
    - date_range: tuple(datetime, datetime)
    - document_type: list[str]
    - ticker: str
    - min_relevance_score: float
    
    Time-aware : boost des documents récents configurable (decay_factor)
    """
src/retrieval/reranker.py
Crée CrossEncoderReRanker :

Modèle : cross-encoder/ms-marco-MiniLM-L-6-v2
Input : query + liste de chunks candidats
Output : chunks re-scorés et re-triés, top-k gardés
Cache des scores pour éviter les re-calculs

3. Couche Agentique (src/agents/)
src/agents/query_decomposer.py
C'est la pièce maîtresse du niveau Excellent :
pythonclass FinancialQueryDecomposer:
    """
    Prend une question complexe du type :
    "Compare la croissance du chiffre d'affaires d'Apple et Microsoft 
    sur les 3 dernières années en tenant compte de l'impact de l'IA"
    
    Et la décompose en sous-requêtes atomiques :
    1. "Chiffre d'affaires Apple 2022 2023 2024"
    2. "Chiffre d'affaires Microsoft 2022 2023 2024"  
    3. "Impact IA sur revenus Apple 2023 2024"
    4. "Impact IA sur revenus Microsoft 2023 2024"
    5. "Comparaison croissance Apple vs Microsoft"
    
    Utilise claude-sonnet-4-20250514 avec un prompt système spécialisé.
    Retourne: List[SubQuery] avec {query, type, time_filter, entities}
    """
src/agents/rag_agent.py
Crée FinancialRAGAgent (LangChain AgentExecutor ou custom) :
pythonclass FinancialRAGAgent:
    """
    Tools disponibles :
    - search_financial_reports(query, date_range)
    - search_news(query, ticker, date_range)
    - search_tables(query, metric_type)
    - calculate_metric(expression, context)  # calculs financiers simples
    - get_document_summary(source_id)
    
    Workflow :
    1. Analyse la question → QueryDecomposer
    2. Execute les sous-requêtes en parallèle (asyncio)
    3. Agrège les résultats avec deduplication
    4. Re-rank global
    5. Génère la réponse finale avec citations
    
    Gestion des erreurs : fallback sur RAG simple si décomposition échoue
    """
4. Génération (src/generation/)
src/generation/generator.py
Crée FinancialAnswerGenerator :
Prompt système (à hardcoder dans le fichier) :
Tu es un analyste financier expert. Tu réponds UNIQUEMENT en te basant sur 
les documents fournis dans le contexte. Chaque affirmation chiffrée DOIT 
être suivie d'une citation [Source: {filename}, p.{page}, {date}].

Format de réponse :
- Réponse directe en 1-2 phrases
- Analyse détaillée avec données chiffrées et citations
- Tableau comparatif si pertinent (markdown)
- Limites et mises en garde si les données sont incomplètes

Si l'information n'est pas dans le contexte, dis-le explicitement.
Ne jamais inventer de chiffres.
Citations vérifiables :
python@dataclass
class Citation:
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    date: Optional[str]
    excerpt: str  # 100 chars max du chunk original
    relevance_score: float
La réponse finale retourne :
python@dataclass  
class FinancialAnswer:
    question: str
    answer: str           # réponse en markdown
    citations: List[Citation]
    sub_queries: List[str]  # si décomposition utilisée
    confidence_score: float
    processing_time: float
    tokens_used: int
5. Évaluation RAGAS (src/evaluation/)
src/evaluation/evaluator.py
Crée RAGASEvaluator :

Metrics : faithfulness, answer_relevancy, context_recall, context_precision
Génère un dataset de test financier minimal (10-20 questions avec ground truth)
Produit un rapport d'évaluation en markdown + CSV
Permet l'évaluation batch ou single-query
Export des résultats vers docs/evaluation_report.md

Dataset de test minimal (data/samples/eval_questions.json) :
json[
  {
    "question": "Quel est le chiffre d'affaires d'Apple en 2023 ?",
    "ground_truth": "Le CA d'Apple en 2023 était de 383,3 milliards de dollars",
    "document": "apple_annual_report_2023.pdf"
  },
  ...
]
6. Interface Utilisateur (src/ui/)
Design et expérience utilisateur — NIVEAU PREMIUM
L'interface doit être visuellement exceptionnelle. Voici les exigences précises :
Palette de couleurs :
css--bg-primary: #0A0E1A        /* Bleu marine très sombre */
--bg-secondary: #111827       /* Gris ardoise foncé */
--bg-card: #1A2236            /* Cartes légèrement plus claires */
--accent-gold: #F0B429        /* Or financier pour accents */
--accent-blue: #3B82F6        /* Bleu électrique */
--accent-green: #10B981       /* Vert pour valeurs positives */
--accent-red: #EF4444         /* Rouge pour valeurs négatives */
--text-primary: #F9FAFB       /* Blanc cassé */
--text-secondary: #9CA3AF     /* Gris clair */
--border: #1F2D45             /* Bordures subtiles */
Typographie :

Titres : DM Serif Display ou Playfair Display (Google Fonts) — élégant et financier
Interface : IBM Plex Sans ou Sora — moderne et lisible
Monospace (chiffres, code) : JetBrains Mono

Composants UI à implémenter :
python# src/ui/app.py — Application Streamlit principale

## SIDEBAR
- Logo + titre "FinRAG Analytics"
- Section "Documents" : liste des fichiers indexés avec badges (PDF/CSV/News)
- Bouton d'upload drag-and-drop avec progress bar
- Statut du vector store (nb documents, nb chunks, espace)
- Section "Filtres" : date range picker, type de document, ticker symbols

## PAGE PRINCIPALE
Layout : colonne principale (75%) + sidebar droite (25%)

### Chat Interface
- Historique des messages avec bulles stylisées
- Message utilisateur : fond bleu foncé, aligné droite
- Message assistant : fond carte, aligné gauche
- Citations inline cliquables → expandable avec extrait du document
- Indicateur de confiance (badge couleur : vert/orange/rouge)
- Métriques par réponse : temps de traitement, tokens, nb sources

### Query Intelligence Panel (sidebar droite)
- Affiche les sous-requêtes si décomposition activée
- Graphe de récupération : quels documents ont été utilisés
- Score de pertinence par source (mini bar chart)

### Documents Panel (onglet séparé)
- Tableau de tous les documents indexés
- Filtres et recherche
- Preview rapide au hover
- Option de suppression

### Analytics Panel (onglet séparé)
- Métriques RAGAS si disponibles (gauge charts Plotly)
- Historique des requêtes
- Documents les plus consultés
- Distribution temporelle des sources
Effets visuels obligatoires :
python# Dans le CSS custom Streamlit :
# - Gradient mesh en arrière-plan (SVG ou CSS)
# - Glassmorphism sur les cartes (backdrop-filter: blur)
# - Animations de chargement personnalisées pendant le RAG
# - Badges colorés pour les types de documents
# - Highlights sur les citations dans le texte de réponse
# - Tooltips personnalisés sur les scores de confiance
# - Streaming de la réponse (affichage token par token)

📁 DONNÉES D'EXEMPLE
Crée impérativement des données d'exemple dans data/samples/ pour que le système soit démontrable immédiatement :
Rapports financiers simulés (PDFs)
Utilise fpdf2 ou reportlab pour générer programmatiquement :

apple_annual_report_2023.pdf — rapport annuel fictif cohérent (10 pages)
microsoft_q4_2024.pdf — rapport trimestriel fictif (5 pages)
market_overview_2024.pdf — rapport macro-économique fictif (8 pages)

Ces PDFs doivent contenir :

Du texte narratif (analyse)
Des tableaux financiers (P&L, bilan, ratios)
Des dates explicites
Des chiffres réalistes cohérents

Articles de news simulés

data/samples/news/ — 5-10 fichiers JSON avec articles financiers fictifs

json{
  "title": "Apple dépasse les attentes au T4 2023",
  "content": "...",
  "source": "Reuters (simulé)",
  "date": "2023-11-02",
  "ticker": "AAPL"
}
Script de génération
Crée data/generate_samples.py qui génère toutes ces données automatiquement.

⚙️ CONFIGURATION
src/config.py — Configuration centralisée
pythonfrom pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # RAG
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 51
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    HYBRID_ALPHA: float = 0.7  # poids dense vs sparse
    TIME_DECAY_FACTOR: float = 0.1
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    COLLECTION_PREFIX: str = "finrag"
    
    # UI
    MAX_HISTORY_MESSAGES: int = 50
    STREAM_RESPONSE: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()
.env.example
bashOPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
# Optionnel - si laissé vide, utilise les embeddings locaux

🧪 TESTS (tests/)
Crée les fichiers de tests suivants :
tests/test_ingestion.py

Test chargement PDF, CSV, JSON
Test extraction de tableaux
Test chunking avec différentes stratégies
Vérification des métadonnées

tests/test_retrieval.py

Test dense retrieval
Test sparse retrieval (BM25)
Test hybrid RRF
Test filtres temporels
Test re-ranking

tests/test_generation.py

Test génération de réponse simple
Test génération avec citations
Test format FinancialAnswer

tests/test_agents.py

Test décomposition de requête simple
Test décomposition requête complexe
Test pipeline complet end-to-end

tests/conftest.py

Fixtures communes : mini vector store de test, documents de test


📓 NOTEBOOKS (notebooks/)
notebooks/01_demo_ingestion.ipynb
Démo complète du pipeline d'ingestion sur les données d'exemple.
notebooks/02_demo_retrieval.ipynb
Comparaison dense vs sparse vs hybrid retrieval avec métriques.
notebooks/03_demo_rag_pipeline.ipynb
Pipeline RAG complet de bout en bout avec exemples de questions financières.
notebooks/04_evaluation_ragas.ipynb
Évaluation RAGAS complète avec visualisations des métriques.
notebooks/05_agentic_rag.ipynb
Démonstration du RAG agentique avec décomposition de requêtes complexes.

📖 DOCUMENTATION
README.md (racine)
Structure obligatoire :
markdown# FinRAG — Système RAG pour l'Analyse Financière

## 🚀 Démarrage rapide (5 minutes)
## 📐 Architecture
## ✨ Fonctionnalités
## 📦 Installation
## 🔧 Configuration
## 💬 Utilisation
## 📊 Évaluation
## 🧪 Tests
## 📁 Structure du projet
## 👥 Équipe
docs/architecture.md

Diagramme ASCII de l'architecture complète
Explication de chaque composant
Flux de données détaillé
Choix techniques justifiés

docs/api.md

Documentation de toutes les classes publiques
Exemples d'utilisation
Guide d'intégration


🚀 SCRIPTS DE LANCEMENT
src/main.py — Point d'entrée CLI
bashpython src/main.py ingest --source data/samples/
python src/main.py query "Quelle est la croissance d'Apple ?"
python src/main.py evaluate
python src/main.py ui  # Lance l'interface Streamlit
Makefile à la racine
makefileinstall:      # pip install -r requirements.txt
generate-data: # python data/generate_samples.py
ingest:       # python src/main.py ingest --source data/samples/
ui:           # streamlit run src/ui/app.py
test:         # pytest tests/ -v
evaluate:     # python src/main.py evaluate

🏆 CRITÈRES DE QUALITÉ — CHECKLIST FINALE
Avant de terminer, vérifie que CHAQUE point est coché :
Fonctionnalités core

 Ingestion PDF avec extraction de tableaux ✓
 Ingestion CSV/Excel ✓
 Ingestion articles news (JSON/RSS) ✓
 Chunking intelligent (adaptatif selon contenu) ✓
 Vector store ChromaDB persistant ✓
 Embeddings OpenAI (+ fallback local) ✓
 Dense retrieval ✓
 Sparse retrieval BM25 ✓
 Hybrid RRF ✓
 Time-aware retrieval ✓
 CrossEncoder re-ranking ✓
 Query decomposition agentique ✓
 Génération avec claude-sonnet-4-20250514 ✓
 Citations vérifiables avec métadonnées complètes ✓
 Évaluation RAGAS ✓

Interface

 Interface Streamlit fonctionnelle ✓
 Design premium (dark theme, palette or/bleu) ✓
 Upload de documents en temps réel ✓
 Affichage des citations inline ✓
 Streaming des réponses ✓
 Panel analytics ✓
 Responsive et fluide ✓

Code quality

 Toutes les classes documentées (docstrings) ✓
 Type hints partout ✓
 Gestion d'erreurs robuste (try/except avec logging) ✓
 Configuration centralisée via pydantic-settings ✓
 Tests unitaires passants ✓
 Pas de secrets hardcodés (utilise .env) ✓

Démarrage

 pip install -r requirements.txt fonctionne sans erreur ✓
 python data/generate_samples.py génère les données ✓
 L'interface démarre et répond à des questions sur les données d'exemple ✓
 README complet et clair ✓


💡 NOTES ET PRIORITÉS
Ordre de développement recommandé

src/config.py + .env.example
data/generate_samples.py + génération des données
src/ingestion/ (loader → table_extractor → chunker)
src/retrieval/ (vector_store → retriever → reranker)
src/generation/generator.py
src/agents/ (query_decomposer → rag_agent)
src/ui/app.py (interface complète)
src/evaluation/evaluator.py
tests/
notebooks/
Documentation finale

Gestion des API Keys manquantes
Si OPENAI_API_KEY n'est pas défini :
→ Utilise sentence-transformers/all-MiniLM-L6-v2 pour les embeddings (gratuit, local)
Si ANTHROPIC_API_KEY n'est pas défini :
→ Utilise ollama avec llama3.2 comme fallback, ou affiche un message d'erreur clair
L'application doit démarrer et être partiellement fonctionnelle même sans clé API.
Performance

Indexation des documents : barre de progression (tqdm)
Requêtes : timeout de 30s max
Cache des embeddings (éviter de recalculer)
Async pour les opérations de retrieval parallèle

Ce qui distingue un projet Excellent

Le RAG agentique avec décomposition fonctionne réellement
Les citations sont cliquables et ramènent au chunk exact
L'interface est digne d'un produit commercial
L'évaluation RAGAS produit des métriques interprétables
Le README permet à quelqu'un d'autre de lancer le projet en 5 minutes


Généré pour le projet groupe-02-C2-JVX — Système RAG Financier — Niveau Excellent