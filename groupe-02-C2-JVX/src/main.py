#!/usr/bin/env python3
"""
FinRAG — Point d'entrée CLI.
Commandes: ingest, query, evaluate, ui, stats.
"""

# -- Import permettant d'utiliser les annotations de type modernes (ex. list[str])
# -- meme dans les versions de Python anterieures a 3.10.
from __future__ import annotations

# -- Modules de la bibliotheque standard :
# --   argparse  : analyse des arguments en ligne de commande (CLI)
# --   sys       : acces aux flux stdin/stdout/stderr et a sys.path
# --   Path      : manipulation orientee objet des chemins de fichiers
import argparse
import sys
from pathlib import Path

# -- Determination du repertoire racine du projet.
# -- __file__ pointe vers ce script (main.py), .parent remonte a src/, .parent.parent remonte a la racine du projet.
# -- On ajoute ensuite ce repertoire au sys.path pour permettre les imports relatifs
# -- du type "from src.xxx import yyy" depuis n'importe quel repertoire de travail.
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# -- loguru : bibliotheque de logging puissante et simple a configurer
# -- settings : objet de configuration centralise (charge depuis config.py) contenant
# --            tous les hyperparametres du systeme (taille des chunks, top_k, etc.)
from loguru import logger
from src.config import settings


# =============================================================================
# CONFIGURATION DU LOGGING
# =============================================================================

def setup_logging():
    """
    Configure le logging.

    FIX MOYEN : création de tous les répertoires nécessaires au premier lancement.
    Sans cela, un run sur installation fraîche lève FileNotFoundError.
    """
    # -- Creation preventive de tous les repertoires requis par l'application.
    # -- Cela evite les erreurs FileNotFoundError lors du tout premier lancement
    # -- sur une installation vierge. parents=True cree les dossiers intermediaires,
    # -- exist_ok=True evite une erreur si le dossier existe deja.
    for dir_path in [
        ROOT_DIR / "logs",                    # -- Dossier pour les fichiers de log
        ROOT_DIR / "data" / "chroma_db",      # -- Base de donnees vectorielle ChromaDB
        ROOT_DIR / "data" / "uploads",        # -- Fichiers uploades via l'interface
        ROOT_DIR / "data" / "embedding_cache", # -- Cache local des embeddings pour eviter les recalculs
        ROOT_DIR / "data" / "samples",        # -- Echantillons de documents pour les tests
        ROOT_DIR / "docs",                    # -- Documentation generee (rapports d'evaluation, etc.)
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # -- Suppression du handler par defaut de loguru pour eviter les doublons
    logger.remove()

    # -- Ajout d'un handler vers la sortie d'erreur standard (stderr).
    # -- Le niveau de log est defini dans settings (ex. INFO, DEBUG).
    # -- Le format affiche l'heure, le niveau, le module et le message avec des couleurs.
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <white>{message}</white>",
    )

    # -- Ajout d'un handler vers un fichier de log persistant sur disque.
    # -- rotation="10 MB" : le fichier est archive et un nouveau est cree quand il depasse 10 Mo.
    # -- retention="7 days" : les anciens fichiers de log sont supprimes apres 7 jours.
    # -- level="DEBUG" : on enregistre tout dans le fichier (plus verbeux que stderr).
    # -- encoding="utf-8" : support des caracteres speciaux (accents francais, etc.).
    logger.add(
        str(ROOT_DIR / "logs" / "finrag.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


# =============================================================================
# CONSTRUCTION DU SYSTEME RAG
# =============================================================================

def _build_system():
    """Initialise et retourne le système RAG complet."""
    # -- Les imports sont places ici (imports paresseux / lazy imports) pour eviter
    # -- de charger les modeles lourds (embeddings, cross-encoder, LLM) au demarrage
    # -- si la commande n'en a pas besoin (ex. la commande "ui" n'utilise pas _build_system).
    from src.retrieval.vector_store import FinancialVectorStore
    from src.retrieval.retriever import HybridFinancialRetriever
    from src.retrieval.reranker import CrossEncoderReRanker
    from src.generation.generator import FinancialAnswerGenerator
    from src.agents.rag_agent import FinancialRAGAgent

    logger.info("Initialisation du système FinRAG...")

    # -- 1) Vector Store : interface avec la base ChromaDB qui stocke les embeddings
    # --    des chunks de documents financiers.
    vs = FinancialVectorStore()

    # -- 2) Retriever hybride : combine la recherche vectorielle (similarite cosinus)
    # --    avec la recherche lexicale BM25. top_k definit le nombre de documents
    # --    recuperes, time_decay_factor penalise les documents anciens.
    retriever = HybridFinancialRetriever(
        vector_store=vs,
        top_k=settings.TOP_K_RETRIEVAL,
        time_decay_factor=settings.TIME_DECAY_FACTOR,
    )

    # -- 3) Re-ranker : utilise un modele Cross-Encoder pour reordonner les resultats
    # --    du retriever par pertinence fine. top_k limite le nombre de documents
    # --    conserves apres le re-ranking.
    reranker = CrossEncoderReRanker(top_k=settings.TOP_K_RERANK)

    # -- 4) Generateur : appelle le LLM (via API) pour produire une reponse
    # --    structuree a partir des documents retrouves.
    generator = FinancialAnswerGenerator()

    # -- 5) Agent RAG : orchestre l'ensemble du pipeline (decomposition de la question,
    # --    recuperation, re-ranking, generation). C'est le composant principal
    # --    qui coordonne tous les autres.
    agent = FinancialRAGAgent(
        vector_store=vs,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    # -- Retourne un dictionnaire contenant tous les composants du systeme.
    # -- Cela permet aux commandes CLI d'acceder individuellement a chaque composant.
    return {
        "vector_store": vs,
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
        "agent": agent,
    }


# =============================================================================
# COMMANDE : INGEST (indexation de documents)
# =============================================================================

def cmd_ingest(args):
    """Commande: indexer des documents."""
    # -- Imports paresseux des modules d'ingestion et de la barre de progression
    from src.ingestion.document_loader import FinancialDocumentLoader
    from src.ingestion.table_extractor import FinancialTableExtractor
    from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy
    from tqdm import tqdm

    # -- Verification que le chemin source (fichier ou repertoire) existe bien
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source introuvable: {source_path}")
        sys.exit(1)

    logger.info(f"Ingestion depuis: {source_path}")

    # -- Le DocumentLoader sait charger differents formats (PDF, CSV, XLSX, JSON, TXT)
    # -- et les convertir en objets Document standardises.
    loader = FinancialDocumentLoader()
    all_docs = []

    # -- Si la source est un fichier unique, on le traite directement.
    # -- Sinon, on parcourt recursivement le repertoire pour trouver
    # -- tous les fichiers ayant une extension supportee.
    if source_path.is_file():
        files = [source_path]
    else:
        supported_exts = {".pdf", ".csv", ".xlsx", ".json", ".txt"}
        files = [f for f in source_path.rglob("*") if f.suffix.lower() in supported_exts]

    logger.info(f"{len(files)} fichiers à traiter")

    # -- Boucle de chargement : pour chaque fichier, on extrait le contenu textuel.
    # -- Pour les fichiers PDF, on effectue en plus une extraction specifique
    # -- des tableaux (donnees financieres structurees) via FinancialTableExtractor.
    # -- tqdm affiche une barre de progression dans le terminal.
    for file_path in tqdm(files, desc="Chargement"):
        docs = loader.load(file_path)
        all_docs.extend(docs)

        # -- Extraction supplementaire des tableaux pour les PDF
        # -- (bilans, comptes de resultat, etc.)
        if file_path.suffix.lower() == ".pdf":
            extractor = FinancialTableExtractor()
            table_docs = extractor.extract_tables_from_pdf(file_path)
            all_docs.extend(table_docs)

    logger.info(f"{len(all_docs)} documents chargés")

    # -- Phase de chunking : decoupe les documents en fragments (chunks) de taille
    # -- controlee, avec chevauchement (overlap) pour ne pas perdre le contexte
    # -- aux frontieres. La strategie HYBRID combine decoupage semantique
    # -- et decoupage par taille fixe pour un equilibre optimal.
    chunker = IntelligentFinancialChunker(
        strategy=ChunkingStrategy.HYBRID,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(all_docs)
    logger.info(f"{len(chunks)} chunks créés")

    # -- Construction du systeme RAG puis ajout des chunks dans le vector store.
    # -- add_documents calcule les embeddings de chaque chunk et les stocke dans ChromaDB.
    system = _build_system()
    added = system["vector_store"].add_documents(chunks)

    # -- Apres l'ajout de nouveaux documents, on invalide le cache BM25 du retriever
    # -- car l'index lexical doit etre reconstruit pour inclure les nouveaux chunks.
    system["retriever"].invalidate_bm25_cache()

    # -- Affichage des statistiques finales d'indexation
    stats = system["vector_store"].get_stats()
    logger.success(
        f"Indexation terminée: {added} chunks ajoutés. "
        f"Total dans le store: {stats['total_chunks']} chunks, "
        f"{stats['total_sources']} sources."
    )


# =============================================================================
# COMMANDE : QUERY (poser une question au systeme)
# =============================================================================

def cmd_query(args):
    """Commande: poser une question."""
    # -- Construction du systeme RAG complet
    system = _build_system()
    agent = system["agent"]

    # -- Recuperation de la question depuis les arguments CLI
    question = args.question
    logger.info(f"Question: {question}")

    # -- Appel a l'agent RAG pour obtenir une reponse.
    # -- use_decomposition=True (par defaut) permet a l'agent de decomposer
    # -- une question complexe en sous-questions plus simples avant de les traiter.
    # -- Le flag --no-decompose desactive cette fonctionnalite.
    answer = agent.answer(
        question=question,
        use_decomposition=not args.no_decompose,
    )

    # -- Affichage formate de la reponse dans le terminal
    print("\n" + "="*70)
    print(f"Q: {question}")
    print("="*70)
    # -- Corps principal de la reponse generee par le LLM
    print(answer.answer)
    print("-"*70)
    # -- Metriques de la reponse :
    # --   confidence_score : score de confiance (0-100%)
    # --   processing_time  : temps total de traitement en secondes
    # --   tokens_used      : nombre de tokens consommes par le LLM
    # --   context_docs_count : nombre de documents utilises comme contexte
    print(
        f"Confiance: {answer.confidence_score:.0%} | "
        f"Temps: {answer.processing_time:.1f}s | "
        f"Tokens: {answer.tokens_used:,} | "
        f"Sources: {answer.context_docs_count}"
    )

    # -- Affichage des citations : extraits des documents sources avec
    # -- le nom du fichier et le numero de page (limite a 5 citations).
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations[:5]:
            print(f"  [{c.source_file}, p.{c.page_number}] {c.excerpt[:80]}...")

    # -- Affichage des sous-requetes generees si la decomposition etait active.
    # -- Cela permet de comprendre comment l'agent a decompose la question initiale.
    if answer.sub_queries:
        print(f"\nSous-requêtes ({len(answer.sub_queries)}):")
        for sq in answer.sub_queries:
            print(f"  → {sq}")


# =============================================================================
# COMMANDE : EVALUATE (evaluation RAGAS du systeme RAG)
# =============================================================================

def cmd_evaluate(args):
    """Commande: évaluation RAGAS."""
    # -- Import de l'evaluateur RAGAS, un framework standardise pour mesurer
    # -- la qualite des systemes RAG selon 4 metriques principales.
    from src.evaluation.evaluator import RAGASEvaluator

    # -- Construction du systeme et de l'evaluateur.
    # -- Le dataset d'evaluation contient des paires (question, reponse attendue)
    # -- utilisees comme reference pour calculer les scores.
    system = _build_system()
    evaluator = RAGASEvaluator(
        agent=system["agent"],
        eval_dataset_path=args.dataset if hasattr(args, "dataset") and args.dataset else None,
    )

    # -- Lancement de l'evaluation sur un nombre limite de questions (max_samples).
    # -- save_report=True genere un fichier Markdown de rapport dans docs/.
    logger.info(f"Démarrage de l'évaluation ({args.max_samples} questions max)...")
    report = evaluator.evaluate_batch(
        max_samples=args.max_samples,
        save_report=True,
    )

    # -- Affichage du rapport d'evaluation avec les 4 metriques RAGAS :
    # --   Faithfulness      : la reponse est-elle fidele aux documents retrouves ? (pas d'hallucination)
    # --   Answer Relevancy  : la reponse repond-elle bien a la question posee ?
    # --   Context Recall    : les documents pertinents ont-ils tous ete retrouves ?
    # --   Context Precision : parmi les documents retrouves, quelle proportion est pertinente ?
    # --   Score Global      : moyenne ponderee des 4 metriques ci-dessus.
    print("\n" + "="*70)
    print("RAPPORT D'ÉVALUATION RAGAS")
    print("="*70)
    print(f"Échantillons: {report.n_samples}")
    print(f"Faithfulness:      {report.avg_faithfulness:.3f}")
    print(f"Answer Relevancy:  {report.avg_answer_relevancy:.3f}")
    print(f"Context Recall:    {report.avg_context_recall:.3f}")
    print(f"Context Precision: {report.avg_context_precision:.3f}")
    print(f"Score Global:      {report.overall_score:.3f}")
    print(f"Durée: {report.evaluation_time:.1f}s")
    print("="*70)
    print("Rapport sauvegardé dans docs/evaluation_report.md")


# =============================================================================
# COMMANDE : UI (lancement de l'interface web Streamlit)
# =============================================================================

def cmd_ui(args):
    """Commande: lancer l'interface Streamlit."""
    import subprocess

    # -- Chemin vers le fichier principal de l'application Streamlit
    ui_path = ROOT_DIR / "src" / "ui" / "app.py"
    logger.info(f"Lancement de l'interface: {ui_path}")

    # -- Lancement de Streamlit en tant que sous-processus.
    # -- sys.executable garantit qu'on utilise le meme interpreteur Python
    # -- que celui qui execute ce script (important si plusieurs versions coexistent).
    # -- check=True leve une exception CalledProcessError si le processus echoue.
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path)],
        check=True,
    )


# =============================================================================
# COMMANDE : STATS (affichage des statistiques du vector store)
# =============================================================================

def cmd_stats(args):
    """Commande: afficher les statistiques du vector store."""
    # -- Construction du systeme pour acceder au vector store
    system = _build_system()
    stats = system["vector_store"].get_stats()

    # -- Affichage des statistiques globales du vector store :
    # --   total_chunks    : nombre total de chunks indexes dans ChromaDB
    # --   total_sources   : nombre de fichiers sources distincts
    # --   embedding_model : nom du modele d'embedding utilise (ex. all-MiniLM-L6-v2)
    # --   persist_dir     : chemin du repertoire de persistance ChromaDB
    print("\n" + "="*50)
    print("STATISTIQUES FINRAG")
    print("="*50)
    print(f"Total chunks:  {stats['total_chunks']:,}")
    print(f"Total sources: {stats['total_sources']}")
    print(f"Modèle:        {stats['embedding_model']}")
    print(f"Répertoire:    {stats['persist_dir']}")

    # -- Detail par collection ChromaDB (ex. "financial", "general", etc.)
    # -- Chaque collection peut regrouper un type de documents specifique.
    print("\nCollections:")
    for coll, count in stats.get("collections", {}).items():
        print(f"  {coll:<10}: {count:,} chunks")
    print("="*50)


# =============================================================================
# POINT D'ENTREE CLI — Analyse des arguments et dispatch des commandes
# =============================================================================

def main():
    # -- Configuration initiale du logging (creation des dossiers + handlers loguru)
    setup_logging()

    # -- Creation du parseur d'arguments principal.
    # -- RawDescriptionHelpFormatter preserve le formatage des exemples dans l'epilog.
    parser = argparse.ArgumentParser(
        description="FinRAG — Système RAG pour l'Analyse Financière",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python src/main.py ingest --source data/samples/
  python src/main.py query "Quel est le CA d'Apple en 2023 ?"
  python src/main.py evaluate --max-samples 10
  python src/main.py ui
  python src/main.py stats
        """,
    )

    # -- Creation des sous-commandes (sous-parseurs).
    # -- Chaque sous-commande a ses propres arguments specifiques.
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # -- Sous-commande "ingest" : indexation de documents dans le vector store.
    # --   --source (obligatoire) : chemin vers un fichier ou un repertoire a indexer.
    p_ingest = subparsers.add_parser("ingest", help="Indexer des documents")
    p_ingest.add_argument("--source", required=True, help="Chemin du fichier ou répertoire")

    # -- Sous-commande "query" : interrogation du systeme RAG.
    # --   question (argument positionnel obligatoire) : la question a poser.
    # --   --no-decompose (optionnel) : desactive la decomposition automatique
    # --     de la question en sous-questions.
    p_query = subparsers.add_parser("query", help="Poser une question")
    p_query.add_argument("question", help="Question financière")
    p_query.add_argument("--no-decompose", action="store_true", help="Désactive la décomposition")

    # -- Sous-commande "evaluate" : evaluation RAGAS de la qualite du systeme.
    # --   --max-samples (optionnel, defaut=10) : nombre maximum de questions a evaluer.
    # --   --dataset (optionnel) : chemin vers un fichier JSON de questions/reponses de reference.
    p_eval = subparsers.add_parser("evaluate", help="Évaluation RAGAS")
    p_eval.add_argument("--max-samples", type=int, default=10, help="Nombre max de questions")
    p_eval.add_argument("--dataset", help="Chemin vers le dataset JSON (optionnel)")

    # -- Sous-commande "ui" : lance l'interface web Streamlit (pas d'arguments supplementaires).
    subparsers.add_parser("ui", help="Lancer l'interface Streamlit")

    # -- Sous-commande "stats" : affiche les statistiques du vector store (pas d'arguments supplementaires).
    subparsers.add_parser("stats", help="Statistiques du vector store")

    # -- Analyse des arguments fournis en ligne de commande
    args = parser.parse_args()

    # -- Si aucune sous-commande n'est specifiee, on affiche l'aide et on quitte.
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # -- Dictionnaire de correspondance entre le nom de la commande (chaine)
    # -- et la fonction Python qui l'implemente. Cela evite un bloc if/elif.
    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "ui": cmd_ui,
        "stats": cmd_stats,
    }

    # -- Execution de la commande correspondante.
    # -- On gere deux cas d'erreur :
    # --   KeyboardInterrupt : l'utilisateur a appuye sur Ctrl+C
    # --   Exception         : toute autre erreur est loguee puis re-levee
    # --                       pour conserver la trace complete (traceback).
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise


# -- Point d'entree standard Python : ce bloc s'execute uniquement quand le script
# -- est lance directement (python src/main.py ...), et non quand il est importe
# -- en tant que module par un autre fichier.
if __name__ == "__main__":
    main()
