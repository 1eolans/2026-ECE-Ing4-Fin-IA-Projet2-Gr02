"""
Retriever hybride (Dense + BM25) avec Reciprocal Rank Fusion.
Filtres temporels, par type de document et ticker.
"""

# -- Ce module implémente un pipeline de recherche hybride pour des documents financiers.
# -- Il combine deux approches complémentaires :
# --   1. La recherche "dense" (sémantique) via des embeddings vectoriels stockés dans ChromaDB,
# --      qui capture le sens et le contexte des mots.
# --   2. La recherche "sparse" (lexicale) via BM25, qui se base sur la fréquence des termes
# --      exacts dans les documents (correspondance mot-à-mot).
# -- Les résultats de ces deux approches sont fusionnés grâce à l'algorithme RRF
# -- (Reciprocal Rank Fusion), qui combine intelligemment les classements pour obtenir
# -- un résultat final plus pertinent que chaque méthode prise individuellement.

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# -- Document est la structure de base de LangChain pour représenter un morceau de texte
# -- accompagné de ses métadonnées (source, date, ticker, type de document, etc.)
from langchain_core.documents import Document
# -- loguru est une bibliothèque de journalisation (logging) plus ergonomique que le module
# -- standard logging de Python. Elle permet de tracer les événements du retriever.
from loguru import logger

# -- On ajoute le répertoire racine du projet au chemin Python pour pouvoir importer
# -- le module de configuration (settings) qui contient les paramètres globaux du projet.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── RRF Implementation ───────────────────────────────────────────────────────

# -- ==========================================================================
# -- RECIPROCAL RANK FUSION (RRF) — Explication détaillée
# -- ==========================================================================
# -- RRF est un algorithme de fusion de classements ("rank aggregation") inventé par
# -- Cormack, Clarke & Buettcher (2009). Son principe est simple mais très efficace :
# --
# -- Pour chaque document apparaissant dans un ou plusieurs classements, on calcule
# -- un score RRF = somme sur chaque classement de : 1 / (k + rang)
# --
# -- Où :
# --   - "rang" est la position du document dans un classement donné (1er = rang 1, etc.)
# --   - "k" est une constante de lissage (typiquement 60) qui empêche les documents
# --     en tête de liste d'avoir un score disproportionnellement élevé.
# --
# -- Pourquoi RRF est pertinent ici :
# --   - La recherche dense (sémantique) peut rater des mots-clés exacts importants
# --     (ex: un ticker boursier précis comme "AAPL").
# --   - La recherche BM25 (lexicale) peut rater le contexte sémantique
# --     (ex: "croissance des revenus" vs "augmentation du chiffre d'affaires").
# --   - RRF combine les deux classements sans avoir besoin de normaliser les scores,
# --     car il ne se base que sur les rangs (positions), pas sur les valeurs absolues.
# --   - Un document bien classé par les DEUX méthodes obtient un score RRF élevé,
# --     ce qui favorise les documents pertinents à la fois sémantiquement et lexicalement.
# -- ==========================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion pour combiner plusieurs classements.

    Args:
        ranked_lists: Listes de (Document, score) déjà triées par pertinence.
        k: Constante RRF (typiquement 60).
        top_n: Nombre de résultats à retourner.

    Returns:
        Liste fusionnée de (Document, rrf_score) triée par score décroissant.
    """
    # -- Dictionnaire pour accumuler le score RRF de chaque document unique.
    # -- La clé est une chaîne composite qui identifie le document de manière unique.
    doc_scores: Dict[str, float] = {}
    # -- Dictionnaire parallèle pour conserver la référence vers l'objet Document
    # -- correspondant à chaque clé, afin de pouvoir le retourner dans les résultats.
    doc_objects: Dict[str, Document] = {}

    # -- On parcourt chaque liste de classement (dense et/ou sparse)
    for ranked_list in ranked_lists:
        # -- Pour chaque document dans cette liste, on récupère son rang (position)
        # -- et l'objet Document. Le score original (_) est ignoré car RRF ne se base
        # -- que sur la position dans le classement, pas sur le score brut.
        for rank, (doc, _) in enumerate(ranked_list):
            # -- On construit une clé unique pour identifier ce document.
            # -- La clé combine : le chemin source du fichier, l'index du chunk,
            # -- et les 50 premiers caractères du contenu. Cela permet d'éviter
            # -- les doublons lorsqu'un même document apparaît dans plusieurs classements.
            doc_key = (
                doc.metadata.get("source", "")
                + "::"
                + str(doc.metadata.get("chunk_index", 0))
                + "::"
                + doc.page_content[:50]
            )

            # -- Formule RRF : score = 1 / (k + rang + 1)
            # -- Le "+1" décale le rang pour commencer à 1 (au lieu de 0).
            # -- Avec k=60 : le 1er document obtient 1/61 ≈ 0.0164,
            # -- le 2ème obtient 1/62 ≈ 0.0161, etc.
            # -- La décroissance est douce, ce qui évite qu'un seul classement domine.
            rrf_score = 1.0 / (k + rank + 1)
            # -- On accumule les scores RRF pour chaque document : si un document
            # -- apparaît dans plusieurs classements, ses scores s'additionnent.
            # -- C'est le coeur de la fusion : un document bien classé partout
            # -- accumule un score total élevé.
            doc_scores[doc_key] = doc_scores.get(doc_key, 0.0) + rrf_score
            # -- On garde (ou met à jour) la référence vers l'objet Document.
            doc_objects[doc_key] = doc

    # -- On trie tous les documents par score RRF décroissant pour obtenir
    # -- le classement final fusionné.
    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # -- On retourne les top_n documents les plus pertinents avec leur score RRF.
    return [
        (doc_objects[key], score)
        for key, score in sorted_items[:top_n]
    ]


# ─── Helpers ─────────────────────────────────────────────────────────────────

# -- Fonction utilitaire pour extraire une année à partir de formats de dates variés
# -- rencontrés dans les documents financiers (rapports annuels, trimestriels, etc.).

def _parse_year(val: str) -> Optional[int]:
    """
    FIX MOYEN : parse robuste d'une année depuis des formats variés.
    Supporte : "2023", "Q4 2024", "FY2023", "2024-01-01", etc.
    Retourne None si le parsing échoue, avec un warning loggé.
    """
    # -- Si la valeur est vide ou None, on retourne immédiatement None.
    if not val:
        return None
    # -- On utilise une expression régulière pour trouver un motif de type "20XX"
    # -- (année commençant par 20, suivie de deux chiffres). Cela capture les années
    # -- de 2000 à 2099, ce qui couvre toutes les données financières actuelles.
    # -- Exemples de correspondances :
    # --   "Q4 2024"       → capture "2024"
    # --   "FY2023"        → capture "2023"
    # --   "2024-01-01"    → capture "2024"
    # --   "Annual 2022"   → capture "2022"
    m = re.search(r'(20\d{2})', str(val))
    if m:
        return int(m.group(1))
    # -- Si aucun motif d'année n'est trouvé, on retourne None.
    # -- Le document ne sera alors pas filtré par date (comportement permissif).
    return None


# ─── BM25 Retriever ───────────────────────────────────────────────────────────

# -- ==========================================================================
# -- BM25 (Best Matching 25) — Explication
# -- ==========================================================================
# -- BM25 est un algorithme classique de recherche d'information basé sur la
# -- fréquence des termes (TF-IDF amélioré). Il évalue la pertinence d'un document
# -- par rapport à une requête en considérant :
# --   1. La fréquence du terme dans le document (TF) — plus un mot apparaît, plus
# --      le document est potentiellement pertinent.
# --   2. La fréquence inverse dans le corpus (IDF) — un mot rare dans le corpus
# --      global est plus discriminant qu'un mot courant.
# --   3. La longueur du document — un document court contenant le terme est
# --      potentiellement plus pertinent qu'un long document.
# -- BM25 est complémentaire à la recherche dense (vectorielle) car il excelle
# -- dans la correspondance exacte de termes (noms propres, tickers, chiffres).
# -- ==========================================================================

class BM25Retriever:
    """BM25 sparse retriever sur le corpus complet."""

    def __init__(self, documents: List[Document]) -> None:
        # -- On importe BM25Okapi de la bibliothèque rank_bm25.
        # -- BM25Okapi est la variante la plus courante de BM25 (paramètres k1=1.5, b=0.75).
        from rank_bm25 import BM25Okapi

        # -- On stocke la liste complète des documents pour pouvoir les retrouver
        # -- par index après le calcul des scores.
        self._documents = documents
        # -- Tokenisation : on découpe chaque document en liste de mots (tokens).
        # -- BM25 travaille sur des listes de tokens, pas sur du texte brut.
        tokenized = [self._tokenize(d.page_content) for d in documents]
        # -- Initialisation de l'index BM25 à partir des documents tokenisés.
        # -- Cette étape calcule les statistiques IDF pour tout le corpus.
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 initialisé sur {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        # -- Tokenisation simple : conversion en minuscules puis découpage par espaces.
        # -- Cette approche est basique mais efficace pour du texte financier.
        # -- Une amélioration possible serait d'ajouter un stemming ou un lemmatiseur,
        # -- ou de retirer les stop words (mots vides comme "le", "de", "est").
        return text.lower().split()

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        # -- On tokenise la requête de la même manière que les documents
        # -- pour assurer la cohérence de la correspondance lexicale.
        tokenized_query = self._tokenize(query)
        # -- BM25 calcule un score de pertinence pour CHAQUE document du corpus.
        # -- Le tableau `scores` a la même taille que le nombre de documents.
        scores = self._bm25.get_scores(tokenized_query)

        # -- On trie les indices des documents par score décroissant
        # -- et on ne garde que les k meilleurs (top-k).
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        # -- On retourne les documents avec leur score BM25.
        # -- On filtre les documents avec un score de 0 (aucun terme en commun
        # -- avec la requête), car ils ne sont pas pertinents du tout.
        return [
            (self._documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


# ─── Main Hybrid Retriever ────────────────────────────────────────────────────

# -- ==========================================================================
# -- PIPELINE DE RECHERCHE HYBRIDE — Vue d'ensemble
# -- ==========================================================================
# -- Le HybridFinancialRetriever orchestre tout le pipeline de recherche :
# --
# -- Étape 1 : Recherche Dense (sémantique)
# --   → Interroge ChromaDB avec les embeddings vectoriels de la requête.
# --   → Retourne les documents dont le sens est proche de la requête.
# --
# -- Étape 2 : Recherche Sparse (BM25)
# --   → Cherche les documents contenant les mots exacts de la requête.
# --   → Complémentaire à la recherche dense pour les termes spécifiques.
# --
# -- Étape 3 : Fusion RRF
# --   → Combine les deux classements via Reciprocal Rank Fusion.
# --   → Produit un classement unique pondéré par les deux méthodes.
# --
# -- Étape 4 : Post-filtrage et Boost temporel
# --   → Filtre par plage de dates si demandé.
# --   → Applique un boost aux documents récents (time decay).
# --   → Élimine les documents en dessous du seuil de pertinence minimum.
# --
# -- Résultat : les top_k documents les plus pertinents, filtrés et reclassés.
# -- ==========================================================================

class HybridFinancialRetriever:
    """
    Retriever hybride pour documents financiers.

    Combine Dense (ChromaDB) + Sparse (BM25) via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        vector_store,
        top_k: int = 10,
        dense_k: int = 20,
        sparse_k: int = 20,
        time_decay_factor: float = 0.1,
    ) -> None:
        # -- vector_store : l'objet ChromaDB qui contient les embeddings vectoriels
        # -- de tous les documents financiers indexés.
        self._vector_store = vector_store
        # -- top_k : nombre final de documents à retourner après fusion et filtrage.
        self._top_k = top_k
        # -- dense_k : nombre de documents à récupérer via la recherche dense (sémantique).
        # -- On récupère plus que top_k pour avoir une marge avant la fusion RRF.
        self._dense_k = dense_k
        # -- sparse_k : nombre de documents à récupérer via BM25 (recherche lexicale).
        # -- Même logique : on en prend plus pour alimenter la fusion.
        self._sparse_k = sparse_k
        # -- time_decay_factor : facteur de décroissance temporelle.
        # -- Plus cette valeur est élevée, plus les documents anciens sont pénalisés.
        # -- Avec 0.1 : un document vieux de 5 ans voit son score multiplié par
        # -- 1/(1 + 0.1*5) = 1/1.5 ≈ 0.67 (perte de ~33% du score).
        # -- Avec 0.0 : pas de pénalité temporelle (tous les documents sont traités
        # -- de manière égale quelle que soit leur date).
        self._time_decay_factor = time_decay_factor
        # -- _bm25 : instance du BM25Retriever, initialisée de manière paresseuse (lazy).
        # -- Elle ne sera créée qu'au premier appel nécessitant BM25, pour éviter
        # -- de charger tout le corpus en mémoire si on ne fait que du dense.
        self._bm25: Optional[BM25Retriever] = None

        logger.info(
            f"HybridRetriever initialisé (dense_k={dense_k}, sparse_k={sparse_k}, top_k={top_k})"
        )

    def _ensure_bm25(self) -> None:
        """
        FIX ÉLEVÉ : initialise BM25 avec plafond mémoire configurable.
        Sur un corpus >10 000 docs, BM25 est tronqué pour éviter les OOM.
        """
        # -- Si BM25 est déjà initialisé, on ne fait rien (pattern lazy singleton).
        if self._bm25 is not None:
            return

        logger.info("Initialisation BM25 (lazy)...")
        # -- On récupère TOUS les documents du vector store pour construire l'index BM25.
        # -- C'est une opération coûteuse en mémoire, d'où l'initialisation paresseuse.
        docs = self._vector_store.get_all_documents()

        # -- Si le vector store est vide, on ne peut pas créer d'index BM25.
        if not docs:
            logger.warning("Vector store vide, BM25 non initialisé")
            return

        # -- Protection contre les dépassements de mémoire (OOM = Out Of Memory).
        # -- BM25_MAX_DOCS est défini dans le fichier .env via settings.
        # -- Si le corpus dépasse cette limite, on tronque pour protéger la RAM.
        max_docs = settings.BM25_MAX_DOCS
        if len(docs) > max_docs:
            logger.warning(
                f"BM25: corpus tronqué à {max_docs} docs "
                f"(total={len(docs)}) pour éviter l'OOM. "
                f"Augmentez BM25_MAX_DOCS dans .env si nécessaire."
            )
            docs = docs[:max_docs]

        # -- Création effective de l'index BM25 sur les documents récupérés.
        self._bm25 = BM25Retriever(docs)

    # -- ==========================================================================
    # -- MÉTHODE PRINCIPALE : retrieve()
    # -- ==========================================================================
    # -- C'est le point d'entrée du pipeline hybride. Elle orchestre toutes les étapes :
    # -- construction des filtres, recherche dense, recherche sparse, fusion RRF,
    # -- puis post-filtrage et boost temporel.
    # -- ==========================================================================

    def retrieve(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        document_type: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        min_relevance_score: float = 0.0,
        use_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """Retrieval hybride avec filtres et RRF."""
        # -- Construction du filtre "where" pour ChromaDB.
        # -- Ce filtre est appliqué AVANT la recherche dense pour réduire l'espace
        # -- de recherche directement au niveau de la base vectorielle.
        where_filter = self._build_where_filter(document_type, ticker)
        # -- Détermination des collections ChromaDB à interroger en fonction
        # -- du type de document demandé (news, reports, tables, csv).
        collection_types = self._infer_collections(document_type)

        # -- ÉTAPE 1 : Recherche Dense (sémantique via embeddings)
        # -- On interroge ChromaDB avec la requête. ChromaDB transforme la requête
        # -- en vecteur d'embedding et cherche les documents dont les vecteurs sont
        # -- les plus proches (similarité cosinus ou distance L2).
        dense_results = self._vector_store.similarity_search(
            query=query,
            k=self._dense_k,
            collection_types=collection_types,
            where=where_filter if where_filter else None,
        )
        logger.debug(f"Dense retrieval: {len(dense_results)} résultats")

        # -- Si le mode hybride est désactivé (use_hybrid=False), on ne fait que
        # -- la recherche dense, on applique les filtres/boosts et on retourne.
        if not use_hybrid:
            return self._apply_filters_and_boost(
                dense_results, date_range, min_relevance_score
            )[:self._top_k]

        # -- ÉTAPE 2 : Recherche Sparse (BM25 — correspondance lexicale)
        # -- On s'assure que l'index BM25 est initialisé (lazy loading).
        self._ensure_bm25()
        sparse_results: List[Tuple[Document, float]] = []

        if self._bm25 is not None:
            # -- Recherche BM25 sur le corpus complet.
            sparse_results = self._bm25.search(query, k=self._sparse_k)
            # -- Filtrage des résultats BM25 par type de document et ticker.
            # -- Contrairement à la recherche dense où le filtre est appliqué
            # -- directement par ChromaDB, ici on filtre manuellement a posteriori
            # -- car BM25 n'a pas de mécanisme de filtrage intégré.
            sparse_results = self._filter_sparse(sparse_results, document_type, ticker)
            logger.debug(f"Sparse retrieval: {len(sparse_results)} résultats")

        # -- ÉTAPE 3 : Fusion RRF (Reciprocal Rank Fusion)
        # -- On prépare les listes non vides à fusionner.
        lists_to_fuse = [lst for lst in [dense_results, sparse_results] if lst]

        if len(lists_to_fuse) == 1:
            # -- Si une seule liste est non vide (ex: BM25 n'a rien trouvé),
            # -- pas besoin de fusion, on prend directement cette liste.
            fused = list(lists_to_fuse[0])
        else:
            # -- Fusion RRF des deux classements (dense + sparse).
            # -- On demande top_k * 2 résultats pour avoir de la marge avant
            # -- le post-filtrage qui va potentiellement éliminer des documents.
            fused = reciprocal_rank_fusion(
                ranked_lists=lists_to_fuse,
                top_n=self._top_k * 2,
            )

        # -- ÉTAPE 4 : Post-filtrage et boost temporel
        # -- On applique le filtre par plage de dates, le seuil de pertinence minimum,
        # -- et le boost favorisant les documents récents.
        final = self._apply_filters_and_boost(fused, date_range, min_relevance_score)

        logger.info(f"Hybrid retrieval: {len(final)} résultats finaux")
        # -- On retourne les top_k meilleurs documents après tous les traitements.
        return final[:self._top_k]

    # -- ==========================================================================
    # -- CONSTRUCTION DU FILTRE "WHERE" POUR CHROMADB
    # -- ==========================================================================
    # -- ChromaDB supporte un filtrage par métadonnées via une clause "where".
    # -- Cette méthode construit ce filtre à partir des critères de l'utilisateur.
    # -- Le filtre est appliqué directement dans ChromaDB, ce qui est plus performant
    # -- que de filtrer a posteriori en Python (moins de données transférées).
    # -- ==========================================================================

    def _build_where_filter(
        self,
        document_type: Optional[List[str]],
        ticker: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        # -- Liste des conditions de filtrage à combiner.
        conditions = []

        # -- Filtre par type de document (ex: "news_article", "annual_report").
        # -- Note : ChromaDB ne supporte pas nativement le "$in" pour les listes,
        # -- donc on ne filtre que si un seul type est demandé. Si plusieurs types
        # -- sont demandés, le filtrage sera fait au niveau des collections.
        if document_type and len(document_type) == 1:
            conditions.append({"document_type": {"$eq": document_type[0]}})

        # -- Filtre par symbole boursier (ticker), ex: "AAPL", "MSFT".
        # -- "$contains" vérifie que le ticker est présent dans le champ ticker_symbols.
        if ticker:
            conditions.append({"ticker_symbols": {"$contains": ticker}})

        # -- Si aucune condition, on retourne None (pas de filtre).
        if not conditions:
            return None
        # -- Si une seule condition, on la retourne directement.
        if len(conditions) == 1:
            return conditions[0]
        # -- Si plusieurs conditions, on les combine avec un opérateur "$and"
        # -- (toutes les conditions doivent être satisfaites simultanément).
        return {"$and": conditions}

    # -- ==========================================================================
    # -- INFÉRENCE DES COLLECTIONS CHROMADB
    # -- ==========================================================================
    # -- ChromaDB organise les documents en collections thématiques.
    # -- Cette méthode détermine quelles collections interroger en fonction
    # -- du type de document demandé par l'utilisateur.
    # -- ==========================================================================

    def _infer_collections(
        self,
        document_type: Optional[List[str]],
    ) -> Optional[List[str]]:
        # -- Si aucun type de document n'est spécifié, on retourne None
        # -- pour interroger TOUTES les collections (recherche globale).
        if not document_type:
            return None

        # -- Table de correspondance entre les types de documents et les noms
        # -- de collections ChromaDB. Chaque type de document est stocké dans
        # -- une collection spécifique pour optimiser les recherches ciblées.
        mapping = {
            "news_article": ["news"],          # -- Articles de presse financière
            "financial_table": ["tables"],      # -- Tableaux financiers (bilans, P&L, etc.)
            "annual_report": ["reports"],       # -- Rapports annuels (10-K, rapport de gestion)
            "quarterly_report": ["reports"],    # -- Rapports trimestriels (10-Q)
            "csv_data": ["csv"],               # -- Données CSV (cours boursiers, etc.)
            "market_overview": ["reports"],     # -- Synthèses de marché
        }

        # -- On utilise un set pour éviter les doublons (ex: annual_report et
        # -- quarterly_report pointent tous les deux vers "reports").
        collections = set()
        for dt in document_type:
            # -- Si le type de document n'est pas dans le mapping, on cherche
            # -- par défaut dans la collection "reports" (comportement permissif).
            colls = mapping.get(dt, ["reports"])
            collections.update(colls)

        return list(collections) if collections else None

    # -- ==========================================================================
    # -- FILTRAGE POST-HOC DES RÉSULTATS BM25
    # -- ==========================================================================
    # -- Contrairement à ChromaDB qui filtre directement via la clause "where",
    # -- BM25 ne dispose pas de mécanisme de filtrage intégré. On doit donc
    # -- filtrer manuellement les résultats après la recherche.
    # -- ==========================================================================

    def _filter_sparse(
        self,
        results: List[Tuple[Document, float]],
        document_type: Optional[List[str]],
        ticker: Optional[str],
    ) -> List[Tuple[Document, float]]:
        filtered = []
        for doc, score in results:
            meta = doc.metadata

            # -- Filtre par type de document : on ne garde que les documents
            # -- dont le type correspond à l'un des types demandés.
            if document_type:
                if meta.get("document_type") not in document_type:
                    continue

            # -- Filtre par ticker : on vérifie que le symbole boursier demandé
            # -- est présent dans la liste des tickers du document.
            if ticker:
                # -- Le champ ticker_symbols peut être une chaîne JSON ou une liste Python.
                # -- On gère les deux cas pour être robuste.
                tickers_raw = meta.get("ticker_symbols", "[]")
                if isinstance(tickers_raw, str):
                    try:
                        # -- Tentative de parsing JSON (ex: '["AAPL", "MSFT"]')
                        tickers = json.loads(tickers_raw)
                    except Exception:
                        # -- Si le parsing échoue, on considère que la liste est vide.
                        tickers = []
                else:
                    tickers = tickers_raw or []
                # -- Si le ticker demandé n'est pas dans la liste, on exclut le document.
                if ticker not in tickers:
                    continue

            # -- Le document a passé tous les filtres, on le conserve.
            filtered.append((doc, score))
        return filtered

    # -- ==========================================================================
    # -- POST-FILTRAGE ET BOOST TEMPOREL
    # -- ==========================================================================
    # -- Cette méthode applique les derniers traitements avant de retourner
    # -- les résultats finaux :
    # --
    # -- 1. Filtre par seuil de pertinence minimum (min_relevance_score) :
    # --    élimine les documents dont le score est trop faible.
    # --
    # -- 2. Filtre par plage de dates (date_range) :
    # --    ne conserve que les documents dont l'année de publication
    # --    est dans la plage [start_year, end_year].
    # --
    # -- 3. Boost temporel (time decay) :
    # --    multiplie le score par un facteur de décroissance basé sur l'ancienneté
    # --    du document. Formule : score_final = score * 1/(1 + decay * ancienneté).
    # --    Cela favorise les documents récents, qui contiennent souvent des
    # --    informations financières plus à jour et donc plus pertinentes.
    # -- ==========================================================================

    def _apply_filters_and_boost(
        self,
        results: List[Tuple[Document, float]],
        date_range: Optional[Tuple[str, str]],
        min_relevance_score: float,
    ) -> List[Tuple[Document, float]]:
        """
        FIX MOYEN : parsing robuste des dates via _parse_year().
        Les formats "Q4 2024", "FY2023" sont maintenant correctement gérés.
        Avant, int("Q4 2"[:4]) levait une ValueError silencieusement ignorée.
        """
        # -- Si la liste de résultats est vide, on retourne une liste vide.
        if not results:
            return []

        # -- Année courante, utilisée pour calculer l'ancienneté des documents.
        current_year = datetime.now().year
        # -- Liste des résultats après application des filtres et du boost.
        boosted: List[Tuple[Document, float]] = []

        # -- Pré-parsing de la plage de dates une seule fois (optimisation).
        # -- On extrait les années de début et de fin à partir des chaînes fournies.
        start_year: Optional[int] = None
        end_year: Optional[int] = None
        if date_range:
            start_year = _parse_year(str(date_range[0]))
            end_year = _parse_year(str(date_range[1]))
            # -- Si le parsing échoue pour l'une des bornes, on désactive
            # -- complètement le filtre temporel plutôt que de risquer un
            # -- filtrage incorrect.
            if start_year is None or end_year is None:
                logger.warning(
                    f"date_range non parseable: {date_range} — filtre temporel désactivé"
                )
                start_year = end_year = None

        for doc, score in results:
            # -- Filtre par seuil de pertinence : on élimine les documents
            # -- dont le score est en dessous du minimum demandé.
            # -- Par défaut min_relevance_score=0.0, donc tous les documents passent.
            if score < min_relevance_score:
                continue

            meta = doc.metadata
            # -- On cherche la période du document dans ses métadonnées.
            # -- Le champ peut s'appeler "date" ou "time_period" selon le type de document.
            doc_period = meta.get("date", "") or meta.get("time_period", "")

            # -- FILTRE PAR PLAGE DE DATES
            # -- On ne filtre que si une plage valide a été fournie ET que le document
            # -- possède une information de date dans ses métadonnées.
            if start_year is not None and end_year is not None and doc_period:
                doc_year = _parse_year(str(doc_period))
                if doc_year is not None:
                    # -- Si l'année du document est en dehors de la plage demandée,
                    # -- on l'exclut des résultats.
                    if not (start_year <= doc_year <= end_year):
                        continue
                # -- Si doc_year n'est pas parseable (None), on garde le document
                # -- par prudence : mieux vaut garder un document potentiellement
                # -- pertinent que de le perdre à cause d'un problème de parsing.

            # -- BOOST TEMPOREL (Time Decay)
            # -- Appliqué seulement si le facteur de décroissance est > 0
            # -- et que le document a une date identifiable.
            if self._time_decay_factor > 0 and doc_period:
                doc_year = _parse_year(str(doc_period))
                if doc_year is not None:
                    # -- Calcul de l'ancienneté en années (minimum 0 pour éviter
                    # -- un boost > 1 pour des documents "futurs").
                    years_ago = max(0, current_year - doc_year)
                    # -- Formule de décroissance hyperbolique :
                    # -- boost = 1 / (1 + decay_factor * years_ago)
                    # -- Exemples avec decay=0.1 :
                    # --   Document de cette année (0 ans) : boost = 1.0 (pas de pénalité)
                    # --   Document de 2 ans : boost = 1/(1+0.2) = 0.83
                    # --   Document de 5 ans : boost = 1/(1+0.5) = 0.67
                    # --   Document de 10 ans : boost = 1/(1+1.0) = 0.50
                    # -- La décroissance est douce et ne pénalise jamais à zéro.
                    boost = 1.0 / (1.0 + self._time_decay_factor * years_ago)
                    # -- On multiplie le score du document par le facteur de boost.
                    score = score * boost

            # -- Le document a passé tous les filtres, on l'ajoute avec son score ajusté.
            boosted.append((doc, score))

        # -- Tri final par score décroissant pour retourner les documents
        # -- les plus pertinents en premier.
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def invalidate_bm25_cache(self) -> None:
        """Invalide le cache BM25 (à appeler après ajout de nouveaux documents)."""
        # -- Remet l'index BM25 à None. Au prochain appel de retrieve() avec
        # -- use_hybrid=True, l'index sera reconstruit automatiquement via
        # -- _ensure_bm25() avec les nouveaux documents.
        self._bm25 = None
        logger.info("Cache BM25 invalidé")
