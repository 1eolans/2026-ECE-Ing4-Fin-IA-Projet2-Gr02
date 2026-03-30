"""
Vector store ChromaDB persistant pour FinRAG.
Supporte embeddings OpenAI (text-embedding-3-small) avec fallback local (all-MiniLM-L6-v2).
Collections séparées par type de document.
"""

# -- Ce module constitue la couche de stockage vectoriel du systeme FinRAG.
# -- Il s'appuie sur ChromaDB, une base de donnees vectorielle persistante,
# -- pour indexer et rechercher des documents financiers par similarite semantique.
# --
# -- Architecture generale :
# --   1. EmbeddingProvider : classe qui encapsule la generation d'embeddings
# --      (vecteurs numeriques representant le sens du texte). Elle tente d'abord
# --      d'utiliser l'API OpenAI, puis bascule sur un modele local HuggingFace
# --      en cas d'echec (strategie de fallback).
# --   2. Fonctions utilitaires : generation d'identifiants uniques, classification
# --      des documents par type, et serialisation des metadonnees.
# --   3. FinancialVectorStore : classe principale qui gere les collections ChromaDB,
# --      l'ajout de documents, la recherche par similarite, la suppression, et
# --      les statistiques.
# --
# -- Strategie de collections :
# --   Les documents sont repartis dans 4 collections distinctes dans ChromaDB :
# --     - "reports"  : rapports financiers (10-K, 10-Q, rapports annuels, etc.)
# --     - "news"     : articles de presse et actualites financieres
# --     - "tables"   : tableaux financiers extraits des documents (bilans, etc.)
# --     - "csv"      : donnees tabulaires importees depuis des fichiers CSV/Excel
# --   Cette separation permet de cibler les recherches sur un type de document
# --   specifique, ameliorant ainsi la pertinence des resultats du RAG.

from __future__ import annotations

import hashlib  # -- Pour generer des identifiants deterministes via SHA-256
import json  # -- Pour serialiser les listes en chaines JSON (metadonnees ChromaDB)
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import chromadb  # -- Client ChromaDB : base de donnees vectorielle open-source
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document  # -- Objet Document de LangChain (contenu + metadonnees)
from loguru import logger  # -- Bibliotheque de logging structuree et coloree
from tqdm import tqdm  # -- Barre de progression pour les boucles longues

# Add src to path for config import
# -- Ajoute le repertoire racine du projet au PYTHONPATH pour pouvoir
# -- importer le module de configuration (settings) depuis src.config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Embedding Functions ──────────────────────────────────────────────────────

# -- ==========================================================================
# -- Classe EmbeddingProvider
# -- ==========================================================================
# -- Cette classe est responsable de transformer du texte brut en vecteurs
# -- numeriques (embeddings). Les embeddings capturent le sens semantique
# -- du texte et permettent de comparer des documents par similarite.
# --
# -- Strategie de fallback :
# --   1. Tentative avec OpenAI (text-embedding-3-small) si configure
# --   2. Si echec ou non configure, bascule sur le modele local
# --      HuggingFace all-MiniLM-L6-v2 (tourne sur CPU, sans API externe)
# --
# -- Cela garantit que le systeme fonctionne meme sans cle API OpenAI.
# -- ==========================================================================

class EmbeddingProvider:
    """Fournisseur d'embeddings avec fallback local."""

    def __init__(self) -> None:
        # -- Attribut prive qui stocke la fonction d'embedding (OpenAI ou HuggingFace)
        self._embed_fn = None
        # -- Nom du modele d'embedding utilise (pour le logging et les stats)
        self._model_name = ""
        # -- Lance l'initialisation du fournisseur d'embeddings
        self._initialize()

    def _initialize(self) -> None:
        # -- Methode d'aiguillage : verifie la configuration pour savoir
        # -- si on doit utiliser OpenAI ou le modele local.
        # -- Le parametre use_openai_embeddings est defini dans settings.
        if settings.use_openai_embeddings:
            self._initialize_openai()
        else:
            self._initialize_local()

    def _initialize_openai(self) -> None:
        # -- Tente d'initialiser les embeddings via l'API OpenAI.
        # -- Utilise le modele configure dans settings.EMBEDDING_MODEL
        # -- (typiquement "text-embedding-3-small", 1536 dimensions).
        # -- En cas d'echec (cle API invalide, reseau indisponible, etc.),
        # -- bascule automatiquement sur le modele local (fallback).
        try:
            from langchain_openai import OpenAIEmbeddings
            self._embed_fn = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            self._model_name = settings.EMBEDDING_MODEL
            logger.info(f"Embeddings OpenAI : {self._model_name}")
        except Exception as e:
            logger.warning(f"OpenAI embeddings échec ({e}), fallback local")
            # -- Fallback : on appelle l'initialisation locale en cas d'erreur
            self._initialize_local()

    def _initialize_local(self) -> None:
        # -- Initialise les embeddings avec un modele HuggingFace local.
        # -- Le modele all-MiniLM-L6-v2 est leger (~80 Mo), fonctionne sur CPU,
        # -- et produit des embeddings de 384 dimensions.
        # -- normalize_embeddings=True normalise les vecteurs (norme L2 = 1),
        # -- ce qui est necessaire pour la distance cosine utilisee par ChromaDB.
        # -- Si meme le modele local echoue, une RuntimeError est levee car
        # -- le systeme ne peut pas fonctionner sans embeddings.
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embed_fn = HuggingFaceEmbeddings(
                model_name=settings.FALLBACK_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},  # -- Force l'execution sur CPU (pas besoin de GPU)
                encode_kwargs={"normalize_embeddings": True},  # -- Normalisation L2 pour distance cosine
            )
            self._model_name = settings.FALLBACK_EMBEDDING_MODEL
            logger.info(f"Embeddings locaux : {self._model_name}")
        except Exception as e:
            logger.error(f"Impossible d'initialiser les embeddings: {e}")
            raise RuntimeError("Aucun fournisseur d'embeddings disponible") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de textes."""
        # -- Prend une liste de textes (documents) et retourne une liste
        # -- de vecteurs d'embeddings. Chaque vecteur est une liste de floats
        # -- dont la dimension depend du modele (1536 pour OpenAI, 384 pour MiniLM).
        # -- Utilise embed_documents de LangChain (optimise pour le batch).
        return self._embed_fn.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Génère l'embedding pour une requête."""
        # -- Genere l'embedding pour une seule requete utilisateur.
        # -- Certains modeles (comme OpenAI) distinguent l'embedding de documents
        # -- et l'embedding de requetes (prefixes differents). C'est pourquoi
        # -- on utilise embed_query plutot que embed_documents pour les recherches.
        return self._embed_fn.embed_query(text)

    @property
    def model_name(self) -> str:
        # -- Propriete en lecture seule qui retourne le nom du modele
        # -- d'embedding actuellement utilise (utile pour les logs et stats).
        return self._model_name

    @property
    def langchain_embeddings(self):
        """Retourne l'objet embeddings compatible LangChain."""
        # -- Expose l'objet d'embedding sous-jacent pour une utilisation
        # -- directe avec d'autres composants LangChain (ex: LangChain VectorStore).
        return self._embed_fn


# ─── ChromaDB Collection Manager ─────────────────────────────────────────────

# -- Liste des types de collections disponibles dans ChromaDB.
# -- Chaque type correspond a une categorie de document financier.
# -- Cette constante est utilisee partout dans le module pour iterer
# -- sur les collections de maniere coherente.
COLLECTION_TYPES = ["reports", "news", "tables", "csv"]


def _make_doc_id(content: str, metadata: Dict[str, Any]) -> str:
    """Génère un ID unique et déterministe pour un document."""
    # -- Fonction utilitaire qui cree un identifiant unique pour chaque chunk
    # -- de document. L'ID est deterministe : le meme contenu avec les memes
    # -- metadonnees produira toujours le meme ID.
    # --
    # -- Logique :
    # --   1. On concatene source (nom du fichier), numero de page, index du chunk,
    # --      et les 100 premiers caracteres du contenu.
    # --   2. On hache cette chaine avec SHA-256 pour obtenir un ID unique.
    # --   3. On tronque a 16 caracteres hexadecimaux (64 bits d'entropie,
    # --      suffisant pour eviter les collisions a notre echelle).
    # --
    # -- Cet ID deterministe permet l'upsert : si on reindexe le meme document,
    # -- les chunks existants seront mis a jour au lieu d'etre dupliques.
    source = metadata.get("source", "")
    page = metadata.get("page_number", 0)
    chunk_idx = metadata.get("chunk_index", 0)
    raw = f"{source}::{page}::{chunk_idx}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _classify_doc_type(metadata: Dict[str, Any]) -> str:
    """Classe un document dans une collection ChromaDB."""
    # -- Fonction utilitaire qui determine dans quelle collection ChromaDB
    # -- un document doit etre stocke, en se basant sur ses metadonnees.
    # --
    # -- Regles de classification (par priorite) :
    # --   1. Si document_type == "financial_table" OU si contains_table == True
    # --      -> collection "tables" (tableaux financiers)
    # --   2. Si document_type == "news_article"
    # --      -> collection "news" (articles de presse)
    # --   3. Si document_type == "csv_data" ou "excel_data"
    # --      -> collection "csv" (donnees tabulaires)
    # --   4. Sinon (par defaut)
    # --      -> collection "reports" (rapports financiers generaux)
    # --
    # -- Retourne le nom du type de collection (cle dans COLLECTION_TYPES).
    doc_type = metadata.get("document_type", "")
    if doc_type in ("financial_table",) or metadata.get("contains_table"):
        return "tables"
    elif doc_type in ("news_article",):
        return "news"
    elif doc_type in ("csv_data", "excel_data"):
        return "csv"
    else:
        return "reports"


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sérialise les métadonnées pour ChromaDB (valeurs scalaires uniquement)."""
    # -- ChromaDB n'accepte que des valeurs scalaires (str, int, float, bool)
    # -- dans les metadonnees. Cette fonction convertit toutes les valeurs
    # -- non-scalaires en representations compatibles :
    # --   - str, int, float, bool : conserves tels quels
    # --   - list : serialisee en chaine JSON (ex: ["a","b"] -> '["a","b"]')
    # --   - None : remplace par une chaine vide ""
    # --   - Autres types : convertis en chaine via str()
    # --
    # -- Retourne un nouveau dictionnaire avec uniquement des valeurs scalaires.
    serialized: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            serialized[k] = v
        elif isinstance(v, list):
            # -- Les listes sont converties en JSON pour etre stockees comme chaines
            serialized[k] = json.dumps(v)
        elif v is None:
            # -- Les valeurs None sont remplacees par des chaines vides
            serialized[k] = ""
        else:
            # -- Tout autre type est converti en sa representation textuelle
            serialized[k] = str(v)
    return serialized


# ─── Main Vector Store ────────────────────────────────────────────────────────

# -- ==========================================================================
# -- Classe FinancialVectorStore
# -- ==========================================================================
# -- Classe principale du module. Elle encapsule toute la logique d'interaction
# -- avec ChromaDB pour le stockage et la recherche de documents financiers.
# --
# -- Responsabilites :
# --   - Gerer un client ChromaDB persistant (donnees sauvegardees sur disque)
# --   - Maintenir 4 collections separees (reports, news, tables, csv)
# --   - Fournir une API CRUD complete :
# --       * add_documents   : ajouter/mettre a jour des documents
# --       * similarity_search : rechercher par similarite semantique
# --       * delete_by_source : supprimer les chunks d'un fichier source
# --       * list_sources    : lister les sources indexees
# --       * get_stats       : obtenir les statistiques globales
# --       * get_all_texts   : recuperer tous les textes (pour BM25)
# --       * get_all_documents: recuperer tous les Documents LangChain
# --
# -- Flux d'indexation :
# --   1. Les documents sont classes par type via _classify_doc_type
# --   2. Chaque groupe est indexe dans sa collection respective
# --   3. Les embeddings sont generes par batch via EmbeddingProvider
# --   4. L'upsert ChromaDB evite les doublons grace aux IDs deterministes
# --
# -- Flux de recherche :
# --   1. La requete est transformee en vecteur via embed_query
# --   2. ChromaDB cherche les k vecteurs les plus proches (distance cosine)
# --   3. Les resultats de toutes les collections sont fusionnes et tries
# -- ==========================================================================

class FinancialVectorStore:
    """
    Vector store ChromaDB persistant pour documents financiers.

    Collections séparées : reports | news | tables | csv
    CRUD complet : add_documents, delete_by_source, list_sources, get_stats
    Embeddings : OpenAI text-embedding-3-small (+ fallback all-MiniLM-L6-v2)
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        """
        Args:
            persist_dir: Répertoire de persistance ChromaDB. Utilise settings si None.
        """
        # -- Determine le repertoire de persistance : soit le parametre fourni,
        # -- soit le chemin par defaut defini dans la configuration (settings).
        self._persist_dir = persist_dir or str(settings.chroma_persist_path)
        # -- Cree le repertoire s'il n'existe pas encore (y compris les parents)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialisation ChromaDB : {self._persist_dir}")

        # -- Initialise le client ChromaDB en mode persistant.
        # -- Les donnees sont stockees sur disque dans _persist_dir et
        # -- survivent aux redemarrages de l'application.
        # -- anonymized_telemetry=False desactive la collecte de telemetrie.
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # -- Cree le fournisseur d'embeddings (OpenAI ou local)
        self._embedding_provider = EmbeddingProvider()
        # -- Dictionnaire qui associe chaque type de collection a son objet ChromaDB
        self._collections: Dict[str, chromadb.Collection] = {}
        # -- Initialise ou recupere les 4 collections dans ChromaDB
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialise (ou récupère) les collections ChromaDB."""
        # -- Parcourt les 4 types de collections (reports, news, tables, csv)
        # -- et cree ou recupere chacune dans ChromaDB.
        # -- Le nom de chaque collection est prefixe par COLLECTION_PREFIX
        # -- (defini dans settings) pour eviter les conflits avec d'autres projets.
        # -- L'espace de distance utilise est "cosine" (similarite cosine),
        # -- ce qui est standard pour la recherche semantique.
        for coll_type in COLLECTION_TYPES:
            name = f"{settings.COLLECTION_PREFIX}_{coll_type}"
            try:
                coll = self._client.get_or_create_collection(
                    name=name,
                    # -- hnsw:space = "cosine" configure l'index HNSW pour utiliser
                    # -- la distance cosine (1 - similarite cosine) comme metrique.
                    metadata={"hnsw:space": "cosine"},
                )
                self._collections[coll_type] = coll
                # -- Log le nombre de documents deja presents dans la collection
                logger.debug(f"Collection '{name}' : {coll.count()} docs")
            except Exception as e:
                logger.error(f"Erreur initialisation collection {name}: {e}")
                raise

    # ── Public API ───────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Ajoute des documents au vector store.

        Args:
            documents: Liste de Documents LangChain avec métadonnées.
            batch_size: Taille des batches pour l'indexation.
            show_progress: Affiche une barre de progression.

        Returns:
            Nombre de documents ajoutés avec succès.
        """
        # -- Methode principale d'indexation. Elle prend une liste de Documents
        # -- LangChain, les classe par type de collection, genere leurs embeddings
        # -- par batch, et les insere dans ChromaDB via upsert.
        # --
        # -- L'upsert (update + insert) permet de mettre a jour les documents
        # -- existants si leurs IDs correspondent, evitant ainsi les doublons
        # -- lors d'une reindexation.

        if not documents:
            # -- Rien a faire si la liste est vide
            return 0

        added_count = 0
        # Group by collection type
        # -- Repartit les documents dans des listes par type de collection.
        # -- Chaque document est classe selon ses metadonnees via _classify_doc_type.
        by_collection: Dict[str, List[Document]] = {t: [] for t in COLLECTION_TYPES}
        for doc in documents:
            coll_type = _classify_doc_type(doc.metadata)
            by_collection[coll_type].append(doc)

        # -- Traite chaque collection separement
        for coll_type, docs in by_collection.items():
            if not docs:
                # -- Passe les collections sans documents
                continue

            collection = self._collections[coll_type]
            # -- Calcule le nombre total de batches pour la barre de progression
            total_batches = (len(docs) + batch_size - 1) // batch_size

            # -- Cree un iterateur sur les indices de debut de chaque batch
            iterator = range(0, len(docs), batch_size)
            if show_progress:
                # -- Encapsule l'iterateur dans tqdm pour afficher la progression
                iterator = tqdm(
                    iterator,
                    total=total_batches,
                    desc=f"Indexation [{coll_type}]",
                )

            for start in iterator:
                # -- Decoupe la liste de documents en batch de taille batch_size
                batch = docs[start: start + batch_size]
                try:
                    # -- Extrait les textes bruts de chaque document du batch
                    texts = [d.page_content for d in batch]
                    # -- Serialise les metadonnees pour les rendre compatibles ChromaDB
                    metadatas = [_serialize_metadata(d.metadata) for d in batch]
                    # -- Genere un ID unique et deterministe pour chaque document
                    ids = [_make_doc_id(d.page_content, d.metadata) for d in batch]

                    # Generate embeddings
                    # -- Genere les vecteurs d'embeddings pour tous les textes du batch
                    # -- en un seul appel (plus efficace que un par un)
                    embeddings = self._embedding_provider.embed_texts(texts)

                    # Upsert (handles duplicates)
                    # -- Insere ou met a jour les documents dans ChromaDB.
                    # -- Si un ID existe deja, le document est mis a jour.
                    # -- Sinon, un nouveau document est cree.
                    collection.upsert(
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    added_count += len(batch)

                except Exception as e:
                    # -- En cas d'erreur sur un batch, on log l'erreur et on continue
                    # -- avec le batch suivant (resilience aux erreurs ponctuelles)
                    logger.error(f"Erreur batch [{coll_type}] {start}-{start+len(batch)}: {e}")
                    continue

        # -- Log le resultat final de l'indexation
        logger.success(f"Indexé {added_count}/{len(documents)} documents")
        return added_count

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        collection_types: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche par similarité cosine.

        Args:
            query: Requête textuelle.
            k: Nombre de résultats par collection.
            collection_types: Collections à interroger (None = toutes).
            where: Filtre ChromaDB (ex: {"document_type": "news_article"}).

        Returns:
            Liste de (Document, score) triés par score décroissant.
        """
        # -- Methode de recherche semantique. Elle transforme la requete en vecteur,
        # -- interroge les collections ChromaDB, et retourne les documents les plus
        # -- similaires avec leur score de similarite.
        # --
        # -- Le score est calcule comme : similarite = 1 - distance_cosine
        # -- Plus le score est proche de 1, plus le document est pertinent.

        # -- Transforme la requete textuelle en vecteur d'embedding
        query_embedding = self._embedding_provider.embed_query(query)
        # -- Determine quelles collections interroger (toutes par defaut)
        collections_to_search = collection_types or COLLECTION_TYPES

        results: List[Tuple[Document, float]] = []

        for coll_type in collections_to_search:
            if coll_type not in self._collections:
                # -- Ignore les types de collection inconnus
                continue

            collection = self._collections[coll_type]
            n_docs = collection.count()
            if n_docs == 0:
                # -- Ignore les collections vides
                continue

            # -- On ne peut pas demander plus de resultats qu'il n'y a de documents
            actual_k = min(k, n_docs)

            try:
                # -- Construit les parametres de la requete ChromaDB
                query_params = {
                    "query_embeddings": [query_embedding],  # -- Vecteur de la requete
                    "n_results": actual_k,  # -- Nombre de resultats souhaites
                    "include": ["documents", "metadatas", "distances"],  # -- Donnees a inclure
                }
                if where:
                    # -- Ajoute un filtre optionnel sur les metadonnees
                    # -- (ex: filtrer par type de document ou par date)
                    query_params["where"] = where

                # -- Execute la requete de recherche par similarite dans ChromaDB
                response = collection.query(**query_params)

                # -- Parcourt les resultats et construit les paires (Document, score)
                for text, meta, distance in zip(
                    response["documents"][0],  # -- Textes des documents trouves
                    response["metadatas"][0],  # -- Metadonnees associees
                    response["distances"][0],  # -- Distances cosine (0 = identique, 2 = oppose)
                ):
                    score = 1.0 - distance  # cosine distance → similarity
                    # -- Convertit la distance cosine en score de similarite
                    # -- distance = 0 -> score = 1.0 (parfaitement similaire)
                    # -- distance = 1 -> score = 0.0 (orthogonal)
                    doc = Document(page_content=text, metadata=meta or {})
                    results.append((doc, score))

            except Exception as e:
                # -- En cas d'erreur sur une collection, on continue avec les autres
                logger.debug(f"Recherche [{coll_type}] erreur: {e}")
                continue

        # Sort by score descending
        # -- Trie tous les resultats (de toutes les collections) par score decroissant
        # -- puis ne garde que les k meilleurs resultats globaux
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def delete_by_source(self, source_filename: str) -> int:
        """
        Supprime tous les chunks d'une source donnée.

        Args:
            source_filename: Nom du fichier source.

        Returns:
            Nombre de chunks supprimés.
        """
        # -- Supprime tous les chunks (fragments de texte) associes a un fichier
        # -- source donne, dans toutes les collections.
        # -- Utile quand un fichier est mis a jour ou supprime du corpus.
        # -- Le filtre utilise le champ "filename" des metadonnees.
        deleted = 0
        for coll_type, collection in self._collections.items():
            try:
                # -- Recherche tous les documents ayant le filename specifie
                results = collection.get(
                    where={"filename": source_filename},
                    include=["documents"],
                )
                if results["ids"]:
                    # -- Supprime les documents trouves par leurs IDs
                    collection.delete(ids=results["ids"])
                    deleted += len(results["ids"])
                    logger.info(f"Supprimé {len(results['ids'])} chunks [{coll_type}] pour {source_filename}")
            except Exception as e:
                # -- Les erreurs sont loguees mais n'interrompent pas le processus
                logger.debug(f"delete_by_source [{coll_type}]: {e}")

        return deleted

    def list_sources(self) -> List[Dict[str, Any]]:
        """
        Liste toutes les sources indexées avec statistiques.

        Returns:
            Liste de dicts {filename, document_type, chunk_count, collection}.
        """
        # -- Parcourt toutes les collections et aggrege les metadonnees
        # -- pour produire un inventaire des sources indexees.
        # -- Chaque source est identifiee par la combinaison (collection, filename)
        # -- et on comptabilise le nombre de chunks par source.
        sources: Dict[str, Dict[str, Any]] = {}

        for coll_type, collection in self._collections.items():
            try:
                if collection.count() == 0:
                    # -- Ignore les collections vides pour eviter des requetes inutiles
                    continue

                # -- Recupere toutes les metadonnees de la collection
                results = collection.get(include=["metadatas"])
                for meta in results.get("metadatas", []):
                    if not meta:
                        continue
                    filename = meta.get("filename", "unknown")
                    # -- Cle unique : combinaison du type de collection et du nom de fichier
                    # -- pour distinguer un meme fichier present dans plusieurs collections
                    key = f"{coll_type}::{filename}"

                    if key not in sources:
                        # -- Premiere rencontre de cette source : initialise l'entree
                        sources[key] = {
                            "filename": filename,
                            "document_type": meta.get("document_type", "unknown"),
                            "chunk_count": 0,
                            "collection": coll_type,
                            "date": meta.get("date", ""),
                        }
                    # -- Incremente le compteur de chunks pour cette source
                    sources[key]["chunk_count"] += 1

            except Exception as e:
                logger.debug(f"list_sources [{coll_type}]: {e}")

        # -- Retourne la liste des sources (sans les cles internes)
        return list(sources.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques globales du vector store.

        Returns:
            Dict avec total_documents, total_chunks, collections, embedding_model.
        """
        # -- Construit un dictionnaire de statistiques sur l'etat du vector store.
        # -- Inclut le modele d'embedding utilise, le repertoire de persistance,
        # -- le nombre de chunks par collection, le total de chunks, et le nombre
        # -- de sources distinctes.
        stats: Dict[str, Any] = {
            "embedding_model": self._embedding_provider.model_name,
            "persist_dir": self._persist_dir,
            "collections": {},
            "total_chunks": 0,
        }

        # -- Parcourt chaque collection pour compter les documents
        for coll_type, collection in self._collections.items():
            count = collection.count()
            stats["collections"][coll_type] = count
            stats["total_chunks"] += count

        # -- Appelle list_sources pour compter le nombre de fichiers sources distincts
        stats["total_sources"] = len(self.list_sources())
        return stats

    def get_all_texts(self, collection_types: Optional[List[str]] = None) -> List[str]:
        """
        Récupère tous les textes du store (pour BM25).

        Args:
            collection_types: Collections à inclure (None = toutes).

        Returns:
            Liste de tous les textes indexés.
        """
        # -- Recupere le contenu textuel brut de tous les documents du store.
        # -- Cette methode est utilisee pour alimenter l'index BM25 (recherche
        # -- lexicale par mots-cles) qui complete la recherche semantique
        # -- dans le cadre d'une strategie de recherche hybride.
        collections_to_use = collection_types or COLLECTION_TYPES
        texts: List[str] = []

        for coll_type in collections_to_use:
            if coll_type not in self._collections:
                continue
            collection = self._collections[coll_type]
            if collection.count() == 0:
                # -- Ignore les collections vides
                continue
            try:
                # -- Recupere uniquement les textes (pas les metadonnees ni les embeddings)
                results = collection.get(include=["documents"])
                texts.extend(results.get("documents", []))
            except Exception as e:
                logger.debug(f"get_all_texts [{coll_type}]: {e}")

        return texts

    def get_all_documents(
        self,
        collection_types: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Récupère tous les Documents du store.

        Args:
            collection_types: Collections à inclure (None = toutes).

        Returns:
            Liste de tous les Documents.
        """
        # -- Recupere tous les Documents LangChain (texte + metadonnees) du store.
        # -- Contrairement a get_all_texts qui ne retourne que le texte brut,
        # -- cette methode retourne des objets Document complets, utiles pour
        # -- les traitements qui necessitent l'acces aux metadonnees (ex: filtrage,
        # -- affichage des sources dans l'interface).
        collections_to_use = collection_types or COLLECTION_TYPES
        docs: List[Document] = []

        for coll_type in collections_to_use:
            if coll_type not in self._collections:
                continue
            collection = self._collections[coll_type]
            if collection.count() == 0:
                # -- Ignore les collections vides
                continue
            try:
                # -- Recupere les textes et les metadonnees de chaque document
                results = collection.get(include=["documents", "metadatas"])
                for text, meta in zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                ):
                    # -- Reconstruit un objet Document LangChain a partir
                    # -- du texte et des metadonnees stockes dans ChromaDB
                    docs.append(Document(page_content=text, metadata=meta or {}))
            except Exception as e:
                logger.debug(f"get_all_documents [{coll_type}]: {e}")

        return docs
