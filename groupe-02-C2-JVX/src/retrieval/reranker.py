"""
CrossEncoder re-ranker pour améliorer la pertinence des résultats.
Modèle : cross-encoder/ms-marco-MiniLM-L-6-v2
Cache des scores pour éviter les re-calculs.
"""

# -- Imports standards et tiers
# -- __future__.annotations : permet d'utiliser les annotations de type modernes
# --   meme sous des versions anterieures de Python (evaluation differee des types).
from __future__ import annotations

# -- hashlib : utilise pour generer des cles de cache deterministes (hash MD5)
# --   a partir de la requete et du texte du document.
import hashlib

# -- sys : utilise pour modifier sys.path afin d'importer le module de configuration
# --   du projet depuis un chemin relatif.
import sys

# -- lru_cache : importe mais non utilise directement ici ; le cache est implemente
# --   manuellement via la classe ScoreCache pour un controle plus fin de l'eviction.
from functools import lru_cache

# -- Path : manipulation de chemins de fichiers, utilise pour construire le chemin
# --   vers le repertoire racine du projet.
from pathlib import Path

# -- Annotations de type pour ameliorer la lisibilite et la verification statique.
from typing import List, Tuple, Optional, Dict, Any

# -- Document : objet LangChain representant un morceau de texte (chunk) avec
# --   son contenu (page_content) et ses metadonnees (metadata).
from langchain_core.documents import Document

# -- loguru.logger : librairie de logging avancee, utilisee pour tracer les evenements
# --   (chargement du modele, erreurs, statistiques de re-ranking).
from loguru import logger

# -- Ajout du repertoire racine du projet au sys.path pour permettre l'import
# --   du module de configuration (src.config.settings).
# --   Path(__file__).parent.parent.parent remonte de 3 niveaux :
# --     reranker.py -> retrieval/ -> src/ -> racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Score Cache ─────────────────────────────────────────────────────────────
# -- STRATEGIE DE CACHE :
# --   Le cache stocke les scores de pertinence deja calcules par le CrossEncoder
# --   pour eviter de recalculer le score d'une meme paire (requete, document).
# --   Cela est particulierement utile lorsque les memes documents apparaissent
# --   dans plusieurs requetes similaires, ou lorsque l'utilisateur reformule
# --   une question deja posee.
# --
# --   Fonctionnement :
# --   1. Chaque paire (requete, texte) est transformee en une cle MD5 unique.
# --   2. Le score est stocke dans un dictionnaire en memoire.
# --   3. Quand le cache atteint sa taille maximale, la moitie la plus ancienne
# --      des entrees est supprimee (politique d'eviction simple).
# --   4. Cela permet un compromis entre vitesse (pas de recalcul) et memoire
# --      (taille bornee du cache).

class ScoreCache:
    """Cache simple LRU pour les scores CrossEncoder."""

    # -- Constructeur : initialise le dictionnaire de cache et la taille maximale.
    # -- Parametre maxsize : nombre maximal d'entrees dans le cache (defaut : 1000).
    # -- Le dictionnaire _cache associe une cle MD5 (str) a un score (float).
    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: Dict[str, float] = {}
        self._maxsize = maxsize

    # -- Genere une cle de cache unique a partir de la requete et du texte.
    # -- Les deux chaines sont tronquees a 200 caracteres puis concatenees avec "::"
    # -- pour former une chaine brute. Un hash MD5 est ensuite calcule sur cette chaine.
    # -- Le MD5 garantit une cle de taille fixe (32 caracteres hexadecimaux) quel que
    # -- soit la longueur des textes d'entree. Le .strip() supprime les espaces superflus
    # -- pour eviter que des variations mineures (espaces en debut/fin) produisent
    # -- des cles differentes pour le meme contenu.
    # -- Retourne : une chaine hexadecimale de 32 caracteres (le hash MD5).
    def _make_key(self, query: str, text: str) -> str:
        raw = f"{query.strip()[:200]}::{text.strip()[:200]}"
        return hashlib.md5(raw.encode()).hexdigest()

    # -- Recherche un score dans le cache pour une paire (requete, texte).
    # -- Retourne le score (float) s'il est present dans le cache, sinon None.
    # -- Cette methode est appelee avant chaque calcul de score pour verifier
    # -- si le resultat est deja disponible.
    def get(self, query: str, text: str) -> Optional[float]:
        key = self._make_key(query, text)
        return self._cache.get(key)

    # -- Ajoute un score dans le cache pour une paire (requete, texte).
    # -- Si le cache a atteint sa taille maximale (_maxsize), une eviction
    # -- est declenchee : la moitie la plus ancienne des entrees est supprimee.
    # -- L'ordre d'insertion dans un dict Python 3.7+ est garanti, donc
    # -- list(self._cache.keys()) retourne les cles dans l'ordre d'insertion,
    # -- et on supprime les premieres (les plus anciennes).
    # -- Ensuite, la nouvelle entree est ajoutee normalement.
    def set(self, query: str, text: str, score: float) -> None:
        if len(self._cache) >= self._maxsize:
            # Simple eviction: remove oldest half
            # -- On recupere toutes les cles dans l'ordre d'insertion
            keys = list(self._cache.keys())
            # -- On supprime la premiere moitie (les plus anciennes entrees)
            for k in keys[: self._maxsize // 2]:
                del self._cache[k]
        # -- On calcule la cle MD5 et on stocke le score
        key = self._make_key(query, text)
        self._cache[key] = score

    # -- Vide entierement le cache. Utile pour liberer la memoire
    # -- ou reinitialiser l'etat apres un changement de modele/donnees.
    def clear(self) -> None:
        self._cache.clear()

    # -- Propriete (accessible comme un attribut) qui retourne le nombre
    # -- d'entrees actuellement stockees dans le cache.
    @property
    def size(self) -> int:
        return len(self._cache)


# ─── CrossEncoder Re-ranker ───────────────────────────────────────────────────
# -- PRINCIPE DU RE-RANKING :
# --   Le re-ranking est une etape de post-traitement dans un pipeline RAG
# --   (Retrieval-Augmented Generation). Apres une premiere recherche par
# --   similarite vectorielle (embedding cosine similarity), les documents
# --   candidats sont re-evalues par un modele CrossEncoder plus precis.
# --
# --   Difference entre Bi-Encoder et CrossEncoder :
# --   - Bi-Encoder : encode la requete et le document separement, puis compare
# --     les vecteurs. Rapide mais moins precis.
# --   - CrossEncoder : prend la paire (requete, document) en entree simultanement
# --     et produit un score de pertinence direct. Plus lent mais plus precis car
# --     il capture les interactions fines entre les mots de la requete et du document.
# --
# --   Le modele utilise (ms-marco-MiniLM-L-6-v2) est un modele leger entraine sur
# --   le dataset MS MARCO, specialise dans le classement de passages par pertinence.
# --
# --   ETAPES DU PROCESSUS :
# --   1. Recevoir les documents candidats (deja filtres par la recherche vectorielle).
# --   2. Pour chaque document, verifier si le score est deja en cache.
# --   3. Pour les documents sans score en cache, preparer les paires (requete, texte).
# --   4. Envoyer les paires au CrossEncoder pour obtenir les scores de pertinence.
# --   5. Stocker les nouveaux scores dans le cache.
# --   6. Trier tous les documents par score decroissant.
# --   7. Retourner les top_k meilleurs documents.

class CrossEncoderReRanker:
    """
    Re-ranker basé sur CrossEncoder pour améliorer la pertinence.

    Modèle par défaut : cross-encoder/ms-marco-MiniLM-L-6-v2
    Input : query + liste de chunks candidats
    Output : chunks re-scorés et re-triés, top-k gardés
    Cache des scores pour éviter les re-calculs.
    """

    # -- Constructeur : configure le re-ranker avec les parametres du modele et du cache.
    # -- Le chargement du modele est differe (lazy loading) : le modele n'est charge
    # -- en memoire que lors du premier appel a rerank(), pas a l'instanciation.
    # -- Cela permet un demarrage plus rapide de l'application.
    # --
    # -- Parametres :
    # --   model_name  : nom du modele HuggingFace CrossEncoder a utiliser.
    # --   top_k       : nombre maximum de documents a retourner apres re-ranking.
    # --   max_length  : longueur maximale en tokens de l'entree (requete + document).
    # --                 Au-dela de cette limite, le texte est tronque par le modele.
    # --   cache_size  : taille maximale du cache de scores (nombre d'entrees).
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        max_length: int = 512,
        cache_size: int = 1000,
    ) -> None:
        """
        Args:
            model_name: Modèle CrossEncoder HuggingFace.
            top_k: Nombre de résultats à retourner après re-ranking.
            max_length: Longueur maximale d'input (query + doc).
            cache_size: Taille du cache de scores.
        """
        self._model_name = model_name
        self._top_k = top_k
        self._max_length = max_length
        # -- Initialisation du cache de scores avec la taille specifiee
        self._cache = ScoreCache(maxsize=cache_size)
        # -- Le modele est initialise a None ; il sera charge lors du premier appel
        self._model = None
        # -- Indicateur pour savoir si on a deja tente de charger le modele
        # -- (evite de retenter indefiniment en cas d'erreur)
        self._model_loaded = False

        logger.info(f"CrossEncoderReRanker initialisé (lazy loading): {model_name}")

    # -- Charge le modele CrossEncoder en memoire (lazy loading).
    # -- Cette methode est appelee automatiquement lors du premier appel a rerank().
    # -- Si le modele est deja charge (_model_loaded == True), on ne fait rien.
    # --
    # -- Gestion des erreurs :
    # -- - ImportError : la librairie sentence_transformers n'est pas installee.
    # --   On met _model a None et _model_loaded a True pour ne pas retenter.
    # -- - Exception generique : toute autre erreur lors du chargement.
    # --   Meme comportement : on marque le modele comme "tente" pour eviter
    # --   une boucle infinie de tentatives de chargement.
    # -- Dans les deux cas d'erreur, les methodes de rerank retourneront
    # -- les documents dans leur ordre original (fallback gracieux).
    def _load_model(self) -> None:
        """Charge le modèle CrossEncoder (lazy loading)."""
        if self._model_loaded:
            return

        try:
            # -- Import local de sentence_transformers pour ne pas forcer
            # -- la dependance au niveau du module entier
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
            )
            self._model_loaded = True
            logger.info(f"CrossEncoder chargé : {self._model_name}")
        except ImportError:
            logger.error("sentence_transformers non installé")
            self._model_loaded = True  # Prevent retry
            self._model = None
        except Exception as e:
            logger.error(f"Erreur chargement CrossEncoder: {e}")
            self._model_loaded = True
            self._model = None

    # -- Methode principale de re-ranking.
    # -- Prend une requete et une liste de tuples (Document, score_initial) en entree.
    # -- Le score_initial provient generalement de la recherche vectorielle (cosine similarity).
    # -- Le CrossEncoder recalcule un score de pertinence plus precis pour chaque document.
    # --
    # -- Deroulement detaille :
    # --   1. Verification que la liste de documents n'est pas vide.
    # --   2. Determination du nombre de resultats a retourner (top_k).
    # --   3. Chargement du modele si necessaire (lazy loading).
    # --   4. Si le modele n'est pas disponible, retour des documents originaux (fallback).
    # --   5. Pour chaque document :
    # --      a. Le texte est tronque a 800 caracteres pour des raisons de performance.
    # --      b. On verifie si le score est deja en cache.
    # --      c. Si oui, on recupere le score du cache (cache hit).
    # --      d. Si non, on ajoute la paire a la liste des paires a scorer (cache miss).
    # --   6. Les paires non cachees sont envoyees au CrossEncoder en un seul batch
    # --      (appel model.predict) pour maximiser l'efficacite GPU/CPU.
    # --   7. Les nouveaux scores sont stockes dans le cache pour les futures requetes.
    # --   8. Tous les scores (caches + nouveaux) sont combines.
    # --   9. Les documents sont tries par score decroissant.
    # --  10. Les top_k meilleurs documents sont retournes.
    # --
    # -- Retourne : une liste de tuples (Document, score_crossencoder) triee par
    # --   pertinence decroissante, limitee a top_k elements.
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-classe les documents selon leur pertinence par rapport à la requête.

        Args:
            query: Requête utilisateur.
            documents: Liste de (Document, score) à re-classer.
            top_k: Nombre de résultats à retourner (utilise self._top_k si None).

        Returns:
            Liste re-classée de (Document, cross_encoder_score).
        """
        # -- Si la liste est vide, on retourne une liste vide immediatement
        if not documents:
            return []

        # -- Utilise le top_k passe en parametre, ou la valeur par defaut de l'instance
        actual_top_k = top_k or self._top_k

        # Load model if needed
        # -- Declenchement du chargement du modele si ce n'est pas deja fait
        self._load_model()

        # -- Si le modele n'a pas pu etre charge (erreur d'import ou autre),
        # -- on retourne les documents dans leur ordre original, tronques a top_k.
        # -- C'est un mecanisme de fallback pour ne pas bloquer le pipeline.
        if self._model is None:
            logger.warning("CrossEncoder indisponible, retour des résultats originaux")
            return documents[:actual_top_k]

        # Prepare pairs for scoring
        # -- pairs_to_score : liste des paires (index, texte) qui ne sont PAS dans le cache
        # --   et qui doivent etre evaluees par le CrossEncoder.
        # -- cached_scores : dictionnaire {index -> score} pour les paires deja en cache.
        pairs_to_score: List[Tuple[int, str]] = []  # (index, text)
        cached_scores: Dict[int, float] = {}

        # -- Parcours de chaque document avec son index
        for idx, (doc, _) in enumerate(documents):
            # -- Troncature du contenu a 800 caracteres pour limiter le temps de calcul.
            # -- Le CrossEncoder a deja une limite max_length en tokens, mais cette
            # -- troncature pre-emptive en caracteres reduit aussi le cout de tokenisation.
            text = doc.page_content[:800]  # Truncate for efficiency
            # -- Verification dans le cache si ce score a deja ete calcule
            cached_score = self._cache.get(query, text)

            if cached_score is not None:
                # -- Cache hit : on reutilise le score deja calcule
                cached_scores[idx] = cached_score
            else:
                # -- Cache miss : on ajoute cette paire a la liste a scorer
                pairs_to_score.append((idx, text))

        # Compute uncached scores
        # -- Calcul des scores pour les paires qui n'etaient pas dans le cache
        new_scores: Dict[int, float] = {}
        if pairs_to_score:
            try:
                # -- Construction du batch de paires [requete, texte] pour le CrossEncoder.
                # -- Le modele attend une liste de listes [query, passage].
                batch_pairs = [[query, text] for _, text in pairs_to_score]
                # -- Appel au modele CrossEncoder qui retourne un score de pertinence
                # -- pour chaque paire. Les scores ne sont pas normalises entre 0 et 1 ;
                # -- ils sont des logits bruts. Un score plus eleve = plus pertinent.
                raw_scores = self._model.predict(batch_pairs)

                # -- Association de chaque score a son index et mise en cache
                for (idx, text), score in zip(pairs_to_score, raw_scores):
                    # -- Conversion en float Python natif (les scores peuvent etre
                    # -- des numpy.float32 retournes par le modele)
                    score_val = float(score)
                    new_scores[idx] = score_val
                    # -- Stockage dans le cache pour les futures requetes
                    self._cache.set(query, text, score_val)

            except Exception as e:
                # -- En cas d'erreur lors de la prediction (ex: memoire insuffisante),
                # -- on retourne les documents dans leur ordre original comme fallback.
                logger.error(f"CrossEncoder predict erreur: {e}")
                return documents[:actual_top_k]

        # Combine all scores
        # -- Fusion des scores caches et des scores nouvellement calcules
        # -- dans un seul dictionnaire {index -> score}.
        all_scores = {**cached_scores, **new_scores}

        # Re-rank
        # -- Construction de la liste finale avec les nouveaux scores.
        # -- Si un document n'a pas de score dans all_scores (cas theoriquement
        # -- impossible sauf erreur), on utilise son score original comme fallback.
        reranked = []
        for idx, (doc, original_score) in enumerate(documents):
            ce_score = all_scores.get(idx, original_score)
            reranked.append((doc, ce_score))

        # -- Tri des documents par score de pertinence decroissant.
        # -- Le document avec le score CrossEncoder le plus eleve arrive en premier.
        reranked.sort(key=lambda x: x[1], reverse=True)

        # -- Log des statistiques de re-ranking pour le debugging :
        # --   nombre de documents en entree, nombre de resultats retournes,
        # --   et ratio de cache hits (combien de scores venaient du cache).
        logger.debug(
            f"Re-ranking: {len(documents)} → {actual_top_k} docs "
            f"(cache hits: {len(cached_scores)}/{len(documents)})"
        )

        # -- Retour des top_k meilleurs documents apres re-ranking
        return reranked[:actual_top_k]

    # -- Methode utilitaire pour re-classer une liste de Documents sans scores initiaux.
    # -- Utile quand on a une liste de Documents bruts (sans score de similarite)
    # -- et qu'on veut simplement les classer par pertinence par rapport a une requete.
    # --
    # -- Fonctionnement :
    # --   1. On attribue un score initial de 1.0 a chaque document (score fictif).
    # --   2. On appelle la methode rerank() qui remplacera ces scores par les
    # --      scores du CrossEncoder.
    # --   3. On extrait uniquement les Documents (sans les scores) du resultat.
    # --
    # -- Retourne : une liste de Documents tries par pertinence decroissante.
    def rerank_documents_only(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Re-classe une liste de Documents (sans scores initiaux).

        Args:
            query: Requête utilisateur.
            documents: Liste de Documents à re-classer.
            top_k: Nombre de résultats à retourner.

        Returns:
            Liste re-classée de Documents.
        """
        # -- Attribution d'un score fictif de 1.0 a chaque document
        pairs = [(doc, 1.0) for doc in documents]
        # -- Appel a la methode principale de re-ranking
        reranked_pairs = self.rerank(query, pairs, top_k=top_k)
        # -- Extraction des Documents uniquement (sans les scores)
        return [doc for doc, _ in reranked_pairs]

    # -- Retourne un dictionnaire contenant les statistiques actuelles du re-ranker.
    # -- Utile pour le monitoring et le debugging :
    # --   - cache_size   : nombre d'entrees actuellement dans le cache
    # --   - model_loaded : indique si le modele a ete charge (ou si le chargement a ete tente)
    # --   - model_name   : nom du modele CrossEncoder utilise
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": self._cache.size,
            "model_loaded": self._model_loaded,
            "model_name": self._model_name,
        }

    # -- Vide entierement le cache des scores CrossEncoder.
    # -- Utile lorsque les donnees sous-jacentes changent (ex: mise a jour
    # -- de la base de connaissances) et que les anciens scores ne sont plus valides.
    def clear_cache(self) -> None:
        """Vide le cache des scores."""
        self._cache.clear()
        logger.info("Cache CrossEncoder vidé")
