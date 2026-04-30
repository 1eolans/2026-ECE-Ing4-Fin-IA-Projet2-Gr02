"""
Générateur de réponses financières avec OpenAI GPT-4o.
Citations vérifiables, format structuré, streaming support.
"""

# -- Import pour permettre l'utilisation d'annotations de type "modernes"
# -- (ex. list[str] au lieu de List[str]) même en Python < 3.10.
from __future__ import annotations

# -- Imports de la bibliothèque standard Python :
# --   sys       : manipulation du chemin d'import (sys.path)
# --   time      : mesure du temps de traitement des requêtes
# --   dataclass : décorateur pour créer des classes de données sans __init__ explicite
# --   field     : permet de définir des valeurs par défaut complexes (listes, dicts, etc.)
# --   Path      : manipulation orientée objet des chemins de fichiers
# --   List, Optional, Iterator : annotations de type pour la lisibilité du code
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator

# -- Document de LangChain : représente un morceau (chunk) de texte avec ses métadonnées
# -- (nom de fichier, numéro de page, date, etc.). C'est le format standard utilisé
# -- par tout le pipeline RAG pour véhiculer les résultats de la recherche vectorielle.
from langchain_core.documents import Document

# -- Loguru : bibliothèque de journalisation (logging) plus ergonomique que le module
# -- standard logging. Permet d'écrire des logs structurés avec niveaux (info, warning, error).
from loguru import logger

# -- On ajoute le répertoire racine du projet au sys.path pour pouvoir importer
# -- le module src.config (qui contient les paramètres de configuration : clé API, etc.)
# -- sans avoir à installer le package. Path(__file__).parent.parent.parent remonte
# -- de generator.py -> generation/ -> src/ -> racine du projet.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# -- Import de la configuration centralisée du projet (settings) qui contient :
# --   - OPENAI_API_KEY : clé d'API OpenAI pour l'appel au modèle GPT-4o
# --   - use_openai_embeddings : booléen indiquant si la clé est disponible
# --   - REQUEST_TIMEOUT : durée maximale d'une requête avant expiration
# --   - et d'autres paramètres partagés par tout le pipeline
from src.config import settings


# ─── Data Models ─────────────────────────────────────────────────────────────

# -- La classe Citation représente une référence vérifiable vers un morceau
# -- (chunk) de document source. Elle permet à l'utilisateur de retrouver
# -- l'origine exacte de chaque information citée dans la réponse générée.
# -- Chaque citation contient :
# --   - chunk_id       : identifiant unique du chunk (format "fichier::index")
# --   - source_file    : nom du fichier d'origine (ex. "rapport_annuel_2024.pdf")
# --   - page_number    : numéro de page dans le document original (si disponible)
# --   - date           : date associée au document (ex. "2024-Q3")
# --   - excerpt        : extrait de 100 caractères max du texte original du chunk
# --   - relevance_score: score de pertinence (0.0 à 1.0) issu de la recherche vectorielle
@dataclass
class Citation:
    """Citation vérifiable pointant vers le chunk source."""
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    date: Optional[str]
    excerpt: str        # 100 chars max du chunk original
    relevance_score: float

    # -- Méthode de formatage : convertit la citation en notation Markdown lisible.
    # -- Produit un texte du type : [Source: **rapport.pdf**, p.12, 2024-Q3]
    # -- Les champs optionnels (page, date) ne sont inclus que s'ils existent.
    def to_markdown(self) -> str:
        parts = [f"**{self.source_file}**"]
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.date:
            parts.append(self.date)
        return f"[Source: {', '.join(parts)}]"


# -- La classe FinancialAnswer représente la réponse structurée complète
# -- retournée par le générateur. Elle encapsule :
# --   - question          : la question posée par l'utilisateur
# --   - answer            : le texte de la réponse générée par le LLM (ou le fallback)
# --   - citations         : liste des citations vérifiables (objets Citation)
# --   - sub_queries       : sous-requêtes générées par la décomposition de la question
# --                         (utile pour le query expansion / multi-query retrieval)
# --   - confidence_score  : score de confiance estimé de 0.0 à 1.0 (voir _estimate_confidence)
# --   - processing_time   : durée totale de génération en secondes
# --   - tokens_used       : nombre total de tokens consommés par l'appel API OpenAI
# --   - context_docs_count: nombre de documents de contexte utilisés pour la réponse
@dataclass
class FinancialAnswer:
    """Réponse financière structurée avec citations et métriques."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    tokens_used: int = 0
    context_docs_count: int = 0

    # -- Sérialise l'objet FinancialAnswer en dictionnaire Python.
    # -- Utilisé pour renvoyer la réponse au format JSON via l'API ou l'interface.
    # -- Chaque citation est elle aussi convertie en dictionnaire imbriqué.
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "date": c.date,
                    "excerpt": c.excerpt,
                    "relevance_score": c.relevance_score,
                }
                for c in self.citations
            ],
            "sub_queries": self.sub_queries,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "context_docs_count": self.context_docs_count,
        }


# ─── Prompt Engineering ───────────────────────────────────────────────────────

# -- SYSTEM_PROMPT : prompt système envoyé au modèle GPT-4o à chaque requête.
# -- Il définit le RÔLE et les CONTRAINTES du LLM dans le cadre de ce système RAG.
# --
# -- Le prompt impose au modèle :
# --   1. De répondre UNIQUEMENT à partir du contexte documentaire fourni
# --      (pas de connaissances générales, pas d'hallucinations).
# --   2. D'ajouter des citations vérifiables après chaque chiffre ou fait
# --      au format [Source: {fichier}, p.{page}, {date}].
# --   3. D'admettre explicitement quand une information n'est pas disponible
# --      dans les documents fournis (transparence).
# --   4. De ne JAMAIS inventer de chiffres ni de faits (interdiction d'halluciner).
# --   5. De distinguer clairement données historiques et projections futures.
# --
# -- Le format de réponse attendu est structuré en 4 parties :
# --   - Synthèse courte (1-2 phrases)
# --   - Analyse détaillée avec données chiffrées et citations
# --   - Tableau comparatif en Markdown si pertinent
# --   - Mises en garde sur les limites des données
# --
# -- Le style demandé est professionnel, avec utilisation du Markdown et des
# -- unités financières françaises (Md$ pour milliards, M$ pour millions, %, bp).
SYSTEM_PROMPT = """Tu es un analyste financier expert spécialisé dans l'analyse de documents financiers (rapports annuels, rapports trimestriels, articles de presse financière, données de marché).

**RÈGLES ABSOLUES :**
1. Tu réponds UNIQUEMENT en te basant sur les documents fournis dans le contexte.
2. Chaque affirmation chiffrée DOIT être suivie d'une citation [Source: {filename}, p.{page}, {date}].
3. Si l'information n'est PAS dans le contexte, dis-le EXPLICITEMENT : "Cette information n'est pas disponible dans les documents fournis."
4. Ne jamais inventer de chiffres ou de faits.
5. Distingue clairement les données historiques des projections.

**FORMAT DE RÉPONSE :**
- Réponse directe en 1-2 phrases (synthèse)
- Analyse détaillée avec données chiffrées et citations
- Tableau comparatif si pertinent (markdown)
- Limites et mises en garde si les données sont incomplètes ou anciennes

**STYLE :**
- Professionnel et précis
- Utilise le markdown pour la mise en forme
- Chiffres toujours avec leur unité (Md$, %, bp, etc.)
- Distingue M$ (millions) de Md$ (milliards)"""


# -- Fonction _build_context : construit la chaîne de texte "contexte" qui sera
# -- injectée dans le prompt utilisateur envoyé au LLM.
# --
# -- Elle prend en entrée la liste de Documents récupérés par la recherche vectorielle
# -- et les assemble en un seul bloc de texte formaté, avec un en-tête descriptif
# -- pour chaque document (nom de fichier, numéro de page, date, type de document).
# --
# -- Paramètres :
# --   - documents         : liste de Documents LangChain (chunks avec métadonnées)
# --   - max_context_chars : limite maximale de caractères pour le contexte (défaut 8000).
# --     Cette limite évite de dépasser la fenêtre de contexte du modèle et contrôle
# --     les coûts en tokens.
# --
# -- Fonctionnement :
# --   - Parcourt les documents dans l'ordre (déjà triés par pertinence)
# --   - Pour chaque document, crée un en-tête formaté [Document N: fichier, p.X, date, type]
# --   - Ajoute le contenu textuel du chunk sous l'en-tête
# --   - Vérifie si l'ajout du prochain chunk dépasse la limite de caractères
# --   - Si oui et qu'il reste plus de 200 caractères, tronque le dernier chunk
# --     avec l'indicateur "[tronqué]"
# --   - Joint tous les morceaux avec un séparateur "---" pour lisibilité
# --
# -- Retourne : une chaîne de caractères prête à être insérée dans le prompt utilisateur.
def _build_context(
    documents: List[Document],
    max_context_chars: int = 8000,
) -> str:
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(documents):
        # -- Extraction des métadonnées du document (nom de fichier, page, date, type)
        meta = doc.metadata
        source = meta.get("filename", meta.get("source", "unknown"))
        page = meta.get("page_number", "")
        date = meta.get("date", "")
        doc_type = meta.get("document_type", "")

        # -- Construction de l'en-tête descriptif du chunk pour le LLM
        # -- Exemple : [Document 1: rapport_annuel.pdf, p.12, 2024-Q3, type=annual_report]
        header = f"[Document {i+1}: {source}"
        if page:
            header += f", p.{page}"
        if date:
            header += f", {date}"
        if doc_type:
            header += f", type={doc_type}"
        header += "]"

        # -- Assemblage : en-tête + saut de ligne + contenu du chunk
        chunk_text = f"{header}\n{doc.page_content}\n"

        # -- Vérification de la limite de caractères : si ajouter ce chunk
        # -- dépasse max_context_chars, on tronque ou on arrête.
        if total_chars + len(chunk_text) > max_context_chars:
            remaining = max_context_chars - total_chars
            # -- On ne tronque que s'il reste au moins 200 caractères utiles,
            # -- sinon ça n'apporte rien et on s'arrête simplement.
            if remaining > 200:
                chunk_text = chunk_text[:remaining] + "...[tronqué]"
                context_parts.append(chunk_text)
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    # -- Les chunks sont séparés par "---" pour que le LLM puisse
    # -- visuellement distinguer les différents documents sources.
    return "\n---\n".join(context_parts)


# -- Fonction _extract_citations : transforme la liste de Documents LangChain
# -- en une liste d'objets Citation exploitables par l'interface utilisateur.
# --
# -- Pour chaque document, elle :
# --   - Génère un identifiant unique (chunk_id) au format "source::index"
# --   - Récupère le nom du fichier source, le numéro de page et la date
# --   - Extrait les 100 premiers caractères du contenu comme aperçu (excerpt)
# --   - Associe le score de pertinence issu de la recherche vectorielle
# --     (par défaut 0.5 si aucun score n'est fourni)
# --
# -- Paramètres :
# --   - documents : liste de Documents récupérés par le retriever
# --   - scores    : liste optionnelle de scores de similarité (flottants entre 0 et 1)
# --
# -- Retourne : liste d'objets Citation prêts à être inclus dans FinancialAnswer.
def _extract_citations(
    documents: List[Document],
    scores: Optional[List[float]] = None,
) -> List[Citation]:
    citations = []
    for i, doc in enumerate(documents):
        meta = doc.metadata
        # -- Si un score de pertinence existe pour ce document, on l'utilise ;
        # -- sinon on attribue un score neutre de 0.5 par défaut.
        score = scores[i] if scores and i < len(scores) else 0.5

        citation = Citation(
            # -- Identifiant unique du chunk : "nom_fichier::index_du_chunk"
            chunk_id=f"{meta.get('source', 'unknown')}::{meta.get('chunk_index', i)}",
            source_file=meta.get("filename", meta.get("source", "unknown")),
            page_number=meta.get("page_number"),
            # -- La date peut provenir du champ "date" ou "time_period" selon le type de document
            date=meta.get("date") or meta.get("time_period"),
            # -- Extrait limité à 100 caractères, retours à la ligne remplacés par des espaces
            excerpt=doc.page_content[:100].replace("\n", " "),
            relevance_score=float(score),
        )
        citations.append(citation)

    return citations


# -- Fonction _estimate_confidence : calcule un score de confiance heuristique
# -- (entre 0.0 et 1.0) pour évaluer la fiabilité de la réponse générée.
# --
# -- Ce score N'EST PAS un calcul statistique rigoureux : c'est une estimation
# -- empirique basée sur plusieurs indicateurs indirects :
# --
# --   1. Présence de citations (+0.3) : si des documents sources ont été trouvés,
# --      la réponse a plus de chances d'être fondée.
# --
# --   2. Nombre de citations >= 3 (+0.2) : une réponse appuyée par au moins 3
# --      sources différentes est considérée comme plus fiable (corroboration).
# --
# --   3. Présence de données chiffrées (+0.2) : si la réponse contient des nombres
# --      avec décimales ou des unités financières (Md, M, %), cela indique une
# --      réponse factuelle et précise plutôt que vague.
# --
# --   4. Nombre de documents de contexte :
# --      - >= 3 documents (+0.2) : bonne couverture documentaire
# --      - >= 1 document (+0.1) : couverture minimale
# --
# --   5. Absence d'aveu d'ignorance (+0.1) : si la réponse ne contient pas de
# --      formule du type "n'est pas disponible" ou "pas dans le contexte",
# --      c'est que le LLM a trouvé de quoi répondre.
# --
# -- Le score final est plafonné à 1.0 (via min).
# --
# -- Paramètres :
# --   - answer   : texte de la réponse générée par le LLM
# --   - citations: liste des citations extraites
# --   - n_docs   : nombre de documents de contexte utilisés
# --
# -- Retourne : un flottant entre 0.0 et 1.0 représentant le niveau de confiance.
def _estimate_confidence(answer: str, citations: List[Citation], n_docs: int) -> float:
    import re
    score = 0.0
    # -- +0.3 si au moins une citation existe (la réponse s'appuie sur des sources)
    if citations:
        score += 0.3
    # -- +0.2 supplémentaire si au moins 3 citations (corroboration multi-sources)
    if len(citations) >= 3:
        score += 0.2
    # -- +0.2 si la réponse contient des données chiffrées (regex : nombres avec
    # -- décimales, ou nombres suivis d'unités Md, M, %)
    if re.search(r'\d+[\.,]\d+|\d+\s*(?:Md|M|%)', answer):
        score += 0.2
    # -- +0.2 ou +0.1 selon le nombre de documents de contexte disponibles
    if n_docs >= 3:
        score += 0.2
    elif n_docs >= 1:
        score += 0.1
    # -- +0.1 si le LLM n'a pas signalé une absence d'information
    # -- (ce qui signifie qu'il a pu répondre à la question)
    if "n'est pas disponible" not in answer.lower() and "pas dans le contexte" not in answer.lower():
        score += 0.1
    # -- Le score est plafonné à 1.0 pour rester dans l'intervalle [0, 1]
    return min(1.0, score)


# ─── Main Generator ───────────────────────────────────────────────────────────

# -- Constante définissant le modèle OpenAI utilisé pour la génération de réponses.
# -- GPT-4o est le modèle multimodal optimisé d'OpenAI, offrant un bon rapport
# -- qualité/coût pour l'analyse de textes financiers complexes.
# Modèle OpenAI utilisé pour la génération
OPENAI_GENERATION_MODEL = "gpt-4o"


# -- Classe principale du module : orchestre la génération de réponses financières.
# --
# -- Architecture :
# --   - Encapsule un client OpenAI (initialisé au constructeur)
# --   - Propose deux modes de génération : synchrone (generate) et streaming (generate_stream)
# --   - Intègre un mécanisme de fallback automatique : si la clé API est absente
# --     ou si l'appel échoue, retourne le contexte brut sans analyse LLM
# --
# -- Cycle de vie :
# --   1. __init__  -> appelle _initialize() pour créer le client OpenAI
# --   2. generate  -> construit le contexte, appelle l'API, extrait citations, estime confiance
# --   3. Retourne un objet FinancialAnswer complet
class FinancialAnswerGenerator:
    """
    Générateur de réponses financières via OpenAI GPT-4o.

    Utilise la même clé API (OPENAI_API_KEY) que le pipeline d'embeddings.
    Fallback automatique si la clé est absente : retourne le contexte brut.
    """

    # -- Constructeur : initialise les attributs internes puis tente de créer
    # -- le client OpenAI. Si l'initialisation échoue, le générateur reste
    # -- fonctionnel en mode dégradé (fallback).
    def __init__(self) -> None:
        # -- _client : instance du client OpenAI (None si non initialisé)
        self._client = None
        # -- _model : nom du modèle à utiliser pour les appels API
        self._model = OPENAI_GENERATION_MODEL
        # -- _initialized : drapeau indiquant si le client est prêt à être utilisé
        self._initialized = False
        self._initialize()

    # -- Méthode _initialize : tente de créer le client OpenAI avec la clé API
    # -- définie dans les settings.
    # --
    # -- Trois cas de figure :
    # --   1. La clé API n'est pas définie -> log warning, mode dégradé
    # --   2. Le package 'openai' n'est pas installé -> log error
    # --   3. Autre erreur (clé invalide, réseau, etc.) -> log error
    # --
    # -- Dans tous les cas d'échec, _initialized reste False et le générateur
    # -- basculera automatiquement en mode fallback lors des appels.
    def _initialize(self) -> None:
        """Initialise le client OpenAI."""
        # -- Vérifie si la clé API OpenAI est disponible dans la configuration
        if not settings.use_openai_embeddings:
            logger.warning(
                "OPENAI_API_KEY non défini. "
                "Le générateur fonctionnera en mode dégradé (contexte brut)."
            )
            return

        try:
            # -- Import et instanciation du client OpenAI avec la clé API
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self._initialized = True
            logger.info(f"Générateur OpenAI initialisé : {self._model}")
        except ImportError:
            logger.error("Package 'openai' non installé. pip install openai")
        except Exception as e:
            logger.error(f"Erreur initialisation OpenAI: {e}")

    # -- Méthode generate : point d'entrée principal pour générer une réponse
    # -- financière complète de manière synchrone (non streaming).
    # --
    # -- Étapes :
    # --   1. Démarre le chronomètre (mesure du temps de traitement)
    # --   2. Vérifie qu'il y a des documents de contexte ; sinon retourne un avertissement
    # --   3. Construit le texte de contexte à partir des documents (_build_context)
    # --   4. Extrait les citations à partir des documents et scores (_extract_citations)
    # --   5. Si le client OpenAI est initialisé, appelle l'API GPT-4o
    # --      Sinon, utilise le mode fallback (contexte brut sans analyse)
    # --   6. Calcule le score de confiance heuristique (_estimate_confidence)
    # --   7. Assemble et retourne l'objet FinancialAnswer complet
    # --
    # -- Paramètres :
    # --   - question          : la question posée par l'utilisateur
    # --   - context_documents : les documents récupérés par la recherche vectorielle
    # --   - context_scores    : scores de similarité associés (optionnel)
    # --   - sub_queries       : sous-requêtes issues de la décomposition (optionnel)
    # --   - max_tokens        : nombre maximum de tokens pour la réponse (défaut 2000)
    # --
    # -- Retourne : un objet FinancialAnswer contenant la réponse, les citations,
    # --            le score de confiance et les métriques de performance.
    def generate(
        self,
        question: str,
        context_documents: List[Document],
        context_scores: Optional[List[float]] = None,
        sub_queries: Optional[List[str]] = None,
        max_tokens: int = 2000,
    ) -> FinancialAnswer:
        """Génère une réponse financière à partir du contexte récupéré."""
        start_time = time.time()

        # -- Cas limite : aucun document trouvé -> retourne un message d'avertissement
        # -- avec un score de confiance de 0.0 (aucune base pour répondre)
        if not context_documents:
            return FinancialAnswer(
                question=question,
                answer="⚠️ Aucun document pertinent trouvé dans la base de connaissances. "
                       "Veuillez d'abord indexer des documents financiers.",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
            )

        # -- Construction du contexte textuel et extraction des citations
        context_text = _build_context(context_documents)
        citations = _extract_citations(context_documents, context_scores)

        # -- Branchement : appel API OpenAI si disponible, sinon mode dégradé
        if self._initialized and self._client:
            answer_text, tokens_used = self._generate_with_openai(
                question=question,
                context=context_text,
                max_tokens=max_tokens,
            )
        else:
            # -- Mode fallback : retourne le contexte brut sans analyse LLM
            answer_text = self._generate_fallback(question, context_documents)
            tokens_used = 0

        processing_time = time.time() - start_time
        # -- Estimation heuristique du score de confiance de la réponse
        confidence = _estimate_confidence(answer_text, citations, len(context_documents))

        # -- Assemblage de l'objet FinancialAnswer final avec toutes les métriques
        return FinancialAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            sub_queries=sub_queries or [],
            confidence_score=confidence,
            processing_time=processing_time,
            tokens_used=tokens_used,
            context_docs_count=len(context_documents),
        )

    # -- Méthode generate_stream : génère une réponse en mode streaming,
    # -- c'est-à-dire token par token (mot par mot), via un itérateur Python.
    # --
    # -- Avantage du streaming : l'utilisateur voit la réponse s'afficher
    # -- progressivement dans l'interface, sans attendre que tout soit généré.
    # -- C'est particulièrement utile pour les réponses longues.
    # --
    # -- Fonctionnement :
    # --   1. Vérifie la présence de documents et du client OpenAI
    # --   2. Construit le contexte et le message utilisateur
    # --   3. Appelle l'API OpenAI avec stream=True
    # --   4. Itère sur les chunks reçus et les renvoie un par un (yield)
    # --
    # -- Gestion d'erreurs détaillée :
    # --   - Timeout        : délai de génération dépassé
    # --   - Rate limit     : trop de requêtes en peu de temps
    # --   - Authentification: clé API invalide ou expirée
    # --   - Quota insuffisant: crédit OpenAI épuisé
    # --   - Autres erreurs : erreur générique avec type d'exception loggé
    # --
    # -- Paramètres :
    # --   - question          : la question de l'utilisateur
    # --   - context_documents : documents de contexte issus de la recherche
    # --   - context_scores    : scores de pertinence (optionnel, non utilisé ici)
    # --   - max_tokens        : limite de tokens pour la réponse
    # --
    # -- Retourne : un Iterator[str] qui produit des fragments de texte successifs.
    def generate_stream(
        self,
        question: str,
        context_documents: List[Document],
        context_scores: Optional[List[float]] = None,
        max_tokens: int = 2000,
    ) -> Iterator[str]:
        """Génère une réponse en streaming (token par token)."""
        # -- Si aucun document n'a été trouvé, on signale immédiatement l'absence
        if not context_documents:
            yield "⚠️ Aucun document pertinent trouvé. Veuillez indexer des documents d'abord."
            return

        # -- Si le client OpenAI n'est pas disponible, on bascule en mode fallback
        if not self._initialized or not self._client:
            yield self._generate_fallback(question, context_documents)
            return

        # -- Construction du contexte textuel à partir des documents récupérés
        context_text = _build_context(context_documents)

        # -- Assemblage du message utilisateur : contexte documentaire + question
        # -- Le format est conçu pour que le LLM comprenne clairement la séparation
        # -- entre le contexte (les sources) et la question à traiter.
        user_content = (
            f"**Contexte documentaire :**\n\n{context_text}\n\n"
            f"---\n\n**Question :** {question}"
        )

        try:
            # -- Appel à l'API OpenAI Chat Completions en mode streaming.
            # -- Le paramètre stream=True active la réception token par token.
            stream = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )
            # -- Itération sur les chunks reçus du flux streaming.
            # -- Chaque chunk contient un delta (fragment) de la réponse.
            # -- On renvoie chaque fragment non-nul à l'appelant via yield.
            for chunk in stream:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    yield delta_content

        except Exception as e:
            # -- Gestion fine des différents types d'erreurs possibles
            # -- pour fournir un message d'erreur adapté à l'utilisateur.
            error_type = type(e).__name__
            error_str = str(e).lower()

            # -- Erreur de timeout : la requête a pris trop de temps
            if "timeout" in error_str or "Timeout" in error_type:
                logger.error(f"Timeout streaming OpenAI après {settings.REQUEST_TIMEOUT}s")
                yield "\n\n⚠️ Délai dépassé : la génération a pris trop de temps. Réessayez."
            # -- Erreur de rate limiting : trop de requêtes simultanées
            elif "ratelimit" in error_str or "rate_limit" in error_str or "RateLimit" in error_type:
                logger.warning("Rate limit OpenAI atteint")
                yield "\n\n⚠️ Limite de requêtes atteinte. Attendez quelques secondes et réessayez."
            # -- Erreur d'authentification : clé API invalide, expirée ou absente
            elif "auth" in error_str or "Authentication" in error_type or "Unauthorized" in error_type:
                logger.error("Erreur d'authentification OpenAI — vérifiez OPENAI_API_KEY")
                yield "\n\n⚠️ Erreur d'authentification. Vérifiez votre OPENAI_API_KEY dans .env."
            # -- Erreur de quota : crédit OpenAI épuisé
            elif "insufficient_quota" in error_str:
                logger.error("Quota OpenAI insuffisant")
                yield "\n\n⚠️ Quota OpenAI insuffisant. Vérifiez votre facturation sur platform.openai.com."
            # -- Toute autre erreur non prévue : on log le type et le message
            else:
                logger.error(f"Erreur streaming OpenAI: {error_type}: {e}")
                yield f"\n\n⚠️ Erreur de génération ({error_type}). Voir les logs pour le détail."

    # -- Méthode _generate_with_openai : effectue l'appel synchrone (non-streaming)
    # -- à l'API OpenAI Chat Completions.
    # --
    # -- Construit les messages (système + utilisateur) et envoie la requête.
    # -- En cas de succès, retourne le texte de la réponse et le nombre de tokens utilisés.
    # -- En cas d'erreur, log l'exception et bascule vers le mode fallback.
    # --
    # -- Paramètres :
    # --   - question   : la question de l'utilisateur
    # --   - context    : le texte de contexte déjà formaté par _build_context
    # --   - max_tokens : nombre maximal de tokens pour la réponse
    # --
    # -- Retourne : un tuple (texte_réponse, nombre_tokens_utilisés)
    def _generate_with_openai(
        self,
        question: str,
        context: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Appel synchrone à l'API OpenAI Chat Completions."""
        try:
            # -- Envoi de la requête avec le prompt système et le message utilisateur
            # -- Le message utilisateur contient le contexte documentaire suivi de la question
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"**Contexte documentaire :**\n\n{context}\n\n"
                            f"---\n\n**Question :** {question}"
                        ),
                    },
                ],
            )
            # -- Extraction du texte de la réponse (premier choix du modèle)
            answer = response.choices[0].message.content or ""
            # -- Extraction du nombre total de tokens consommés (prompt + réponse)
            tokens = response.usage.total_tokens if response.usage else 0
            return answer, tokens

        except Exception as e:
            # -- En cas d'erreur API, on log et on bascule en mode dégradé
            logger.error(f"Erreur API OpenAI: {e}")
            return self._generate_fallback_from_context(question, context), 0

    # -- Méthode _generate_fallback : mode dégradé utilisé lorsque le client
    # -- OpenAI n'est pas disponible (clé absente ou erreur d'initialisation).
    # --
    # -- Elle construit le contexte avec une limite réduite (3000 caractères
    # -- au lieu de 8000) pour garder la réponse concise, puis délègue
    # -- à _generate_fallback_from_context pour le formatage final.
    def _generate_fallback(self, question: str, documents: List[Document]) -> str:
        context = _build_context(documents, max_context_chars=3000)
        return self._generate_fallback_from_context(question, context)

    # -- Méthode _generate_fallback_from_context : formate la réponse en mode dégradé.
    # --
    # -- Au lieu d'une analyse par le LLM, affiche directement :
    # --   - Un avertissement indiquant le mode dégradé
    # --   - La question posée par l'utilisateur
    # --   - Le contexte brut (tronqué à 2000 caractères)
    # --   - Une instruction pour configurer la clé API
    # --
    # -- C'est la solution de repli minimale : l'utilisateur peut au moins
    # -- lire les extraits de documents pertinents, même sans analyse LLM.
    def _generate_fallback_from_context(self, question: str, context: str) -> str:
        return (
            f"⚠️ **Mode dégradé** (OPENAI_API_KEY non configurée)\n\n"
            f"**Question :** {question}\n\n"
            f"**Contexte disponible :**\n\n{context[:2000]}\n\n"
            "*Configurez OPENAI_API_KEY dans votre fichier .env pour obtenir une réponse analysée.*"
        )
