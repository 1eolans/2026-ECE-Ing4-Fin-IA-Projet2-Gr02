"""
Chunker intelligent pour documents financiers.
Plusieurs stratégies : sémantique, table-aware, sentence, hybrid.

# -- Ce module fournit un systeme de decoupage (chunking) adaptatif specialise
# -- pour les documents financiers. Il permet de segmenter de longs documents
# -- en morceaux plus petits (chunks) exploitables par un modele de langage (LLM)
# -- ou un systeme de recherche vectorielle (RAG).
# --
# -- Quatre strategies de decoupage sont proposees :
# --   1. SEMANTIC  : decoupe recursive basee sur des separateurs hierarchiques
# --   2. TABLE_AWARE : preserve l'integrite des tableaux markdown
# --   3. SENTENCE  : decoupe au niveau des frontieres de phrases
# --   4. HYBRID    : detection automatique du type de contenu pour choisir
# --                  la strategie la plus adaptee (table ou semantique)
# --
# -- Chaque chunk produit est enrichi de metadonnees financieres :
# --   - presence de tableaux, donnees chiffrees, periodes temporelles,
# --     entites financieres detectees, longueur du chunk, etc.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional, Dict, Any

# -- Import du type Document de LangChain, qui encapsule le contenu textuel
# -- et les metadonnees associees a chaque morceau de texte.
from langchain_core.documents import Document

# -- RecursiveCharacterTextSplitter est le splitter principal de LangChain.
# -- Il tente de decouper le texte en utilisant une hierarchie de separateurs
# -- (paragraphes, lignes, phrases, mots) afin de produire des chunks
# -- dont la taille est proche de la cible specifiee.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -- Loguru est une bibliotheque de logging avancee pour Python.
# -- Elle est utilisee ici pour tracer les erreurs et les statistiques de chunking.
from loguru import logger


# -- Classe d'enumeration qui definit les quatre strategies de decoupage
# -- disponibles dans ce module. L'utilisateur peut choisir l'une d'elles
# -- lors de l'instanciation du chunker, ou laisser HYBRID par defaut
# -- pour une detection automatique.
class ChunkingStrategy(Enum):
    """Stratégies de chunking disponibles."""
    SEMANTIC = "semantic"        # RecursiveCharacterTextSplitter avec overlap
    TABLE_AWARE = "table_aware"  # Préserve les tableaux entiers
    SENTENCE = "sentence"        # Découpage par phrases
    HYBRID = "hybrid"            # Auto-détection selon le contenu


# ─── Helpers ─────────────────────────────────────────────────────────────────
# -- Cette section regroupe les fonctions utilitaires (helpers) utilisees
# -- pour analyser le contenu des chunks et en extraire des metadonnees
# -- pertinentes pour le domaine financier.

# -- Liste de reference des entites financieres connues.
# -- Elle contient des noms d'entreprises majeures (Apple, Microsoft, etc.),
# -- des indicateurs financiers cles (EBITDA, EPS, marge brute, etc.),
# -- des periodes temporelles (Q1, FY2023, T1, etc.)
# -- et des indices boursiers (S&P 500, CAC 40, etc.).
# -- Cette liste est utilisee par _extract_financial_entities() pour tagger
# -- chaque chunk avec les entites qu'il mentionne.
FINANCIAL_ENTITIES = [
    # Companies
    "Apple", "Microsoft", "NVIDIA", "Google", "Alphabet", "Amazon", "Meta",
    "Tesla", "JPMorgan", "Goldman Sachs", "Morgan Stanley",
    # Metrics
    "chiffre d'affaires", "bénéfice net", "marge brute", "EBITDA", "EPS",
    "revenue", "net income", "gross margin", "cash flow", "dividende",
    "résultat opérationnel", "résultat net", "capitalisation",
    # Time periods
    "Q1", "Q2", "Q3", "Q4", "FY2023", "FY2024", "T1", "T2", "T3", "T4",
    # Indices
    "S&P 500", "Nasdaq", "CAC 40", "Dow Jones", "MSCI",
]

# -- Expression reguliere pour detecter les periodes temporelles dans le texte.
# -- Reconnait les formats suivants :
# --   - Q1 2024, Q2 2023, etc. (trimestres anglais)
# --   - FY2024 (annee fiscale complete)
# --   - T1 2024, T2 2023, etc. (trimestres francais)
# --   - 2024 (annee seule)
# -- Le \b assure la correspondance sur des frontieres de mots pour eviter
# -- les faux positifs au milieu d'un nombre plus long.
TIME_PERIOD_PATTERN = re.compile(
    r'\b(Q[1-4]\s*20\d{2}|FY20\d{2}|T[1-4]\s*20\d{2}|20\d{2})\b'
)

# -- Expression reguliere pour detecter les valeurs numeriques significatives.
# -- Reconnait les nombres suivis d'unites optionnelles :
# --   - % (pourcentage), Md (milliards en francais), M (millions),
# --     B (billions en anglais), K (milliers),
# --     million, billion, trillion (unites en toutes lettres)
# -- Les separateurs de milliers (virgules) et decimaux (points) sont autorises.
NUMBER_PATTERN = re.compile(r'\b\d+[\d,\.]*\s*(%|Md|M|B|K|million|billion|trillion)?\b')


def _contains_table(text: str) -> bool:
    """Détecte si le texte contient un tableau markdown."""
    # -- Un tableau markdown se caracterise par la presence conjointe de :
    # --   - le caractere pipe "|" (delimiteur de colonnes)
    # --   - la sequence "---" (ligne de separation entre l'en-tete et le corps)
    # -- Cette heuristique simple est rapide et suffisante pour la majorite
    # -- des tableaux generes par des parsers de PDF ou rediges manuellement.
    # -- Retourne True si le texte contient un tableau, False sinon.
    return "|" in text and "---" in text


def _contains_numbers(text: str) -> bool:
    """Détecte si le texte contient des données chiffrées significatives."""
    # -- Recherche toutes les occurrences de valeurs numeriques dans le texte
    # -- grace a la regex NUMBER_PATTERN.
    matches = NUMBER_PATTERN.findall(text)
    # -- On considere que le texte contient des donnees chiffrees "significatives"
    # -- s'il y a au moins 3 nombres. Ce seuil evite de tagger des chunks
    # -- qui ne contiennent qu'une ou deux valeurs isolees et peu informatives.
    # -- Retourne True si >= 3 nombres trouves, False sinon.
    return len(matches) >= 3


def _extract_financial_entities(text: str) -> List[str]:
    """Extrait les entités financières mentionnées dans le texte."""
    # -- Parcourt la liste de reference FINANCIAL_ENTITIES et verifie
    # -- la presence de chaque entite dans le texte (comparaison insensible
    # -- a la casse grace a .lower()).
    found = []
    text_lower = text.lower()
    for entity in FINANCIAL_ENTITIES:
        if entity.lower() in text_lower:
            found.append(entity)
    # -- Limite le resultat aux 10 premieres entites trouvees pour eviter
    # -- de surcharger les metadonnees avec trop d'entites.
    # -- Retourne une liste de chaines contenant les noms des entites detectees.
    return found[:10]  # Limit to top 10


def _extract_time_period(text: str) -> str:
    """Extrait la période temporelle principale du chunk."""
    # -- Recherche la premiere occurrence d'une periode temporelle dans le texte
    # -- en utilisant la regex TIME_PERIOD_PATTERN.
    match = TIME_PERIOD_PATTERN.search(text)
    # -- Retourne la periode trouvee sous forme de chaine (ex: "Q1 2024", "FY2023")
    # -- ou une chaine vide si aucune periode n'est detectee.
    return match.group(1) if match else ""


def _build_chunk_metadata(
    chunk_text: str,
    base_metadata: Dict[str, Any],
    chunk_index: int,
    strategy: str,
) -> Dict[str, Any]:
    """Construit les métadonnées enrichies d'un chunk."""
    # -- Cette fonction assemble les metadonnees finales pour un chunk donne.
    # -- Elle part des metadonnees de base du document parent (source, titre, etc.)
    # -- et les enrichit avec des informations specifiques au chunk.

    # -- Copie des metadonnees de base pour ne pas modifier le dictionnaire original
    meta = dict(base_metadata)

    # -- Ajout des metadonnees specifiques au chunk :
    # --   chunk_index        : position du chunk dans la sequence (0, 1, 2, ...)
    # --   chunk_strategy     : nom de la strategie utilisee pour ce chunk
    # --   contains_table     : booleen indiquant la presence d'un tableau markdown
    # --   contains_numbers   : booleen indiquant la presence de donnees chiffrees
    # --   time_period        : periode temporelle principale detectee (ou "")
    # --   financial_entities : liste des entites financieres mentionnees
    # --   chunk_length       : longueur du chunk en nombre de caracteres
    meta.update({
        "chunk_index": chunk_index,
        "chunk_strategy": strategy,
        "contains_table": _contains_table(chunk_text),
        "contains_numbers": _contains_numbers(chunk_text),
        "time_period": _extract_time_period(chunk_text),
        "financial_entities": _extract_financial_entities(chunk_text),
        "chunk_length": len(chunk_text),
    })
    # -- Retourne le dictionnaire complet de metadonnees enrichies
    return meta


# ─── Main Chunker ─────────────────────────────────────────────────────────────

# -- Classe principale du module. Elle orchestre le decoupage des documents
# -- en chunks selon la strategie choisie. Elle est concue pour etre instanciee
# -- une seule fois puis reutilisee pour traiter plusieurs documents.
# -- Le chunker enrichit automatiquement chaque chunk avec des metadonnees
# -- financieres (entites, periodes, presence de tableaux/chiffres).
class IntelligentFinancialChunker:
    """
    Chunker adaptatif pour documents financiers.

    Stratégies :
    - SEMANTIC : RecursiveCharacterTextSplitter (texte narratif, 512 tokens)
    - TABLE_AWARE : Préserve les tableaux entiers
    - SENTENCE : Découpage par phrases
    - HYBRID : Auto-détection (table → TABLE_AWARE, texte → SEMANTIC)

    Tagging automatique : {contains_table, contains_numbers,
                            time_period, financial_entities}
    Filtrage des chunks trop courts ou non informatifs.
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        min_chunk_length: int = 50,
    ) -> None:
        """
        Args:
            strategy: Stratégie de chunking.
            chunk_size: Taille cible des chunks (en caractères ~= tokens).
            chunk_overlap: Overlap entre chunks consécutifs.
            min_chunk_length: Longueur minimale pour qu'un chunk soit gardé.
        """
        # -- Initialisation du constructeur avec les parametres de configuration.
        # -- strategy : la strategie de decoupage a utiliser (HYBRID par defaut)
        self.strategy = strategy

        # -- chunk_size : taille cible en caracteres pour chaque chunk.
        # -- 512 est un bon compromis entre granularite et contexte suffisant
        # -- pour un modele de langage.
        self.chunk_size = chunk_size

        # -- chunk_overlap : nombre de caracteres de chevauchement entre chunks
        # -- consecutifs. Ce chevauchement permet de ne pas couper une idee
        # -- en plein milieu et assure une continuite contextuelle entre chunks.
        self.chunk_overlap = chunk_overlap

        # -- min_chunk_length : longueur minimale en caracteres pour conserver
        # -- un chunk. Les chunks trop courts (< 50 car.) sont consideres
        # -- comme non informatifs et sont filtres.
        self.min_chunk_length = min_chunk_length

        # -- Splitter semantique standard : utilise RecursiveCharacterTextSplitter
        # -- avec des separateurs hierarchiques classiques :
        # --   "\n\n" (double saut de ligne = paragraphe)
        # --   "\n"   (saut de ligne simple)
        # --   ". "   (fin de phrase)
        # --   " "    (espace = entre mots)
        # --   ""     (dernier recours : coupe caractere par caractere)
        # -- Le splitter essaie le premier separateur, puis descend dans la
        # -- hierarchie si les chunks sont encore trop grands.
        # -- keep_separator=True conserve le separateur dans le texte du chunk.
        self._semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

        # -- Splitter financier : variante du splitter semantique avec un
        # -- separateur supplementaire "\n\n\n" (triple saut de ligne).
        # -- Ce separateur permet de mieux respecter les sections de documents
        # -- financiers qui utilisent souvent de grands espacements entre
        # -- les blocs (resume, tableaux, notes de bas de page, etc.).
        # -- Ce splitter est utilise par les strategies SEMANTIC et TABLE_AWARE.
        self._financial_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe une liste de documents en chunks.

        Args:
            documents: Documents LangChain à découper.

        Returns:
            Liste de chunks (Documents) avec métadonnées enrichies.
        """
        # -- Methode principale d'entree. Elle prend une liste de documents
        # -- LangChain et retourne une liste de chunks (egalement des Documents).
        # -- Chaque document est traite individuellement par _chunk_single().

        # -- Accumulateur pour tous les chunks produits a partir de tous les documents
        all_chunks: List[Document] = []

        for doc in documents:
            try:
                # -- Decoupe le document courant selon la strategie choisie
                chunks = self._chunk_single(doc)
                # -- Ajoute les chunks obtenus a la liste globale
                all_chunks.extend(chunks)
            except Exception as e:
                # -- En cas d'erreur sur un document, on logue l'erreur
                # -- et on passe au document suivant sans interrompre le processus.
                # -- Cela garantit la robustesse du pipeline meme si un document
                # -- est malformate ou pose probleme.
                logger.error(f"Erreur chunking doc {doc.metadata.get('source', '?')}: {e}")
                continue

        # -- Log du bilan : nombre de documents en entree -> nombre de chunks en sortie
        logger.info(f"Chunking : {len(documents)} docs → {len(all_chunks)} chunks")
        return all_chunks

    def _chunk_single(self, doc: Document) -> List[Document]:
        """Découpe un seul document selon la stratégie choisie."""
        # -- Methode interne qui traite un document individuel.
        # -- Elle determine la strategie effective a appliquer et
        # -- delegue le decoupage a la methode specialisee correspondante.

        # -- Extraction du contenu textuel et des metadonnees du document
        text = doc.page_content
        meta = doc.metadata

        # -- Determination de la strategie effective a utiliser.
        # -- Si la strategie globale est HYBRID, on analyse le contenu
        # -- pour choisir automatiquement entre TABLE_AWARE et SEMANTIC.
        effective_strategy = self.strategy
        if self.strategy == ChunkingStrategy.HYBRID:
            # -- Si le texte contient un tableau markdown, on utilise TABLE_AWARE
            # -- pour preserver l'integrite du tableau dans un seul chunk.
            if _contains_table(text):
                effective_strategy = ChunkingStrategy.TABLE_AWARE
            else:
                # -- Sinon, on utilise SEMANTIC qui est la strategie generique
                # -- la plus performante pour du texte narratif.
                effective_strategy = ChunkingStrategy.SEMANTIC

        # -- Dispatch vers la methode de chunking appropriee
        if effective_strategy == ChunkingStrategy.TABLE_AWARE:
            # -- Decoupage preservant les tableaux markdown
            return self._chunk_table_aware(text, meta)
        elif effective_strategy == ChunkingStrategy.SENTENCE:
            # -- Decoupage par phrases completes
            return self._chunk_by_sentence(text, meta)
        else:  # SEMANTIC
            # -- Decoupage semantique recursif (strategie par defaut)
            return self._chunk_semantic(text, meta)

    def _chunk_semantic(self, text: str, base_meta: Dict[str, Any]) -> List[Document]:
        """Chunking sémantique standard."""
        # -- Strategie SEMANTIC : utilise le _financial_splitter pour decouper
        # -- le texte de maniere recursive en respectant les separateurs
        # -- hierarchiques (paragraphes > lignes > phrases > mots).
        # -- C'est la strategie la plus adaptee pour le texte narratif
        # -- (commentaires d'analystes, descriptions, notes, etc.).

        # -- Decoupe le texte brut en une liste de chaines de caracteres
        raw_chunks = self._financial_splitter.split_text(text)

        # -- Construit les objets Document enrichis a partir des chaines brutes
        # -- et retourne la liste finale de chunks.
        return self._build_docs(raw_chunks, base_meta, "semantic")

    def _chunk_table_aware(self, text: str, base_meta: Dict[str, Any]) -> List[Document]:
        """
        Chunking qui préserve les tableaux markdown entiers.
        Découpe le texte en sections (hors tableaux) + tableaux complets.
        """
        # -- Strategie TABLE_AWARE : cette methode garantit que les tableaux
        # -- markdown ne sont jamais coupes en plusieurs chunks.
        # -- Le texte est segmente en 3 types de morceaux :
        # --   1. Le texte avant chaque tableau (decoupe semantiquement)
        # --   2. Chaque tableau entier (conserve tel quel dans un seul chunk)
        # --   3. Le texte apres le dernier tableau (decoupe semantiquement)

        # -- Liste accumulatrice pour tous les morceaux de texte
        parts: List[str] = []

        # -- Position courante dans le texte (curseur de parcours)
        current_pos = 0

        # -- Expression reguliere pour detecter les blocs de tableaux markdown.
        # -- Un tableau markdown est compose de :
        # --   - Lignes de donnees : |col1|col2|col3|
        # --   - Ligne separatrice : |---|---|---|
        # -- La regex capture l'ensemble du bloc (en-tete + separateur + lignes).
        table_pattern = re.compile(
            r'((?:\|[^\n]+\|\n)+(?:\|[-| :]+\|\n)(?:\|[^\n]+\|\n)*)',
            re.MULTILINE,
        )

        # -- Parcourt toutes les occurrences de tableaux dans le texte
        for match in table_pattern.finditer(text):
            # -- Extrait le texte situe AVANT le tableau courant
            # -- (entre la fin du dernier tableau et le debut de celui-ci)
            before = text[current_pos:match.start()].strip()
            if before:
                # -- Decoupe ce texte intermediaire de maniere semantique
                # -- car il s'agit de texte narratif classique
                sub_chunks = self._financial_splitter.split_text(before)
                parts.extend(sub_chunks)

            # -- Conserve le tableau entier comme un seul chunk indivisible.
            # -- Cela permet de garder la coherence des donnees tabulaires
            # -- (en-tetes alignes avec les valeurs).
            table_text = match.group(0).strip()
            if table_text:
                parts.append(table_text)

            # -- Avance le curseur apres la fin du tableau traite
            current_pos = match.end()

        # -- Traite le texte restant apres le dernier tableau
        remaining = text[current_pos:].strip()
        if remaining:
            # -- Decoupe le texte restant de maniere semantique
            sub_chunks = self._financial_splitter.split_text(remaining)
            parts.extend(sub_chunks)

        # -- Cas de secours : si aucun tableau n'a ete detecte et que parts
        # -- est vide, on applique le decoupage semantique standard sur tout le texte.
        if not parts:
            parts = self._financial_splitter.split_text(text)

        # -- Construit et retourne les Documents enrichis
        return self._build_docs(parts, base_meta, "table_aware")

    def _chunk_by_sentence(self, text: str, base_meta: Dict[str, Any]) -> List[Document]:
        """Chunking par phrases (respecte les frontières de phrases)."""
        # -- Strategie SENTENCE : decoupe le texte en respectant les frontieres
        # -- de phrases. Les phrases sont regroupees dans des chunks dont la
        # -- taille ne depasse pas chunk_size. Cette strategie est utile pour
        # -- les textes tres structures ou chaque phrase porte un sens complet
        # -- (ex: listes de recommandations, points cles, etc.).

        # -- Decoupe le texte en phrases individuelles en utilisant les ponctuations
        # -- de fin de phrase (point, point d'exclamation, point d'interrogation)
        # -- suivies d'un ou plusieurs espaces. Le lookbehind (?<=...) conserve
        # -- la ponctuation dans la phrase precedente.
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # -- Accumulation des phrases dans des chunks de taille bornee
        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            # -- Si l'ajout de la phrase courante ne depasse pas chunk_size,
            # -- on l'ajoute au chunk en cours de construction
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # -- Sinon, on finalise le chunk courant et on en demarre un nouveau
                # -- avec la phrase qui n'a pas pu etre ajoutee
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # -- Ne pas oublier le dernier chunk en cours de construction
        if current_chunk:
            chunks.append(current_chunk.strip())

        # -- Construit et retourne les Documents enrichis
        return self._build_docs(chunks, base_meta, "sentence")

    def _build_docs(
        self,
        texts: List[str],
        base_meta: Dict[str, Any],
        strategy: str,
    ) -> List[Document]:
        """Construit les objets Document avec métadonnées enrichies."""
        # -- Methode utilitaire partagee par toutes les strategies de chunking.
        # -- Elle transforme une liste de chaines de texte brut en une liste
        # -- d'objets Document LangChain, chacun enrichi de metadonnees
        # -- financieres (entites, periodes, tableaux, chiffres, etc.).
        # -- Les chunks trop courts sont filtres pour eviter le bruit.

        # -- Liste de sortie contenant les Documents finalises
        docs: List[Document] = []

        # -- Compteur d'index pour numeroter les chunks au sein du document
        chunk_index = 0

        for text in texts:
            # -- Nettoyage des espaces en debut et fin de chunk
            text = text.strip()

            # -- Filtrage : on ignore les chunks dont la longueur est inferieure
            # -- a min_chunk_length (50 caracteres par defaut). Ces chunks
            # -- sont generalement des residus de decoupage (titres isoles,
            # -- lignes vides, fragments non informatifs).
            if len(text) < self.min_chunk_length:
                continue  # Filter out too-short chunks

            # -- Construction des metadonnees enrichies pour ce chunk
            # -- en appelant le helper _build_chunk_metadata qui agrege
            # -- les metadonnees du document parent et les nouvelles
            # -- metadonnees specifiques au chunk.
            meta = _build_chunk_metadata(
                chunk_text=text,
                base_metadata=base_meta,
                chunk_index=chunk_index,
                strategy=strategy,
            )

            # -- Creation de l'objet Document LangChain avec le contenu
            # -- textuel du chunk et ses metadonnees enrichies
            docs.append(Document(page_content=text, metadata=meta))

            # -- Incrementation de l'index pour le prochain chunk
            chunk_index += 1

        # -- Retourne la liste complete de Documents enrichis pour ce document
        return docs
