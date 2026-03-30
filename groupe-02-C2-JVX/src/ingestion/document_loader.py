"""
Chargeur de documents financiers multi-format.
Supporte PDF, CSV, XLSX, TXT, JSON, URLs (news JSON).
"""

# -- Importation de annotations pour permettre l'utilisation de types comme
# -- "str | Path" dans les signatures de fonctions (PEP 604), compatible Python 3.7+.
from __future__ import annotations

# -- Bibliotheques standard Python :
# --   json   : lecture/ecriture de fichiers JSON
# --   re     : expressions regulieres pour detecter tickers et dates
# --   datetime : horodatage de l'ingestion des documents
# --   Path   : manipulation de chemins de fichiers de maniere portable
# --   typing : annotations de types pour une meilleure lisibilite du code
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# -- pandas : manipulation de donnees tabulaires (CSV, Excel)
import pandas as pd
# -- Document de LangChain : objet standard contenant page_content (texte) et
# -- metadata (dictionnaire de metadonnees associees au document)
from langchain_core.documents import Document
# -- loguru : bibliotheque de journalisation (logging) avec des niveaux colores
# -- (info, warning, error, success) pour faciliter le suivi des operations
from loguru import logger


# ─── Metadata helpers ────────────────────────────────────────────────────────
# -- Cette section definit les outils (regex, constantes, fonctions) permettant
# -- d'extraire automatiquement des metadonnees a partir du contenu textuel
# -- des documents financiers charges.

# -- TICKER_PATTERN : expression reguliere qui detecte les mots composes de
# -- 2 a 5 lettres majuscules entourees de limites de mots (\b).
# -- Exemples de correspondances : "AAPL", "MSFT", "NVDA", "META", "GS"
# -- Les limites \b empechent de capturer des sous-chaines a l'interieur de mots.
TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')

# -- KNOWN_TICKERS : ensemble des symboles boursiers connus et valides.
# -- Utilise pour filtrer les faux positifs de TICKER_PATTERN (par exemple,
# -- les mots anglais en majuscules comme "THE", "FOR", "AND" ne sont pas
# -- dans cet ensemble et seront donc exclus).
# -- Organise par secteur :
# --   - Tech     : AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
# --   - Finance  : JPM, BAC, GS, MS, WFC, C, BRK
# --   - Sante    : JNJ, UNH, PFE, ABBV, LLY
# --   - Energie  : XOM, CVX, COP
# --   - Conso    : WMT, PG, KO, PEP, MCD
# --   - ETFs     : SPY, QQQ, DIA, IWM
KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BRK",
    "JNJ", "UNH", "PFE", "ABBV", "LLY",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO", "PEP", "MCD",
    "SPY", "QQQ", "DIA", "IWM",
}

# -- DATE_PATTERNS : liste de regex pour extraire des references temporelles
# -- dans les documents financiers. Chaque pattern cible un format different :
# --
# -- 1) r'\b(20\d{2})\b'       -> Annee simple au format 20XX
# --    Exemples : "2023", "2024", "2025"
# --    \b garantit que ce n'est pas un sous-nombre (ex: "120234" ne matche pas)
# --
# -- 2) r'\b(Q[1-4]\s+20\d{2})\b' -> Trimestre au format "Q1 2024"
# --    Exemples : "Q1 2024", "Q3 2023", "Q4 2025"
# --    [1-4] restreint aux trimestres valides, \s+ permet un ou plusieurs espaces
# --
# -- 3) r'\b(FY20\d{2})\b'     -> Annee fiscale au format "FY2024"
# --    Exemples : "FY2023", "FY2024"
# --    Utile pour les rapports annuels qui mentionnent l'exercice fiscal
DATE_PATTERNS = [
    re.compile(r'\b(20\d{2})\b'),
    re.compile(r'\b(Q[1-4]\s+20\d{2})\b'),
    re.compile(r'\b(FY20\d{2})\b'),
]

# -- DOCUMENT_TYPE_KEYWORDS : dictionnaire associant chaque type de document
# -- financier a une liste de mots-cles (en anglais et en francais).
# -- Utilise par _detect_document_type() pour classifier automatiquement
# -- un document selon son contenu et son nom de fichier.
# --
# -- Types supportes :
# --   "annual_report"    : rapports annuels (10-K SEC, rapports annuels FR)
# --   "quarterly_report" : rapports trimestriels (10-Q SEC, Q1-Q4)
# --   "news_article"     : articles de presse et communiques
# --   "market_overview"  : analyses macroeconomiques et vues de marche
# --   "research_report"  : notes d'analystes et rapports de recherche
# --   "csv_data"         : donnees tabulaires brutes (portefeuilles, cours)
DOCUMENT_TYPE_KEYWORDS = {
    "annual_report": ["annual report", "rapport annuel", "10-K", "annual"],
    "quarterly_report": ["quarterly", "trimestriel", "Q1", "Q2", "Q3", "Q4", "10-Q"],
    "news_article": ["article", "news", "press release", "communiqué"],
    "market_overview": ["overview", "market", "marché", "macro"],
    "research_report": ["research", "analyse", "analyst", "rating"],
    "csv_data": ["csv", "portfolio", "données"],
}


def _detect_document_type(content: str, filename: str) -> str:
    """Détecte automatiquement le type de document."""
    # -- Convertit le contenu et le nom de fichier en minuscules pour une
    # -- comparaison insensible a la casse avec les mots-cles.
    content_lower = content.lower()
    filename_lower = filename.lower()
    # -- Combine le contenu et le nom de fichier en un seul texte de recherche.
    # -- Ainsi, un fichier nomme "10-K_AAPL.pdf" sera detecte comme annual_report
    # -- meme si le contenu ne contient pas explicitement "10-K".
    text = content_lower + " " + filename_lower

    # -- Algorithme de scoring : pour chaque type de document, on compte combien
    # -- de mots-cles associes apparaissent dans le texte. Le type avec le plus
    # -- grand nombre de correspondances l'emporte (best_score).
    # -- Si aucun mot-cle ne correspond, le type retourne sera "unknown".
    best_type = "unknown"
    best_score = 0
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_score = score
            best_type = doc_type
    return best_type


def _extract_tickers(text: str) -> List[str]:
    """Extrait les symboles boursiers du texte."""
    # -- Etape 1 : TICKER_PATTERN.findall() recupere toutes les sequences de
    # -- 2 a 5 lettres majuscules dans le texte (ex: "AAPL", "THE", "MSFT").
    found = set(TICKER_PATTERN.findall(text))
    # -- Etape 2 : On filtre en ne gardant que les symboles qui existent dans
    # -- KNOWN_TICKERS (intersection d'ensembles). Cela elimine les faux positifs
    # -- comme "THE", "AND", "FOR" qui matchent la regex mais ne sont pas des tickers.
    # -- Le resultat est trie alphabetiquement pour un affichage coherent.
    return sorted(found & KNOWN_TICKERS)


def _extract_dates(text: str) -> Optional[str]:
    """Extrait la date/période principale du document."""
    # -- Parcourt les patterns de date dans l'ordre de priorite :
    # -- 1) Annee simple (20XX) — le plus courant
    # -- 2) Trimestre (Q1 2024) — plus precis si present
    # -- 3) Annee fiscale (FY2024) — specifique aux rapports financiers
    # -- Retourne la premiere correspondance trouvee, ou None si aucune date
    # -- n'est detectee dans le texte.
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


# ─── Main Loader ─────────────────────────────────────────────────────────────

class FinancialDocumentLoader:
    """
    Chargeur universel de documents financiers.

    Supporte : PDF, CSV, XLSX, TXT, JSON, URLs (news JSON).
    Extrait les métadonnées : source, filename, date, document_type,
    ticker_symbols, page_count.
    Retourne des objets langchain.schema.Document enrichis.
    """

    def __init__(self, chunk_on_load: bool = False) -> None:
        # -- chunk_on_load : parametre reserve pour un eventuel decoupage
        # -- automatique du texte en morceaux (chunks) lors du chargement.
        # -- Actuellement non utilise dans les methodes de chargement,
        # -- mais prevu pour une integration future avec un TextSplitter.
        self.chunk_on_load = chunk_on_load

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, source: str | Path) -> List[Document]:
        """
        Charge un document depuis un chemin fichier ou URL.

        Args:
            source: Chemin vers le fichier (str ou Path).

        Returns:
            Liste de Documents LangChain enrichis.
        """
        # -- Convertit la source en objet Path pour manipuler facilement
        # -- l'extension et le nom du fichier.
        path = Path(source)
        # -- Recupere l'extension du fichier en minuscules (ex: ".pdf", ".csv")
        ext = path.suffix.lower()

        logger.info(f"Chargement : {path.name} (type={ext})")

        # -- Dictionnaire de dispatch : associe chaque extension de fichier
        # -- a la methode de chargement privee correspondante.
        # -- Cela permet d'eviter une longue serie de if/elif et facilite
        # -- l'ajout de nouveaux formats (il suffit d'ajouter une entree).
        loaders = {
            ".pdf": self._load_pdf,     # -- Documents PDF (rapports, articles)
            ".csv": self._load_csv,     # -- Fichiers CSV (donnees tabulaires)
            ".xlsx": self._load_xlsx,   # -- Fichiers Excel format .xlsx
            ".xls": self._load_xlsx,    # -- Fichiers Excel format .xls (ancien)
            ".txt": self._load_txt,     # -- Fichiers texte brut
            ".json": self._load_json,   # -- Fichiers JSON (articles de news, donnees)
        }

        # -- Recherche la fonction de chargement correspondant a l'extension.
        # -- Si l'extension n'est pas supportee, retourne une liste vide avec un warning.
        loader_fn = loaders.get(ext)
        if loader_fn is None:
            logger.warning(f"Extension non supportée : {ext} — {path.name}")
            return []

        # -- Appelle la fonction de chargement appropriee dans un bloc try/except
        # -- pour capturer toute erreur inattendue (fichier corrompu, encodage, etc.)
        # -- et eviter que le programme ne plante.
        try:
            docs = loader_fn(path)
            logger.success(f"Chargé {len(docs)} documents depuis {path.name}")
            return docs
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {path.name}: {e}")
            return []

    def load_directory(self, directory: str | Path) -> List[Document]:
        """
        Charge tous les documents supportés depuis un répertoire.

        Args:
            directory: Chemin du répertoire à scanner.

        Returns:
            Liste consolidée de Documents LangChain.
        """
        directory = Path(directory)
        # -- Verifie que le repertoire existe avant de tenter le scan.
        if not directory.exists():
            logger.error(f"Répertoire introuvable : {directory}")
            return []

        all_docs: List[Document] = []
        # -- Ensemble des extensions prises en charge pour filtrer les fichiers.
        supported_exts = {".pdf", ".csv", ".xlsx", ".xls", ".txt", ".json"}

        # -- rglob("*") parcourt recursivement tous les fichiers du repertoire
        # -- et de ses sous-repertoires. On filtre pour ne garder que ceux dont
        # -- l'extension (en minuscules) est dans supported_exts.
        files = [f for f in directory.rglob("*") if f.suffix.lower() in supported_exts]
        logger.info(f"Répertoire {directory} : {len(files)} fichiers trouvés")

        # -- Charge chaque fichier individuellement via la methode load() qui
        # -- dispatche vers le bon loader selon l'extension.
        for file_path in files:
            docs = self.load(file_path)
            all_docs.extend(docs)

        logger.info(f"Total chargé : {len(all_docs)} documents")
        return all_docs

    # ── Private Loaders ──────────────────────────────────────────────────────

    def _build_metadata(
        self,
        filename: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construit les métadonnées enrichies pour un document."""
        # -- Construction du dictionnaire de metadonnees de base.
        # -- Chaque document LangChain possede un champ metadata qui contient :
        # --   "source"              : nom du fichier d'origine (pour tracabilite)
        # --   "filename"            : idem, redondant mais utile pour certains pipelines
        # --   "date"                : date ou periode extraite du contenu (ex: "2024", "Q3 2023")
        # --                           Si aucune date trouvee, tente le champ extra["date"], sinon ""
        # --   "document_type"       : type detecte automatiquement (ex: "annual_report", "news_article")
        # --   "ticker_symbols"      : liste des symboles boursiers trouves dans le contenu (ex: ["AAPL", "MSFT"])
        # --   "page_count"          : nombre de pages (pertinent pour les PDF, 0 par defaut)
        # --   "ingestion_timestamp" : horodatage UTC au format ISO de l'ingestion (pour audit)
        meta: Dict[str, Any] = {
            "source": filename,
            "filename": filename,
            "date": _extract_dates(content) or extra.get("date", "") if extra else _extract_dates(content) or "",
            "document_type": _detect_document_type(content, filename),
            "ticker_symbols": _extract_tickers(content),
            "page_count": 0,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }
        # -- Si des metadonnees supplementaires (extra) sont fournies par le loader,
        # -- elles sont fusionnees dans le dictionnaire de base. Les cles en double
        # -- sont ecrasees par les valeurs de extra (ex: page_count, document_type).
        if extra:
            meta.update(extra)
        return meta

    def _load_pdf(self, path: Path) -> List[Document]:
        """Charge un PDF avec pdfplumber (fallback PyMuPDF)."""
        docs: List[Document] = []

        # -- Strategie en deux etapes pour maximiser la compatibilite :
        # -- 1) Tentative avec pdfplumber (meilleure extraction de tableaux)
        # -- 2) Si pdfplumber echoue ou n'est pas installe, fallback sur PyMuPDF (fitz)

        # -- Tentative avec pdfplumber en premier (bibliotheque preferee)
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                # -- Recupere le nombre total de pages pour les metadonnees
                page_count = len(pdf.pages)
                # -- Parcourt chaque page en numerotant a partir de 1 (convention humaine)
                for page_num, page in enumerate(pdf.pages, start=1):
                    # -- extract_text() retourne le texte brut de la page.
                    # -- Peut retourner None si la page est une image ou vide.
                    text = page.extract_text() or ""
                    # -- Ignore les pages sans contenu textuel exploitable
                    if not text.strip():
                        continue

                    # -- Construit les metadonnees pour cette page specifique.
                    # -- Chaque page du PDF devient un Document LangChain distinct,
                    # -- ce qui permet un decoupage plus fin pour la recherche vectorielle.
                    # -- extra contient : page_number (numero de la page), page_count (total),
                    # -- et document_type (re-detecte a partir du texte de cette page).
                    meta = self._build_metadata(
                        filename=path.name,
                        content=text,
                        extra={
                            "page_number": page_num,
                            "page_count": page_count,
                            "document_type": _detect_document_type(text, path.name),
                        },
                    )
                    docs.append(Document(page_content=text, metadata=meta))

            return docs

        except ImportError:
            # -- pdfplumber n'est pas installe dans l'environnement
            logger.warning("pdfplumber non disponible, tentative avec PyMuPDF")
        except Exception as e:
            # -- pdfplumber est installe mais a rencontre une erreur sur ce PDF
            logger.warning(f"pdfplumber échec sur {path.name}: {e}, tentative PyMuPDF")

        # -- Fallback : PyMuPDF (importe sous le nom "fitz")
        # -- PyMuPDF est generalement plus robuste pour les PDF complexes ou
        # -- proteges, mais moins performant pour l'extraction de tableaux.
        try:
            import fitz  # PyMuPDF

            pdf_doc = fitz.open(str(path))
            page_count = len(pdf_doc)

            # -- PyMuPDF numerote les pages a partir de 0 (convention Python)
            for page_num in range(page_count):
                page = pdf_doc[page_num]
                # -- get_text() extrait le texte brut de la page
                text = page.get_text()
                # -- Ignore les pages vides (meme logique que pdfplumber)
                if not text.strip():
                    continue

                # -- page_num + 1 pour aligner sur la convention humaine (page 1, 2, 3...)
                meta = self._build_metadata(
                    filename=path.name,
                    content=text,
                    extra={
                        "page_number": page_num + 1,
                        "page_count": page_count,
                    },
                )
                docs.append(Document(page_content=text, metadata=meta))

            # -- Ferme explicitement le document PDF pour liberer la memoire
            pdf_doc.close()
            return docs

        except ImportError:
            # -- Ni pdfplumber ni PyMuPDF ne sont disponibles dans l'environnement.
            # -- L'utilisateur doit installer au moins l'un des deux :
            # -- pip install pdfplumber   ou   pip install PyMuPDF
            logger.error("Ni pdfplumber ni PyMuPDF ne sont installés.")
            return []
        except Exception as e:
            logger.error(f"Impossible de charger le PDF {path.name}: {e}")
            return []

    def _load_csv(self, path: Path) -> List[Document]:
        """Charge un CSV et le convertit en document textuel."""
        try:
            # -- Lecture du fichier CSV avec pandas. L'encodage UTF-8 est utilise
            # -- par defaut, avec errors="replace" pour remplacer les caracteres
            # -- invalides plutot que de lever une exception.
            df = pd.read_csv(path, encoding="utf-8", errors="replace")

            # -- Construction d'une representation textuelle du CSV.
            # -- Le texte genere contient :
            # --   - Le nom du dataset (derive du nom de fichier sans extension)
            # --   - La liste des colonnes
            # --   - Le nombre total de lignes
            # --   - Un apercu des 50 premieres lignes au format tableau
            # -- Ce texte sera ensuite indexe dans la base vectorielle pour
            # -- permettre la recherche semantique sur les donnees tabulaires.
            content = f"Dataset: {path.stem}\n\n"
            content += f"Colonnes: {', '.join(df.columns.tolist())}\n"
            content += f"Lignes: {len(df)}\n\n"
            content += "Aperçu des données:\n"
            content += df.head(50).to_string(index=False)

            # -- Detection des tickers : si une colonne "ticker" existe dans le CSV,
            # -- on extrait directement les valeurs uniques de cette colonne.
            # -- Sinon, on utilise la detection par regex sur le contenu textuel.
            if "ticker" in df.columns:
                tickers = df["ticker"].dropna().unique().tolist()
            else:
                tickers = _extract_tickers(content)

            # -- Metadonnees specifiques aux CSV :
            # --   "document_type" : force a "csv_data" (pas de detection automatique)
            # --   "row_count"     : nombre de lignes dans le DataFrame
            # --   "columns"       : liste des noms de colonnes
            # --   "ticker_symbols": liste des tickers detectes
            meta = self._build_metadata(
                filename=path.name,
                content=content,
                extra={
                    "document_type": "csv_data",
                    "row_count": len(df),
                    "columns": df.columns.tolist(),
                    "ticker_symbols": tickers,
                },
            )
            return [Document(page_content=content, metadata=meta)]

        except Exception as e:
            logger.error(f"Erreur CSV {path.name}: {e}")
            return []

    def _load_xlsx(self, path: Path) -> List[Document]:
        """Charge un fichier Excel (toutes les feuilles)."""
        try:
            # -- Ouvre le fichier Excel avec le moteur openpyxl (supporte .xlsx).
            # -- ExcelFile permet de lire plusieurs feuilles sans reouvrir le fichier.
            xl = pd.ExcelFile(path, engine="openpyxl")
            docs: List[Document] = []

            # -- Parcourt chaque feuille du classeur Excel.
            # -- Chaque feuille devient un Document LangChain distinct,
            # -- ce qui permet de traiter separement les differents onglets
            # -- (ex: "Bilan", "Compte de resultat", "Flux de tresorerie").
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                # -- Construction du contenu textuel, similaire au CSV,
                # -- mais avec le nom de la feuille en plus.
                content = f"Fichier: {path.stem} | Feuille: {sheet_name}\n\n"
                content += f"Colonnes: {', '.join(str(c) for c in df.columns.tolist())}\n"
                content += f"Lignes: {len(df)}\n\n"
                content += df.head(50).to_string(index=False)

                # -- Metadonnees specifiques aux fichiers Excel :
                # --   "document_type" : force a "excel_data"
                # --   "sheet_name"    : nom de la feuille (onglet) d'origine
                # --   "row_count"     : nombre de lignes dans cette feuille
                meta = self._build_metadata(
                    filename=path.name,
                    content=content,
                    extra={
                        "document_type": "excel_data",
                        "sheet_name": sheet_name,
                        "row_count": len(df),
                    },
                )
                docs.append(Document(page_content=content, metadata=meta))

            return docs

        except Exception as e:
            logger.error(f"Erreur XLSX {path.name}: {e}")
            return []

    def _load_txt(self, path: Path) -> List[Document]:
        """Charge un fichier texte brut."""
        try:
            # -- Lecture integrale du fichier texte en UTF-8.
            # -- errors="replace" remplace les caracteres invalides par le caractere
            # -- de remplacement Unicode (U+FFFD) pour eviter les erreurs d'encodage.
            content = path.read_text(encoding="utf-8", errors="replace")
            # -- Pour un fichier texte, les metadonnees sont construites uniquement
            # -- a partir du contenu et du nom de fichier (pas d'extra specifique).
            # -- Le type de document sera detecte automatiquement par _detect_document_type.
            meta = self._build_metadata(filename=path.name, content=content)
            return [Document(page_content=content, metadata=meta)]
        except Exception as e:
            logger.error(f"Erreur TXT {path.name}: {e}")
            return []

    def _load_json(self, path: Path) -> List[Document]:
        """Charge un fichier JSON (article de news ou données)."""
        try:
            # -- Lecture et parsing du fichier JSON complet en memoire.
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # -- Cas 1 : Format "article de news" (dictionnaire avec cles "title" et "content")
            # -- Ce format est typiquement produit par un scraper d'actualites financieres.
            # -- Structure attendue :
            # --   { "title": "...", "content": "...", "source": "...",
            # --     "date": "...", "ticker": "...", "category": "...", "url": "..." }
            if isinstance(data, dict) and "title" in data and "content" in data:
                # -- Reconstruction d'un contenu textuel structure a partir des champs JSON.
                # -- Le titre, la source, la date et le ticker sont places en en-tete,
                # -- suivis du contenu complet de l'article.
                content = f"Titre: {data.get('title', '')}\n\n"
                content += f"Source: {data.get('source', '')}\n"
                content += f"Date: {data.get('date', '')}\n"
                content += f"Ticker: {data.get('ticker', '')}\n\n"
                content += data.get("content", "")

                # -- Metadonnees enrichies pour les articles de news :
                # --   "document_type"  : force a "news_article"
                # --   "date"           : date de publication extraite du JSON
                # --   "ticker_symbols" : ticker mentionne dans l'article (liste a un element)
                # --   "news_source"    : source de l'article (ex: "Reuters", "Bloomberg")
                # --   "news_title"     : titre de l'article pour affichage
                # --   "category"       : categorie de l'article (ex: "earnings", "merger")
                # --   "url"            : lien vers l'article original
                meta = self._build_metadata(
                    filename=path.name,
                    content=content,
                    extra={
                        "document_type": "news_article",
                        "date": data.get("date", ""),
                        "ticker_symbols": [data["ticker"]] if data.get("ticker") else [],
                        "news_source": data.get("source", ""),
                        "news_title": data.get("title", ""),
                        "category": data.get("category", ""),
                        "url": data.get("url", ""),
                    },
                )
                return [Document(page_content=content, metadata=meta)]

            # -- Cas 2 : Liste d'articles (tableau JSON contenant plusieurs objets)
            # -- Chaque element de la liste ayant une cle "content" est traite comme
            # -- un document distinct. Le contenu est serialise en JSON indente
            # -- pour conserver la structure originale.
            elif isinstance(data, list):
                docs: List[Document] = []
                for item in data:
                    if isinstance(item, dict) and "content" in item:
                        # -- Serialise chaque article en JSON lisible (indentation 2 espaces)
                        # -- ensure_ascii=False preserve les caracteres accentues francais
                        content = json.dumps(item, ensure_ascii=False, indent=2)
                        meta = self._build_metadata(filename=path.name, content=content)
                        docs.append(Document(page_content=content, metadata=meta))
                return docs

            # -- Cas 3 : JSON generique (ni article, ni liste d'articles)
            # -- Le contenu est simplement serialise en texte JSON formate.
            # -- Utile pour les fichiers de configuration, de donnees structurees, etc.
            else:
                content = json.dumps(data, ensure_ascii=False, indent=2)
                meta = self._build_metadata(filename=path.name, content=content)
                return [Document(page_content=content, metadata=meta)]

        except Exception as e:
            logger.error(f"Erreur JSON {path.name}: {e}")
            return []
