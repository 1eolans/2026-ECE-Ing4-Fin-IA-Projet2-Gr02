"""
Extracteur de tableaux financiers depuis les PDFs.
Utilise camelot (lattice + stream) avec fallback pdfplumber.
Convertit les tableaux en markdown structuré lisible par LLM.
"""

# -- Ce module a pour objectif d'extraire les tableaux contenus dans des fichiers PDF
# -- financiers (rapports annuels, comptes de résultat, bilans, etc.) et de les convertir
# -- en un format markdown structuré que les modèles de langage (LLM) peuvent facilement
# -- interpréter. Deux bibliothèques d'extraction sont utilisées selon une stratégie
# -- de fallback : camelot en priorité, puis pdfplumber si camelot échoue ou n'est pas
# -- installé. Les tableaux extraits sont encapsulés dans des objets Document de LangChain,
# -- enrichis de métadonnées (source, page, méthode d'extraction, dimensions, précision).

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from langchain_core.documents import Document
from loguru import logger


# ─── Helpers ─────────────────────────────────────────────────────────────────

# -- Expression régulière compilée pour détecter les unités financières courantes
# -- dans le contenu des cellules de tableaux. Elle reconnaît les formats suivants :
# --   - M€, Md€, M$, Md$ : millions / milliards d'euros ou de dollars
# --   - % : pourcentages (marges, ratios, rendements)
# --   - bp : points de base (basis points), utilisés pour les taux d'intérêt
# --   - K€, K$ : milliers d'euros ou de dollars
# --   - billion, million, trillion : unités anglo-saxonnes en toutes lettres
# -- Le drapeau re.IGNORECASE permet de capturer "Million", "MILLION", etc.
FINANCIAL_UNITS = re.compile(
    r'(\d[\d\s,\.]+)\s*(M€|Md€|M\$|Md\$|%|bp|K€|K\$|billion|million|trillion)',
    re.IGNORECASE,
)


def _clean_cell(cell: str) -> str:
    """Nettoie le contenu d'une cellule de tableau."""
    # -- Cette fonction prend en entrée le texte brut d'une cellule de tableau PDF
    # -- et le normalise pour produire un texte propre utilisable en markdown.
    # -- Les opérations effectuées sont :
    # --   1. Vérification du type : si la valeur n'est pas une chaîne (ex. int, float, None),
    # --      elle est convertie en chaîne via str().
    # --   2. strip() : suppression des espaces/tabulations en début et fin de cellule.
    # --   3. Remplacement des sauts de ligne internes (\n) par des espaces, car les
    # --      cellules PDF contiennent parfois des retours à la ligne indésirables.
    # --   4. Remplacement des doubles espaces par un espace simple pour éviter les
    # --      trous visuels dans le markdown final.
    if not isinstance(cell, str):
        return str(cell)
    cleaned = cell.strip().replace('\n', ' ').replace('  ', ' ')
    return cleaned


def _df_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convertit un DataFrame en tableau markdown avec titre."""
    # -- Cette fonction transforme un DataFrame pandas (représentant un tableau extrait
    # -- d'un PDF) en une chaîne de caractères au format markdown. Ce format est choisi
    # -- car il est facilement lisible par les LLM et conserve la structure tabulaire.
    # -- Le processus est le suivant :
    # --   1. Si le DataFrame est vide, on retourne une chaîne vide (pas de tableau à afficher).
    # --   2. Si un titre est fourni, il est ajouté en tant qu'en-tête markdown de niveau 3 (###).
    # --   3. Toutes les cellules du DataFrame sont nettoyées via _clean_cell().
    # --   4. La ligne d'en-tête est construite avec les noms de colonnes séparés par " | ".
    # --   5. Une ligne de séparation markdown (| --- | --- | ...) est ajoutée.
    # --   6. Chaque ligne de données est formatée de la même manière.
    if df.empty:
        return ""

    lines: List[str] = []
    if title:
        # -- Ajout du titre en tant qu'en-tête markdown de niveau 3
        lines.append(f"### {title}")
        # -- Ligne vide après le titre pour respecter la syntaxe markdown
        lines.append("")

    # FIX ÉLEVÉ : applymap() supprimé en pandas ≥ 2.1, utiliser map() ou applymap selon version
    # Compatibilité pandas 2.0 (applymap) ET pandas 2.1+ (map)
    # -- Vérification de la version de pandas pour choisir la bonne méthode d'application
    # -- élément par élément. Depuis pandas 2.1, applymap() est déprécié au profit de map().
    # -- On vérifie dynamiquement si df.map existe et est appelable ; sinon on utilise applymap.
    _apply = df.map if hasattr(df, "map") and callable(getattr(df, "map")) else df.applymap
    # -- Application de _clean_cell() à chaque cellule du DataFrame. Les valeurs NaN/None
    # -- sont remplacées par une chaîne vide "" pour éviter les "nan" dans le markdown.
    df = _apply(lambda x: _clean_cell(str(x)) if pd.notna(x) else "")

    # Header
    # -- Construction de la ligne d'en-tête du tableau markdown à partir des noms de colonnes
    headers = [str(col) for col in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    # -- Ligne de séparation obligatoire en markdown pour délimiter l'en-tête du corps du tableau.
    # -- Chaque colonne reçoit "---" comme séparateur.
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Rows
    # -- Itération sur chaque ligne du DataFrame pour construire les lignes markdown du tableau.
    # -- Chaque valeur est convertie en chaîne et les cellules sont séparées par " | ".
    for _, row in df.iterrows():
        cells = [str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")

    # -- Jointure de toutes les lignes avec des sauts de ligne pour produire le markdown final
    return "\n".join(lines)


def _infer_table_title(page_text: str, table_bbox: Optional[Tuple] = None) -> str:
    """Tente d'inférer le titre d'un tableau depuis le texte de la page."""
    # -- Cette fonction essaie de deviner le titre d'un tableau en analysant le texte
    # -- de la page PDF où le tableau a été trouvé. C'est une heuristique utile car
    # -- les bibliothèques d'extraction ne fournissent pas directement le titre du tableau.
    # -- Le paramètre table_bbox (bounding box) est prévu pour une future amélioration
    # -- qui permettrait de chercher le titre juste au-dessus de la zone du tableau.

    # -- Si aucun texte de page n'est disponible, on retourne un titre générique
    if not page_text:
        return "Tableau Financier"

    # -- Deux patterns regex sont définis pour chercher un titre probable :
    # --   1. Premier pattern : cherche des mots-clés financiers courants comme "Tableau",
    # --      "Table", "Compte de résultat", "Bilan", "Cash-flow", "Revenus", "Résultats",
    # --      "Performance", "Summary", "Overview", "Ratios", suivis de jusqu'à 80 caractères
    # --      sur la même ligne. Ce pattern est le plus fiable car il cible des termes
    # --      spécifiques au domaine financier.
    # --   2. Deuxième pattern : cherche une ligne commençant par une majuscule, de 10 à 60
    # --      caractères, qui pourrait être un titre générique (ex. "Indicateurs clés").
    # --      C'est un pattern plus permissif utilisé en dernier recours.
    title_patterns = [
        re.compile(
            r'((?:Tableau|Table|Compte de résultat|Bilan|Cash[- ]flow|Revenus?|'
            r'Résultats?|Performance|Summary|Overview|Ratios?)[^\n]{0,80})',
            re.I,
        ),
        re.compile(r'^([A-Z][^\n]{10,60})$', re.MULTILINE),
    ]

    # -- On cherche un titre uniquement dans les 500 premiers caractères de la page,
    # -- car les titres de tableaux apparaissent généralement en haut de la page.
    for pattern in title_patterns:
        match = pattern.search(page_text[:500])
        if match:
            return match.group(1).strip()

    # -- Si aucun pattern ne correspond, on retourne le titre générique par défaut
    return "Tableau Financier"


# ─── Main Extractor ───────────────────────────────────────────────────────────

class FinancialTableExtractor:
    """
    Extracteur de tableaux financiers depuis les PDFs.

    Stratégie :
    1. camelot lattice (tableaux avec bordures nettes)
    2. camelot stream (tableaux sans bordures)
    3. pdfplumber extract_tables (fallback)

    Convertit en markdown structuré avec métadonnées de position.
    Gère les tableaux multi-pages.
    """
    # -- Cette classe orchestre l'extraction de tableaux depuis un fichier PDF.
    # -- Elle encapsule trois stratégies d'extraction ordonnées par fiabilité :
    # --   - camelot "lattice" : fonctionne le mieux sur les tableaux avec des lignes
    # --     de bordure visibles (traits horizontaux et verticaux). C'est la méthode
    # --     la plus précise quand le PDF contient des tableaux bien structurés.
    # --   - camelot "stream" : utilise l'espacement entre les mots pour détecter les
    # --     colonnes, adapté aux tableaux sans bordures (souvent dans les rapports modernes).
    # --   - pdfplumber : bibliothèque de fallback plus tolérante, utilisée lorsque
    # --     camelot n'est pas installé ou ne parvient pas à extraire de tableaux.
    # -- Chaque tableau extrait est converti en Document LangChain contenant :
    # --   - page_content : le tableau en format markdown
    # --   - metadata : informations contextuelles (source, page, méthode, précision, etc.)

    def __init__(
        self,
        min_rows: int = 2,
        min_cols: int = 2,
        accuracy_threshold: float = 80.0,
    ) -> None:
        # -- min_rows : nombre minimum de lignes pour qu'un tableau soit retenu.
        # --   Valeur par défaut 2 : élimine les faux positifs à 1 ligne qui ne sont
        # --   généralement pas de vrais tableaux mais du texte mal interprété.
        self.min_rows = min_rows
        # -- min_cols : nombre minimum de colonnes pour qu'un tableau soit retenu.
        # --   Valeur par défaut 2 : élimine les listes à une seule colonne qui ne sont
        # --   pas des tableaux au sens financier (pas de données structurées en colonnes).
        self.min_cols = min_cols
        # -- accuracy_threshold : seuil de précision camelot (0-100) en dessous duquel
        # --   un tableau est rejeté. Valeur par défaut 80.0 : on ne garde que les tableaux
        # --   dont camelot est raisonnablement sûr de la structure. Un seuil trop bas
        # --   risque d'inclure des tableaux mal parsés avec des cellules fusionnées ou décalées.
        self.accuracy_threshold = accuracy_threshold

    def extract_tables_from_pdf(self, pdf_path: str | Path) -> List[Document]:
        """
        Extrait tous les tableaux d'un PDF et retourne des Documents LangChain.
        """
        # -- Point d'entrée principal de l'extraction. Cette méthode :
        # --   1. Convertit le chemin en objet Path pour la compatibilité multi-OS.
        # --   2. Tente d'abord l'extraction via camelot (lattice puis stream).
        # --   3. Si camelot ne retourne aucun résultat (bibliothèque absente, pas de
        # --      tableaux détectés, ou erreur), bascule vers pdfplumber en fallback.
        # --   4. Journalise le nombre de tableaux extraits et retourne la liste de Documents.
        pdf_path = Path(pdf_path)
        logger.info(f"Extraction tableaux : {pdf_path.name}")

        docs: List[Document] = []

        # -- Tentative d'extraction via camelot (méthode prioritaire, plus précise)
        camelot_docs = self._extract_with_camelot(pdf_path)
        if camelot_docs:
            # -- camelot a réussi à extraire des tableaux : on les ajoute à la liste
            docs.extend(camelot_docs)
        else:
            # -- camelot n'a rien trouvé ou a échoué : on passe au fallback pdfplumber
            logger.info(f"Fallback pdfplumber pour {pdf_path.name}")
            docs.extend(self._extract_with_pdfplumber(pdf_path))

        logger.info(f"Tableaux extraits : {len(docs)} depuis {pdf_path.name}")
        return docs

    def _extract_with_camelot(self, pdf_path: Path) -> List[Document]:
        """Extraction via camelot (lattice puis stream)."""
        # -- Cette méthode tente d'extraire les tableaux en utilisant la bibliothèque camelot.
        # -- camelot est spécialisé dans l'extraction de tableaux PDF et offre deux modes :
        # --   - "lattice" : détecte les lignes de bordure du tableau (traits horizontaux et
        # --     verticaux) via le traitement d'image. Très fiable pour les tableaux avec grille.
        # --   - "stream" : analyse l'espacement entre les caractères pour inférer la structure
        # --     des colonnes. Utile pour les tableaux sans bordures visibles.
        # -- On essaie d'abord lattice (plus fiable), puis stream si lattice ne donne rien.
        # -- Si camelot n'est pas installé, on retourne une liste vide silencieusement.
        try:
            import camelot
        except ImportError:
            # -- camelot n'est pas installé dans l'environnement : on log un avertissement
            # -- et on retourne une liste vide pour déclencher le fallback pdfplumber.
            logger.warning("camelot non installé, skip")
            return []

        docs: List[Document] = []
        filename = pdf_path.name

        # -- Boucle sur les deux modes d'extraction (lattice d'abord, puis stream)
        for flavor in ("lattice", "stream"):
            try:
                # -- read_pdf lit toutes les pages du PDF ("all") et extrait les tableaux
                # -- selon le mode (flavor) spécifié. Retourne un objet TableList.
                tables = camelot.read_pdf(
                    str(pdf_path),
                    pages="all",
                    flavor=flavor,
                )
                logger.debug(f"camelot {flavor}: {tables.n} tableaux trouvés")

                # -- Itération sur chaque tableau détecté par camelot
                for i, table in enumerate(tables):
                    # -- Vérification du score de précision : camelot attribue un score
                    # -- de 0 à 100 indiquant sa confiance dans la qualité de l'extraction.
                    # -- On rejette les tableaux en dessous du seuil configuré.
                    if table.accuracy < self.accuracy_threshold:
                        continue

                    # -- table.df contient le DataFrame pandas du tableau extrait
                    df = table.df
                    # -- Vérification des dimensions minimales : on rejette les tableaux
                    # -- trop petits (moins de min_rows lignes ou min_cols colonnes)
                    # -- car ils sont probablement des faux positifs.
                    if df.shape[0] < self.min_rows or df.shape[1] < self.min_cols:
                        continue

                    # -- Détection heuristique de l'en-tête : si la première ligne du
                    # -- tableau ressemble à un en-tête (majoritairement du texte, pas
                    # -- des chiffres), on l'utilise comme noms de colonnes du DataFrame
                    # -- et on la supprime des données pour éviter la duplication.
                    if self._looks_like_header(df.iloc[0].tolist()):
                        df.columns = df.iloc[0]
                        df = df[1:].reset_index(drop=True)

                    # -- Génération d'un titre descriptif incluant le numéro du tableau
                    # -- et la page d'origine pour faciliter le repérage dans le document.
                    title = f"Tableau {i+1} (page {table.page})"
                    # -- Conversion du DataFrame en markdown structuré
                    markdown = _df_to_markdown(df, title=title)

                    # -- Si la conversion markdown a produit une chaîne vide (tableau
                    # -- vide après nettoyage), on passe au tableau suivant.
                    if not markdown:
                        continue

                    # -- Création d'un objet Document LangChain contenant le markdown du
                    # -- tableau et des métadonnées riches pour le contexte RAG :
                    # --   - source / filename : nom du fichier PDF d'origine
                    # --   - document_type : "financial_table" pour le filtrage dans le vectorstore
                    # --   - table_index : index du tableau dans la page (0-based)
                    # --   - page_number : numéro de la page PDF contenant le tableau
                    # --   - extraction_method : "camelot_lattice" ou "camelot_stream"
                    # --   - accuracy : score de confiance de camelot (0-100)
                    # --   - contains_table / contains_numbers : flags booléens pour le filtrage
                    # --   - shape : dimensions du tableau (lignes x colonnes)
                    # --   - ticker_symbols : liste vide, prévue pour enrichissement ultérieur
                    doc = Document(
                        page_content=markdown,
                        metadata={
                            "source": filename,
                            "filename": filename,
                            "document_type": "financial_table",
                            "table_index": i,
                            "page_number": table.page,
                            "extraction_method": f"camelot_{flavor}",
                            "accuracy": table.accuracy,
                            "contains_table": True,
                            "contains_numbers": True,
                            "shape": f"{df.shape[0]}x{df.shape[1]}",
                            "ticker_symbols": [],
                        },
                    )
                    docs.append(doc)

                # -- Si on a trouvé des tableaux avec ce mode (lattice ou stream),
                # -- on retourne immédiatement sans essayer le mode suivant.
                # -- Cela évite les doublons : si lattice a fonctionné, pas besoin de stream.
                if docs:
                    return docs

            except Exception as e:
                # -- En cas d'erreur (fichier corrompu, format non supporté, etc.),
                # -- on log l'erreur en mode debug et on essaie le mode suivant.
                logger.debug(f"camelot {flavor} échec: {e}")
                continue

        # -- Si aucun des deux modes n'a produit de résultat, on retourne une liste vide
        return docs

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Document]:
        """Extraction via pdfplumber (fallback)."""
        # -- Cette méthode est le plan de secours (fallback) utilisé quand camelot
        # -- n'est pas disponible ou n'a pas réussi à extraire de tableaux.
        # -- pdfplumber est une bibliothèque plus généraliste pour l'analyse de PDF.
        # -- Elle détecte les tableaux en analysant l'alignement des caractères et
        # -- des lignes dans le PDF. Bien que moins spécialisée que camelot pour les
        # -- tableaux, elle est plus robuste face aux formats PDF variés.
        # -- De plus, elle permet d'extraire le texte de la page en parallèle, ce qui
        # -- sert à inférer le titre du tableau via _infer_table_title().
        try:
            import pdfplumber
        except ImportError:
            # -- pdfplumber n'est pas installé : on log une erreur (plus grave que
            # -- l'absence de camelot car c'est le dernier recours) et on retourne vide.
            logger.error("pdfplumber non installé")
            return []

        docs: List[Document] = []
        filename = pdf_path.name

        try:
            # -- Ouverture du fichier PDF via le context manager de pdfplumber
            # -- (assure la fermeture propre du fichier même en cas d'erreur)
            with pdfplumber.open(pdf_path) as pdf:
                # -- Itération sur chaque page du PDF (numérotation à partir de 1
                # -- pour correspondre aux numéros de pages humainement lisibles)
                for page_num, page in enumerate(pdf.pages, start=1):
                    # -- extract_tables() retourne une liste de tableaux détectés sur la page.
                    # -- Chaque tableau est une liste de listes (lignes de cellules).
                    tables = page.extract_tables()
                    # -- extract_text() retourne le texte brut de la page, utilisé pour
                    # -- inférer le titre du tableau. Si aucun texte n'est extractible,
                    # -- on utilise une chaîne vide.
                    page_text = page.extract_text() or ""

                    # -- Itération sur chaque tableau trouvé sur cette page
                    for table_idx, table_data in enumerate(tables):
                        # -- Vérification de base : le tableau ne doit pas être None ou vide,
                        # -- et il doit avoir au moins min_rows lignes pour être considéré valide.
                        if not table_data or len(table_data) < self.min_rows:
                            continue

                        try:
                            # -- Conversion des données brutes (liste de listes) en DataFrame
                            # -- pandas pour faciliter le traitement et la conversion en markdown.
                            df = pd.DataFrame(table_data)
                            # -- Vérification du nombre minimum de colonnes
                            if df.shape[1] < self.min_cols:
                                continue

                            # -- Même heuristique de détection d'en-tête que pour camelot :
                            # -- si la première ligne semble être un en-tête (peu de valeurs
                            # -- numériques), on l'utilise comme noms de colonnes.
                            if self._looks_like_header(df.iloc[0].tolist()):
                                # -- Conversion des valeurs de la première ligne en noms de colonnes.
                                # -- Si une cellule est vide/None, on génère un nom par défaut
                                # -- "Col_0", "Col_1", etc. pour éviter les colonnes sans nom.
                                df.columns = [
                                    str(v) if v else f"Col_{idx}"
                                    for idx, v in enumerate(df.iloc[0])
                                ]
                                # -- Suppression de la première ligne (devenue les en-têtes)
                                # -- et réinitialisation de l'index du DataFrame.
                                df = df[1:].reset_index(drop=True)

                            # -- Tentative d'inférer le titre du tableau à partir du texte
                            # -- de la page (recherche de mots-clés financiers, etc.)
                            title = _infer_table_title(page_text)
                            # -- Conversion du DataFrame en markdown, avec le titre inféré
                            # -- et le numéro de page ajouté entre parenthèses.
                            markdown = _df_to_markdown(df, title=f"{title} (p.{page_num})")

                            # -- Si le markdown est vide (tableau vide après traitement), on passe
                            if not markdown:
                                continue

                            # -- Création du Document LangChain avec les métadonnées.
                            # -- Les métadonnées sont similaires à celles de camelot, sauf :
                            # --   - extraction_method est "pdfplumber" au lieu de "camelot_*"
                            # --   - pas de champ "accuracy" car pdfplumber ne fournit pas
                            # --     de score de confiance pour les tableaux extraits.
                            doc = Document(
                                page_content=markdown,
                                metadata={
                                    "source": filename,
                                    "filename": filename,
                                    "document_type": "financial_table",
                                    "table_index": table_idx,
                                    "page_number": page_num,
                                    "extraction_method": "pdfplumber",
                                    "contains_table": True,
                                    "contains_numbers": True,
                                    "shape": f"{df.shape[0]}x{df.shape[1]}",
                                    "ticker_symbols": [],
                                },
                            )
                            docs.append(doc)

                        except Exception as e:
                            # -- En cas d'erreur sur un tableau individuel (données malformées,
                            # -- cellules fusionnées problématiques, etc.), on log en debug
                            # -- et on continue avec le tableau suivant sans interrompre le processus.
                            logger.debug(f"Table parse error p.{page_num}: {e}")
                            continue

        except Exception as e:
            # -- Erreur globale sur le fichier PDF (fichier corrompu, protégé par mot de passe,
            # -- format non supporté, etc.). On log en erreur car c'est le dernier recours.
            logger.error(f"pdfplumber échec {filename}: {e}")

        return docs

    def _looks_like_header(self, row: List[Any]) -> bool:
        """Heuristique pour détecter si une ligne est un en-tête."""
        # -- Cette heuristique détermine si la première ligne d'un tableau correspond à
        # -- un en-tête (noms de colonnes) plutôt qu'à une ligne de données.
        # -- Le principe est simple : dans un tableau financier, les en-têtes contiennent
        # -- principalement du texte ("Revenus", "2023", "Q1", "Variation"), tandis que
        # -- les lignes de données contiennent principalement des chiffres (1 234, -5.6%, etc.).
        # -- La règle appliquée : si MOINS de 50% des cellules non vides sont purement
        # -- numériques, la ligne est considérée comme un en-tête.

        # -- Si la ligne est vide, ce n'est pas un en-tête
        if not row:
            return False
        # -- Filtrage des cellules non vides/None et conversion en chaînes
        str_row = [str(c) for c in row if c]
        # -- Si toutes les cellules sont vides, ce n'est pas un en-tête
        if not str_row:
            return False
        # -- Comptage des cellules qui correspondent au pattern numérique :
        # -- ^\d[\d\s,\.\-\(\)]+$ reconnaît les formats numériques courants :
        # --   - "1234" : entier simple
        # --   - "1 234,56" : nombre avec séparateur de milliers et décimale
        # --   - "(1,234.56)" : notation anglo-saxonne avec parenthèses pour les négatifs
        # --   - "-5.6" : nombre négatif avec point décimal
        # --   - "1,234,567" : grands nombres avec séparateurs de milliers
        numeric_count = sum(1 for c in str_row if re.match(r'^[\d\s,\.\-\(\)]+$', c))
        # -- Si moins de 50% des cellules sont numériques, c'est probablement un en-tête
        return numeric_count < len(str_row) * 0.5

    def tables_to_context(self, docs: List[Document]) -> str:
        """Agrège les tableaux extraits en un contexte textuel pour le LLM."""
        # -- Cette méthode prend en entrée la liste de Documents LangChain contenant
        # -- les tableaux extraits et les combine en une seule chaîne de caractères
        # -- formatée, prête à être injectée comme contexte dans le prompt d'un LLM.
        # -- Le format de sortie est :
        # --   - Un titre global indiquant le nombre de tableaux trouvés
        # --   - Pour chaque tableau : un séparateur "---", la source et le numéro de page
        # --     en italique, puis le contenu markdown du tableau.
        # -- Si aucun tableau n'a été extrait, un message explicite est retourné.
        if not docs:
            return "Aucun tableau financier trouvé."

        # -- En-tête global indiquant le nombre total de tableaux extraits, en gras markdown
        parts = [f"**Tableaux financiers extraits ({len(docs)} tableaux)**\n"]
        for doc in docs:
            meta = doc.metadata
            # -- Séparateur horizontal markdown suivi des informations de source :
            # -- nom du fichier et numéro de page pour que le LLM puisse référencer
            # -- l'origine exacte de chaque tableau dans ses réponses.
            parts.append(
                f"\n---\n*Source: {meta.get('filename', 'unknown')}, "
                f"Page {meta.get('page_number', '?')}*\n"
            )
            # -- Ajout du contenu markdown du tableau
            parts.append(doc.page_content)

        # -- Jointure de toutes les parties avec des sauts de ligne
        return "\n".join(parts)
