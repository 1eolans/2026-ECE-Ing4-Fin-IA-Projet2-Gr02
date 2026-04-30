"""
FinRAG — Interface Streamlit Professionnelle
Design : fond blanc, typographie Barlow, bordures carrees, palette noir/blanc/vert.
Inspire d'un design enterprise epure.

Ce fichier constitue le point d'entree de l'interface utilisateur Streamlit.
Il orchestre l'affichage des 4 vues principales :
  - Hub d'Ingestion (upload + filtres)
  - Base Documentaire (liste des documents indexes)
  - Assistant IA (chat RAG avec panel de sources)
  - Analytics (metriques, historique, evaluation RAGAS)
"""

from __future__ import annotations

import sys
import time
import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import plotly.graph_objects as go
from loguru import logger

# -- Ajout du repertoire racine au path pour pouvoir importer les modules src.*
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# -- Import de la configuration centralisee (cles API, parametres RAG, etc.)
from src.config import settings

# ─── Page config ─────────────────────────────────────────────────────────────
# Configuration globale de la page Streamlit :
# - layout "wide" pour utiliser toute la largeur
# - sidebar masquee car non utilisee (navigation par onglets)
st.set_page_config(
    page_title="FinRAG — Intelligence Financiere",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": "FinRAG — Systeme RAG Financier"},
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
# Feuille de style personnalisee injectee via st.markdown.
# Utilise les polices Barlow (texte), Barlow Condensed (titres/labels)
# et JetBrains Mono (chiffres/metriques).

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Barlow+Condensed:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables de couleurs ── */
:root {
    --white:     #FFFFFF;
    --black:     #0C0C0C;
    --g50:       #FAFAFA;
    --g100:      #F5F5F5;
    --g200:      #EBEBEB;
    --g300:      #D4D4D4;
    --g400:      #A3A3A3;
    --g600:      #5C5C5C;
    --g800:      #2C2C2C;
    --green:     #16A34A;
    --green-bg:  #F0FDF4;
    --green-txt: #15803D;
    --amber:     #D97706;
    --amber-bg:  #FFFBEB;
    --red:       #DC2626;
    --red-bg:    #FEF2F2;
    --border:    #EBEBEB;
    --shadow:    0 1px 3px rgba(0,0,0,0.06);
}

/* ── Reset Streamlit (suppression des elements par defaut) ── */
html, body, [class*="css"] { font-family: 'Barlow', sans-serif !important; }
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.reportview-container .main .block-container { padding-top: 0 !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: var(--white) !important; border-bottom: 1px solid var(--border); height: 0 !important; }

/* ── Fond de l'application ── */
.stApp { background: var(--white) !important; color: var(--black) !important; }
.stApp > div { background: var(--white) !important; }

/* ── Sidebar masquee (navigation par onglets uniquement) ── */
[data-testid="stSidebar"] { display: none !important; }

/* ── Boutons (style carre, bordure noire, hover inverse) ── */
.stButton > button {
    border-radius: 0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 9px 18px !important;
    border: 1px solid var(--black) !important;
    background: var(--white) !important;
    color: var(--black) !important;
    transition: background 0.12s ease, color 0.12s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: var(--black) !important;
    color: var(--white) !important;
    border-color: var(--black) !important;
}
/* Bouton primaire : fond noir par defaut */
.stButton > button[kind="primary"] {
    background: var(--black) !important;
    color: var(--white) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--g800) !important;
}
.stButton > button:focus { box-shadow: none !important; }

/* ── Champs de saisie (text input, text area) ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 0 !important;
    border: 1px solid var(--g200) !important;
    background: var(--white) !important;
    color: var(--black) !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 14px !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--black) !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div, .stTextArea > div > div { border-radius: 0 !important; }

/* ── Selectbox (style carre) ── */
.stSelectbox > div > div {
    border-radius: 0 !important;
    border: 1px solid var(--g200) !important;
    background: var(--white) !important;
}

/* ── Onglets (navigation principale en haut) ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--white) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 32px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--g400) !important;
    padding: 14px 28px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    margin-bottom: -1px !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--g800) !important; background: transparent !important; }
.stTabs [aria-selected="true"] {
    color: var(--black) !important;
    border-bottom: 2px solid var(--black) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── Cartes de metriques ── */
[data-testid="stMetric"] {
    background: var(--g50) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] { font-family: 'Barlow Condensed', sans-serif !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--g400) !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 24px !important; font-weight: 500 !important; color: var(--black) !important; }

/* ── Expander (accordeon) ── */
.streamlit-expander { border: 1px solid var(--border) !important; border-radius: 0 !important; }
.streamlit-expander > div:first-child { border-radius: 0 !important; background: var(--g50) !important; border-bottom: 1px solid var(--border) !important; }

/* ── Zone d'upload de fichier ── */
.stFileUploader > div {
    border-radius: 0 !important;
    border: 1px dashed var(--g300) !important;
    background: var(--g50) !important;
    padding: 16px !important;
}
.stFileUploader > div:hover { border-color: var(--black) !important; }

/* ── Toggle switch ── */
.stToggle { font-family: 'Barlow', sans-serif !important; font-size: 13px !important; }

/* ── Spinner de chargement ── */
.stSpinner > div { border-color: var(--black) transparent transparent transparent !important; }

/* ── Barres de defilement personnalisees ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--g100); }
::-webkit-scrollbar-thumb { background: var(--g300); }
::-webkit-scrollbar-thumb:hover { background: var(--g400); }

/* ── Alertes / warnings ── */
.stAlert { border-radius: 0 !important; border-left-width: 3px !important; }

/* ── Composants FinRAG personnalises ── */

/* Barre superieure de chaque vue */
.top-bar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 28px 32px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--white);
}
/* Label "VUE 0X" au-dessus du titre */
.view-meta {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--g400);
    margin-bottom: 6px;
}
/* Titre principal de la vue */
.view-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 26px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--black);
    margin: 0;
    text-transform: uppercase;
}
/* Partie droite de la barre superieure */
.top-bar-right { display: flex; align-items: center; gap: 20px; }
.doc-counter { text-align: right; }
.doc-count-num { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 500; color: var(--black); display: block; line-height: 1; }
.doc-count-label { font-size: 9px; color: var(--g400); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Barlow Condensed', sans-serif; margin-top: 2px; display: block; }
/* Badge vert "Base Active" */
.active-badge {
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--green-bg);
    border: 1px solid #BBF7D0;
    padding: 5px 10px;
}
.badge-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); }
.badge-text { font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--green-txt); font-family: 'Barlow Condensed', sans-serif; }

/* Zone de chat (messages) */
.chat-area { padding: 24px 32px; }

/* Style des messages utilisateur et agent */
.msg-wrap { margin-bottom: 28px; }
.msg-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 8px; }
.msg-sender-user { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--black); }
.msg-sender-agent { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--g400); }
.msg-time { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); }
.msg-user { background: var(--g100); padding: 14px 16px; border-left: 2px solid var(--black); font-size: 14px; color: var(--black); line-height: 1.5; }
.msg-agent { background: var(--g50); padding: 16px 18px; border-left: 2px solid var(--g200); font-size: 14px; color: var(--black); line-height: 1.6; }

/* Badges de confiance (vert/jaune/rouge) */
.conf-high { display: inline-flex; align-items: center; gap: 4px; background: var(--green-bg); padding: 2px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--green-txt); font-family: 'Barlow Condensed', sans-serif; }
.conf-medium { display: inline-flex; align-items: center; gap: 4px; background: var(--amber-bg); padding: 2px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--amber); font-family: 'Barlow Condensed', sans-serif; }
.conf-low { display: inline-flex; align-items: center; gap: 4px; background: var(--red-bg); padding: 2px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--red); font-family: 'Barlow Condensed', sans-serif; }

/* Metriques en bas d'un message agent (temps, tokens, sources) */
.msg-metrics { display: flex; gap: 20px; padding: 8px 0; border-top: 1px solid var(--border); margin-top: 10px; }
.msg-metric { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); }

/* Panel de source documentaire (colonne droite du chat) */
.source-panel-header {
    padding: 20px 20px 14px;
    border-bottom: 1px solid var(--border);
}
.source-panel-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--g400);
    margin-bottom: 8px;
}
.source-filename {
    font-size: 12px;
    font-weight: 500;
    color: var(--black);
    display: flex;
    align-items: center;
    gap: 6px;
}
.source-section {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--g400);
}
/* Extrait du document source */
.source-excerpt {
    padding: 14px 20px;
    font-size: 12px;
    line-height: 1.7;
    color: var(--g600);
    border-left: 2px solid var(--g200);
    margin: 12px 16px;
    background: var(--g50);
}
/* Navigation entre sources */
.source-nav {
    padding: 8px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--g600);
}
/* Barre de score de pertinence */
.source-score-bar { margin: 12px 20px; }
.score-label { font-size: 10px; color: var(--g400); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Barlow Condensed', sans-serif; margin-bottom: 4px; }
.score-val { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--black); }

/* Item de citation cliquable */
.citation-item {
    padding: 10px 14px;
    border: 1px solid var(--border);
    margin-bottom: 6px;
    cursor: pointer;
    transition: border-color 0.1s ease;
}
.citation-item:hover { border-color: var(--black); }
.citation-source-name { font-size: 12px; font-weight: 600; color: var(--black); }
.citation-meta { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); margin-top: 2px; }
.citation-excerpt { font-size: 11px; color: var(--g600); margin-top: 6px; line-height: 1.5; }

/* Zone de saisie en bas du chat */
.input-area {
    padding: 20px 32px;
    border-top: 1px solid var(--border);
    background: var(--white);
}
.input-hint { font-size: 11px; color: var(--g400); margin-top: 6px; font-family: 'Barlow', sans-serif; }

/* Sous-requete decomposee */
.subq-item {
    padding: 7px 12px;
    border-left: 2px solid var(--g200);
    font-size: 12px;
    color: var(--g600);
    margin-bottom: 5px;
    background: var(--g50);
}
.subq-label { font-family: 'Barlow Condensed', sans-serif; font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g400); display: block; margin-bottom: 3px; }

/* Etat vide (aucun document, aucune question posee) */
.empty-state { text-align: center; padding: 60px 20px; }
.empty-icon { font-size: 28px; margin-bottom: 12px; opacity: 0.3; }
.empty-title { font-family: 'Barlow Condensed', sans-serif; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--g400); margin-bottom: 6px; }
.empty-sub { font-size: 13px; color: var(--g400); }

/* Exemples de questions affichees par defaut */
.example-label { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g400); margin-bottom: 10px; }
.example-grid { display: flex; flex-direction: column; gap: 6px; }

/* Zone d'upload dans la vue Ingestion */
.upload-section { padding: 12px 16px; }
.upload-label { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g400); margin-bottom: 8px; display: block; }

/* Ligne de document dans la liste documentaire */
.doc-row { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.doc-row:hover { background: var(--g50); }
.doc-row-left { display: flex; align-items: center; gap: 10px; }
/* Pilule de type de document (PDF, CSV, JSON, XLSX, TABLE) */
.doc-type-pill { font-family: 'Barlow Condensed', sans-serif; font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 2px 6px; border: 1px solid; }
.pill-pdf { color: var(--black); border-color: var(--black); }
.pill-csv { color: var(--green-txt); border-color: var(--green); }
.pill-json { color: #7C3AED; border-color: #7C3AED; }
.pill-xlsx { color: var(--amber); border-color: var(--amber); }
.pill-table { color: var(--g600); border-color: var(--g300); }
.doc-name { font-size: 13px; color: var(--black); font-weight: 500; }
.doc-chunks { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--g400); }
.doc-date { font-size: 11px; color: var(--g400); }

/* Section Analytics */
.analytics-section { padding: 0 32px 32px; }
.section-heading { font-family: 'Barlow Condensed', sans-serif; font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--g400); padding: 24px 0 12px; border-top: 1px solid var(--border); margin-top: 24px; }
/* Ligne d'historique de requete */
.history-row { padding: 14px 16px; border: 1px solid var(--border); margin-bottom: 6px; }
.history-q { font-size: 13px; color: var(--black); margin-bottom: 6px; font-weight: 500; }
.history-meta { display: flex; gap: 16px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); }

/* Separateur horizontal dans le contenu */
.content-separator { border: none; border-top: 1px solid var(--border); margin: 0 32px; }

/* Carte de metrique RAGAS */
.ragas-card { padding: 20px; border: 1px solid var(--border); text-align: center; background: var(--g50); }
.ragas-score { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 500; color: var(--black); display: block; }
.ragas-label { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--g400); margin-top: 4px; display: block; }
.ragas-bar { height: 2px; background: var(--g200); margin-top: 12px; }
.ragas-fill { height: 100%; background: var(--black); }

/* Separateur dans la sidebar (conserve pour d'eventuelles reutilisations) */
.sidebar-divider { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
</style>
"""

# -- Injection du CSS dans la page
st.markdown(CSS, unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
# Initialisation des variables de session Streamlit.
# Ces variables persistent entre les reruns (interactions utilisateur).

def _init_state():
    """Initialise les variables de session avec des valeurs par defaut."""
    defaults = {
        "messages": [],                # Historique des messages du chat
        "query_history": [],           # Historique des requetes (pour analytics)
        "total_tokens": 0,             # Compteur cumule de tokens utilises
        "total_queries": 0,            # Compteur cumule de requetes envoyees
        "filters": {                   # Filtres de recherche actifs
            "doc_type": "Tous",
            "ticker": "",
            "date_from": None,
            "date_to": None,
        },
        "use_decomposition": True,     # Active/desactive la decomposition de requetes
        "last_citations": [],          # Citations de la derniere reponse (pour le panel source)
        "last_sub_queries": [],        # Sous-requetes de la derniere decomposition
        "active_citation_idx": 0,      # Index de la citation active dans le panel source
        "active_view": "assistant",    # Vue active (upload/documents/assistant/analytics)
    }
    # On n'ecrase pas une valeur deja existante (permet de garder l'etat entre reruns)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -- Appel immediat a l'initialisation
_init_state()


# ─── System init ─────────────────────────────────────────────────────────────
# Initialisation du systeme RAG complet (vector store, retriever, reranker, generator, agent).
# Le decorateur @st.cache_resource garantit que le systeme n'est instancie qu'une seule fois
# et reutilise entre les reruns de Streamlit.

@st.cache_resource(show_spinner=False)
def _build_system():
    """
    Construit et retourne le systeme RAG complet.

    Returns:
        dict avec les cles : vs (vector store), retriever, reranker, generator, agent, ok (bool).
        En cas d'erreur : dict avec ok=False et error (str).
    """
    try:
        # -- Import des composants du pipeline RAG
        from src.retrieval.vector_store import FinancialVectorStore
        from src.retrieval.retriever import HybridFinancialRetriever
        from src.retrieval.reranker import CrossEncoderReRanker
        from src.generation.generator import FinancialAnswerGenerator
        from src.agents.rag_agent import FinancialRAGAgent

        # -- Instanciation de chaque composant
        vs = FinancialVectorStore()
        retriever = HybridFinancialRetriever(
            vector_store=vs,
            top_k=settings.TOP_K_RETRIEVAL,
            time_decay_factor=settings.TIME_DECAY_FACTOR,
        )
        reranker = CrossEncoderReRanker(top_k=settings.TOP_K_RERANK)
        generator = FinancialAnswerGenerator()
        agent = FinancialRAGAgent(vs, retriever, reranker, generator)

        return {"vs": vs, "retriever": retriever, "reranker": reranker,
                "generator": generator, "agent": agent, "ok": True}
    except Exception as e:
        logger.error(f"Init erreur: {e}")
        return {"ok": False, "error": str(e)}


# ─── Helpers ─────────────────────────────────────────────────────────────────
# Fonctions utilitaires pour le formatage et l'affichage.

def _fmt_time(seconds: float) -> str:
    """Formate une duree en millisecondes ou secondes selon la valeur."""
    return f"{seconds*1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def _now() -> str:
    """Retourne l'heure courante au format HH:MM:SS."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def _conf_badge(score: float) -> str:
    """
    Genere le HTML d'un badge de confiance colore selon le score.
    - >= 0.7 : vert (haute confiance)
    - >= 0.4 : jaune/ambre (confiance moyenne)
    - < 0.4  : rouge (faible confiance)
    """
    if score >= 0.7:
        return f'<span class="conf-high"><span style="width:5px;height:5px;border-radius:50%;background:#16A34A;display:inline-block;"></span>{score:.0%}</span>'
    elif score >= 0.4:
        return f'<span class="conf-medium"><span style="width:5px;height:5px;border-radius:50%;background:#D97706;display:inline-block;"></span>{score:.0%}</span>'
    else:
        return f'<span class="conf-low"><span style="width:5px;height:5px;border-radius:50%;background:#DC2626;display:inline-block;"></span>{score:.0%}</span>'


def _pill(doc_type: str) -> str:
    """
    Retourne le HTML d'une pilule de type de document (PDF, CSV, NEWS, etc.).
    La couleur depend du type de document.
    """
    mapping = {
        "annual_report":    ("PDF", "pill-pdf"),
        "quarterly_report": ("PDF", "pill-pdf"),
        "market_overview":  ("PDF", "pill-pdf"),
        "financial_table":  ("TABLE", "pill-table"),
        "news_article":     ("NEWS", "pill-json"),
        "csv_data":         ("CSV", "pill-csv"),
        "excel_data":       ("XLSX", "pill-xlsx"),
    }
    label, css = mapping.get(doc_type, ("DOC", "pill-table"))
    return f'<span class="doc-type-pill {css}">{label}</span>'


def _serialize_citation(c: Any) -> Dict[str, Any]:
    """
    Convertit un objet Citation (dataclass) en dictionnaire serialisable.
    Supporte les dataclasses, les dicts natifs et les objets avec __dict__.
    """
    try:
        return asdict(c)
    except TypeError:
        return c if isinstance(c, dict) else vars(c)


def _index_samples(system: dict, samples_dir: Path):
    """
    Indexe les documents d'exemple situes dans le repertoire samples_dir.
    Charge les fichiers, extrait les tableaux des PDFs, chunke et indexe dans ChromaDB.
    """
    from src.ingestion.document_loader import FinancialDocumentLoader
    from src.ingestion.table_extractor import FinancialTableExtractor
    from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

    loader = FinancialDocumentLoader()
    all_docs = []

    # -- Parcours recursif de tous les fichiers supportes
    for fp in samples_dir.rglob("*"):
        if fp.suffix.lower() in [".pdf", ".csv", ".json", ".xlsx", ".txt"]:
            all_docs.extend(loader.load(fp))
            # -- Extraction supplementaire des tableaux pour les PDFs
            if fp.suffix.lower() == ".pdf":
                all_docs.extend(FinancialTableExtractor().extract_tables_from_pdf(fp))

    if not all_docs:
        st.warning("Aucun document trouve. Lancez : python data/generate_samples.py")
        return

    # -- Chunking hybride (auto-detection table/texte)
    chunks = IntelligentFinancialChunker(strategy=ChunkingStrategy.HYBRID).chunk_documents(all_docs)
    # -- Indexation dans le vector store ChromaDB
    added = system["vs"].add_documents(chunks, show_progress=False)
    # -- Invalidation du cache BM25 apres ajout de nouveaux documents
    system["retriever"].invalidate_bm25_cache()

    n = len(set(d.metadata.get("filename", "") for d in all_docs))
    st.success(f"Done — {added} chunks indexes — {n} fichiers")


def _upload_file(uploaded_file, system: dict) -> bool:
    """
    Traite un fichier uploade par l'utilisateur :
    1. Sauvegarde sur disque dans data/uploads/
    2. Charge et extrait le contenu (+ tableaux pour les PDFs)
    3. Chunke et indexe dans ChromaDB
    4. Invalide le cache BM25

    Returns:
        True si l'indexation a reussi, False sinon.
    """
    from src.ingestion.document_loader import FinancialDocumentLoader
    from src.ingestion.table_extractor import FinancialTableExtractor
    from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

    # -- Sauvegarde du fichier sur disque
    upload_dir = ROOT_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    fp = upload_dir / uploaded_file.name
    fp.write_bytes(uploaded_file.getbuffer())

    # -- Chargement du document
    loader = FinancialDocumentLoader()
    docs = loader.load(fp)

    # -- Extraction des tableaux pour les PDFs
    if fp.suffix.lower() == ".pdf":
        docs.extend(FinancialTableExtractor().extract_tables_from_pdf(fp))

    if not docs:
        st.error(f"Aucun contenu extrait de {uploaded_file.name}")
        return False

    # -- Chunking et indexation
    chunks = IntelligentFinancialChunker(strategy=ChunkingStrategy.HYBRID).chunk_documents(docs)
    added = system["vs"].add_documents(chunks, show_progress=False)
    system["retriever"].invalidate_bm25_cache()
    st.success(f"Done — {added} chunks indexes — {uploaded_file.name}")
    return True


# ─── Top bar ─────────────────────────────────────────────────────────────────
# Barre superieure affichee en haut de chaque vue, avec le titre de la vue
# et les indicateurs de statut (nombre de documents, badge actif/inactif).

# -- Mapping vue → (meta label, titre)
VIEW_META = {
    "upload":    ("VUE 01 — INGESTION",    "Hub d'Ingestion"),
    "documents": ("VUE 02 — DOCUMENTAIRE", "Base Documentaire"),
    "assistant": ("VUE 03 — INTELLIGENCE", "Agent FinRAG"),
    "analytics": ("VUE 04 — ANALYTIQUE",   "Analytics"),
}

def render_top_bar(system: dict):
    """Affiche la barre superieure de la vue active avec les metriques du systeme."""
    view = st.session_state.active_view
    meta, title = VIEW_META.get(view, ("", ""))

    # -- Recuperation des stats du vector store
    chunks = 0
    sources = 0
    if system.get("ok"):
        stats = system["vs"].get_stats()
        chunks = stats.get("total_chunks", 0)
        sources = stats.get("total_sources", 0)

    # -- Badge vert si des documents sont indexes, gris sinon
    badge_html = (
        '<div class="active-badge"><span class="badge-dot"></span>'
        '<span class="badge-text">Base Active</span></div>'
        if chunks > 0
        else '<div style="font-size:10px;color:#A3A3A3;font-family:\'Barlow Condensed\';'
             'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">En Attente</div>'
    )

    # -- Rendu HTML de la barre superieure
    st.markdown(f"""
    <div class="top-bar">
        <div>
            <div class="view-meta">{meta}</div>
            <h1 class="view-title">{title}</h1>
        </div>
        <div class="top-bar-right">
            <div class="doc-counter">
                <span class="doc-count-num">{sources}</span>
                <span class="doc-count-label">Documents indexes</span>
            </div>
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Source panel ─────────────────────────────────────────────────────────────
# Panneau lateral affiche a droite du chat dans la vue Assistant.
# Montre le detail de la source documentaire utilisee pour la reponse courante.

def render_source_panel():
    """
    Affiche le panneau de source documentaire dans la vue Assistant.
    Montre l'extrait pertinent, le score de pertinence et permet de naviguer
    entre les differentes citations.
    """
    citations = st.session_state.last_citations
    if not citations:
        # -- Etat vide : aucune citation disponible
        st.markdown("""
        <div style="padding: 32px 20px; text-align: center;">
            <div style="font-size:24px; margin-bottom:10px; opacity:0.2;">▪</div>
            <div style="font-family:'Barlow Condensed';font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;">Source documentaire</div>
            <div style="font-size:12px;color:#A3A3A3;margin-top:6px;">Posez une question pour voir les sources utilisees</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # -- Affichage de la citation active
    active_idx = min(st.session_state.active_citation_idx, len(citations) - 1)
    cit = citations[active_idx]

    # -- Extraction des metadonnees de la citation
    source = cit.get("source_file", "unknown")
    page = cit.get("page_number")
    date = cit.get("date", "")
    excerpt = cit.get("excerpt", "")
    score = cit.get("relevance_score", 0)

    page_str = f"p. {page}" if page else ""
    n = len(citations)

    # -- Rendu HTML du panel de source
    st.markdown(f"""
    <div style="border-left: 1px solid #EBEBEB; height: 100%;">
        <div class="source-panel-header">
            <div class="source-panel-label">Source documentaire</div>
            <div class="source-filename">▪ {source}</div>
        </div>
        <div class="source-section">Extrait pertinent</div>
        <div style="font-family:'JetBrains Mono';font-size:10px;color:#A3A3A3;padding:6px 20px 0;display:flex;justify-content:space-between;">
            <span>{page_str}{" · " + date if date else ""}</span>
            <span>{active_idx + 1} / {n}</span>
        </div>
        <div class="source-excerpt">{excerpt}</div>
        <div class="source-score-bar">
            <div class="score-label">Score de pertinence</div>
            <div class="score-val">{score:.3f}</div>
            <div style="height:2px;background:#EBEBEB;margin-top:6px;">
                <div style="height:100%;background:#0C0C0C;width:{min(score, 1)*100:.0f}%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- Boutons de navigation entre citations (affiche seulement s'il y en a plus d'une)
    if n > 1:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Prec.", key="src_prev", width="stretch",
                         disabled=active_idx == 0):
                st.session_state.active_citation_idx = max(0, active_idx - 1)
                st.rerun()
        with c2:
            if st.button("Suiv.", key="src_next", width="stretch",
                         disabled=active_idx >= n - 1):
                st.session_state.active_citation_idx = min(n - 1, active_idx + 1)
                st.rerun()


# ─── Intelligence panel (sub-queries + stats) ────────────────────────────────
# Panneau affiche sous le panel de source dans la vue Assistant.
# Montre les sous-requetes decomposees et les statistiques de la session.

def render_intel_sidebar():
    """Affiche les sous-requetes decomposees et les stats de session."""
    sqs = st.session_state.last_sub_queries

    # -- Affichage des sous-requetes si presentes
    if sqs:
        st.markdown("""
        <div style="padding: 16px 20px 8px;">
            <div style="font-family:'Barlow Condensed';font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;margin-bottom:8px;">Requetes decomposees</div>
        </div>
        """, unsafe_allow_html=True)
        for sq in sqs:
            # -- Troncature a 80 caracteres pour l'affichage
            st.markdown(
                f'<div class="subq-item"><span class="subq-label">Sous-requete</span>'
                f'{sq[:80]}{"..." if len(sq) > 80 else ""}</div>',
                unsafe_allow_html=True,
            )

    # -- Statistiques de la session courante (nombre de requetes, tokens utilises)
    st.markdown(f"""
    <div style="padding: 12px 20px; border-top: 1px solid #EBEBEB; margin-top: 12px;">
        <div style="font-family:'Barlow Condensed';font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;margin-bottom:8px;">Session</div>
        <div style="font-family:'JetBrains Mono';font-size:11px;color:#5C5C5C;">
            {st.session_state.total_queries} requete{"s" if st.session_state.total_queries != 1 else ""}<br>
            {st.session_state.total_tokens:,} tokens
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Message rendering ────────────────────────────────────────────────────────
# Rendu d'un message individuel dans le chat (utilisateur ou agent).

def _render_message(msg: dict):
    """
    Affiche un message dans la zone de chat.

    Args:
        msg: dict avec les cles role, content, meta (optionnel), ts (timestamp).
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    meta = msg.get("meta", {})
    ts = msg.get("ts", "")

    if role == "user":
        # -- Message utilisateur : fond gris, bordure noire a gauche
        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-header">
                <span class="msg-sender-user">Utilisateur</span>
                <span class="msg-time">{ts}</span>
            </div>
            <div class="msg-user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # -- Message agent : fond clair, bordure grise, metriques en pied
        conf = meta.get("confidence_score", 0)
        proc = meta.get("processing_time", 0)
        tokens = meta.get("tokens_used", 0)
        n_src = meta.get("context_docs_count", 0)
        conf_b = _conf_badge(conf)

        # -- En-tete du message agent avec badge de confiance
        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-header">
                <span class="msg-sender-agent">Agent IA</span>
                <div style="display:flex;align-items:center;gap:10px;">
                    {conf_b}
                    <span class="msg-time">{ts}</span>
                </div>
            </div>
            <div class="msg-agent">
        """, unsafe_allow_html=True)

        # -- Contenu de la reponse (rendu markdown natif Streamlit)
        st.markdown(content)

        # -- Pied du message avec metriques (temps, tokens, nombre de sources)
        st.markdown(f"""
            <div class="msg-metrics">
                <span class="msg-metric">time {_fmt_time(proc)}</span>
                <span class="msg-metric">tokens {tokens:,}</span>
                <span class="msg-metric">sources {n_src}</span>
            </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Query processing ─────────────────────────────────────────────────────────
# Traitement d'une question utilisateur : envoi au pipeline RAG et stockage
# de la reponse dans l'historique de session.

def _process_query(query: str, system: dict):
    """
    Traite une question utilisateur via le pipeline RAG complet :
    1. Ajoute le message utilisateur a l'historique
    2. Applique les filtres actifs (type doc, ticker, dates)
    3. Appelle l'agent RAG pour obtenir une reponse
    4. Stocke la reponse, les citations et les metriques dans la session

    Args:
        query: Question financiere de l'utilisateur.
        system: Dict contenant les composants du systeme RAG.
    """
    ts = _now()
    # -- Ajout du message utilisateur a l'historique
    st.session_state.messages.append({"role": "user", "content": query, "ts": ts})

    # -- Verification que le systeme est initialise
    if not system.get("ok"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Systeme non initialise.",
            "meta": {}, "ts": _now(),
        })
        return

    # -- Construction des filtres a partir du state
    f = st.session_state.filters
    date_range = (f["date_from"], f["date_to"]) if f.get("date_from") and f.get("date_to") else None

    # -- Mapping du filtre type de document (label UI → types internes)
    doc_type_map = {
        "Rapports": ["annual_report", "quarterly_report", "market_overview"],
        "News": ["news_article"],
        "Tableaux": ["financial_table"],
        "CSV": ["csv_data"],
    }
    document_type = doc_type_map.get(f.get("doc_type", "Tous"))
    ticker = f.get("ticker") or None

    # -- Appel a l'agent RAG avec les filtres
    answer = system["agent"].answer(
        question=query,
        use_decomposition=st.session_state.use_decomposition,
        date_range=date_range,
        document_type=document_type,
        ticker=ticker,
    )

    # -- Serialisation des citations pour stockage en session
    cits = [_serialize_citation(c) for c in answer.citations]

    # -- Ajout de la reponse agent a l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.answer,
        "meta": {
            "confidence_score": answer.confidence_score,
            "processing_time": answer.processing_time,
            "tokens_used": answer.tokens_used,
            "context_docs_count": answer.context_docs_count,
            "citations": cits,
            "sub_queries": answer.sub_queries,
        },
        "ts": _now(),
    })

    # -- Mise a jour des citations et sous-requetes pour le panel source
    st.session_state.last_citations = cits
    st.session_state.last_sub_queries = answer.sub_queries
    st.session_state.active_citation_idx = 0

    # -- Mise a jour des compteurs de session
    st.session_state.total_queries += 1
    st.session_state.total_tokens += answer.tokens_used

    # -- Ajout a l'historique pour le tableau de bord analytics
    st.session_state.query_history.append({
        "question": query,
        "time": time.time(),
        "confidence": answer.confidence_score,
        "processing_time": answer.processing_time,
        "tokens": answer.tokens_used,
        "n_sources": answer.context_docs_count,
    })


# ─── Views ────────────────────────────────────────────────────────────────────
# Les 4 vues principales de l'application :
#   1. view_upload    : Hub d'Ingestion (upload de fichiers + filtres)
#   2. view_documents : Base Documentaire (liste des documents indexes)
#   3. view_assistant : Assistant IA (chat RAG + panel de sources)
#   4. view_analytics : Analytics (metriques, historique, RAGAS)

def view_upload(system: dict):
    """
    Vue 01 — Hub d'Ingestion.
    Permet d'uploader des fichiers, de configurer les filtres de recherche
    et d'activer/desactiver la decomposition de requetes.
    """
    st.markdown('<div style="padding: 32px 32px 0;">', unsafe_allow_html=True)

    # -- Section upload de documents (deplacee depuis l'ancienne sidebar)
    if system.get("ok"):
        st.markdown("""
        <div style="font-family:'Barlow Condensed';font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#A3A3A3;margin-bottom:16px;">
            Importer des documents
        </div>
        """, unsafe_allow_html=True)

        # -- Widget d'upload multi-fichiers (PDF, CSV, XLSX, JSON, TXT)
        uploaded = st.file_uploader(
            "Importer des fichiers",
            type=["pdf", "csv", "xlsx", "json", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            # -- Bouton d'indexation : lance le traitement de tous les fichiers uploades
            if st.button("Indexer les fichiers", width="stretch", type="primary"):
                with st.spinner("Indexation en cours..."):
                    for f in uploaded:
                        _upload_file(f, system)
                st.rerun()

        # -- Bouton d'indexation des documents d'exemple (si le repertoire samples existe)
        samples_dir = ROOT_DIR / "data" / "samples"
        if samples_dir.exists() and list(samples_dir.glob("*.pdf")):
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            if st.button("Indexer les donnees d'exemple", width="stretch"):
                with st.spinner("Indexation..."):
                    _index_samples(system, samples_dir)
                st.rerun()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # -- Section filtres de recherche
    st.markdown("""
    <div style="font-family:'Barlow Condensed';font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#A3A3A3;margin-bottom:16px;">
        Filtres de recherche
    </div>
    """, unsafe_allow_html=True)

    # -- 4 colonnes de filtres : type de document, ticker, annee debut, annee fin
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        doc_t = st.selectbox("Type de document", ["Tous", "Rapports", "News", "Tableaux", "CSV"],
                             label_visibility="collapsed")
    with c2:
        ticker = st.text_input("Ticker boursier", placeholder="AAPL, MSFT...",
                               label_visibility="collapsed")
    with c3:
        yf = st.text_input("Annee debut", placeholder="2022", label_visibility="collapsed")
    with c4:
        yt = st.text_input("Annee fin", placeholder="2024", label_visibility="collapsed")

    # -- Mise a jour des filtres dans le state
    st.session_state.filters = {
        "doc_type": doc_t,
        "ticker": ticker.strip().upper() if ticker else "",
        "date_from": yf.strip() if yf else None,
        "date_to": yt.strip() if yt else None,
    }

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # -- Section options RAG
    st.markdown(
        '<div style="font-family:\'Barlow Condensed\';font-size:11px;font-weight:700;'
        'letter-spacing:0.12em;text-transform:uppercase;color:#A3A3A3;margin-bottom:16px;">'
        'Options RAG</div>',
        unsafe_allow_html=True,
    )

    # -- Toggle pour activer/desactiver la decomposition intelligente des requetes
    st.session_state.use_decomposition = st.toggle(
        "Decomposition intelligente des requetes",
        value=st.session_state.use_decomposition,
        help="Decompose les questions complexes en sous-requetes atomiques",
    )

    # -- Indicateur de statut OpenAI
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    if settings.use_openai_embeddings:
        st.markdown(
            '<span style="font-size:10px;color:#16A34A;font-family:\'Barlow Condensed\';'
            'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">'
            'OpenAI GPT-4o actif</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="font-size:10px;color:#D97706;font-family:\'Barlow Condensed\';'
            'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">'
            'Mode degrade — cle manquante</span>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def view_documents(system: dict):
    """
    Vue 02 — Base Documentaire.
    Affiche la liste de tous les documents indexes dans ChromaDB
    avec possibilite de filtrer par nom et de supprimer des documents.
    """
    if not system.get("ok"):
        st.warning("Systeme non initialise")
        return

    # -- Recuperation de la liste des sources indexees
    sources = system["vs"].list_sources()

    st.markdown('<div style="padding: 0 32px;">', unsafe_allow_html=True)

    # -- Etat vide : aucun document indexe
    if not sources:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">▪</div>
            <div class="empty-title">Base documentaire vide</div>
            <div class="empty-sub">Importez des documents via l'onglet Ingestion ou indexez les donnees d'exemple.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # -- Champ de recherche pour filtrer les documents par nom de fichier
    # FIX : label non vide pour eviter le warning d'accessibilite Streamlit
    search = st.text_input("Filtrer par nom de fichier", placeholder="Filtrer par nom de fichier...",
                           label_visibility="collapsed")
    if search:
        # -- Filtrage case-insensitive sur le nom de fichier
        sources = [s for s in sources if search.lower() in s.get("filename", "").lower()]

    # -- Compteur de documents affiches
    st.markdown(
        f'<div style="font-family:\'Barlow Condensed\';font-size:10px;font-weight:600;'
        f'letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;padding: 12px 0 8px;">'
        f'{len(sources)} document{"s" if len(sources) != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Affichage de chaque document sous forme de ligne
    for src in sources:
        fname = src.get("filename", "unknown")
        dtype = src.get("document_type", "unknown")
        count = src.get("chunk_count", 0)
        date = src.get("date", "")
        coll = src.get("collection", "unknown")
        pill = _pill(dtype)

        # -- 5 colonnes : nom + pilule, type, nb chunks, date, bouton supprimer
        c1, c2, c3, c4, c5 = st.columns([4, 2, 1, 1, 1])
        with c1:
            st.markdown(
                f'<div style="padding:12px 32px;font-size:13px;font-weight:500;color:#0C0C0C;">'
                f'{pill} {fname}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div style="padding:12px 0;font-size:12px;color:#5C5C5C;">'
                f'{dtype.replace("_", " ").title()}</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div style="padding:12px 0;font-family:\'JetBrains Mono\';'
                f'font-size:11px;color:#A3A3A3;">{count}</div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div style="padding:12px 0;font-size:11px;color:#A3A3A3;">{date}</div>',
                unsafe_allow_html=True,
            )
        with c5:
            # -- Bouton de suppression d'un document (supprime tous ses chunks)
            if st.button("X", key=f"del_{fname}_{coll}", help=f"Supprimer {fname}"):
                deleted = system["vs"].delete_by_source(fname)
                if deleted > 0:
                    system["retriever"].invalidate_bm25_cache()
                    st.rerun()
        # -- Separateur horizontal entre chaque ligne
        st.markdown(
            '<hr style="border:none;border-top:1px solid #EBEBEB;margin:0 32px;">',
            unsafe_allow_html=True,
        )


def view_assistant(system: dict):
    """
    Vue 03 — Assistant IA.
    Interface de chat avec le pipeline RAG. Affiche :
    - A gauche : la zone de chat (historique + zone de saisie)
    - A droite : le panel de source documentaire + sous-requetes
    """
    # -- Disposition en 2 colonnes : chat (3/4) + panel source (1/4)
    col_chat, col_src = st.columns([3, 1], gap="small")

    # ── Colonne chat ──────────────────────────────────────────────────────────
    with col_chat:
        # -- Affichage de l'historique des messages
        if st.session_state.messages:
            st.markdown('<div class="chat-area">', unsafe_allow_html=True)
            for msg in st.session_state.messages:
                _render_message(msg)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # -- Etat vide : message d'accueil + exemples de questions
            st.markdown("""
            <div style="padding: 48px 32px 24px;">
                <div class="empty-state" style="padding: 40px 0 32px;">
                    <div style="font-family:'Barlow Condensed';font-size:10px;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:#A3A3A3;margin-bottom:6px;">Agent FinRAG — Intelligence Artificielle</div>
                    <div style="font-size:15px;color:#5C5C5C;max-width:400px;margin:0 auto;line-height:1.5;">
                        Posez vos questions sur les documents financiers indexes. Les reponses sont generees par GPT-4o et sourcees documentairement.
                    </div>
                </div>
                <div class="example-label">Questions d'exemple</div>
            </div>
            """, unsafe_allow_html=True)

            # -- Grille de boutons d'exemples de questions
            examples = [
                "Quel est le chiffre d'affaires d'Apple en FY2023 ?",
                "Comparez la croissance d'Apple et Microsoft sur leurs derniers exercices",
                "Quelle est la marge operationnelle de Microsoft au T4 FY2024 ?",
                "Quel est l'impact de l'IA sur les revenus de Microsoft en 2024 ?",
            ]
            st.markdown('<div style="padding: 0 32px;">', unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            for i, ex in enumerate(examples):
                with (ec1 if i % 2 == 0 else ec2):
                    if st.button(ex, key=f"ex_{i}", width="stretch"):
                        _process_query(ex, system)
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # -- Zone de saisie de question en bas du chat
        st.markdown('<div class="content-separator"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="input-area">', unsafe_allow_html=True)

            # -- Champ de saisie multiligne pour la question
            query = st.text_area(
                "Votre question",
                placeholder="Posez votre question financiere...",
                height=72,
                label_visibility="collapsed",
                key="chat_input",
            )

            # -- Boutons Envoyer et Effacer
            c_send, c_clear = st.columns([4, 1])
            with c_send:
                if st.button("Envoyer", width="stretch", type="primary"):
                    if query and query.strip():
                        _process_query(query.strip(), system)
                        st.rerun()
                    else:
                        st.warning("Saisissez une question avant d'envoyer.")
            with c_clear:
                if st.button("Effacer", width="stretch"):
                    # -- Reset de l'historique de chat et des citations
                    st.session_state.messages = []
                    st.session_state.last_citations = []
                    st.session_state.last_sub_queries = []
                    st.rerun()

            st.markdown(
                '<div class="input-hint">GPT-4o — Reponses sourcees documentairement</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Colonne source panel ─────────────────────────────────────────────────
    with col_src:
        render_source_panel()
        st.markdown(
            '<hr class="sidebar-divider" style="margin: 16px 0;">',
            unsafe_allow_html=True,
        )
        render_intel_sidebar()


def view_analytics(system: dict):
    """
    Vue 04 — Analytics.
    Affiche les statistiques du vector store, l'historique des requetes,
    les metriques RAGAS (demo ou reelles) et le rapport d'evaluation.
    """
    if not system.get("ok"):
        st.warning("Systeme non initialise")
        return

    # -- Recuperation des stats globales du vector store
    stats = system["vs"].get_stats()

    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-heading" style="margin-top:0;padding-top:24px;border-top:none;">'
        'Statistiques du vector store</div>',
        unsafe_allow_html=True,
    )

    # -- 4 metriques principales en colonnes
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Chunks indexes", f"{stats.get('total_chunks', 0):,}")
    with c2:
        st.metric("Sources", stats.get("total_sources", 0))
    with c3:
        st.metric("Requetes (session)", st.session_state.total_queries)
    with c4:
        st.metric("Tokens utilises", f"{st.session_state.total_tokens:,}")

    # -- Graphique de repartition par collection ChromaDB
    colls = stats.get("collections", {})
    if any(v > 0 for v in colls.values()):
        st.markdown(
            '<div class="section-heading">Collections ChromaDB</div>',
            unsafe_allow_html=True,
        )
        # -- Graphique a barres horizontales (Plotly)
        fig = go.Figure(go.Bar(
            x=list(colls.values()),
            y=list(colls.keys()),
            orientation='h',
            marker_color=['#0C0C0C', '#5C5C5C', '#A3A3A3', '#D4D4D4'][:len(colls)],
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', size=11)),
            yaxis=dict(showgrid=False, tickfont=dict(family='Barlow Condensed', size=12, color='#5C5C5C')),
            font=dict(family='Barlow', color='#0C0C0C'),
        )
        # FIX : remplacement de use_container_width par width (deprecation Streamlit)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # -- Historique des requetes de la session
    if st.session_state.query_history:
        st.markdown(
            '<div class="section-heading">Historique des requetes</div>',
            unsafe_allow_html=True,
        )
        # -- Affichage des 10 dernieres requetes (plus recentes en premier)
        for q in st.session_state.query_history[-10:][::-1]:
            conf = q.get("confidence", 0)
            # -- Couleur du point de confiance selon le score
            conf_color = "#16A34A" if conf >= 0.7 else "#D97706" if conf >= 0.4 else "#DC2626"
            st.markdown(f"""
            <div class="history-row">
                <div class="history-q">{q['question'][:100]}{"..." if len(q['question']) > 100 else ""}</div>
                <div class="history-meta">
                    <span style="color:{conf_color};">conf {conf:.0%}</span>
                    <span>time {_fmt_time(q.get('processing_time', 0))}</span>
                    <span>tokens {q.get('tokens', 0):,}</span>
                    <span>sources {q.get('n_sources', 0)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -- Metriques RAGAS de demonstration
    st.markdown(
        '<div class="section-heading">Metriques RAGAS (demonstration)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Lancez `python src/main.py evaluate` pour obtenir des metriques reelles")

    # -- Valeurs de demonstration pour les 4 metriques RAGAS
    demo = {"Faithfulness": 0.87, "Answer Relevancy": 0.82, "Context Recall": 0.78, "Context Precision": 0.84}
    cols_ragas = st.columns(4)
    for i, (name, val) in enumerate(demo.items()):
        with cols_ragas[i]:
            st.markdown(f"""
            <div class="ragas-card">
                <span class="ragas-score">{val*100:.0f}%</span>
                <span class="ragas-label">{name}</span>
                <div class="ragas-bar">
                    <div class="ragas-fill" style="width:{val*100:.0f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -- Rapport d'evaluation RAGAS reel (si disponible sur disque)
    rpt = ROOT_DIR / "docs" / "evaluation_report.md"
    if rpt.exists():
        st.markdown(
            '<div class="section-heading">Rapport d\'evaluation</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Afficher le rapport complet", expanded=False):
            st.markdown(rpt.read_text(encoding="utf-8"))

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
# Point d'entree principal de l'application Streamlit.
# Initialise le systeme RAG, puis affiche les 4 onglets de navigation.

def main():
    """
    Fonction principale : initialise le systeme RAG et affiche l'interface.
    La navigation se fait par onglets (Ingestion, Documents, Assistant, Analytics).
    """
    # -- Initialisation du systeme RAG (cache apres le premier appel)
    with st.spinner("Initialisation du systeme..."):
        system = _build_system()

    # -- Affichage d'une erreur si l'initialisation a echoue
    if not system.get("ok"):
        st.error(f"Erreur d'initialisation : {system.get('error', 'Erreur inconnue')}")
        st.code("pip install -r requirements.txt\nstreamlit run src/ui/app.py")
        return

    # -- Definition des onglets de navigation
    tab_labels = [
        "01 — Ingestion",
        "02 — Documents",
        "03 — Assistant",
        "04 — Analytics",
    ]
    tab_views = ["upload", "documents", "assistant", "analytics"]

    # -- Creation des onglets Streamlit
    tabs = st.tabs(tab_labels)

    # -- Rendu de chaque vue dans son onglet correspondant
    for tab, view_key in zip(tabs, tab_views):
        with tab:
            # -- Mise a jour de la vue active dans le state
            st.session_state.active_view = view_key
            # -- Affichage de la barre superieure (titre + stats)
            render_top_bar(system)

            # -- Appel de la vue correspondante
            if view_key == "upload":
                view_upload(system)
            elif view_key == "documents":
                view_documents(system)
            elif view_key == "assistant":
                view_assistant(system)
            elif view_key == "analytics":
                view_analytics(system)


# -- Lancement de l'application
if __name__ == "__main__":
    main()
