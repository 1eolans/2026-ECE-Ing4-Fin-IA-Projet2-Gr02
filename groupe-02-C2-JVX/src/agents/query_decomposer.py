"""
Décomposeur de requêtes financières complexes en sous-requêtes atomiques.
Utilise Claude pour analyser et décomposer les questions multi-dimensionnelles.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class SubQuery:
    """Sous-requête atomique issue de la décomposition."""
    query: str
    type: str                    # "temporal", "metric", "comparison", "entity"
    time_filter: Optional[str]   # Ex: "2023", "Q4 2024"
    entities: List[str]          # Tickers ou noms d'entreprises
    priority: int                # Ordre d'exécution (1 = première)


# ─── Decomposition Prompt ─────────────────────────────────────────────────────

DECOMPOSITION_SYSTEM_PROMPT = """Tu es un expert en analyse financière et en recherche d'information.
Ton rôle est de décomposer des questions financières complexes en sous-requêtes atomiques simples.

Pour chaque question, retourne un JSON avec la structure suivante :
{
  "needs_decomposition": true/false,
  "sub_queries": [
    {
      "query": "sous-requête simple et précise",
      "type": "temporal|metric|comparison|entity|macro",
      "time_filter": "2023" ou "Q4 2024" ou null,
      "entities": ["AAPL", "Apple"] ou [],
      "priority": 1
    }
  ],
  "reasoning": "explication courte de la décomposition"
}

Règles :
- Une sous-requête doit être simple et recherchable dans un seul document
- Maximum 6 sous-requêtes par question
- Si la question est simple, retourne needs_decomposition: false et une seule sub_query
- Sois précis sur les filtres temporels (année, trimestre)
- Identifie les tickers boursiers (AAPL, MSFT, etc.) dans les entités"""

DECOMPOSITION_EXAMPLES = """
Exemples de décomposition :

Q: "Compare la croissance du CA d'Apple et Microsoft sur 3 ans"
→ Sub-queries:
  1. "Chiffre d'affaires Apple 2022" (type=metric, entities=["AAPL"])
  2. "Chiffre d'affaires Apple 2023" (type=metric, entities=["AAPL"])
  3. "Chiffre d'affaires Apple 2024" (type=metric, entities=["AAPL"])
  4. "Chiffre d'affaires Microsoft 2022" (type=metric, entities=["MSFT"])
  5. "Chiffre d'affaires Microsoft 2023" (type=metric, entities=["MSFT"])
  6. "Comparaison croissance Apple vs Microsoft" (type=comparison)

Q: "Quel est l'EPS d'Apple en 2023 ?"
→ needs_decomposition: false
  1. "EPS dilué Apple FY2023" (type=metric, entities=["AAPL"])
"""


# ─── Rule-based Decomposer (Fallback) ────────────────────────────────────────

COMPARISON_PATTERNS = [
    re.compile(r'compar(?:e|er|aison)', re.I),
    re.compile(r'vs\.?\s+|versus|par rapport à', re.I),
    re.compile(r'différence entre|évolution de', re.I),
]

TEMPORAL_PATTERNS = [
    re.compile(r'\d+\s+derni(?:ère|er)s?\s+ann(?:ée|ée)s?', re.I),
    re.compile(r'sur\s+\d+\s+ans?', re.I),
    re.compile(r'Q[1-4]\s*\d{4}|FY\d{4}|20\d{2}', re.I),
]

COMPANY_PATTERNS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Google": "GOOGL",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Tesla": "TSLA",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
}


def _extract_entities(text: str) -> List[str]:
    """Extrait les entités financières d'un texte."""
    entities = []
    for name, ticker in COMPANY_PATTERNS.items():
        if name.lower() in text.lower():
            if ticker not in entities:
                entities.append(ticker)
    return entities


def _rule_based_decompose(question: str) -> List[SubQuery]:
    """Décomposition basée sur des règles heuristiques (fallback sans API)."""
    entities = _extract_entities(question)
    is_comparison = any(p.search(question) for p in COMPARISON_PATTERNS)
    has_temporal = any(p.search(question) for p in TEMPORAL_PATTERNS)

    sub_queries: List[SubQuery] = []

    if is_comparison and len(entities) >= 2:
        # Create one sub-query per entity
        for i, entity in enumerate(entities):
            sub_queries.append(SubQuery(
                query=f"{question.split('et')[0].strip()} {entity}",
                type="comparison",
                time_filter=None,
                entities=[entity],
                priority=i + 1,
            ))
        # Add comparison query
        sub_queries.append(SubQuery(
            query=f"Comparaison {' vs '.join(entities)}",
            type="comparison",
            time_filter=None,
            entities=entities,
            priority=len(entities) + 1,
        ))
    elif has_temporal and entities:
        # Create queries per year mentioned or per entity
        years_match = re.findall(r'20\d{2}', question)
        years = years_match if years_match else ["2023", "2024"]

        for i, year in enumerate(years[:3]):
            for entity in entities[:2]:
                sub_queries.append(SubQuery(
                    query=f"{question} {entity} {year}",
                    type="temporal",
                    time_filter=year,
                    entities=[entity],
                    priority=i + 1,
                ))
    else:
        # Simple query
        sub_queries.append(SubQuery(
            query=question,
            type="metric",
            time_filter=None,
            entities=entities,
            priority=1,
        ))

    return sub_queries if sub_queries else [
        SubQuery(
            query=question,
            type="metric",
            time_filter=None,
            entities=entities,
            priority=1,
        )
    ]


# ─── Main Decomposer ──────────────────────────────────────────────────────────

class FinancialQueryDecomposer:
    """
    Décomposeur de requêtes financières complexes.

    Utilise Claude claude-sonnet-4-20250514 pour décomposer intelligemment les questions
    complexes en sous-requêtes atomiques recherchables.

    Fallback sur des règles heuristiques si l'API est indisponible.

    Exemple :
        "Compare la croissance du CA d'Apple et Microsoft sur les 3 dernières années"
        →
        1. "Chiffre d'affaires Apple 2022 2023 2024"
        2. "Chiffre d'affaires Microsoft 2022 2023 2024"
        3. "Comparaison croissance Apple vs Microsoft"
    """

    def __init__(self) -> None:
        self._client = None
        self._model = settings.LLM_MODEL
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialise le client Anthropic."""
        if not settings.use_anthropic:
            logger.info("QueryDecomposer en mode règles (pas d'API Anthropic)")
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self._initialized = True
            logger.info("QueryDecomposer Anthropic initialisé")
        except Exception as e:
            logger.warning(f"QueryDecomposer API indisponible: {e}")

    def decompose(self, question: str) -> List[SubQuery]:
        """
        Décompose une question financière en sous-requêtes atomiques.

        Args:
            question: Question complexe de l'utilisateur.

        Returns:
            Liste de SubQuery triées par priorité.
        """
        if not question.strip():
            return []

        logger.info(f"Décomposition: '{question[:80]}...'")

        if self._initialized and self._client:
            try:
                sub_queries = self._decompose_with_llm(question)
                if sub_queries:
                    logger.info(f"Décomposition LLM: {len(sub_queries)} sous-requêtes")
                    return sorted(sub_queries, key=lambda x: x.priority)
            except Exception as e:
                logger.warning(f"Décomposition LLM échouée, fallback règles: {e}")

        # Fallback: rule-based
        sub_queries = _rule_based_decompose(question)
        logger.info(f"Décomposition règles: {len(sub_queries)} sous-requêtes")
        return sorted(sub_queries, key=lambda x: x.priority)

    def needs_decomposition(self, question: str) -> bool:
        """
        Détermine si une question nécessite une décomposition.

        Args:
            question: Question à analyser.

        Returns:
            True si la question est complexe et nécessite une décomposition.
        """
        # Simple heuristics
        is_comparison = any(p.search(question) for p in COMPARISON_PATTERNS)
        has_multiple_entities = len(_extract_entities(question)) >= 2
        is_long = len(question.split()) > 15
        has_temporal_range = bool(re.search(r'\d+\s+(derni\w+\s+ann|ans?)', question, re.I))

        return is_comparison or (has_multiple_entities and is_long) or has_temporal_range

    def _decompose_with_llm(self, question: str) -> List[SubQuery]:
        """Décomposition via API Anthropic."""
        prompt = (
            f"{DECOMPOSITION_EXAMPLES}\n\n"
            f"Question à décomposer : {question}\n\n"
            "Retourne UNIQUEMENT le JSON, sans texte autour."
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=800,
            system=DECOMPOSITION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        # Extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("Pas de JSON dans la réponse LLM")

        data = json.loads(json_match.group(0))

        if not data.get("needs_decomposition", True):
            sqs = data.get("sub_queries", [])
            if not sqs:
                return []
        else:
            sqs = data.get("sub_queries", [])

        result: List[SubQuery] = []
        for sq_data in sqs:
            result.append(SubQuery(
                query=sq_data.get("query", question),
                type=sq_data.get("type", "metric"),
                time_filter=sq_data.get("time_filter"),
                entities=sq_data.get("entities", []),
                priority=sq_data.get("priority", 1),
            ))

        return result
