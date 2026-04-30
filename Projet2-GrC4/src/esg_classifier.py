"""
ESG Multi-label Classification System
Projet C4 - ECE Paris 2026

Classification de risque ESG (Environnemental, Social, Gouvernance)
avec LLM local via Ollama.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ollama

# ─── Constantes ESG ────────────────────────────────────────────────────────────

ESG_CATEGORIES = {
    "E": {
        "label": "Environnemental",
        "subcategories": [
            "changement_climatique",
            "emissions_carbone",
            "energie_renouvelable",
            "biodiversite",
            "eau",
            "dechets",
            "pollution",
        ],
    },
    "S": {
        "label": "Social",
        "subcategories": [
            "conditions_travail",
            "droits_humains",
            "diversite_inclusion",
            "sante_securite",
            "relations_communaute",
            "protection_consommateur",
            "supply_chain",
        ],
    },
    "G": {
        "label": "Gouvernance",
        "subcategories": [
            "structure_conseil",
            "remuneration_dirigeants",
            "droits_actionnaires",
            "transparence",
            "ethique_conformite",
            "lutte_corruption",
            "fiscalite",
        ],
    },
}

RISK_LEVELS = ["faible", "modere", "eleve", "critique"]

# ─── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ESGMention:
    """Une mention ESG extraite d'un document."""
    category: str           # "E", "S", ou "G"
    subcategory: str        # sous-catégorie spécifique
    risk_level: str         # "faible", "modere", "eleve", "critique"
    citation: str           # extrait textuel original
    explanation: str        # explication du classifieur
    confidence: float       # [0.0, 1.0]
    page_hint: Optional[str] = None


@dataclass
class ESGDocumentResult:
    """Résultat de classification pour un document complet."""
    document_id: str
    document_text: str
    mentions: List[ESGMention] = field(default_factory=list)
    labels: Dict[str, bool] = field(default_factory=dict)        # {"E": True, "S": False, "G": True}
    risk_scores: Dict[str, float] = field(default_factory=dict)  # {"E": 0.8, "S": 0.0, "G": 0.6}
    summary: str = ""
    processing_time: float = 0.0


@dataclass
class CompanyESGProfile:
    """Profil ESG agrégé d'une entreprise sur plusieurs documents."""
    company_name: str
    documents: List[ESGDocumentResult] = field(default_factory=list)
    aggregated_scores: Dict[str, float] = field(default_factory=dict)
    overall_risk: str = "non_evalue"
    esg_rating: str = ""

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en analyse ESG (Environnemental, Social, Gouvernance)
spécialisé dans l'évaluation de risques pour la finance durable et la réglementation SFDR (EU).

Tes rôles :
1. Identifier et extraire toutes les mentions ESG dans un texte
2. Classifier chaque mention selon le pilier E, S ou G
3. Évaluer le niveau de risque de chaque mention (faible/modéré/élevé/critique)
4. Estimer ta confiance dans chaque classification

Critères ESG selon la taxonomie européenne :
- E (Environnemental): changement climatique, émissions CO2, énergie, biodiversité, eau, déchets, pollution
- S (Social): conditions de travail, droits humains, diversité, santé-sécurité, communauté, consommateurs, chaîne d'approvisionnement
- G (Gouvernance): conseil d'administration, rémunération, droits actionnaires, transparence, éthique, anti-corruption, fiscalité

Niveaux de risque :
- faible: impact mineur, bien géré
- modere: impact notable, gestion partielle
- eleve: impact significatif, gestion insuffisante
- critique: impact majeur, absence de gestion ou violation

Tu dois répondre UNIQUEMENT en JSON valide, sans texte avant ni après."""


def build_classification_prompt(text: str, chunk_id: int = 0) -> str:
    """Construit le prompt de classification pour un chunk de texte."""
    return f"""Analyse ce document financier (segment {chunk_id}) et extrait toutes les mentions ESG.

TEXTE À ANALYSER:
\"\"\"
{text}
\"\"\"

Réponds avec ce JSON exactement (sans markdown, sans texte autour):
{{
  "mentions": [
    {{
      "category": "E" | "S" | "G",
      "subcategory": "string (ex: emissions_carbone, conditions_travail, transparence)",
      "risk_level": "faible" | "modere" | "eleve" | "critique",
      "citation": "extrait exact du texte (max 150 mots)",
      "explanation": "pourquoi cette mention est ESG et ce niveau de risque",
      "confidence": 0.0-1.0
    }}
  ],
  "document_summary": "Résumé ESG global en 2-3 phrases",
  "dominant_pillar": "E" | "S" | "G" | "aucun",
  "overall_risk": "faible" | "modere" | "eleve" | "critique"
}}

Si aucune mention ESG n'est trouvée, retourne {{"mentions": [], "document_summary": "Aucun contenu ESG identifié.", "dominant_pillar": "aucun", "overall_risk": "faible"}}"""

# ─── Classifieur Principal ─────────────────────────────────────────────────────

class ESGClassifier:
    """
    Classifieur ESG multi-label basé sur Ollama (Local LLM).

    Utilise Ollama pour extraire et classifier les mentions ESG
    dans des documents financiers selon le cadre SFDR/taxonomie UE.
    """

    def __init__(self, model: str = "mistral", chunk_size: int = 2000):
        self.model = model
        self.chunk_size = chunk_size

    def _call_api(self, prompt: str, max_tokens: int = 2000) -> dict:
        """Appelle l'API locale Ollama et retourne le JSON parsé."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "num_predict": max_tokens,
                "temperature": 0.0,
            },
        )
        raw = response["message"]["content"].strip()
        # Nettoyer les éventuels blocs markdown
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    def _chunk_text(self, text: str) -> List[str]:
        """Découpe le texte en chunks chevauchants."""
        words = text.split()
        chunks = []
        step = int(self.chunk_size * 0.8)  # 20% overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    def classify_document(
        self, text: str, document_id: str = "doc_0"
    ) -> ESGDocumentResult:
        """Classifie un document complet en mentions ESG multi-label."""
        start = time.time()
        result = ESGDocumentResult(document_id=document_id, document_text=text)
        chunks = self._chunk_text(text)
        all_mentions: List[ESGMention] = []
        summaries: List[str] = []

        for i, chunk in enumerate(chunks):
            prompt = build_classification_prompt(chunk, chunk_id=i)
            try:
                data = self._call_api(prompt)
                for m in data.get("mentions", []):
                    mention = ESGMention(
                        category=m.get("category", "G"),
                        subcategory=m.get("subcategory", "autre"),
                        risk_level=m.get("risk_level", "faible"),
                        citation=m.get("citation", ""),
                        explanation=m.get("explanation", ""),
                        confidence=float(m.get("confidence", 0.5)),
                        page_hint=f"chunk_{i}",
                    )
                    all_mentions.append(mention)
                if data.get("document_summary"):
                    summaries.append(data["document_summary"])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"  [Avertissement] Erreur parsing chunk {i}: {e}")

        result.mentions = all_mentions
        result.labels = self._compute_labels(all_mentions)
        result.risk_scores = self._compute_risk_scores(all_mentions)
        result.summary = " | ".join(summaries) if summaries else "Analyse non disponible."
        result.processing_time = time.time() - start
        return result

    def _compute_labels(self, mentions: List[ESGMention]) -> Dict[str, bool]:
        """Calcule les labels binaires E/S/G."""
        labels: Dict[str, bool] = {"E": False, "S": False, "G": False}
        for m in mentions:
            if m.category in labels and m.confidence >= 0.5:
                labels[m.category] = True
        return labels

    def _compute_risk_scores(self, mentions: List[ESGMention]) -> Dict[str, float]:
        """Calcule le score de risque agrégé par pilier (0–1)."""
        risk_map = {"faible": 0.2, "modere": 0.5, "eleve": 0.8, "critique": 1.0}
        scores: Dict[str, List[float]] = {"E": [], "S": [], "G": []}
        for m in mentions:
            if m.category in scores:
                score = risk_map.get(m.risk_level, 0.0) * m.confidence
                scores[m.category].append(score)
        return {
            cat: round(sum(vals) / len(vals), 3) if vals else 0.0
            for cat, vals in scores.items()
        }

    def classify_company(
        self, company_name: str, documents: List[Dict]
    ) -> CompanyESGProfile:
        """Agrège les résultats ESG de plusieurs documents pour une entreprise."""
        profile = CompanyESGProfile(company_name=company_name)
        for doc in documents:
            print(f"  Analyse de '{doc['id']}'...")
            result = self.classify_document(doc["text"], doc["id"])
            profile.documents.append(result)

        # Agrégation : moyenne pondérée par nombre de mentions
        agg: Dict[str, List[float]] = {"E": [], "S": [], "G": []}
        for doc in profile.documents:
            for cat, score in doc.risk_scores.items():
                if doc.labels.get(cat):
                    agg[cat].append(score)

        profile.aggregated_scores = {
            cat: round(sum(vals) / len(vals), 3) if vals else 0.0
            for cat, vals in agg.items()
        }

        # Rating global
        max_score = max(profile.aggregated_scores.values(), default=0.0)
        if max_score >= 0.8:
            profile.overall_risk = "critique"
            profile.esg_rating = "CCC"
        elif max_score >= 0.6:
            profile.overall_risk = "eleve"
            profile.esg_rating = "B"
        elif max_score >= 0.4:
            profile.overall_risk = "modere"
            profile.esg_rating = "BB"
        elif max_score >= 0.2:
            profile.overall_risk = "faible"
            profile.esg_rating = "BBB"
        else:
            profile.overall_risk = "negligeable"
            profile.esg_rating = "A"

        return profile
