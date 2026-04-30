#!/usr/bin/env python3
"""
Générateur de données d'exemple pour FinRAG.
Génère des PDFs financiers fictifs, articles de news et données CSV.
"""

import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Ensure we're in the right directory
BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "samples"
NEWS_DIR = SAMPLES_DIR / "news"

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)


def generate_pdfs():
    """Génère les PDFs financiers fictifs avec reportlab."""
    try:
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        print("reportlab non installé. Génération de PDFs ignorée.")
        print("Installez avec: pip install reportlab")
        return

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1a3a5c'),
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a3a5c'),
        spaceAfter=12,
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#2c5f8a'),
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    def make_table(data, col_widths=None, header_color=colors.HexColor('#1a3a5c')):
        """Crée un tableau stylisé."""
        t = Table(data, colWidths=col_widths)
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f8fb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f8fb'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
        t.setStyle(style)
        return t

    # =========================================================
    # PDF 1: Apple Annual Report 2023
    # =========================================================
    print("Génération de apple_annual_report_2023.pdf...")
    doc = SimpleDocTemplate(
        str(SAMPLES_DIR / "apple_annual_report_2023.pdf"),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    story = []

    # Cover
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("APPLE INC.", title_style))
    story.append(Paragraph("Rapport Annuel 2023", heading_style))
    story.append(Paragraph("Exercice fiscal clos le 30 septembre 2023", body_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a3a5c')))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Ticker: AAPL | Bourse: NASDAQ | ISIN: US0378331005", body_style))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("1. Résumé Exécutif", heading_style))
    story.append(Paragraph(
        "Apple Inc. a réalisé une performance remarquable au cours de l'exercice fiscal 2023, "
        "malgré un environnement macroéconomique difficile caractérisé par une inflation persistante "
        "et des tensions géopolitiques mondiales. Le chiffre d'affaires total a atteint 383,3 milliards "
        "de dollars, légèrement en retrait de 2,8% par rapport à l'exercice précédent (394,3 Md$), "
        "principalement dû à la faiblesse du segment Mac et aux défis logistiques en Chine.",
        body_style
    ))
    story.append(Paragraph(
        "Le bénéfice net s'est établi à 97,0 milliards de dollars, représentant une marge nette "
        "exceptionnelle de 25,3%. Le bénéfice par action dilué (EPS) a progressé de 13% à 6,13 dollars, "
        "porté par le programme de rachat d'actions agressif d'Apple. La trésorerie nette disponible "
        "atteint 55,7 milliards de dollars après déduction de la dette à long terme.",
        body_style
    ))
    story.append(Paragraph(
        "L'iPhone reste le moteur principal de la croissance avec un chiffre d'affaires de 200,6 Md$, "
        "représentant 52,3% du total. Les Services, segment à forte marge, ont enregistré une croissance "
        "de 16,1% à 85,2 Md$, attestant du succès de la stratégie de monétisation de l'écosystème Apple.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # Key metrics table
    story.append(Paragraph("Indicateurs Clés de Performance (FY2023)", subheading_style))
    kpi_data = [
        ['Indicateur', 'FY2023', 'FY2022', 'Variation'],
        ['Chiffre d\'affaires total', '383,3 Md$', '394,3 Md$', '-2,8%'],
        ['Bénéfice net', '97,0 Md$', '99,8 Md$', '-2,8%'],
        ['EBITDA', '130,2 Md$', '133,1 Md$', '-2,2%'],
        ['Marge brute', '44,1%', '43,3%', '+0,8 pp'],
        ['Marge nette', '25,3%', '25,3%', '0,0 pp'],
        ['EPS dilué', '6,13 $', '6,11 $', '+0,3%'],
        ['Free Cash Flow', '111,4 Md$', '111,4 Md$', '0,0%'],
        ['Rachat d\'actions', '77,6 Md$', '89,4 Md$', '-13,2%'],
        ['Dividendes versés', '15,0 Md$', '14,8 Md$', '+1,4%'],
    ]
    story.append(make_table(kpi_data, col_widths=[6*cm, 4*cm, 4*cm, 3*cm]))
    story.append(PageBreak())

    # Revenue breakdown
    story.append(Paragraph("2. Analyse des Revenus par Segment", heading_style))
    story.append(Paragraph(
        "Apple décompose ses revenus en cinq segments opérationnels principaux. L'iPhone demeure "
        "le segment dominant malgré une légère contraction, tandis que les Services affichent la "
        "plus forte dynamique de croissance, confirmant la transition vers un modèle récurrent.",
        body_style
    ))

    rev_data = [
        ['Segment', 'FY2023 (Md$)', 'FY2022 (Md$)', 'Croissance', '% du CA'],
        ['iPhone', '200,6', '205,5', '-2,4%', '52,3%'],
        ['Services', '85,2', '73,4', '+16,1%', '22,2%'],
        ['Mac', '29,4', '40,2', '-26,9%', '7,7%'],
        ['iPad', '28,3', '29,3', '-3,4%', '7,4%'],
        ['Wearables & Accessories', '39,8', '41,2', '-3,4%', '10,4%'],
        ['TOTAL', '383,3', '394,3', '-2,8%', '100,0%'],
    ]
    story.append(make_table(rev_data, col_widths=[5.5*cm, 3.5*cm, 3.5*cm, 3*cm, 3*cm]))
    story.append(Spacer(1, 0.3*inch))

    # Geographic breakdown
    story.append(Paragraph("Répartition Géographique des Revenus", subheading_style))
    geo_data = [
        ['Zone Géographique', 'FY2023 (Md$)', 'FY2022 (Md$)', 'Croissance'],
        ['Amériques', '162,1', '169,7', '-4,5%'],
        ['Europe', '94,3', '95,1', '-0,8%'],
        ['Grande Chine', '72,6', '74,2', '-2,2%'],
        ['Japon', '24,3', '25,9', '-6,2%'],
        ['Reste de l\'Asie-Pacifique', '30,0', '29,4', '+2,0%'],
        ['TOTAL', '383,3', '394,3', '-2,8%'],
    ]
    story.append(make_table(geo_data, col_widths=[6*cm, 4*cm, 4*cm, 3*cm]))
    story.append(PageBreak())

    # P&L
    story.append(Paragraph("3. Compte de Résultat Consolidé", heading_style))
    story.append(Paragraph(
        "Le compte de résultat consolidé d'Apple pour l'exercice fiscal 2023 (clos le 30 septembre 2023) "
        "illustre la solidité exceptionnelle des marges du groupe, notamment grâce à la montée en puissance "
        "du segment Services à haute valeur ajoutée.",
        body_style
    ))

    pl_data = [
        ['Poste (en millions $)', 'FY2023', 'FY2022', 'FY2021'],
        ['Chiffre d\'affaires net', '383 285', '394 328', '365 817'],
        ['Coût des ventes', '214 137', '223 546', '212 981'],
        ['Marge brute', '169 148', '170 782', '152 836'],
        ['% Marge brute', '44,1%', '43,3%', '41,8%'],
        ['Frais de R&D', '29 915', '26 251', '21 914'],
        ['Frais commerciaux & admin.', '24 932', '25 094', '21 973'],
        ['Résultat opérationnel', '114 301', '119 437', '108 949'],
        ['% Marge opérationnelle', '29,8%', '30,3%', '29,8%'],
        ['Produits financiers nets', '1 025', '-201', '258'],
        ['Résultat avant impôts', '115 326', '119 236', '109 207'],
        ['Charge d\'impôts', '18 325', '19 300', '14 527'],
        ['Résultat net', '96 995', '99 803', '94 680'],
        ['% Marge nette', '25,3%', '25,3%', '25,9%'],
    ]
    story.append(make_table(pl_data, col_widths=[6*cm, 3.5*cm, 3.5*cm, 3.5*cm]))
    story.append(PageBreak())

    # Balance Sheet
    story.append(Paragraph("4. Bilan Consolidé (au 30 septembre 2023)", heading_style))
    bs_data = [
        ['Actif (en millions $)', 'FY2023', 'Passif & Capitaux propres', 'FY2023'],
        ['Trésorerie & équivalents', '29 965', 'Dettes fournisseurs', '62 611'],
        ['Titres négociables CT', '31 590', 'Dettes fiscales', '1 307'],
        ['Créances clients', '29 508', 'Autres dettes CT', '58 897'],
        ['Stocks', '6 331', 'Dette LT (part CT)', '9 822'],
        ['Autres actifs CT', '14 695', 'Total Passif CT', '145 308'],
        ['Total Actif CT', '143 566', '', ''],
        ['Actifs LT nets', '43 715', 'Dette long terme', '95 281'],
        ['Goodwill', '3 514', 'Autres passifs LT', '49 848'],
        ['Titres négociables LT', '100 544', 'Total Passif LT', '145 129'],
        ['Autres actifs LT', '64 758', 'Capitaux propres', '62 146'],
        ['Total Actif', '352 583', 'Total Passif & CP', '352 583'],
    ]
    story.append(make_table(bs_data, col_widths=[5*cm, 3*cm, 5.5*cm, 3*cm]))
    story.append(PageBreak())

    # Financial Ratios
    story.append(Paragraph("5. Ratios Financiers et Valorisation", heading_style))
    ratio_data = [
        ['Ratio', 'Valeur FY2023', 'Benchmark Secteur'],
        ['Price/Earnings (P/E)', '29,4x', '25,0x'],
        ['Price/Sales (P/S)', '7,5x', '5,2x'],
        ['Price/Book (P/B)', '44,2x', '15,3x'],
        ['EV/EBITDA', '22,8x', '18,5x'],
        ['Ratio de liquidité courante', '0,99', '1,50'],
        ['Ratio d\'endettement (D/E)', '2,05', '0,80'],
        ['ROE (Return on Equity)', '171,9%', '35,0%'],
        ['ROA (Return on Assets)', '27,5%', '12,0%'],
        ['ROIC', '58,5%', '25,0%'],
        ['Rendement dividende', '0,53%', '1,20%'],
        ['Taux de croissance EPS (3 ans)', '+8,2%', '+10,0%'],
    ]
    story.append(make_table(ratio_data, col_widths=[7*cm, 4*cm, 4.5*cm]))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("6. Perspectives et Stratégie 2024", heading_style))
    story.append(Paragraph(
        "Pour l'exercice fiscal 2024, Apple anticipe une reprise progressive des ventes d'iPhone "
        "portée par le cycle de remplacement iPhone 15 et l'expansion dans les marchés émergents. "
        "Le lancement de l'Apple Vision Pro marque l'entrée du groupe dans l'informatique spatiale, "
        "un segment potentiellement transformateur à moyen terme. Le segment Services devrait "
        "continuer sa trajectoire de croissance à deux chiffres, visant les 100 Md$ de revenus annuels.",
        body_style
    ))
    story.append(Paragraph(
        "L'intégration de l'IA générative dans l'ensemble de l'écosystème Apple (iOS 18, macOS Sequoia) "
        "constitue le principal catalyseur de différenciation compétitive pour 2024-2025. Apple Intelligence, "
        "développé en partenariat avec OpenAI pour certaines fonctionnalités cloud, devrait stimuler "
        "les ventes de nouveaux appareils compatibles et renforcer la rétention des utilisateurs.",
        body_style
    ))

    doc.build(story)
    print("  ✓ apple_annual_report_2023.pdf généré")

    # =========================================================
    # PDF 2: Microsoft Q4 2024
    # =========================================================
    print("Génération de microsoft_q4_2024.pdf...")
    doc2 = SimpleDocTemplate(
        str(SAMPLES_DIR / "microsoft_q4_2024.pdf"),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    story2 = []

    story2.append(Spacer(1, 1.5*inch))
    story2.append(Paragraph("MICROSOFT CORPORATION", title_style))
    story2.append(Paragraph("Résultats T4 et Exercice Fiscal 2024", heading_style))
    story2.append(Paragraph("Trimestre clos le 30 juin 2024", body_style))
    story2.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a3a5c')))
    story2.append(Paragraph("Ticker: MSFT | Bourse: NASDAQ | ISIN: US5949181045", body_style))
    story2.append(PageBreak())

    story2.append(Paragraph("1. Faits Marquants T4 FY2024", heading_style))
    story2.append(Paragraph(
        "Microsoft a publié des résultats record pour son quatrième trimestre de l'exercice fiscal 2024, "
        "avec un chiffre d'affaires de 64,7 milliards de dollars (+15,2% YoY), dépassant les attentes "
        "des analystes de 1,2 Md$. La croissance est tirée par l'accélération de Microsoft Cloud, "
        "qui a atteint 36,8 Md$ au trimestre (+21,0% YoY).",
        body_style
    ))
    story2.append(Paragraph(
        "Azure continue de gagner des parts de marché face à AWS et Google Cloud, avec une croissance "
        "de 29% en glissement annuel (contre 17% pour AWS). L'intégration de Copilot dans Microsoft 365 "
        "génère une valeur ajoutée significative avec plus de 1,8 million de sièges Copilot déployés "
        "dans les entreprises, à un prix de 30$/mois par utilisateur.",
        body_style
    ))

    q4_kpi = [
        ['Indicateur', 'T4 FY2024', 'T4 FY2023', 'Croissance'],
        ['Chiffre d\'affaires', '64,7 Md$', '56,2 Md$', '+15,2%'],
        ['Microsoft Cloud', '36,8 Md$', '30,3 Md$', '+21,0%'],
        ['Azure & Cloud Services', '28,5 Md$', '22,1 Md$', '+29,0%'],
        ['Résultat opérationnel', '27,9 Md$', '24,3 Md$', '+14,8%'],
        ['Marge opérationnelle', '43,1%', '43,2%', '-0,1 pp'],
        ['Bénéfice net', '22,0 Md$', '20,1 Md$', '+9,7%'],
        ['EPS dilué', '2,95 $', '2,69 $', '+9,7%'],
    ]
    story2.append(make_table(q4_kpi, col_widths=[6*cm, 4*cm, 4*cm, 3*cm]))
    story2.append(PageBreak())

    story2.append(Paragraph("2. Analyse par Segment", heading_style))
    seg_data = [
        ['Segment', 'CA T4 FY2024', 'CA T4 FY2023', 'Croissance'],
        ['Productivity & Business Processes', '20,3 Md$', '18,3 Md$', '+11,0%'],
        ['  dont Microsoft 365 Commercial', '15,2 Md$', '13,3 Md$', '+14,3%'],
        ['  dont LinkedIn', '4,3 Md$', '3,9 Md$', '+10,3%'],
        ['Intelligent Cloud', '28,5 Md$', '24,0 Md$', '+18,8%'],
        ['  dont Azure', '19,7 Md$', '15,3 Md$', '+28,8%'],
        ['  dont Server Products', '7,1 Md$', '6,9 Md$', '+2,9%'],
        ['More Personal Computing', '15,9 Md$', '13,9 Md$', '+14,4%'],
        ['  dont Windows OEM', '4,4 Md$', '3,5 Md$', '+25,7%'],
        ['  dont Xbox & Gaming', '5,1 Md$', '3,6 Md$', '+44,4%'],
        ['TOTAL', '64,7 Md$', '56,2 Md$', '+15,2%'],
    ]
    story2.append(make_table(seg_data, col_widths=[6.5*cm, 3.5*cm, 3.5*cm, 3*cm]))
    story2.append(Spacer(1, 0.3*inch))

    story2.append(Paragraph("3. Impact de l'Intelligence Artificielle", heading_style))
    story2.append(Paragraph(
        "L'investissement massif de Microsoft dans l'IA générative via son partenariat avec OpenAI "
        "commence à se matérialiser dans les résultats financiers. Copilot for Microsoft 365 est désormais "
        "disponible pour toutes les entreprises et génère un revenu additionnel de 30$/mois/utilisateur.",
        body_style
    ))

    ai_data = [
        ['Produit IA', 'Métrique Clé', 'Valeur T4 FY2024'],
        ['GitHub Copilot', 'Abonnés payants', '1,8 million'],
        ['Microsoft 365 Copilot', 'Sièges d\'entreprise', '1,8 million'],
        ['Azure OpenAI Service', 'Clients actifs', '65 000+'],
        ['Bing Chat Enterprise', 'Utilisateurs actifs/mois', '5 million'],
        ['Copilot Studio', 'Organisations utilisant', '50 000+'],
    ]
    story2.append(make_table(ai_data, col_widths=[5.5*cm, 5.5*cm, 5.5*cm]))
    story2.append(PageBreak())

    story2.append(Paragraph("4. Compte de Résultat FY2024 (Annuel)", heading_style))
    annual_pl = [
        ['Poste (en millions $)', 'FY2024', 'FY2023', 'Variation'],
        ['Chiffre d\'affaires', '245 122', '211 915', '+15,7%'],
        ['Coût des ventes', '74 114', '65 863', '+12,5%'],
        ['Marge brute', '171 008', '146 052', '+17,1%'],
        ['% Marge brute', '69,8%', '68,9%', '+0,9 pp'],
        ['R&D', '29 510', '27 195', '+8,5%'],
        ['Marketing & Commercial', '24 456', '22 759', '+7,5%'],
        ['Général & Administratif', '7 607', '7 575', '+0,4%'],
        ['Résultat opérationnel', '109 433', '88 523', '+23,6%'],
        ['% Marge opérationnelle', '44,6%', '41,8%', '+2,8 pp'],
        ['Produits financiers nets', '3 268', '1 625', '+101,1%'],
        ['Bénéfice avant impôts', '112 701', '90 148', '+25,0%'],
        ['Impôts', '17 868', '16 950', '+5,4%'],
        ['Bénéfice net', '88 136', '72 361', '+21,8%'],
    ]
    story2.append(make_table(annual_pl, col_widths=[6*cm, 3.5*cm, 3.5*cm, 3.5*cm]))

    doc2.build(story2)
    print("  ✓ microsoft_q4_2024.pdf généré")

    # =========================================================
    # PDF 3: Market Overview 2024
    # =========================================================
    print("Génération de market_overview_2024.pdf...")
    doc3 = SimpleDocTemplate(
        str(SAMPLES_DIR / "market_overview_2024.pdf"),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    story3 = []

    story3.append(Spacer(1, 1.5*inch))
    story3.append(Paragraph("VUE D'ENSEMBLE DES MARCHÉS", title_style))
    story3.append(Paragraph("Rapport Macro-Économique & Sectoriel 2024", heading_style))
    story3.append(Paragraph("Publié le 15 juillet 2024 — FinRAG Research", body_style))
    story3.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a3a5c')))
    story3.append(PageBreak())

    story3.append(Paragraph("1. Contexte Macro-Économique", heading_style))
    story3.append(Paragraph(
        "L'année 2024 est marquée par une normalisation progressive de la politique monétaire "
        "américaine, avec la Fed ayant procédé à deux baisses de taux de 25bp chacune au second "
        "semestre, portant le taux des Fed Funds à une fourchette de 4,75%-5,00%. L'inflation "
        "PCE a reflué à 2,6% en juin, se rapprochant de l'objectif de 2%.",
        body_style
    ))

    macro_data = [
        ['Indicateur Macro', 'T1 2024', 'T2 2024', 'Prévision T3 2024'],
        ['PIB USA (croissance QoQ)', '+1,4%', '+2,8%', '+2,5%'],
        ['Inflation CPI (YoY)', '3,5%', '3,0%', '2,8%'],
        ['Taux Fed Funds', '5,25-5,50%', '5,25-5,50%', '5,00-5,25%'],
        ['Taux chômage USA', '3,7%', '4,1%', '4,2%'],
        ['USD Index (DXY)', '104,5', '105,2', '103,8'],
        ['Or ($/oz)', '2 083', '2 331', '2 400'],
        ['Pétrole Brent ($/bbl)', '83,2', '85,0', '82,5'],
        ['VIX (Volatilité S&P 500)', '13,1', '16,4', '15,0'],
    ]
    story3.append(make_table(macro_data, col_widths=[6.5*cm, 3*cm, 3*cm, 4*cm]))
    story3.append(PageBreak())

    story3.append(Paragraph("2. Performance des Marchés Actions", heading_style))
    market_data = [
        ['Indice', 'Niveau (30/06/2024)', 'Performance YTD', 'Variation 12 mois'],
        ['S&P 500', '5 460', '+14,5%', '+22,1%'],
        ['Nasdaq Composite', '17 732', '+18,1%', '+29,6%'],
        ['Dow Jones', '39 118', '+3,8%', '+12,8%'],
        ['MSCI World', '3 486', '+11,8%', '+18,9%'],
        ['Euro Stoxx 50', '4 895', '+8,2%', '+15,3%'],
        ['Nikkei 225', '39 583', '+18,3%', '+25,2%'],
        ['CAC 40', '7 527', '+2,4%', '+8,6%'],
        ['MSCI EM', '1 075', '+6,5%', '+9,8%'],
    ]
    story3.append(make_table(market_data, col_widths=[4.5*cm, 4*cm, 3.5*cm, 4.5*cm]))
    story3.append(Spacer(1, 0.3*inch))

    story3.append(Paragraph("3. Performance Sectorielle S&P 500 (YTD 2024)", heading_style))
    sector_data = [
        ['Secteur', 'Performance YTD', 'P/E Médian', 'Recommandation'],
        ['Technologie', '+28,2%', '32x', 'Surpondérer'],
        ['Communication', '+22,7%', '22x', 'Surpondérer'],
        ['Services aux collectivités', '+17,4%', '18x', 'Neutre'],
        ['Financier', '+20,1%', '14x', 'Surpondérer'],
        ['Industriel', '+10,5%', '22x', 'Neutre'],
        ['Santé', '+7,2%', '20x', 'Neutre'],
        ['Consommation discr.', '+3,1%', '28x', 'Sous-pondérer'],
        ['Matériaux', '-0,4%', '18x', 'Neutre'],
        ['Énergie', '+10,8%', '12x', 'Neutre'],
        ['Immobilier (REIT)', '-2,7%', '35x', 'Sous-pondérer'],
        ['Consommation de base', '+7,8%', '22x', 'Neutre'],
    ]
    story3.append(make_table(sector_data, col_widths=[5*cm, 4*cm, 3.5*cm, 4*cm]))
    story3.append(PageBreak())

    story3.append(Paragraph("4. Thème Majeur : Intelligence Artificielle", heading_style))
    story3.append(Paragraph(
        "L'intelligence artificielle générative constitue le mégatrend dominant de 2024, "
        "portant les valorisations du secteur technologique à des niveaux historiques. "
        "Les dépenses mondiales en infrastructure IA (GPU, data centers, énergie) devraient "
        "atteindre 200 Md$ en 2024, avec une croissance projetée à 500 Md$ d'ici 2027.",
        body_style
    ))

    ai_market = [
        ['Entreprise', 'Ticker', 'Expo. IA', 'Perf. YTD', 'Capitalisation'],
        ['NVIDIA', 'NVDA', 'Très élevée', '+149%', '3 100 Md$'],
        ['Microsoft', 'MSFT', 'Très élevée', '+18%', '3 300 Md$'],
        ['Alphabet', 'GOOGL', 'Élevée', '+29%', '2 200 Md$'],
        ['Meta Platforms', 'META', 'Élevée', '+42%', '1 400 Md$'],
        ['Amazon', 'AMZN', 'Élevée', '+23%', '1 900 Md$'],
        ['Apple', 'AAPL', 'Modérée', '+10%', '3 200 Md$'],
        ['Oracle', 'ORCL', 'Modérée', '+25%', '430 Md$'],
        ['Salesforce', 'CRM', 'Modérée', '-4%', '235 Md$'],
    ]
    story3.append(make_table(ai_market, col_widths=[4*cm, 2*cm, 3.5*cm, 3*cm, 4*cm]))
    story3.append(Spacer(1, 0.3*inch))

    story3.append(Paragraph("5. Perspectives et Risques", heading_style))
    story3.append(Paragraph(
        "Scénario central (60% de probabilité) : Atterrissage en douceur avec maintien de la croissance "
        "américaine autour de 2,5%, baisse progressive de l'inflation vers 2% et assouplissement monétaire "
        "graduel. Le S&P 500 devrait terminer 2024 entre 5 500 et 5 800 points.",
        body_style
    ))
    story3.append(Paragraph(
        "Risques principaux : (1) Résurgence inflationniste liée aux coûts de l'IA et de l'énergie, "
        "(2) Tensions géopolitiques Chine-Taiwan affectant les chaînes d'approvisionnement semi-conducteurs, "
        "(3) Valorisations extrêmes du secteur IA pouvant mener à une correction technique, "
        "(4) Ralentissement du marché de l'emploi plus prononcé qu'anticipé.",
        body_style
    ))

    doc3.build(story3)
    print("  ✓ market_overview_2024.pdf généré")


def generate_news_articles():
    """Génère les articles de news fictifs."""
    print("Génération des articles de news...")

    articles = [
        {
            "id": "article_001",
            "title": "Apple dépasse les attentes au T4 2023 avec des Services en forte croissance",
            "content": (
                "Apple Inc. (AAPL) a publié jeudi des résultats trimestriels supérieurs aux attentes de Wall Street, "
                "portés par une croissance exceptionnelle de son segment Services. Le chiffre d'affaires du T4 FY2023 "
                "s'est établi à 89,5 milliards de dollars (+1% YoY), légèrement au-dessus du consensus à 89,2 Md$. "
                "Le segment Services a enregistré un record historique à 22,3 milliards de dollars (+16,3% YoY), "
                "avec l'App Store, Apple Music, iCloud et AppleTV+ comme principaux moteurs. "
                "Le PDG Tim Cook a déclaré : 'Nos Services atteignent des sommets record. Nous avons maintenant "
                "plus d'un milliard d'abonnements payants sur notre plateforme, un jalon historique.' "
                "Les ventes d'iPhone ont atteint 43,8 Md$ (+3% YoY), dépassant les prévisions de 43,1 Md$, "
                "tirées par le succès de l'iPhone 15 Pro dans les marchés premium. "
                "La marge brute a progressé à 45,2%, un record pour un trimestre fiscal Q4. "
                "Apple a annoncé un programme de rachat d'actions supplémentaire de 90 Md$ et une augmentation "
                "du dividende de 4% à 0,24$/action par trimestre. "
                "L'action AAPL a progressé de 2,1% en after-hours suite à la publication."
            ),
            "source": "Reuters (simulé)",
            "date": "2023-11-02",
            "ticker": "AAPL",
            "url": "https://example.com/news/apple-q4-2023",
            "category": "earnings"
        },
        {
            "id": "article_002",
            "title": "Microsoft : Azure accélère grâce à l'IA, revenus cloud record au T3 FY2024",
            "content": (
                "Microsoft (MSFT) a publié des résultats du troisième trimestre fiscal 2024 en forte hausse, "
                "avec un chiffre d'affaires de 61,9 milliards de dollars (+17% YoY), dépassant les attentes "
                "des analystes de 60,8 Md$. Le segment Intelligent Cloud a été le grand gagnant avec des revenus "
                "de 26,7 Md$ (+21% YoY), portés par Azure qui a affiché une croissance de 31% en glissement annuel. "
                "La PDG Satya Nadella a souligné l'accélération de l'adoption de l'IA générative : "
                "'Copilot est en train de transformer la façon dont les gens travaillent. Nous voyons des clients "
                "qui doublent leur engagement Azure dans les 6 mois suivant le déploiement de Copilot.' "
                "Microsoft 365 Copilot, lancé en novembre 2023 pour les entreprises, compte désormais "
                "plus de 1,5 million d'utilisateurs actifs dans 40 000 organisations. "
                "Le revenu par utilisateur de Microsoft 365 a augmenté de 9% grâce à la montée en gamme vers "
                "les formules incluant Copilot. "
                "Le bénéfice net a progressé de 20% à 21,9 Md$ avec une marge nette de 35,4%. "
                "L'action MSFT a gagné 4,5% en after-hours."
            ),
            "source": "Bloomberg (simulé)",
            "date": "2024-04-25",
            "ticker": "MSFT",
            "url": "https://example.com/news/microsoft-q3-2024",
            "category": "earnings"
        },
        {
            "id": "article_003",
            "title": "NVIDIA devient la première capitalisation mondiale avec 3 300 Md$ grâce à la demande IA",
            "content": (
                "NVIDIA Corporation (NVDA) a brièvement dépassé Microsoft pour devenir la plus grande capitalisation "
                "boursière mondiale, atteignant 3 335 milliards de dollars le 18 juin 2024, portée par une demande "
                "insatiable pour ses processeurs graphiques dédiés à l'intelligence artificielle. "
                "L'action NVDA a progressé de 197% depuis le début de l'année 2024, après un gain de 239% en 2023. "
                "Les GPU H100 et H200 de NVIDIA sont au coeur des infrastructures d'entraînement et d'inférence "
                "des grands modèles de langage (LLM) comme GPT-4, Gemini et Claude. "
                "La demande excède largement l'offre disponible, avec des délais de livraison de 8 à 12 mois "
                "pour les GPU H100 et une liste d'attente de plusieurs milliards de dollars. "
                "Au dernier trimestre (T1 FY2025, clos en avril 2024), NVIDIA a réalisé 26,0 Md$ de chiffre "
                "d'affaires (+262% YoY) avec une marge brute record de 78,4%. "
                "Le PDG Jensen Huang a affirmé : 'Nous sommes au début d'un changement de paradigme computing. "
                "Les centres de données du monde entier sont en train d'être réinventés pour l'ère de l'IA.' "
                "Les analystes estiment que le marché des accélérateurs IA atteindra 200 Md$ en 2024."
            ),
            "source": "Financial Times (simulé)",
            "date": "2024-06-19",
            "ticker": "NVDA",
            "url": "https://example.com/news/nvidia-market-cap-2024",
            "category": "market"
        },
        {
            "id": "article_004",
            "title": "Apple Intelligence : la stratégie IA d'Apple pour regagner le leadership technologique",
            "content": (
                "Apple a présenté lors de la WWDC 2024 sa stratégie d'intelligence artificielle baptisée "
                "'Apple Intelligence', intégrée à iOS 18, iPadOS 18 et macOS Sequoia. "
                "Contrairement à ses concurrents, Apple mise sur une approche hybride : traitement local "
                "sur les appareils Apple Silicon pour les données sensibles, et recours optionnel à OpenAI "
                "GPT-4o pour les requêtes complexes nécessitant plus de puissance. "
                "Les fonctionnalités phares incluent : la réécriture intelligente de texte, la génération "
                "d'images (Image Playground), un Siri considérablement amélioré avec compréhension du contexte "
                "et intégration cross-apps, et la gestion prioritaire des notifications par IA. "
                "Certains analystes estiment qu'Apple Intelligence pourrait déclencher un 'super cycle' "
                "de renouvellement d'iPhone, les utilisateurs souhaitant accéder aux nouvelles fonctionnalités "
                "IA réservées aux iPhone 15 Pro et futurs iPhone 16. "
                "Dan Ives de Wedbush Securities a relevé son objectif de cours sur AAPL à 285$ (vs 225$ auparavant), "
                "estimant que l'IA pourrait ajouter 30-40 Md$ de revenus incrémentaux d'ici 2026. "
                "La fonctionnalité sera disponible en bêta dès septembre 2024 en anglais américain."
            ),
            "source": "The Verge (simulé)",
            "date": "2024-06-10",
            "ticker": "AAPL",
            "url": "https://example.com/news/apple-intelligence-wwdc-2024",
            "category": "technology"
        },
        {
            "id": "article_005",
            "title": "La Fed réduit ses taux directeurs pour la première fois depuis 2020",
            "content": (
                "La Réserve Fédérale américaine (Fed) a abaissé ses taux directeurs de 25 points de base "
                "lors de sa réunion du 17-18 septembre 2024, portant la fourchette cible des Fed Funds "
                "à 5,00%-5,25%, la première baisse depuis mars 2020 en pleine pandémie COVID-19. "
                "Le président Jerome Powell a justifié cette décision par les 'progrès substantiels' "
                "réalisés dans la lutte contre l'inflation et les signes de modération du marché de l'emploi. "
                "L'inflation PCE (mesure préférée de la Fed) s'établit à 2,2% en août 2024, proche de "
                "l'objectif de 2%. Le taux de chômage est remonté à 4,3%, contre un point bas de 3,4% en 2023. "
                "Les marchés actions ont réagi positivement : le S&P 500 a progressé de 1,7% et le Nasdaq de 2,5%. "
                "Les obligations d'État ont également bondi, avec le taux 10 ans américain qui s'est replié "
                "de 12pb à 3,68%. "
                "Le 'dot plot' de la Fed anticipe deux baisses supplémentaires de 25pb d'ici fin 2024 "
                "et quatre autres en 2025, ramenant les taux vers 3,25%-3,50%."
            ),
            "source": "Wall Street Journal (simulé)",
            "date": "2024-09-18",
            "ticker": "SPY",
            "url": "https://example.com/news/fed-rate-cut-september-2024",
            "category": "macro"
        },
        {
            "id": "article_006",
            "title": "Microsoft-OpenAI : l'alliance stratégique qui redessine le cloud computing",
            "content": (
                "L'investissement total de Microsoft dans OpenAI, estimé à 13 milliards de dollars depuis 2019, "
                "se révèle être l'un des paris technologiques les plus rentables de la décennie. "
                "L'intégration d'Azure OpenAI Service a permis à Microsoft de différencier son offre cloud "
                "de manière substantielle face à Amazon Web Services et Google Cloud Platform. "
                "Azure OpenAI Service compte désormais plus de 65 000 clients actifs, dont 65% des entreprises "
                "Fortune 500, générant des revenus estimés à 3-4 Md$ sur une base annuelle en 2024. "
                "L'avantage compétitif de Microsoft réside dans la distribution : chaque employé utilisant "
                "Office 365 est un client potentiel de Copilot, créant un levier de croissance sans équivalent. "
                "À 30$/mois par utilisateur, si seulement 10% des 400 millions d'utilisateurs M365 adoptent "
                "Copilot, cela représente un potentiel de revenu additionnel de 14,4 Md$/an. "
                "Bernstein Research maintient une recommandation 'Outperform' sur MSFT avec un objectif de 500$, "
                "soulignant que l'IA représente le plus grand catalyseur de croissance depuis l'invention du cloud."
            ),
            "source": "CNBC (simulé)",
            "date": "2024-05-15",
            "ticker": "MSFT",
            "url": "https://example.com/news/microsoft-openai-partnership-2024",
            "category": "technology"
        },
        {
            "id": "article_007",
            "title": "Résultats S&P 500 T2 2024 : la saison des résultats dépasse les attentes",
            "content": (
                "La saison des résultats du deuxième trimestre 2024 du S&P 500 s'achève sur une note "
                "très positive, avec 79% des entreprises ayant publié des bénéfices supérieurs aux attentes "
                "des analystes, au-dessus de la moyenne historique de 74%. "
                "La croissance agrégée des bénéfices par action du S&P 500 ressort à +11,5% YoY, "
                "la meilleure performance depuis le T1 2022, portée par les secteurs Technologie (+20%) "
                "et Communication (+16%). "
                "Les 'Magnificent 7' (Apple, Microsoft, Alphabet, Amazon, Meta, NVIDIA, Tesla) ont "
                "collectivement augmenté leurs bénéfices de 35% YoY, représentant 35% de la capitalisation "
                "totale du S&P 500. "
                "Les revenus agrégés ont progressé de +5,2% YoY, avec une marge opérationnelle médiane "
                "en hausse à 15,8% contre 14,9% au T2 2023. "
                "Les analystes de Goldman Sachs ont relevé leurs prévisions de BPA S&P 500 pour 2024 "
                "à 255$ (+6%) et pour 2025 à 278$ (+9%), impliquant un P/E forward de 21,5x à 5 500 points."
            ),
            "source": "FactSet (simulé)",
            "date": "2024-08-09",
            "ticker": "SPY",
            "url": "https://example.com/news/sp500-q2-2024-earnings",
            "category": "market"
        },
        {
            "id": "article_008",
            "title": "Apple : objectif de cours relevé à 250$ après l'annonce du programme IA",
            "content": (
                "JPMorgan Chase a relevé son objectif de cours sur l'action Apple (AAPL) de 225$ à 250$, "
                "maintenant sa recommandation 'Overweight', suite à l'annonce détaillée d'Apple Intelligence "
                "et aux perspectives de cycle de renouvellement d'iPhone porté par l'IA. "
                "Les analystes de JPMorgan estiment qu'Apple Intelligence pourrait déclencher un cycle de "
                "renouvellement de 2 ans touchant 20-30% de la base installée d'iPhone (estimée à 1,2 milliard "
                "d'appareils actifs). Cela représenterait 240-360 millions d'iPhones vendus sur 2024-2026, "
                "soit une progression de 10-20% par rapport aux niveaux actuels de ventes annuelles. "
                "La note souligne également la montée en puissance du segment Services : avec un P/E de 35x "
                "pour les Services contre 28x pour le groupe, toute accélération de la croissance Services "
                "est immédiatement relutive pour la valorisation globale d'Apple. "
                "Sur les 52 analystes couvrant AAPL, 38 ont une recommandation 'Buy/Overweight', "
                "12 'Hold/Neutral' et 2 'Underweight', avec un objectif moyen de cours à 228$. "
                "L'action AAPL évolue actuellement à 210$, offrant un potentiel de hausse de 8-19% "
                "par rapport aux objectifs de cours du consensus."
            ),
            "source": "JPMorgan Research (simulé)",
            "date": "2024-06-25",
            "ticker": "AAPL",
            "url": "https://example.com/news/apple-price-target-2024",
            "category": "analyst"
        }
    ]

    for article in articles:
        filepath = NEWS_DIR / f"{article['id']}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {article['id']}.json généré")


def generate_eval_questions():
    """Génère les questions d'évaluation RAGAS."""
    print("Génération de eval_questions.json...")

    questions = [
        {
            "id": "q001",
            "question": "Quel est le chiffre d'affaires total d'Apple en FY2023 ?",
            "ground_truth": "Le chiffre d'affaires total d'Apple en FY2023 était de 383,3 milliards de dollars (383 285 millions), en baisse de 2,8% par rapport à FY2022.",
            "document": "apple_annual_report_2023.pdf",
            "category": "revenue"
        },
        {
            "id": "q002",
            "question": "Quelle est la marge nette d'Apple en FY2023 ?",
            "ground_truth": "La marge nette d'Apple en FY2023 est de 25,3%, avec un bénéfice net de 97,0 milliards de dollars.",
            "document": "apple_annual_report_2023.pdf",
            "category": "profitability"
        },
        {
            "id": "q003",
            "question": "Quel segment génère le plus de revenus pour Apple en FY2023 ?",
            "ground_truth": "L'iPhone est le segment générant le plus de revenus pour Apple en FY2023, avec 200,6 milliards de dollars, représentant 52,3% du chiffre d'affaires total.",
            "document": "apple_annual_report_2023.pdf",
            "category": "segments"
        },
        {
            "id": "q004",
            "question": "Quelle est la croissance du segment Services d'Apple en FY2023 ?",
            "ground_truth": "Le segment Services d'Apple a crû de 16,1% en FY2023, passant de 73,4 milliards de dollars en FY2022 à 85,2 milliards en FY2023.",
            "document": "apple_annual_report_2023.pdf",
            "category": "growth"
        },
        {
            "id": "q005",
            "question": "Quel est l'EPS dilué d'Apple en FY2023 ?",
            "ground_truth": "Le bénéfice par action (EPS) dilué d'Apple en FY2023 est de 6,13 dollars, en progression de 0,3% par rapport à 6,11 dollars en FY2022.",
            "document": "apple_annual_report_2023.pdf",
            "category": "per_share"
        },
        {
            "id": "q006",
            "question": "Quel est le chiffre d'affaires de Microsoft au T4 FY2024 ?",
            "ground_truth": "Microsoft a réalisé un chiffre d'affaires de 64,7 milliards de dollars au T4 FY2024, en hausse de 15,2% par rapport au T4 FY2023 (56,2 Md$).",
            "document": "microsoft_q4_2024.pdf",
            "category": "revenue"
        },
        {
            "id": "q007",
            "question": "Quelle est la croissance d'Azure au T4 FY2024 ?",
            "ground_truth": "Azure et les services cloud de Microsoft ont affiché une croissance de 29% au T4 FY2024, avec des revenus de 28,5 milliards de dollars contre 22,1 Md$ au T4 FY2023.",
            "document": "microsoft_q4_2024.pdf",
            "category": "growth"
        },
        {
            "id": "q008",
            "question": "Combien Microsoft a-t-il de sièges Copilot déployés dans les entreprises ?",
            "ground_truth": "Microsoft a déployé plus de 1,8 million de sièges Copilot dans les entreprises au T4 FY2024, à un prix de 30 dollars par mois par utilisateur.",
            "document": "microsoft_q4_2024.pdf",
            "category": "ai"
        },
        {
            "id": "q009",
            "question": "Quelle est la marge brute annuelle de Microsoft en FY2024 ?",
            "ground_truth": "La marge brute annuelle de Microsoft en FY2024 est de 69,8% (171 008 millions de dollars), en hausse de 0,9 point de pourcentage par rapport à FY2023 (68,9%).",
            "document": "microsoft_q4_2024.pdf",
            "category": "profitability"
        },
        {
            "id": "q010",
            "question": "Quelle est la performance du secteur technologique sur le S&P 500 YTD 2024 ?",
            "ground_truth": "Le secteur technologique du S&P 500 a enregistré une performance de +28,2% depuis le début de l'année 2024, avec un P/E médian de 32x et une recommandation Surpondérer.",
            "document": "market_overview_2024.pdf",
            "category": "market"
        },
        {
            "id": "q011",
            "question": "Quel est le niveau du S&P 500 au 30 juin 2024 et sa performance YTD ?",
            "ground_truth": "Le S&P 500 était à 5 460 points au 30 juin 2024, avec une performance YTD de +14,5% et une variation sur 12 mois de +22,1%.",
            "document": "market_overview_2024.pdf",
            "category": "market"
        },
        {
            "id": "q012",
            "question": "Quelle est la croissance du PIB américain au T2 2024 ?",
            "ground_truth": "Le PIB américain a crû de 2,8% en rythme trimestriel annualisé au T2 2024, après une croissance de 1,4% au T1 2024.",
            "document": "market_overview_2024.pdf",
            "category": "macro"
        },
        {
            "id": "q013",
            "question": "Apple a-t-il racheté des actions en FY2023 ? Pour quel montant ?",
            "ground_truth": "Apple a racheté pour 77,6 milliards de dollars de ses propres actions en FY2023, en baisse de 13,2% par rapport aux 89,4 Md$ en FY2022.",
            "document": "apple_annual_report_2023.pdf",
            "category": "capital_allocation"
        },
        {
            "id": "q014",
            "question": "Quelle est la capitalisation boursière de Microsoft mentionnée dans les données disponibles ?",
            "ground_truth": "La capitalisation boursière de Microsoft (MSFT) est de 3 300 milliards de dollars selon le rapport de marché 2024.",
            "document": "market_overview_2024.pdf",
            "category": "valuation"
        },
        {
            "id": "q015",
            "question": "Comparez la croissance du chiffre d'affaires d'Apple et de Microsoft pour leurs derniers exercices disponibles.",
            "ground_truth": "Apple a vu son CA baisser de 2,8% en FY2023 (383,3 Md$), tandis que Microsoft a enregistré une forte croissance de +15,7% en FY2024 (245,1 Md$). Microsoft affiche une dynamique de croissance nettement supérieure, notamment grâce à l'accélération d'Azure et de l'IA.",
            "document": "apple_annual_report_2023.pdf,microsoft_q4_2024.pdf",
            "category": "comparison"
        }
    ]

    filepath = SAMPLES_DIR / "eval_questions.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"  ✓ eval_questions.json généré ({len(questions)} questions)")


def generate_portfolio_csv():
    """Génère un fichier CSV de portefeuille simulé."""
    print("Génération de portfolio.csv...")

    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']

    # Base prices (approximate 2023-2024)
    base_prices = {
        'AAPL': 175.0, 'MSFT': 375.0, 'NVDA': 480.0,
        'GOOGL': 140.0, 'AMZN': 185.0, 'META': 490.0, 'TSLA': 245.0
    }

    rows = []
    start_date = datetime(2023, 1, 1)

    prices = dict(base_prices)

    for i in range(365):
        date = start_date + timedelta(days=i)
        if date.weekday() >= 5:  # Skip weekends
            continue

        for ticker in tickers:
            change = random.gauss(0.0003, 0.018)  # Daily return
            prices[ticker] = prices[ticker] * (1 + change)

            market_caps = {
                'AAPL': 2800e9, 'MSFT': 2800e9, 'NVDA': 1200e9,
                'GOOGL': 1800e9, 'AMZN': 1900e9, 'META': 1200e9, 'TSLA': 800e9
            }

            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'price': round(prices[ticker], 2),
                'volume': random.randint(20_000_000, 80_000_000),
                'market_cap_billions': round(market_caps[ticker] * (prices[ticker] / base_prices[ticker]) / 1e9, 1),
                'pe_ratio': round(random.gauss(28, 5), 1),
                'dividend_yield': round(random.uniform(0.0, 0.02), 4)
            })

    filepath = SAMPLES_DIR / "portfolio.csv"
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ portfolio.csv généré ({len(rows)} lignes)")


if __name__ == "__main__":
    print("=" * 60)
    print("FinRAG — Générateur de Données d'Exemple")
    print("=" * 60)
    print()

    generate_pdfs()
    print()
    generate_news_articles()
    print()
    generate_eval_questions()
    print()
    generate_portfolio_csv()

    print()
    print("=" * 60)
    print("✅ Génération terminée !")
    print()
    print("Fichiers générés dans data/samples/ :")
    for f in sorted(Path(SAMPLES_DIR).rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(SAMPLES_DIR)} ({size:,} bytes)")
    print("=" * 60)
