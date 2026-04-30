"""
dashboard.py - Interface graphique interactive TrueSkill
=========================================================
Projet ECE Paris ING4 Groupe 02

Lancement :
    cd groupe-02-trueskill-matchmaking/
    streamlit run src/dashboard.py

L'interface se compose de 4 onglets :
  🎯 Niveau Minimum : convergence µ/σ, classement TrueSkill
  🏆 Niveau Bon     : équipes, matchs nuls, comparaison ELO
  🚀 Niveau Excellent : saisons + decay, TrueSkill 2
  🎮 Matchmaking Live : simulateur interactif de matchs
"""

import sys
import os
import random

# Ajout du dossier src au chemin Python pour les imports
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Imports locaux
from simulation  import creer_joueurs, simuler_tournoi, classement_final
from matchmaking import probabilite_victoire, qualite_match
from elo         import creer_joueurs_elo, simuler_tournoi_elo, classement_elo, comparer_classements
from equipes     import simuler_tournoi_equipes
from dynamique   import simuler_saisons, analyser_progression
from trueskill2  import (creer_joueurs_ts2_depuis_dicts, simuler_tournoi_ts2,
                         classement_ts2, comparer_ts1_ts2, tableau_stats_ts2)

# =============================================================================
# Configuration générale de la page Streamlit
# =============================================================================

st.set_page_config(
    page_title  = "TrueSkill Dashboard — ECE ING4 Gr02",
    page_icon   = "🎮",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Style CSS personnalisé pour embellir l'interface
st.markdown("""
<style>
    /* Titre principal */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    /* Sous-titre */
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* Carte métrique personnalisée */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #667eea;
        margin: 0.3rem 0;
    }
    /* Badge de niveau */
    .badge-minimum  { background:#28a745; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-bon      { background:#fd7e14; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-excellent{ background:#dc3545; color:white; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# En-tête principal
# =============================================================================

st.markdown('<div class="main-title">🎮 TrueSkill Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">ECE Paris ING4 — Groupe 02 | '
    'Matchmaking Compétitif & Inférence Bayésienne</div>',
    unsafe_allow_html=True
)
st.divider()


# =============================================================================
# Barre latérale : paramètres de simulation
# =============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/TrueSkill-graphic.png/220px-TrueSkill-graphic.png",
             use_container_width=True, caption="Graphe de facteurs TrueSkill")
    st.markdown("## ⚙️ Paramètres")

    nb_joueurs = st.slider(
        "Nombre de joueurs",
        min_value=4, max_value=20, value=10, step=1,
        help="Nombre de joueurs dans la simulation"
    )
    nb_matchs = st.slider(
        "Matchs 1v1 (niveau min.)",
        min_value=50, max_value=500, value=200, step=50,
        help="Plus il y a de matchs, meilleure est la convergence"
    )
    graine = st.number_input(
        "Graine aléatoire",
        min_value=0, max_value=9999, value=42,
        help="Changer la graine change les résultats (reproductible)"
    )

    st.divider()
    st.markdown("### 🏆 Niveau Bon")
    taille_equipe = st.slider(
        "Joueurs par équipe",
        min_value=2, max_value=4, value=2,
        help="2 = 2v2, 3 = 3v3, 4 = 4v4"
    )
    draw_prob = st.slider(
        "Probabilité de match nul",
        min_value=0.0, max_value=0.5, value=0.15, step=0.05,
        help="0 = jamais de nul, 0.5 = très fréquent"
    )

    st.divider()
    st.markdown("### 🚀 Niveau Excellent")
    nb_saisons = st.slider(
        "Nombre de saisons",
        min_value=2, max_value=5, value=3,
        help="Nombre de saisons avec decay inter-saison"
    )
    matchs_par_saison = st.slider(
        "Matchs par saison",
        min_value=30, max_value=150, value=70, step=10,
    )
    decay_rate = st.slider(
        "Taux de decay inter-saison",
        min_value=0.05, max_value=0.40, value=0.15, step=0.05,
        help="σ augmente de X% entre deux saisons"
    )

    st.divider()
    lancer = st.button("🚀 Lancer la simulation", type="primary", use_container_width=True)
    st.caption("Cliquez pour (re)lancer avec ces paramètres")


# =============================================================================
# Fonction de simulation principale (mise en cache)
# =============================================================================

@st.cache_data
def lancer_simulation(nb_joueurs, nb_matchs, graine,
                      taille_equipe, draw_prob,
                      nb_saisons, matchs_par_saison, decay_rate):
    """
    Lance toutes les simulations et retourne les données.
    Le cache Streamlit évite de relancer si les paramètres n'ont pas changé.
    """
    random.seed(graine)

    # ── Niveau Minimum ──────────────────────────────────────────────────────
    joueurs   = creer_joueurs(nb_joueurs=nb_joueurs, mu_min=10, mu_max=50)
    historique = simuler_tournoi(joueurs, nb_matchs=nb_matchs)
    classes    = classement_final(joueurs)

    # ── Niveau Bon : ELO ────────────────────────────────────────────────────
    random.seed(graine)
    joueurs_elo = creer_joueurs_elo(joueurs)
    simuler_tournoi_elo(joueurs_elo, historique)
    classes_elo = classement_elo(joueurs_elo)
    comparaison_classements = comparer_classements(classes, classes_elo)

    # ── Niveau Bon : équipes ────────────────────────────────────────────────
    random.seed(graine)
    joueurs_eq = creer_joueurs(nb_joueurs=nb_joueurs, mu_min=10, mu_max=50)
    hist_eq, stats_eq = simuler_tournoi_equipes(
        joueurs_eq,
        nb_matchs=max(50, nb_matchs // 2),
        taille=taille_equipe,
        draw_prob=draw_prob,
        equilibre=True,
    )

    # ── Niveau Excellent : saisons ──────────────────────────────────────────
    random.seed(graine)
    joueurs_dyn = creer_joueurs(nb_joueurs=nb_joueurs, mu_min=10, mu_max=50)
    hist_dyn, info_saisons, breakpoints_saisons = simuler_saisons(
        joueurs_dyn,
        nb_saisons=nb_saisons,
        matchs_par_saison=matchs_par_saison,
        decay_inter_saison=decay_rate,
    )

    # ── Niveau Excellent : TrueSkill 2 ─────────────────────────────────────
    random.seed(graine)
    joueurs_ts2 = creer_joueurs_ts2_depuis_dicts(joueurs)
    simuler_tournoi_ts2(joueurs_ts2, nb_matchs=nb_matchs)
    classes_ts2   = classement_ts2(joueurs_ts2)
    comparaison_ts = comparer_ts1_ts2(classes_ts2, classes)
    stats_ts2     = tableau_stats_ts2(classes_ts2)

    return {
        # Minimum
        'joueurs':     joueurs,
        'historique':  historique,
        'classes':     classes,
        # ELO
        'joueurs_elo': joueurs_elo,
        'classes_elo': classes_elo,
        'comp_classements': comparaison_classements,
        # Équipes
        'joueurs_eq':  joueurs_eq,
        'hist_eq':     hist_eq,
        'stats_eq':    stats_eq,
        # Saisons
        'joueurs_dyn':       joueurs_dyn,
        'info_saisons':      info_saisons,
        'breakpoints_saisons': breakpoints_saisons,
        # TrueSkill 2
        'classes_ts2':  classes_ts2,
        'comp_ts':      comparaison_ts,
        'stats_ts2':    stats_ts2,
    }


# =============================================================================
# Lancement automatique au démarrage
# =============================================================================

with st.spinner("⏳ Simulation en cours..."):
    data = lancer_simulation(
        nb_joueurs, nb_matchs, int(graine),
        taille_equipe, draw_prob,
        nb_saisons, matchs_par_saison, decay_rate
    )

st.success(f"✅ Simulation terminée — {nb_matchs} matchs, {nb_joueurs} joueurs")


# =============================================================================
# Couleurs pour les graphes (une couleur par joueur)
# =============================================================================

PALETTE = px.colors.qualitative.Plotly


def couleur(i):
    return PALETTE[i % len(PALETTE)]


# =============================================================================
# ONGLETS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Niveau Minimum",
    "🏆 Niveau Bon",
    "🚀 Niveau Excellent",
    "🎮 Matchmaking Live",
])


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1 — NIVEAU MINIMUM
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## 🎯 Niveau Minimum — TrueSkill 1v1")
    st.markdown(
        "Simulation de **matchs 1v1** avec mise à jour TrueSkill. "
        "On observe comment µ converge vers la vraie compétence et σ diminue."
    )

    joueurs   = data['joueurs']
    classes   = data['classes']
    historique = data['historique']

    # ── Métriques rapides ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Joueurs", nb_joueurs)
    col2.metric("⚔️ Matchs joués", nb_matchs)
    top1 = classes[0]
    col3.metric("🥇 Meilleur joueur", top1['nom'], f"µ={top1['rating'].mu:.1f}")
    sigma_moy = sum(j['rating'].sigma for j in joueurs) / len(joueurs)
    col4.metric("📉 σ moyen final", f"{sigma_moy:.2f}", "↓ depuis 8.33")

    st.divider()

    # ── Graphe 1 : Convergence de µ ───────────────────────────────────────────
    st.markdown("### 📈 Convergence de µ vers la vraie compétence")
    st.caption("Lignes pleines = µ estimé | Lignes pointillées = vraie compétence cachée")

    fig_mu = go.Figure()
    for i, j in enumerate(joueurs):
        x = list(range(len(j['historique_mu'])))
        c = couleur(i)
        # Courbe µ
        fig_mu.add_trace(go.Scatter(
            x=x, y=j['historique_mu'],
            mode='lines', name=j['nom'],
            line=dict(color=c, width=2),
            legendgroup=j['nom'],
        ))
        # Ligne de compétence cachée (pointillée)
        fig_mu.add_trace(go.Scatter(
            x=[0, len(j['historique_mu']) - 1],
            y=[j['competence'], j['competence']],
            mode='lines',
            line=dict(color=c, dash='dot', width=1),
            showlegend=False,
            legendgroup=j['nom'],
            hovertemplate=f"{j['nom']} — vraie compétence: {j['competence']:.1f}",
        ))

    fig_mu.update_layout(
        xaxis_title="Nombre de matchs joués",
        yaxis_title="µ (estimation TrueSkill)",
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_dark',
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_mu, use_container_width=True)

    # ── Graphe 2 : Convergence de σ ───────────────────────────────────────────
    st.markdown("### 📉 Diminution de σ (certitude croissante)")
    st.caption("σ → 0 signifie que le système est de plus en plus certain du niveau du joueur")

    fig_sigma = go.Figure()
    for i, j in enumerate(joueurs):
        x = list(range(len(j['historique_sigma'])))
        fig_sigma.add_trace(go.Scatter(
            x=x, y=j['historique_sigma'],
            mode='lines', name=j['nom'],
            line=dict(color=couleur(i), width=2),
            fill='tozeroy',
            fillcolor=couleur(i).replace('rgb', 'rgba').replace(')', ',0.05)') if 'rgb' in couleur(i) else couleur(i),
        ))

    fig_sigma.update_layout(
        xaxis_title="Nombre de matchs joués",
        yaxis_title="σ (incertitude)",
        height=380,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_dark',
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_sigma, use_container_width=True)

    # ── Graphe 3 : Classement final ───────────────────────────────────────────
    st.markdown("### 🏅 Classement final TrueSkill")

    noms      = [j['nom'] for j in classes]
    mus       = [j['rating'].mu for j in classes]
    sigmas    = [j['rating'].sigma for j in classes]
    scores_ts = [j['rating'].mu - 3 * j['rating'].sigma for j in classes]
    competences = [j['competence'] for j in classes]

    fig_class = go.Figure()
    # Barres µ avec barre d'erreur ±σ
    fig_class.add_trace(go.Bar(
        x=noms, y=mus,
        error_y=dict(type='data', array=sigmas, visible=True),
        name='µ ± σ', marker_color='steelblue', opacity=0.75,
    ))
    # Points vraie compétence
    fig_class.add_trace(go.Scatter(
        x=noms, y=competences,
        mode='markers', name='Vraie compétence',
        marker=dict(color='tomato', size=12, symbol='diamond'),
    ))
    # Points score conservateur
    fig_class.add_trace(go.Scatter(
        x=noms, y=scores_ts,
        mode='markers', name='Score conservateur (µ-3σ)',
        marker=dict(color='gold', size=10, symbol='circle'),
    ))

    fig_class.update_layout(
        xaxis_title="Joueur (trié par rang)",
        yaxis_title="Niveau",
        height=420,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=40, b=40),
        bargap=0.3,
    )
    st.plotly_chart(fig_class, use_container_width=True)

    # ── Tableau classement ─────────────────────────────────────────────────────
    st.markdown("### 📊 Tableau de classement")
    rows = []
    for rang, j in enumerate(classes, 1):
        score = j['rating'].mu - 3 * j['rating'].sigma
        erreur = abs(j['rating'].mu - j['competence'])
        rows.append({
            'Rang':          rang,
            'Joueur':        j['nom'],
            'µ (estimé)':   round(j['rating'].mu, 2),
            'σ (incert.)':  round(j['rating'].sigma, 2),
            'Score (µ-3σ)': round(score, 2),
            'Vrai niveau':  round(j['competence'], 2),
            'Erreur |µ-vrai|': round(erreur, 2),
        })
    df_class = pd.DataFrame(rows)
    st.dataframe(df_class, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2 — NIVEAU BON
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("## 🏆 Niveau Bon — Équipes & ELO")

    # ── Section A : Comparaison ELO vs TrueSkill ──────────────────────────────
    st.markdown("### ⚔️ Comparaison ELO vs TrueSkill")
    st.info(
        "Les deux systèmes ont joué **exactement les mêmes matchs**. "
        "On compare comment ils classent les joueurs au final."
    )

    comp = data['comp_classements']
    joueurs_elo = data['joueurs_elo']
    classes_elo = data['classes_elo']

    col_a, col_b = st.columns(2)

    with col_a:
        # Évolution ELO dans le temps
        fig_elo = go.Figure()
        for i, j in enumerate(joueurs_elo):
            fig_elo.add_trace(go.Scatter(
                x=list(range(len(j['historique_elo']))),
                y=j['historique_elo'],
                mode='lines', name=j['nom'],
                line=dict(color=couleur(i), width=1.8),
            ))
            # Compétence cachée (en points, non en ELO, mais normalisée)
        fig_elo.update_layout(
            title="Évolution du rating ELO",
            xaxis_title="Matchs joués",
            yaxis_title="Rating ELO",
            height=380, template='plotly_dark',
            legend=dict(font=dict(size=9)),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_elo, use_container_width=True)

    with col_b:
        # Comparaison des classements
        df_comp = pd.DataFrame(comp)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=df_comp['joueur'], y=df_comp['rang_ts'],
            name='Rang TrueSkill', marker_color='steelblue',
        ))
        fig_comp.add_trace(go.Bar(
            x=df_comp['joueur'], y=df_comp['rang_elo'],
            name='Rang ELO', marker_color='tomato',
        ))
        fig_comp.update_layout(
            title="Rang TrueSkill vs Rang ELO",
            barmode='group',
            xaxis_title="Joueur",
            yaxis_title="Rang (1 = meilleur)",
            yaxis=dict(autorange='reversed'),
            height=380, template='plotly_dark',
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Tableau comparatif
    st.markdown("#### 📋 Écarts de classement ELO vs TrueSkill")
    df_comp_display = pd.DataFrame(comp).rename(columns={
        'joueur': 'Joueur',
        'rang_ts': 'Rang TrueSkill',
        'rang_elo': 'Rang ELO',
        'ecart': 'Écart de rang',
    })
    st.dataframe(
        df_comp_display,
        use_container_width=True, hide_index=True
    )

    # Explication pédagogique
    ecart_moy = sum(c['ecart'] for c in comp) / len(comp)
    st.markdown(f"""
    > **Interprétation** : L'écart moyen de classement entre ELO et TrueSkill est de **{ecart_moy:.1f} positions**.
    > TrueSkill converge plus vite car il modélise l'**incertitude σ** — chaque match apporte plus d'information.
    """)

    st.divider()

    # ── Section B : Matchs par équipes avec matchs nuls ───────────────────────
    st.markdown("### 👥 Matchs par équipes avec marge de nul")

    stats_eq  = data['stats_eq']
    hist_eq   = data['hist_eq']
    joueurs_eq = data['joueurs_eq']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⚔️ Matchs joués",       stats_eq['nb_matchs'])
    col2.metric("🤝 Matchs nuls",        stats_eq['nb_nuls'], f"{stats_eq['pct_nuls']}%")
    col3.metric("🏆 Victoires équipe 1", stats_eq['nb_e1_gagne'])
    col4.metric("🏆 Victoires équipe 2", stats_eq['nb_e2_gagne'])

    col_left, col_right = st.columns(2)

    with col_left:
        # Graphe camembert des résultats
        fig_pie = go.Figure(go.Pie(
            labels=['Équipe 1 gagne', 'Nul', 'Équipe 2 gagne'],
            values=[stats_eq['nb_e1_gagne'], stats_eq['nb_nuls'], stats_eq['nb_e2_gagne']],
            marker_colors=['steelblue', 'gold', 'tomato'],
            hole=0.4,
        ))
        fig_pie.update_layout(
            title=f"Résultats des matchs {taille_equipe}v{taille_equipe}",
            height=340, template='plotly_dark',
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Évolution µ après matchs d'équipe
        classes_eq = sorted(joueurs_eq, key=lambda j: j['rating'].mu - 3 * j['rating'].sigma, reverse=True)
        fig_eq = go.Figure()
        for i, j in enumerate(joueurs_eq):
            fig_eq.add_trace(go.Scatter(
                x=list(range(len(j['historique_mu']))),
                y=j['historique_mu'],
                mode='lines', name=j['nom'],
                line=dict(color=couleur(i), width=1.5),
            ))
        fig_eq.update_layout(
            title=f"Convergence µ en mode {taille_equipe}v{taille_equipe}",
            xaxis_title="Matchs", yaxis_title="µ estimé",
            height=340, template='plotly_dark',
            legend=dict(font=dict(size=8)),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # Tableau des 10 premiers matchs d'équipe
    if hist_eq:
        st.markdown("#### 📋 Aperçu des matchs par équipes")
        rows_eq = []
        for m in hist_eq[:15]:
            rows_eq.append({
                'Match': m['match_num'],
                'Équipe 1': ', '.join(m['equipe1']),
                'Équipe 2': ', '.join(m['equipe2']),
                'Résultat': ('🤝 Nul' if m['resultat'] == 'nul'
                             else ('✅ E1' if m['resultat'] == 'equipe1' else '✅ E2')),
            })
        st.dataframe(pd.DataFrame(rows_eq), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — NIVEAU EXCELLENT
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## 🚀 Niveau Excellent — Dynamique temporelle & TrueSkill 2")

    # ── Section A : Saisons et decay ──────────────────────────────────────────
    st.markdown("### 🗓️ Simulation par saisons avec dégradation inter-saison")
    st.info(
        f"**{nb_saisons} saisons** de {matchs_par_saison} matchs chacune. "
        f"Entre chaque saison, σ augmente de **{decay_rate*100:.0f}%** (inactivité). "
        "Les compétences cachées évoluent légèrement entre les saisons."
    )

    joueurs_dyn          = data['joueurs_dyn']
    info_saisons         = data['info_saisons']
    # breakpoints[s] = indice réel dans l'historique individuel où la saison s finit
    breakpoints          = data['breakpoints_saisons']

    # Graphe σ avec marqueurs de saison
    fig_dyn_sigma = go.Figure()
    for i, j in enumerate(joueurs_dyn):
        fig_dyn_sigma.add_trace(go.Scatter(
            x=list(range(len(j['historique_sigma']))),
            y=j['historique_sigma'],
            mode='lines', name=j['nom'],
            line=dict(color=couleur(i), width=1.8),
        ))

    # Lignes verticales aux vraies positions inter-saison (sauf après la dernière saison)
    for s_idx, bp in enumerate(breakpoints[:-1]):
        fig_dyn_sigma.add_vline(
            x=bp,
            line=dict(color='rgba(255,200,0,0.6)', dash='dash', width=2),
            annotation_text=f"Inter-saison {s_idx+1}→{s_idx+2}",
            annotation_position="top",
        )

    fig_dyn_sigma.update_layout(
        title="Évolution de σ par joueur (sauts jaunes = decay inter-saison)",
        xaxis_title="Matchs joués par ce joueur (toutes saisons)",
        yaxis_title="σ (incertitude)",
        height=400, template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=9)),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_dyn_sigma, use_container_width=True)

    # Graphe µ sur les saisons
    fig_dyn_mu = go.Figure()
    for i, j in enumerate(joueurs_dyn):
        x_range = list(range(len(j['historique_mu'])))
        fig_dyn_mu.add_trace(go.Scatter(
            x=x_range,
            y=j['historique_mu'],
            mode='lines', name=j['nom'],
            line=dict(color=couleur(i), width=1.8),
        ))
        fig_dyn_mu.add_trace(go.Scatter(
            x=[0, len(j['historique_mu']) - 1],
            y=[j['competence'], j['competence']],
            mode='lines',
            line=dict(color=couleur(i), dash='dot', width=1),
            showlegend=False,
        ))
    for s_idx, bp in enumerate(breakpoints[:-1]):
        fig_dyn_mu.add_vline(
            x=bp,
            line=dict(color='rgba(255,200,0,0.4)', dash='dash', width=2),
            annotation_text=f"S{s_idx+1}→S{s_idx+2}",
            annotation_position="top right",
        )

    fig_dyn_mu.update_layout(
        title="Convergence de µ sur plusieurs saisons",
        xaxis_title="Matchs cumulés",
        yaxis_title="µ (estimation)",
        height=400, template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=9)),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_dyn_mu, use_container_width=True)

    # Classements par saison (heatmap)
    st.markdown("#### 🗺️ Évolution des classements saison par saison")
    saison_rangs = {}
    for snap in info_saisons:
        for i, j_info in enumerate(snap['joueurs']):
            if j_info['nom'] not in saison_rangs:
                saison_rangs[j_info['nom']] = {}
            saison_rangs[j_info['nom']][f"Saison {snap['saison']}"] = i + 1

    df_rangs = pd.DataFrame(saison_rangs).T
    df_rangs = df_rangs.sort_values(df_rangs.columns[-1])  # Tri par dernier rang

    fig_heat = go.Figure(go.Heatmap(
        z=df_rangs.values,
        x=df_rangs.columns.tolist(),
        y=df_rangs.index.tolist(),
        colorscale='RdYlGn_r',
        showscale=True,
        colorbar=dict(title="Rang"),
        text=df_rangs.values,
        texttemplate="%{text}",
    ))
    fig_heat.update_layout(
        title="Rang par saison (1 = meilleur, vert = bon rang)",
        height=350, template='plotly_dark',
        xaxis_title="Saison",
        yaxis_title="Joueur",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Section B : TrueSkill 2 ───────────────────────────────────────────────
    st.markdown("### 🧠 TrueSkill 2 — Consistance & Activité")
    st.info(
        "TrueSkill 2 (Microsoft, 2018) enrichit le score avec :\n"
        "- **Consistance** : un joueur régulier est mieux classé qu'un joueur imprévisible\n"
        "- **Activité** : un joueur qui joue beaucoup a un score plus fiable"
    )

    classes_ts2 = data['classes_ts2']
    comp_ts     = data['comp_ts']
    stats_ts2   = data['stats_ts2']

    # Graphe : Score TS1 vs Score TS2
    noms_ts2   = [r['joueur'] for r in comp_ts]
    scores_ts1 = [r['score_ts1'] for r in comp_ts]
    scores_ts2_vals = [r['score_ts2'] for r in comp_ts]
    consistances = [r['consistance'] for r in comp_ts]

    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        fig_ts_comp = go.Figure()
        fig_ts_comp.add_trace(go.Bar(
            y=noms_ts2, x=scores_ts1,
            name='Score TrueSkill 1', orientation='h',
            marker_color='steelblue',
        ))
        fig_ts_comp.add_trace(go.Bar(
            y=noms_ts2, x=scores_ts2_vals,
            name='Score TrueSkill 2', orientation='h',
            marker_color='mediumpurple',
        ))
        fig_ts_comp.update_layout(
            title="Score TS1 vs TS2",
            barmode='group',
            xaxis_title="Score",
            height=400, template='plotly_dark',
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_ts_comp, use_container_width=True)

    with col_ts2:
        # Graphe consistance & activité par joueur
        noms_sorted = [r['joueur'] for r in sorted(comp_ts, key=lambda x: x['consistance'], reverse=True)]
        consis_sorted = [r['consistance'] for r in sorted(comp_ts, key=lambda x: x['consistance'], reverse=True)]
        activ_sorted  = [r['activite']     for r in sorted(comp_ts, key=lambda x: x['consistance'], reverse=True)]

        fig_ts2_detail = go.Figure()
        fig_ts2_detail.add_trace(go.Bar(
            x=noms_sorted, y=consis_sorted,
            name='Consistance', marker_color='mediumseagreen',
        ))
        fig_ts2_detail.add_trace(go.Bar(
            x=noms_sorted, y=activ_sorted,
            name='Facteur activité', marker_color='coral',
        ))
        fig_ts2_detail.update_layout(
            title="Consistance & Activité par joueur",
            barmode='group',
            yaxis=dict(range=[0, 1.1]),
            height=400, template='plotly_dark',
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_ts2_detail, use_container_width=True)

    # Tableau complet TrueSkill 2
    st.markdown("#### 📊 Classement TrueSkill 2 complet")
    df_ts2 = pd.DataFrame(stats_ts2).rename(columns={
        'rang': 'Rang', 'joueur': 'Joueur',
        'mu': 'µ', 'sigma': 'σ',
        'score_ts1': 'Score TS1', 'score_ts2': 'Score TS2',
        'consistance': 'Consistance', 'activite': 'Activité',
        'nb_victoires': 'Victoires', 'nb_defaites': 'Défaites',
        'taux_victoire': '% Victoires',
        'competence_reelle': 'Vrai niveau',
    })
    st.dataframe(df_ts2, use_container_width=True, hide_index=True)

    # Explication pédagogique
    st.markdown("""
    > **Pourquoi le score TS2 peut différer de TS1 ?**
    > - Un joueur avec µ élevé mais **inconsistant** voit son score baisser
    > - Un joueur qui a joué **peu de matchs** a un facteur activité bas → score réduit
    > - Formule : `Score TS2 = (µ - 3σ) × consistance × facteur_activité`
    """)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4 — MATCHMAKING LIVE
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("## 🎮 Matchmaking Live — Simulation interactive")
    st.info("Sélectionnez deux joueurs et visualisez leurs statistiques et probabilité de victoire.")

    joueurs   = data['joueurs']
    classes   = data['classes']

    noms_joueurs = [j['nom'] for j in joueurs]

    col_j1, col_j2 = st.columns(2)
    with col_j1:
        choix_j1 = st.selectbox("🔵 Joueur 1", noms_joueurs, index=0)
    with col_j2:
        choix_j2 = st.selectbox("🔴 Joueur 2", noms_joueurs, index=min(1, len(noms_joueurs)-1))

    if choix_j1 == choix_j2:
        st.warning("⚠️ Sélectionnez deux joueurs différents !")
    else:
        j1 = next(j for j in joueurs if j['nom'] == choix_j1)
        j2 = next(j for j in joueurs if j['nom'] == choix_j2)

        prob = probabilite_victoire(j1, j2)
        qualite = qualite_match(j1, j2)

        # ── Métriques du match ─────────────────────────────────────────────────
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"P({j1['nom']} gagne)", f"{prob:.1%}")
        col2.metric(f"P({j2['nom']} gagne)", f"{(1-prob):.1%}")
        col3.metric("Qualité du match", f"{qualite:.3f}",
                    help="1.0 = match parfaitement équilibré, 0 = déséquilibré")

        # Jauge de probabilité
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={'text': f"Probabilité de victoire de {j1['nom']} (%)"},
            delta={'reference': 50, 'increasing': {'color': 'steelblue'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "steelblue"},
                'steps': [
                    {'range': [0, 30],  'color': 'tomato'},
                    {'range': [30, 70], 'color': 'gold'},
                    {'range': [70, 100],'color': 'mediumseagreen'},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': 50,
                }
            }
        ))
        fig_gauge.update_layout(
            height=300, template='plotly_dark',
            margin=dict(t=60, b=20, l=40, r=40),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Comparaison des profils ────────────────────────────────────────────
        st.markdown("### 👤 Comparaison des profils")
        col_l, col_r = st.columns(2)

        def carte_joueur(col, joueur, couleur_accent):
            rang = next(i + 1 for i, j in enumerate(classes) if j['nom'] == joueur['nom'])
            score_ts = joueur['rating'].mu - 3 * joueur['rating'].sigma
            with col:
                st.markdown(f"**{joueur['nom']}**")
                st.metric("µ estimé",         f"{joueur['rating'].mu:.2f}")
                st.metric("σ (incertitude)",  f"{joueur['rating'].sigma:.2f}")
                st.metric("Score (µ-3σ)",     f"{score_ts:.2f}")
                st.metric("Vraie compétence", f"{joueur['competence']:.2f}")
                st.metric("Rang TrueSkill",   f"#{rang}")

        carte_joueur(col_l, j1, 'steelblue')
        carte_joueur(col_r, j2, 'tomato')

        # ── Graphe radar : comparaison visuelle ───────────────────────────────
        st.markdown("### 🕸️ Radar de comparaison")

        def normaliser(val, vmin, vmax):
            return max(0, min(1, (val - vmin) / (vmax - vmin))) if vmax > vmin else 0.5

        mu_max_global = max(j['rating'].mu for j in joueurs)
        mu_min_global = min(j['rating'].mu for j in joueurs)

        categories = ['µ estimé', 'Certitude (1/σ)', 'Score conservateur', 'Vraie compétence', 'µ normalisé']

        def valeurs_radar(joueur):
            return [
                normaliser(joueur['rating'].mu, mu_min_global, mu_max_global),
                normaliser(1.0 / joueur['rating'].sigma, 0.1, 0.7),
                normaliser(joueur['rating'].mu - 3 * joueur['rating'].sigma,
                           min(j['rating'].mu - 3*j['rating'].sigma for j in joueurs),
                           max(j['rating'].mu - 3*j['rating'].sigma for j in joueurs)),
                normaliser(joueur['competence'], 10, 50),
                normaliser(joueur['rating'].mu, mu_min_global, mu_max_global),
            ]

        vals_j1 = valeurs_radar(j1)
        vals_j2 = valeurs_radar(j2)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_j1 + [vals_j1[0]],
            theta=categories + [categories[0]],
            fill='toself', name=j1['nom'],
            line_color='steelblue', fillcolor='rgba(70,130,180,0.2)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_j2 + [vals_j2[0]],
            theta=categories + [categories[0]],
            fill='toself', name=j2['nom'],
            line_color='tomato', fillcolor='rgba(255,99,71,0.2)',
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400, template='plotly_dark',
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Évolutions comparées ───────────────────────────────────────────────
        st.markdown("### 📈 Évolution µ comparée")
        fig_evol = go.Figure()

        for j, c in [(j1, 'steelblue'), (j2, 'tomato')]:
            x = list(range(len(j['historique_mu'])))
            mus_j    = j['historique_mu']
            sigmas_j = j['historique_sigma']

            fig_evol.add_trace(go.Scatter(
                x=x, y=[m + s for m, s in zip(mus_j, sigmas_j)],
                mode='lines', showlegend=False,
                line=dict(color=c, width=0),
                fill=None,
            ))
            fig_evol.add_trace(go.Scatter(
                x=x, y=[m - s for m, s in zip(mus_j, sigmas_j)],
                mode='lines', showlegend=False,
                fill='tonexty',
                fillcolor=f'rgba({",".join(str(v) for v in px.colors.hex_to_rgb(c) if isinstance(c, str) and c.startswith("#"))},0.15)'
                    if isinstance(c, str) and c.startswith('#') else 'rgba(70,130,180,0.1)',
                line=dict(color=c, width=0),
            ))
            fig_evol.add_trace(go.Scatter(
                x=x, y=mus_j,
                mode='lines', name=j['nom'],
                line=dict(color=c, width=2.5),
            ))
            fig_evol.add_hline(
                y=j['competence'], line_dash='dot', line_color=c,
                annotation_text=f"Vrai niveau {j['nom']}: {j['competence']:.1f}",
            )

        fig_evol.update_layout(
            xaxis_title="Matchs joués", yaxis_title="µ",
            height=380, template='plotly_dark',
            hovermode='x unified',
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_evol, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#888; font-size:0.85rem;">
        ECE Paris ING4 — Groupe 02 | Projet TrueSkill & Matchmaking Compétitif<br>
        Références : Herbrich et al. (2006), Minka et al. (2018), MBML Book
    </div>
    """, unsafe_allow_html=True)
