"""
Hierarchical Bayesian Poisson model for Premier League goal prediction.

Model specification (Baio & Blangiardo, 2010):
    goals_home ~ Poisson(theta_home)
    goals_away ~ Poisson(theta_away)

    log(theta_home) = intercept + home_adv + attack[h] - defense[a]
    log(theta_away) = intercept            + attack[a] - defense[h]

Hierarchical priors:
    attack[k]  ~ Normal(mu_att, sigma_att)   for each team k
    defense[k] ~ Normal(mu_def, sigma_def)
    mu_att, mu_def ~ Normal(0, 0.5)
    sigma_att, sigma_def ~ HalfNormal(0.5)
    home_adv   ~ Normal(0.1, 0.2)
    intercept  ~ Normal(0, 1)

Identifiability: sum-to-zero constraint on attack parameters.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pandas as pd


def build_model(df: pd.DataFrame, n_teams: int) -> pm.Model:
    """
    Build the static hierarchical Poisson model.

    Parameters
    ----------
    df : DataFrame with columns home_idx, away_idx, FTHG, FTAG
    n_teams : total number of teams

    Returns
    -------
    PyMC model (not yet sampled)
    """
    home_idx  = df["home_idx"].values
    away_idx  = df["away_idx"].values
    obs_home  = df["FTHG"].values.astype(int)
    obs_away  = df["FTAG"].values.astype(int)

    with pm.Model() as model:
        # --- Hyperpriors ---
        mu_att    = pm.Normal("mu_att",    mu=0,   sigma=0.5)
        mu_def    = pm.Normal("mu_def",    mu=0,   sigma=0.5)
        sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
        sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)

        # --- Team-level parameters ---
        attack_raw = pm.Normal("attack_raw", mu=mu_att, sigma=sigma_att, shape=n_teams)
        defense    = pm.Normal("defense",    mu=mu_def, sigma=sigma_def, shape=n_teams)

        # Sum-to-zero constraint on attack (identifiability)
        attack = pm.Deterministic("attack", attack_raw - pt.mean(attack_raw))

        # --- Fixed effects ---
        home_adv  = pm.Normal("home_adv",  mu=0.1, sigma=0.2)
        intercept = pm.Normal("intercept", mu=0,   sigma=1.0)

        # --- Expected goals ---
        log_mu_home = intercept + home_adv + attack[home_idx] - defense[away_idx]
        log_mu_away = intercept            + attack[away_idx] - defense[home_idx]

        mu_home = pm.math.exp(log_mu_home)
        mu_away = pm.math.exp(log_mu_away)

        # --- Likelihood ---
        pm.Poisson("goals_home", mu=mu_home, observed=obs_home)
        pm.Poisson("goals_away", mu=mu_away, observed=obs_away)

    return model


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> az.InferenceData:
    """Run NUTS sampler and return InferenceData."""
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata


def predict_match(
    idata: az.InferenceData,
    home_idx: int,
    away_idx: int,
    n_samples: int = 4000,
) -> dict:
    """
    Posterior predictive for a single match.

    Returns dict with:
        home_goals, away_goals : arrays of simulated scorelines
        p_home, p_draw, p_away : win/draw/loss probabilities
        home_goals_mean, away_goals_mean : expected goals
        home_goals_hdi, away_goals_hdi : 94% HDI intervals
    """
    posterior = idata.posterior
    attack = posterior["attack"].values.reshape(-1, posterior["attack"].shape[-1])
    defense = posterior["defense"].values.reshape(-1, posterior["defense"].shape[-1])
    home_adv  = posterior["home_adv"].values.flatten()
    intercept = posterior["intercept"].values.flatten()

    idx = np.random.choice(len(intercept), size=n_samples, replace=False)

    log_mu_h = (intercept[idx] + home_adv[idx]
                + attack[idx, home_idx] - defense[idx, away_idx])
    log_mu_a = (intercept[idx]
                + attack[idx, away_idx] - defense[idx, home_idx])

    mu_h = np.exp(log_mu_h)
    mu_a = np.exp(log_mu_a)

    sim_home = np.random.poisson(mu_h)
    sim_away = np.random.poisson(mu_a)

    return {
        "home_goals": sim_home,
        "away_goals": sim_away,
        "p_home": (sim_home > sim_away).mean(),
        "p_draw": (sim_home == sim_away).mean(),
        "p_away": (sim_home < sim_away).mean(),
        "home_goals_mean": mu_h.mean(),
        "away_goals_mean": mu_a.mean(),
        "home_goals_hdi": az.hdi(sim_home.astype(float), hdi_prob=0.94),
        "away_goals_hdi": az.hdi(sim_away.astype(float), hdi_prob=0.94),
    }


def predict_all_matches(
    idata: az.InferenceData,
    df: pd.DataFrame,
    teams: list,
) -> pd.DataFrame:
    """
    Generate posterior predictions for every match in df.
    Returns dataframe with predicted probabilities and expected goals.
    """
    results = []
    for _, row in df.iterrows():
        pred = predict_match(idata, int(row["home_idx"]), int(row["away_idx"]))
        results.append({
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "Date": row["Date"],
            "actual_home": row["FTHG"],
            "actual_away": row["FTAG"],
            "actual_result": row["FTR"],
            "pred_home_goals": pred["home_goals_mean"],
            "pred_away_goals": pred["away_goals_mean"],
            "p_home": pred["p_home"],
            "p_draw": pred["p_draw"],
            "p_away": pred["p_away"],
        })
    return pd.DataFrame(results)


def team_strengths(idata: az.InferenceData, teams: list) -> pd.DataFrame:
    """
    Summarise posterior attack/defense parameters per team.
    """
    posterior = idata.posterior
    attack  = posterior["attack"].values.reshape(-1, len(teams))
    defense = posterior["defense"].values.reshape(-1, len(teams))

    rows = []
    for i, team in enumerate(teams):
        rows.append({
            "team": team,
            "attack_mean":  attack[:, i].mean(),
            "attack_hdi_lo": az.hdi(attack[:, i], hdi_prob=0.94)[0],
            "attack_hdi_hi": az.hdi(attack[:, i], hdi_prob=0.94)[1],
            "defense_mean":  defense[:, i].mean(),
            "defense_hdi_lo": az.hdi(defense[:, i], hdi_prob=0.94)[0],
            "defense_hdi_hi": az.hdi(defense[:, i], hdi_prob=0.94)[1],
        })
    return pd.DataFrame(rows).sort_values("attack_mean", ascending=False)
