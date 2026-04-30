"""
Dynamic Bayesian Poisson model — team strengths evolve over matchweeks.

Each team's attack and defense parameter follows a Gaussian Random Walk
(GRW) over time, capturing form dynamics (injuries, new signings, etc.).

Model:
    attack[k, t]  = attack[k, t-1]  + noise_att[k, t],  noise ~ N(0, sigma_walk_att)
    defense[k, t] = defense[k, t-1] + noise_def[k, t],  noise ~ N(0, sigma_walk_def)

    log(mu_home) = intercept + home_adv + attack[h, t] - defense[a, t]
    log(mu_away) = intercept            + attack[a, t] - defense[h, t]
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pandas as pd


def build_dynamic_model(df: pd.DataFrame, n_teams: int, n_weeks: int) -> pm.Model:
    """
    Build the dynamic hierarchical Poisson model.

    Parameters
    ----------
    df        : DataFrame with home_idx, away_idx, matchweek (1-indexed), FTHG, FTAG
    n_teams   : number of teams
    n_weeks   : total number of matchweeks
    """
    home_idx  = df["home_idx"].values
    away_idx  = df["away_idx"].values
    week_idx  = (df["matchweek"].values - 1).astype(int)   # 0-indexed
    obs_home  = df["FTHG"].values.astype(int)
    obs_away  = df["FTAG"].values.astype(int)

    with pm.Model() as model:
        # --- Random-walk volatility ---
        sigma_walk_att = pm.HalfNormal("sigma_walk_att", sigma=0.1)
        sigma_walk_def = pm.HalfNormal("sigma_walk_def", sigma=0.1)

        # --- Initial team strengths (t=0) ---
        mu_att_0    = pm.Normal("mu_att_0",    mu=0,   sigma=0.5)
        mu_def_0    = pm.Normal("mu_def_0",    mu=0,   sigma=0.5)
        sigma_att_0 = pm.HalfNormal("sigma_att_0", sigma=0.5)
        sigma_def_0 = pm.HalfNormal("sigma_def_0", sigma=0.5)

        att_init = pm.Normal("att_init", mu=mu_att_0, sigma=sigma_att_0, shape=n_teams)
        def_init = pm.Normal("def_init", mu=mu_def_0, sigma=sigma_def_0, shape=n_teams)

        # --- Gaussian Random Walk innovations (weeks × teams) ---
        # Shape: (n_weeks-1, n_teams) innovations, then cumsum to get trajectory
        att_innovations = pm.Normal(
            "att_innovations",
            mu=0,
            sigma=sigma_walk_att,
            shape=(n_weeks - 1, n_teams),
        )
        def_innovations = pm.Normal(
            "def_innovations",
            mu=0,
            sigma=sigma_walk_def,
            shape=(n_weeks - 1, n_teams),
        )

        # Full trajectory: shape (n_weeks, n_teams)
        att_traj = pm.Deterministic(
            "att_traj",
            pt.concatenate(
                [att_init[None, :], att_init[None, :] + pt.cumsum(att_innovations, axis=0)],
                axis=0,
            ),
        )
        def_traj = pm.Deterministic(
            "def_traj",
            pt.concatenate(
                [def_init[None, :], def_init[None, :] + pt.cumsum(def_innovations, axis=0)],
                axis=0,
            ),
        )

        # --- Fixed effects ---
        home_adv  = pm.Normal("home_adv",  mu=0.1, sigma=0.2)
        intercept = pm.Normal("intercept", mu=0,   sigma=1.0)

        # --- Per-match parameters ---
        att_home = att_traj[week_idx, home_idx]
        def_home = def_traj[week_idx, home_idx]
        att_away = att_traj[week_idx, away_idx]
        def_away = def_traj[week_idx, away_idx]

        log_mu_home = intercept + home_adv + att_home - def_away
        log_mu_away = intercept            + att_away - def_home

        mu_home = pm.math.exp(log_mu_home)
        mu_away = pm.math.exp(log_mu_away)

        # --- Likelihood ---
        pm.Poisson("goals_home", mu=mu_home, observed=obs_home)
        pm.Poisson("goals_away", mu=mu_away, observed=obs_away)

    return model


def sample_dynamic_model(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> az.InferenceData:
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


def get_latest_strengths(idata: az.InferenceData, teams: list) -> pd.DataFrame:
    """
    Return the posterior distribution of team strength at the LAST matchweek.
    Useful for predicting upcoming fixtures.
    """
    att_traj = idata.posterior["att_traj"].values  # (chains, draws, weeks, teams)
    def_traj = idata.posterior["def_traj"].values

    # Flatten chains × draws, take last week
    n_c, n_d, n_w, n_t = att_traj.shape
    att_last = att_traj.reshape(n_c * n_d, n_w, n_t)[:, -1, :]
    def_last = def_traj.reshape(n_c * n_d, n_w, n_t)[:, -1, :]

    rows = []
    for i, team in enumerate(teams):
        rows.append({
            "team": team,
            "attack_mean":   att_last[:, i].mean(),
            "attack_std":    att_last[:, i].std(),
            "defense_mean":  def_last[:, i].mean(),
            "defense_std":   def_last[:, i].std(),
        })
    return pd.DataFrame(rows).sort_values("attack_mean", ascending=False)
