# IMPORTS & CONFIG
import warnings, logging, random, os, sys, subprocess
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import norm, chi2
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_pinball_loss
import lightgbm as lgb
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

def _ensure(pkg, name=None):
    try: __import__(name or pkg)
    except ImportError:
        print(f"Installing {pkg}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("xgboost")
try:    import xgboost as xgb;  HAS_XGB = True
except: HAS_XGB = False

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("CPPS-V4")

OUT = Path("results_v4"); OUT.mkdir(exist_ok=True)

ALPHA           = 0.10
COVERAGE_TGT    = 0.90
ACI_GAMMA       = 0.12
ACI_GAMMA_CRISIS= 0.30
ACI_RECOVERY_STEP= 0.010
ACI_RECOVERY_LAG = 15
CRISIS_VOL_THRESH= 1.8
MIN_CRISIS_DD   = -0.06
RETRAIN_FREQ    = 10
RF_TREES        = 300
LSTM_EPOCHS     = 25
LSTM_HIDDEN     = 64
LSTM_SEQ_LEN    = 20
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
VOL_AUG_SCALES  = [1.5, 2.0, 2.5, 3.5]
ENS_WINDOW      = 20

# Online CP
ONLINE_LAMBDA   = 0.025   # forgetting rate: half-life ≈ 28 days
ONLINE_WINDOW   = 400     # max history length

# Multi-period VIX buckets
VIX_BINS        = [0, 15, 25, 40, 9999]
VIX_LABELS      = ["calm", "normal", "elevated", "crisis"]

ASSETS = ["SPY","QQQ","IWM","EFA","EEM","TLT","LQD","GLD","USO","VNQ"]

CRISIS_PERIODS = {
    "COVID_2020":     ("2020-02-01", "2020-04-30"),
    "Inflation_2022": ("2022-01-01", "2022-06-30"),
}
CRISIS_START = CRISIS_PERIODS["COVID_2020"][0]
CRISIS_END   = CRISIS_PERIODS["COVID_2020"][1]

DATA_START  = "2015-01-01"
DATA_END    = "2022-12-31"
TRAIN_END   = "2017-12-31"
CAL_END     = "2019-12-31"
TEST_START  = "2020-01-01"

PALETTE = {
    "OnlineCPPS":"#26C6DA","CPPS":"#2196F3","Bayesian":"#FF9800",
    "QR":"#9C27B0","AdaptCQR":"#4CAF50","ConfEnsemble":"#FF5722",
    "ScoreHybrid":"#00BCD4",
}
plt.rcParams.update({
    "font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
    "figure.facecolor":"#0d1117","axes.facecolor":"#0d1117","text.color":"white",
    "axes.labelcolor":"white","xtick.color":"white","ytick.color":"white",
    "axes.edgecolor":"#30363d","grid.color":"#21262d","grid.alpha":0.5,
})

def fetch_data(assets, start, end):
    log.info("Fetching %d assets [%s → %s] …", len(assets), start, end)
    raw = yf.download(assets, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(axis=1, thresh=int(len(raw)*0.95)).ffill().bfill()
    returns = np.log(raw / raw.shift(1)).dropna()
    log.info("Returns shape: %s", returns.shape)
    return returns

def fetch_vix(returns: pd.DataFrame) -> pd.Series:
    """Download VIX; fall back to realized vol × 100 if unavailable."""
    try:
        vix_raw = yf.download("^VIX",
                               start=returns.index[0].strftime("%Y-%m-%d"),
                               end=returns.index[-1].strftime("%Y-%m-%d"),
                               auto_adjust=True, progress=False)["Close"]
        vix = vix_raw.reindex(returns.index).ffill().bfill().squeeze()
        if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
        log.info("VIX downloaded: %d days", vix.notna().sum())
    except Exception as e:
        log.warning("VIX download failed (%s). Using realized vol proxy.", e)
        vix = returns.mean(axis=1).rolling(20).std() * 100 * np.sqrt(252)
        vix = vix.reindex(returns.index).ffill().bfill()
    return vix.rename("VIX")

def engineer_features(returns: pd.DataFrame, vix: pd.Series) -> dict:
    """10 return-based features + vol_regime + vix + vix_regime per asset."""
    vol_bins = np.array([0.005, 0.012, 0.020])
    vix_arr  = np.asarray(vix).flatten()
    fd = {}
    for asset in returns.columns:
        r  = returns[asset]; df = pd.DataFrame(index=r.index)
        df["mom_5"]     = r.rolling(5).sum()
        df["mom_20"]    = r.rolling(20).sum()
        df["mom_60"]    = r.rolling(60).sum()
        df["vol_20"]    = r.rolling(20).std()
        df["vol_60"]    = r.rolling(60).std()
        df["skew_20"]   = r.rolling(20).skew()
        df["kurt_20"]   = r.rolling(20).kurt()
        df["vol_ratio"] = df["vol_20"] / (df["vol_60"] + 1e-9)
        df["ret_lag1"]  = r.shift(1)
        df["ret_lag2"]  = r.shift(2)
        vol_arr = df["vol_20"].fillna(0.015).values
        df["vol_regime"] = np.digitize(vol_arr, vol_bins).astype(float)
        # VIX features (aligned to returns index)
        aligned_vix = vix.reindex(r.index).ffill().bfill().values.flatten()
        df["vix"]        = aligned_vix
        df["vix_regime"] = np.digitize(aligned_vix,
                                        np.array([15., 25., 40.])).astype(float)
        fd[asset] = df.dropna()
    return fd

def split_indices(idx: pd.DatetimeIndex):
    train = idx <= pd.Timestamp(TRAIN_END)
    cal   = (idx > pd.Timestamp(TRAIN_END)) & (idx <= pd.Timestamp(CAL_END))
    test  = idx >= pd.Timestamp(TEST_START)
    cs, ce = pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END)
    in_test = ((idx >= cs) & (idx <= ce) & test).sum()
    print("\n" + "─"*60)
    print("  V4 Data Split")
    print("─"*60)
    print(f"  Train : {idx[train][0].date()} → {idx[train][-1].date()}  ({train.sum()}d)")
    print(f"  Cal   : {idx[cal][0].date()}   → {idx[cal][-1].date()}   ({cal.sum()}d)")
    print(f"  Test  : {idx[test][0].date()}  → {idx[test][-1].date()}  ({test.sum()}d)")
    for name, (cs_s, ce_s) in CRISIS_PERIODS.items():
        n = ((idx >= cs_s) & (idx <= ce_s) & test).sum()
        print(f"  {'✓' if n>=10 else '✗'} {name}: {n} crisis days in test")
    if in_test < 50: raise RuntimeError("Too few COVID days in test!")
    print("─"*60 + "\n")
    return train, cal, test

def detect_crisis_regime(returns: pd.DataFrame) -> pd.Series:
    avg = returns.mean(axis=1)
    vol20 = avg.rolling(20).std()
    spike = vol20 > CRISIS_VOL_THRESH * vol20.rolling(60).mean()
    cum = (1 + avg).cumprod()
    dd  = (cum / cum.cummax() - 1) < MIN_CRISIS_DD
    c   = (spike | dd).fillna(False)
    for lag in range(1, 11): c = c | c.shift(lag).fillna(False)
    return c.fillna(False)

def conf_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0: return float("inf")
    idx = min(int(np.ceil((1-alpha)*(n+1)))-1, n-1)
    return float(np.sort(np.abs(scores))[max(idx, 0)])

def augment_calibration(scores: np.ndarray, vols: np.ndarray,
                         scales=VOL_AUG_SCALES) -> np.ndarray:
    thr = np.percentile(vols, 80)
    aug = list(scores)
    if (vols > thr).sum() < 50:
        base = scores[vols <= np.percentile(vols, 50)]
        for s in scales: aug.extend((np.abs(base)*s).tolist())
    return np.array(aug)

class OnlineConformalPrediction:
    """Online conformal prediction with exponential forgetting."""

    def __init__(self, lambda_decay=ONLINE_LAMBDA, window=ONLINE_WINDOW):
        self.lam     = lambda_decay
        self.window  = window
        self.scores_: list = []

    def reset(self): self.scores_ = []

    def push(self, score: float):
        """Add a new residual observation."""
        self.scores_.append(float(score))
        if len(self.scores_) > self.window:
            self.scores_ = self.scores_[-self.window:]

    def weighted_quantile(self, alpha: float) -> float:
        """Return the (1-α) weighted quantile of stored scores."""
        if not self.scores_: return float("inf")
        n = len(self.scores_)
        s = np.array(self.scores_)
        w = np.exp(-self.lam * np.arange(n-1, -1, -1))   # recent = higher weight
        w /= w.sum()
        order = np.argsort(s)
        cumw  = np.cumsum(w[order])
        hit   = np.searchsorted(cumw, 1 - alpha)
        hit   = min(hit, n-1)
        return float(s[order[hit]])

    def get_q(self, static_q: float, alpha: float) -> float:
        """Return max(static quantile, online quantile) for conservatism."""
        oq = self.weighted_quantile(alpha)
        return max(static_q, oq) if np.isfinite(oq) else static_q

class MultiPeriodConformal:
    """Maintains separate calibration score sets for 4 VIX regimes:."""

    def __init__(self, alpha=ALPHA):
        self.alpha  = alpha
        self.buckets_: dict = {i: {} for i in range(4)}  # regime → {asset: scores}
        self.q_:     dict = {i: {} for i in range(4)}

    def _vix_regime(self, vix_val: float) -> int:
        return int(np.digitize(vix_val, [15., 25., 40.]))

    def fit(self, cal_scores: dict, cal_vix: np.ndarray):
        """
        cal_scores : {asset: np.ndarray of cal residuals (pre-augmentation)}
        cal_vix    : 1D array of VIX values aligned to calibration dates
        """
        vix_arr = np.asarray(cal_vix).flatten()
        regimes = np.digitize(vix_arr, [15., 25., 40.])

        for regime in range(4):
            mask = regimes == regime
            for asset, sc in cal_scores.items():
                n = min(len(sc), len(vix_arr))
                sub = sc[-n:][mask[-n:]]
                self.buckets_[regime][asset] = sub

        for asset in cal_scores:
            for regime in range(4):
                sc = self.buckets_[regime].get(asset, np.array([]))
                if len(sc) < 20 and regime == 3:
                    # Crisis bucket sparse → 2× the elevated-regime quantile
                    sc_elev = self.buckets_[2].get(asset, cal_scores[asset])
                    sc = np.abs(sc_elev) * 2.0
                if len(sc) < 5:
                    sc = np.abs(cal_scores.get(asset, np.array([0.02])))
                self.q_[regime][asset] = conf_q(sc, self.alpha)

    def get_q(self, asset: str, current_vix: float) -> float:
        regime = self._vix_regime(current_vix)
        return self.q_.get(regime, {}).get(asset, float("inf"))

class VaRCalibrator:
    """Dedicated ONE-SIDED conformal VaR calibration."""

    def __init__(self, alpha=ALPHA):
        self.alpha  = alpha
        self.lo_:  dict = {}
        self.sc_:  dict = {}
        self.eta_: dict = {}

    def _lgb_p(self):
        return {"objective":"quantile","metric":"quantile",
                "alpha":self.alpha,"n_estimators":200,
                "learning_rate":0.05,"random_state":SEED,"verbose":-1}

    def fit(self, returns, fd, train, cal):
        log.info("[VaRCal] Fitting 1-sided conformal VaR …")
        self.assets_ = list(returns.columns)
        for a in self.assets_:
            feat = fd[a]; common = returns.index.intersection(feat.index)
            r_   = returns.loc[common, a].values
            X_   = feat.loc[common].values
            pos  = [i for i, d in enumerate(returns.index) if d in common]
            tr   = np.array([train[i] for i in pos])
            ca   = np.array([cal[i]   for i in pos])
            sc   = StandardScaler()
            Xts  = sc.fit_transform(X_[tr]); self.sc_[a] = sc
            Xcs  = sc.transform(X_[ca])
            # Fit quantile regression at α
            lo_m = lgb.LGBMRegressor(**self._lgb_p())
            lo_m.fit(Xts, r_[tr]); self.lo_[a] = lo_m
            # Calibrate: one-sided scores
            lo_c = lo_m.predict(Xcs)
            sc_i = np.maximum(lo_c - r_[ca], 0)
            # 1-sided conformal quantile at coverage (1-α)
            self.eta_[a] = conf_q(sc_i, self.alpha)
        return self

    def predict_var(self, returns, fd, test) -> dict:
        res = {}
        for a in self.assets_:
            feat = fd[a]; common = returns.index.intersection(feat.index)
            r_   = returns.loc[common, a].values
            X_   = feat.loc[common].values
            pos  = [i for i, d in enumerate(returns.index) if d in common]
            te   = np.array([test[i] for i in pos])
            Xts  = self.sc_[a].transform(X_[te])
            lo_p = self.lo_[a].predict(Xts)
            # var_lo = q_alpha(X_t) - eta_var   (correct 1-sided VaR)
            var_lo = lo_p - self.eta_[a]
            res[a] = {"var_lo": var_lo, "actual": r_[te],
                      "dates": common[te]}
        return res

class AdaptiveRetrainingTrigger:
    """
    Monitors rolling 20-day coverage; logs dates where retraining would be triggered.
    Does NOT actually retrain (computationally expensive in backtest),
    but records trigger dates for analysis and plot 14.
    """

    def __init__(self, target=COVERAGE_TGT, threshold=0.75, gap=20):
        self.target    = target
        self.threshold = threshold
        self.gap       = gap
        self.history_: list = []   # (covered: bool)
        self.triggers_: list = []  # (date, rolling_cov)
        self._last: int = -999

    def update(self, covered: bool, date, step_idx: int):
        self.history_.append(covered)
        if len(self.history_) >= 20:
            rc = np.mean(self.history_[-20:])
            if rc < self.threshold and (step_idx - self._last) >= self.gap:
                self.triggers_.append((date, rc))
                self._last = step_idx

class CrisisAwareACI_V4:
    """V4 ACI: same as V3 + stores crisis flag history for plot 14."""
    def __init__(self, alpha=ALPHA, gamma=ACI_GAMMA, gamma_c=ACI_GAMMA_CRISIS):
        self.alpha = alpha; self.gamma = gamma; self.gamma_c = gamma_c
        self.target = COVERAGE_TGT
        self.alphas_: list = []; self._crisis = False; self._consec = 0
        self.crisis_flags_: list = []

    def set_crisis(self, flag):
        self._crisis = flag
        if flag: self._consec = 0

    def update(self, score, covered):
        g = self.gamma_c if self._crisis else self.gamma
        self.alpha -= g * (self.target - float(covered))
        if not self._crisis and covered: self._consec += 1
        else: self._consec = 0
        if self._consec > ACI_RECOVERY_LAG and self.alpha < ALPHA - 0.01:
            self.alpha = min(self.alpha + ACI_RECOVERY_STEP, ALPHA)
        self.alpha = float(np.clip(self.alpha, 0.005, 0.50))
        self.alphas_.append(self.alpha)
        self.crisis_flags_.append(self._crisis)
        return self.alpha

    def get_q(self, cal_scores):
        n   = len(cal_scores)
        idx = min(int(np.ceil((1-self.alpha)*(n+1)))-1, n-1)
        return float(np.sort(cal_scores)[max(idx, 0)])

class LSTMReg(nn.Module):
    def __init__(self, d, h=LSTM_HIDDEN, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d, h, 1, batch_first=True)
        self.drop = nn.Dropout(drop); self.fc = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(self.drop(o[:,-1,:])).squeeze(-1)

def _seqs(X, y, L=LSTM_SEQ_LEN):
    Xs, ys = [], []
    for i in range(L, len(y)): Xs.append(X[i-L:i]); ys.append(y[i])
    return np.array(Xs, np.float32), np.array(ys, np.float32)

def train_lstm(Xtr, ytr):
    sx, sy = _seqs(Xtr, ytr)
    if len(sx) < 2: return LSTMReg(Xtr.shape[1]).to(DEVICE)
    dl = DataLoader(TensorDataset(torch.tensor(sx), torch.tensor(sy)),
                    batch_size=64, shuffle=True, drop_last=True)
    m   = LSTMReg(sx.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    L   = nn.HuberLoss(delta=0.01); best, pat = float("inf"), 0
    for _ in range(LSTM_EPOCHS):
        m.train(); ep = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = L(m(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); ep += loss.item()
        ep /= max(len(dl), 1)
        if ep < best: best, pat = ep, 0
        else:
            pat += 1
            if pat >= 5: break
    return m

def lstm_pred(model, X):
    sx, _ = _seqs(X, np.zeros(len(X)))
    if len(sx) == 0: return np.zeros(0)
    model.eval()
    with torch.no_grad(): return model(torch.tensor(sx).to(DEVICE)).cpu().numpy()

class CPPSModel:
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.sc_: dict={}; self.rf_: dict={}; self.xgb_: dict={}
        self.lstm_: dict={}; self.cal_raw_: dict={}
        self.cal_aug_: dict={}; self.cal_vols_: dict={}
        self.var_q_: dict={}; self.aci_: dict={}
        self.cal_vix_: dict = {}   # VIX values during calibration per asset

    def _align(self, asset, returns, fd, mask):
        feat = fd[asset]; common = returns.index.intersection(feat.index)
        r = returns.loc[common, asset].values; X = feat.loc[common].values
        pos = [i for i, d in enumerate(returns.index) if d in common]
        m   = np.array([mask[i] for i in pos])
        return X[m], r[m], common[m]

    def _ens(self, asset, Xs):
        p_rf   = self.rf_[asset].predict(Xs)
        p_xgb  = self.xgb_[asset].predict(Xs)
        p_lstm = lstm_pred(self.lstm_[asset], Xs)
        n = len(p_lstm)
        return 0.35*p_rf[-n:] + 0.35*p_xgb[-n:] + 0.30*p_lstm

    def _fit_asset(self, asset, Xtr, ytr, Xca, yca):
        sc = StandardScaler(); Xts = sc.fit_transform(Xtr)
        Xcs = sc.transform(Xca); self.sc_[asset] = sc
        rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=6,
                                    n_jobs=-1, random_state=SEED)
        rf.fit(Xts, ytr); self.rf_[asset] = rf
        if HAS_XGB:
            xg = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=SEED, verbosity=0)
        else:
            xg = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=SEED)
        xg.fit(Xts, ytr); self.xgb_[asset] = xg
        self.lstm_[asset] = train_lstm(Xts, ytr)
        p_cal = self._ens(asset, Xcs); nd = len(yca) - len(p_cal)
        yca_a = yca[nd:]; Xcs_a = Xcs[nd:]
        raw   = np.abs(yca_a - p_cal)
        vols  = np.std(Xcs_a, axis=1)
        self.cal_raw_[asset]  = raw
        self.cal_aug_[asset]  = augment_calibration(raw, vols)
        self.cal_vols_[asset] = vols
        down = yca_a - p_cal; dl = -down[down < 0]
        self.var_q_[asset] = conf_q(dl if len(dl) > 0 else raw, ALPHA)
        self.aci_[asset]   = CrisisAwareACI_V4()

    def fit(self, returns, fd, train, cal, vix: pd.Series = None):
        log.info("[CPPS] Fitting vol-augmented ensemble …")
        self.assets_ = list(returns.columns)
        for a in self.assets_:
            Xtr, ytr, _  = self._align(a, returns, fd, train)
            Xca, yca, dc = self._align(a, returns, fd, cal)
            # Store VIX during calibration for MultiPeriodConformal
            if vix is not None:
                self.cal_vix_[a] = vix.reindex(dc).ffill().bfill().values.flatten()
            self._fit_asset(a, Xtr, ytr, Xca, yca)
        log.info("[CPPS] Done.")
        return self

    def predict_intervals(self, returns, fd, test, crisis_s=None):
        res = {}
        for a in self.assets_:
            Xte, yte, dates = self._align(a, returns, fd, test)
            Xts = self.sc_[a].transform(Xte)
            p   = self._ens(a, Xts)
            nd  = len(yte) - len(p); yts = yte[nd:]; dts = dates[nd:]
            aci = self.aci_[a]; aug_sc = self.cal_aug_[a]
            lo, hi, pts, var_lo, aci_a = [], [], [], [], []
            for i, (pt, ya) in enumerate(zip(p, yts)):
                if crisis_s is not None and dts[i] in crisis_s.index:
                    aci.set_crisis(bool(crisis_s.loc[dts[i]]))
                q = aci.get_q(aug_sc)
                lo.append(pt - q); hi.append(pt + q); pts.append(pt)
                var_lo.append(pt - self.var_q_[a])
                aci.update(abs(ya - pt),
                           covered=bool((pt-q) <= ya <= (pt+q)))
                aci_a.append(aci.alphas_[-1] if aci.alphas_ else ALPHA)
            res[a] = {"lower":np.array(lo),"upper":np.array(hi),
                      "point":np.array(pts),"actual":yts,"dates":dts,
                      "var_lo":np.array(var_lo),"aci_alpha":np.array(aci_a)}
        return res

class OnlineCPPSModel:
    """Wraps CPPSModel with per-asset OnlineConformalPrediction."""

    def __init__(self, cpps: CPPSModel, alpha=ALPHA):
        self.cpps  = cpps
        self.alpha = alpha
        self.online_: dict = {}   # asset → OnlineConformalPrediction

    def _get_or_create(self, asset):
        if asset not in self.online_:
            self.online_[asset] = OnlineConformalPrediction(ONLINE_LAMBDA, ONLINE_WINDOW)
        return self.online_[asset]

    def predict_intervals(self, returns, fd, test, crisis_s=None,
                          var_cal: VaRCalibrator = None,
                          multi_period: MultiPeriodConformal = None,
                          vix: pd.Series = None):
        res = {}
        for a in self.cpps.assets_:
            Xte, yte, dates = self.cpps._align(a, returns, fd, test)
            Xts = self.cpps.sc_[a].transform(Xte)
            p   = self.cpps._ens(a, Xts)
            nd  = len(yte) - len(p); yts = yte[nd:]; dts = dates[nd:]
            aci    = self.cpps.aci_[a]
            aug_sc = self.cpps.cal_aug_[a]
            ocp    = self._get_or_create(a); ocp.reset()
            trigger= AdaptiveRetrainingTrigger()
            lo, hi, pts, var_lo_a, aci_a = [], [], [], [], []
            for i, (pt, ya) in enumerate(zip(p, yts)):
                if crisis_s is not None and dts[i] in crisis_s.index:
                    aci.set_crisis(bool(crisis_s.loc[dts[i]]))
                # Base quantile from ACI
                q_static = aci.get_q(aug_sc)
                # Online + multi-period enhancement
                q_eff = ocp.get_q(q_static, self.alpha)
                if multi_period is not None and vix is not None:
                    curr_vix = float(vix.reindex([dts[i]]).ffill().iloc[0]) \
                               if dts[i] in vix.index else 20.0
                    q_mp  = multi_period.get_q(a, curr_vix)
                    q_eff = max(q_eff, q_mp)
                lo.append(pt - q_eff); hi.append(pt + q_eff); pts.append(pt)
                # 1-sided VaR (from VaRCalibrator if available, else fallback)
                if var_cal is not None and a in var_cal.lo_:
                    Xts_i = Xts[i:i+1]
                    lo_pred = var_cal.lo_[a].predict(
                        var_cal.sc_[a].transform(Xte[i:i+1]))
                    vlo = float(lo_pred[0]) - var_cal.eta_[a]
                else:
                    vlo = pt - self.cpps.var_q_[a]
                var_lo_a.append(vlo)
                # ACI + online update
                cov = bool((pt - q_eff) <= ya <= (pt + q_eff))
                aci.update(abs(ya - pt), covered=cov)
                ocp.push(abs(ya - pt))   # ← feeds online calibration
                trigger.update(cov, dts[i], i)
                aci_a.append(aci.alphas_[-1] if aci.alphas_ else ALPHA)
            res[a] = {"lower":np.array(lo),"upper":np.array(hi),
                      "point":np.array(pts),"actual":yts,"dates":dts,
                      "var_lo":np.array(var_lo_a),"aci_alpha":np.array(aci_a),
                      "trigger_dates":[t[0] for t in trigger.triggers_]}
        return res

class AdaptiveCQR:
    """CQR with crisis-specific eta AND corrected var_lo using VaRCalibrator."""

    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.lo_: dict={}; self.hi_: dict={}; self.sc_: dict={}
        self.eta_: dict={}; self.crisis_eta_: dict={}

    def _p(self, q):
        return {"objective":"quantile","metric":"quantile","alpha":q,
                "n_estimators":300,"learning_rate":0.05,
                "random_state":SEED,"verbose":-1}

    def fit(self, returns, fd, train, cal):
        log.info("[AdaptCQR] Fitting …"); self.assets_ = list(returns.columns)
        for a in self.assets_:
            feat = fd[a]; common = returns.index.intersection(feat.index)
            r_ = returns.loc[common, a].values; X_ = feat.loc[common].values
            pos = [i for i, d in enumerate(returns.index) if d in common]
            tr  = np.array([train[i] for i in pos])
            ca  = np.array([cal[i]   for i in pos])
            sc  = StandardScaler(); Xts = sc.fit_transform(X_[tr])
            Xcs = sc.transform(X_[ca]); self.sc_[a] = sc
            lom = lgb.LGBMRegressor(**self._p(self.alpha/2))
            him = lgb.LGBMRegressor(**self._p(1-self.alpha/2))
            lom.fit(Xts, r_[tr]); him.fit(Xts, r_[tr])
            self.lo_[a] = lom; self.hi_[a] = him
            lc = lom.predict(Xcs); hc = him.predict(Xcs)
            scores = np.maximum(lc - r_[ca], r_[ca] - hc)
            self.eta_[a] = conf_q(scores, self.alpha)
            vols = np.abs(r_[ca]).cumsum() / (np.arange(len(r_[ca]))+1)
            hv   = vols > np.percentile(vols, 70)
            self.crisis_eta_[a] = conf_q(scores[hv], self.alpha) \
                                   if hv.sum() >= 30 else self.eta_[a]*1.5
        return self

    def predict_intervals(self, returns, fd, test, crisis_s=None,
                           var_cal: VaRCalibrator = None):
        res = {}
        for a in self.assets_:
            feat = fd[a]; common = returns.index.intersection(feat.index)
            r_ = returns.loc[common, a].values; X_ = feat.loc[common].values
            pos = [i for i, d in enumerate(returns.index) if d in common]
            te  = np.array([test[i] for i in pos])
            Xts = self.sc_[a].transform(X_[te]); dates = common[te]
            lo_b = self.lo_[a].predict(Xts); hi_b = self.hi_[a].predict(Xts)
            lo_a, hi_a = [], []
            for i, dt in enumerate(dates):
                in_c = (crisis_s is not None and dt in crisis_s.index
                        and bool(crisis_s.loc[dt]))
                eta = self.crisis_eta_[a] if in_c else self.eta_[a]
                lo_a.append(lo_b[i] - eta); hi_a.append(hi_b[i] + eta)
            lo_a = np.array(lo_a); hi_a = np.array(hi_a)
            pt   = (lo_a + hi_a) / 2
            if var_cal is not None and a in var_cal.lo_:
                lo_var = var_cal.lo_[a].predict(Xts) - var_cal.eta_[a]
            else:
                lo_var = lo_a  # fallback (old behaviour)
            res[a] = {"lower":lo_a,"upper":hi_a,"point":pt,
                      "actual":r_[te],"dates":dates,"var_lo":lo_var}
        return res

class BayesianModel:
    def __init__(self, alpha=ALPHA):
        self.alpha=alpha; self.m_:dict={}; self.sc_:dict={}
    def fit(self, returns, fd, train, cal):
        log.info("[Bayes] Fitting …"); self.assets_=list(returns.columns)
        for a in self.assets_:
            feat=fd[a]; common=returns.index.intersection(feat.index)
            r_=returns.loc[common,a].values; X_=feat.loc[common].values
            pos=[i for i,d in enumerate(returns.index) if d in common]
            tr=np.array([train[i] for i in pos])
            sc=StandardScaler(); Xts=sc.fit_transform(X_[tr]); self.sc_[a]=sc
            br=BayesianRidge(compute_score=True); br.fit(Xts,r_[tr]); self.m_[a]=br
        return self
    def predict_intervals(self, returns, fd, test, crisis_s=None,
                          var_cal=None):
        z=norm.ppf(1-self.alpha/2); res={}
        for a in self.assets_:
            feat=fd[a]; common=returns.index.intersection(feat.index)
            r_=returns.loc[common,a].values; X_=feat.loc[common].values
            pos=[i for i,d in enumerate(returns.index) if d in common]
            te=np.array([test[i] for i in pos])
            Xts=self.sc_[a].transform(X_[te])
            mu,std=self.m_[a].predict(Xts,return_std=True)
            lo_var = (var_cal.lo_[a].predict(
                var_cal.sc_[a].transform(X_[te])) - var_cal.eta_[a]
                if var_cal and a in var_cal.lo_ else mu-norm.ppf(0.95)*std)
            res[a]={"lower":mu-z*std,"upper":mu+z*std,"point":mu,
                    "actual":r_[te],"dates":common[te],"var_lo":lo_var}
        return res

class QRModel:
    def __init__(self, alpha=ALPHA):
        self.alpha=alpha; self.lo_:dict={}; self.hi_:dict={}; self.sc_:dict={}
    def _p(self,q): return {"objective":"quantile","metric":"quantile","alpha":q,
                "n_estimators":300,"learning_rate":0.05,"random_state":SEED,"verbose":-1}
    def fit(self, returns, fd, train, cal):
        log.info("[QR] Fitting …"); self.assets_=list(returns.columns)
        for a in self.assets_:
            feat=fd[a]; common=returns.index.intersection(feat.index)
            r_=returns.loc[common,a].values; X_=feat.loc[common].values
            pos=[i for i,d in enumerate(returns.index) if d in common]
            tr=np.array([train[i] for i in pos])
            sc=StandardScaler(); Xts=sc.fit_transform(X_[tr]); self.sc_[a]=sc
            lo=lgb.LGBMRegressor(**self._p(self.alpha/2)); lo.fit(Xts,r_[tr])
            hi=lgb.LGBMRegressor(**self._p(1-self.alpha/2)); hi.fit(Xts,r_[tr])
            self.lo_[a]=lo; self.hi_[a]=hi
        return self
    def predict_intervals(self, returns, fd, test, crisis_s=None, var_cal=None):
        res={}
        for a in self.assets_:
            feat=fd[a]; common=returns.index.intersection(feat.index)
            r_=returns.loc[common,a].values; X_=feat.loc[common].values
            pos=[i for i,d in enumerate(returns.index) if d in common]
            te=np.array([test[i] for i in pos])
            Xts=self.sc_[a].transform(X_[te])
            lo=self.lo_[a].predict(Xts); hi=self.hi_[a].predict(Xts)
            lo_var=(var_cal.lo_[a].predict(var_cal.sc_[a].transform(X_[te]))
                    - var_cal.eta_[a] if var_cal and a in var_cal.lo_ else lo)
            res[a]={"lower":lo,"upper":hi,"point":(lo+hi)/2,
                    "actual":r_[te],"dates":common[te],"var_lo":lo_var}
        return res

class ScoreBlendedHybrid:
    """V4: aligned blended-quantile hybrid (same as V3.1 fixed)."""
    def __init__(self, cpps: CPPSModel, cqr_model, alpha=ALPHA):
        self.cpps=cpps; self.cqr=cqr_model; self.alpha=alpha; self.bq_:dict={}
    def _bq(self, a):
        raw=self.cpps.cal_raw_.get(a, np.array([])); eta=self.cqr.eta_.get(a,0.02)
        cq=np.full(len(raw), eta) if len(raw)>0 else np.array([eta])
        n=min(len(raw),len(cq)); bl=0.5*np.abs(raw[-n:])+0.5*cq[-n:]
        aug=list(bl)
        for s in VOL_AUG_SCALES: aug.extend((bl*s).tolist())
        return conf_q(np.array(aug), self.alpha)
    def predict_intervals(self, returns, fd, test, crisis_s=None, var_cal=None):
        rc=self.cpps.predict_intervals(returns,fd,test,crisis_s)
        rq=self.cqr.predict_intervals(returns,fd,test,crisis_s,var_cal)
        res={}
        for a in self.cpps.assets_:
            if a not in rc or a not in rq: continue
            dc=rc[a]; dq=rq[a]
            dc_d=pd.DatetimeIndex(dc["dates"]); dq_d=pd.DatetimeIndex(dq["dates"])
            common=dc_d.intersection(dq_d)
            if not self.bq_.get(a): self.bq_[a]=self._bq(a)
            q=self.bq_[a]
            cm={d:i for i,d in enumerate(dc_d)}; qm={d:i for i,d in enumerate(dq_d)}
            lo,hi,pt=[],[],[]
            for dt in common:
                ic,iq=cm[dt],qm[dt]; inc=(crisis_s is not None and dt in crisis_s.index
                                           and bool(crisis_s.loc[dt]))
                wp=0.30 if inc else 0.70
                p=wp*dc["point"][ic]+(1-wp)*dq["point"][iq]
                lo.append(p-q); hi.append(p+q); pt.append(p)
            idx=[cm[d] for d in common]; lo_a=np.array(lo); hi_a=np.array(hi)
            vl=(rq[a]["var_lo"][np.array([qm[d] for d in common])]
                if "var_lo" in rq[a] else lo_a)
            res[a]={"lower":lo_a,"upper":hi_a,"point":np.array(pt),
                    "actual":dc["actual"][idx],"dates":common,"var_lo":vl,
                    "aci_alpha":dc.get("aci_alpha",np.full(len(common),ALPHA))[idx]}
        return res

class ConformalEnsemble:
    """Dynamic weight ensemble (same as V3)."""
    def predict_intervals(self, all_res, crisis_s=None):
        methods=list(all_res.keys()); assets=list(all_res[methods[0]].keys())
        results={}
        for a in assets:
            cov_h={m:[] for m in methods}; w_h={m:1/len(methods) for m in methods}
            n_min=min(len(all_res[m][a]["actual"]) for m in methods if a in all_res[m])
            dates_a=all_res[methods[0]][a]["dates"][-n_min:]
            lo_o,hi_o=np.zeros(n_min),np.zeros(n_min)
            for i in range(n_min):
                if i>0 and i%ENS_WINDOW==0:
                    pf={m:max(1-abs(COVERAGE_TGT-np.mean(cov_h[m][-ENS_WINDOW:])),0.01)
                        for m in methods if a in all_res[m] and cov_h[m]}
                    tot=sum(pf.values())+1e-9; w_h={m:pf[m]/tot for m in pf}
                in_c=(crisis_s is not None and i<len(dates_a)
                      and dates_a[i] in crisis_s.index
                      and bool(crisis_s.loc[dates_a[i]]))
                wa=dict(w_h)
                if in_c and "OnlineCPPS" in wa:
                    wa["OnlineCPPS"]=min(wa["OnlineCPPS"]*1.5,0.50)
                    tot=sum(wa.values()); wa={m:v/tot for m,v in wa.items()}
                li=hi=tw=0.0
                for m in methods:
                    if a not in all_res[m]: continue
                    d=all_res[m][a]; idx=-(n_min-i)
                    lv=d["lower"][idx] if abs(idx)<=len(d["lower"]) else 0.
                    hv=d["upper"][idx] if abs(idx)<=len(d["upper"]) else 0.
                    w=wa.get(m,1/len(methods))
                    li+=w*lv; hi+=w*hv; tw+=w
                    ya=d["actual"][idx] if abs(idx)<=len(d["actual"]) else 0.
                    cov_h[m].append(float(lv<=ya<=hv))
                lo_o[i]=li/(tw+1e-12); hi_o[i]=hi/(tw+1e-12)
            ref=all_res[methods[0]][a]; act=ref["actual"][-n_min:]
            results[a]={"lower":lo_o,"upper":hi_o,"point":(lo_o+hi_o)/2,
                        "actual":act,"dates":dates_a,"var_lo":lo_o}
        return results

def build_portfolio(all_res: dict, returns: pd.DataFrame) -> dict:
    portfolios = {}
    for method, res in all_res.items():
        assets_= [a for a in res if "actual" in res[a]]
        if not assets_: continue
        dates = pd.DatetimeIndex(res[assets_[0]]["dates"])
        n = len(dates); W = np.zeros((n, len(assets_)))
        for i in range(n):
            widths = []
            for j, a in enumerate(assets_):
                d = res[a]; idx = -(n - i)
                lo = float(d["lower"][idx]) if abs(idx) <= len(d["lower"]) else 0.
                hi = float(d["upper"][idx]) if abs(idx) <= len(d["upper"]) else 0.
                widths.append(hi - lo + 1e-9)
            inv = 1.0 / np.array(widths); W[i] = inv / inv.sum()
        port_ret = np.array([
            sum(W[i, j] * float(res[a]["actual"][-(n-i)])
                for j, a in enumerate(assets_)
                if abs(n-i) <= len(res[a]["actual"]))
            for i in range(n)])
        portfolios[method] = {"weights": W,"returns": port_ret,"dates": dates.values}
    return portfolios

def coverage_at(lo, hi, actual):
    if len(lo) == 0: return float("nan")
    return np.mean((lo <= actual) & (actual <= hi))

def worst_case_cov(lo, hi, actual, window=20):
    n = len(actual); worst = 1.0
    for i in range(0, n - window + 1):
        c = coverage_at(lo[i:i+window], hi[i:i+window], actual[i:i+window])
        if np.isfinite(c): worst = min(worst, c)
    return worst

def recovery_days(aci_alphas, crisis_end_idx, target=ALPHA, tol=0.02):
    for i, a in enumerate(aci_alphas[crisis_end_idx:]):
        if a >= target - tol: return i
    return len(aci_alphas) - crisis_end_idx

def kupiec_pof(violations, n, alpha=ALPHA):
    v = np.sum(violations); p = v / n
    if v == 0 or v == n: return (float("inf"), 0.0)
    ll = v*np.log(p/alpha) + (n-v)*np.log((1-p)/(1-alpha))
    stat = -2 * ll; pval = 1 - chi2.cdf(stat, 1)
    return float(stat), float(pval)

def christoffersen_cc(violations):
    vi = np.array(violations, dtype=int)
    n00=n01=n10=n11 = 0
    for t in range(1, len(vi)):
        if vi[t-1]==0 and vi[t]==0: n00+=1
        elif vi[t-1]==0 and vi[t]==1: n01+=1
        elif vi[t-1]==1 and vi[t]==0: n10+=1
        else: n11+=1
    p01 = n01/(n00+n01+1e-12); p11 = n11/(n10+n11+1e-12)
    p   = (n01+n11)/(n00+n01+n10+n11+1e-12)
    def _ll(p0, p1): return (n00*np.log(max(1-p0,1e-12))+n01*np.log(max(p0,1e-12))
                              +n10*np.log(max(1-p1,1e-12))+n11*np.log(max(p1,1e-12)))
    stat = -2*(_ll(p,p)-_ll(p01,p11))
    return max(float(stat),0.), float(1-chi2.cdf(max(stat,0.),1))

def dm_test(e1, e2):
    d = e1**2 - e2**2
    if d.std() < 1e-12: return 0., 1.
    t = d.mean() / (d.std()/np.sqrt(len(d))+1e-12)
    return float(t), float(2*(1-norm.cdf(abs(t))))

def compute_metrics(all_res, portfolios, crisis_series, var_res=None):
    metrics = {}
    for method, res in all_res.items():
        m = {}; n_assets = 0
        all_cov=[]; all_wid=[]; all_act=[]; all_lo=[]; all_hi=[]
        for a in res:
            d = res[a]
            if "actual" not in d or len(d["actual"]) == 0: continue
            lo,hi,act = d["lower"],d["upper"],d["actual"]
            all_cov.append(coverage_at(lo,hi,act))
            all_wid.append(np.mean(hi - lo)); n_assets += 1
            all_act.extend(act.tolist()); all_lo.extend(lo.tolist())
            all_hi.extend(hi.tolist())
        m["marginal_cov"]  = float(np.mean(all_cov)) if all_cov else float("nan")
        m["avg_width"]     = float(np.mean(all_wid)) if all_wid else float("nan")
        all_lo_=np.array(all_lo); all_hi_=np.array(all_hi); all_act_=np.array(all_act)
        m["worst_cov"]     = worst_case_cov(all_lo_,all_hi_,all_act_)
        for cname,(cs,ce) in CRISIS_PERIODS.items():
            c_lo,c_hi,c_act=[],[],[]
            for a in res:
                d=res[a]; dates=pd.DatetimeIndex(d["dates"]); n=len(dates)
                mask_tmp=((dates>=cs)&(dates<=ce)); mask=mask_tmp.values if hasattr(mask_tmp,"values") else np.array(mask_tmp)
                li=len(d["lower"]); shift=li-n
                if mask.sum()==0: continue
                idx=[i for i,fl in enumerate(mask) if fl]
                si=[i+shift for i in idx if i+shift>=0 and i+shift<li]
                if si: c_lo.extend(d["lower"][si]); c_hi.extend(d["upper"][si]); c_act.extend(d["actual"][[i for i in idx if i<len(d["actual"])]])
            m[f"crisis_cov_{cname}"] = coverage_at(np.array(c_lo),np.array(c_hi),np.array(c_act)) if c_lo else float("nan")
        viol=[]; n_var=0
        for a in res:
            d=res[a]
            if "var_lo" not in d or "actual" not in d: continue
            vl=d["var_lo"]; act=d["actual"]
            n=min(len(vl),len(act)); n_var += n
            viol.extend((act[:n] < vl[:n]).tolist())
        m["var_viol_rate"] = float(np.mean(viol)) if viol else float("nan")
        viol_a = np.array(viol, dtype=int)
        kstat,kp = kupiec_pof(viol_a, max(n_var,1))
        m["kupiec_stat"]=kstat; m["kupiec_p"]=kp
        if len(viol_a)>1:
            cstat,cp = christoffersen_cc(viol_a); m["cc_stat"]=cstat; m["cc_p"]=cp
        else:
            m["cc_stat"]=float("nan"); m["cc_p"]=float("nan")
        # Portfolio metrics
        if method in portfolios:
            pr   = pd.Series(portfolios[method]["returns"])
            pr_d = portfolios[method]["dates"]
            m["sharpe"] = float(pr.mean()/(pr.std()+1e-9)*np.sqrt(252))
            cum  = (1 + pr).cumprod(); m["max_dd"] = float((cum / cum.cummax() - 1).min())
            m["turnover"] = float(np.mean(np.abs(np.diff(portfolios[method]["weights"],axis=0)))) if len(pr)>1 else float("nan")
            pb=[]
            for a in res:
                d=res[a]; n=len(d["actual"])
                if n==0: continue
                q=np.clip(d.get("aci_alpha",np.full(n,ALPHA)), 0.001, 0.999)
                pb.append(float(np.mean(mean_pinball_loss(d["actual"],d["point"],alpha=q.mean()))))
            m["pinball"] = float(np.mean(pb)) if pb else float("nan")
            for cname,(cs,ce) in CRISIS_PERIODS.items():
                pr_d_ = pd.DatetimeIndex(pr_d)
                mask  = (pr_d_ >= cs) & (pr_d_ <= ce)
                cpr   = pr.values[mask] if hasattr(pr, 'values') else pr[mask]
                m[f"sharpe_{cname}"] = float(np.mean(cpr)/(np.std(cpr)+1e-9)*np.sqrt(252)) if len(cpr)>1 else float("nan")
        metrics[method] = m
    return metrics

STRESS_SCENARIOS = {
    "2008_crash": {
        "pre_days":5,"crisis_days":20,"post_days":10,
        "pre_mu":-0.001,"pre_sig":0.010,
        "cri_mu":-0.025,"cri_sig":0.040,
        "post_mu":0.005,"post_sig":0.015,
    },
    "COVID_2020": {
        "pre_days":5,"crisis_days":23,"post_days":10,
        "pre_mu":-0.002,"pre_sig":0.012,
        "cri_mu":-0.040,"cri_sig":0.055,
        "post_mu":0.010,"post_sig":0.020,
    },
    "Inflation_2022": {
        "pre_days":10,"crisis_days":60,"post_days":20,
        "pre_mu":-0.001,"pre_sig":0.012,
        "cri_mu":-0.003,"cri_sig":0.018,
        "post_mu":0.000,"post_sig":0.014,
    },
}

def generate_stress_returns(params, seed=SEED):
    """Generate returns for [pre_days + crisis_days + post_days], matching STRESS_SCENARIOS dict."""
    rng = np.random.default_rng(seed)
    specs = [
        (params["pre_days"],    params["pre_mu"],  params["pre_sig"]),
        (params["crisis_days"], params["cri_mu"],  params["cri_sig"]),
        (params["post_days"],   params["post_mu"], params["post_sig"]),
    ]
    return np.concatenate([rng.normal(mu, sig, nd) for nd, mu, sig in specs])

def stress_test_all(all_results: dict) -> pd.DataFrame:
    """Test each method's in-sample quantile on synthetic stress scenarios."""
    rows = []
    for scenario, params in STRESS_SCENARIOS.items():
        syn = generate_stress_returns(params)
        crisis_start = params["pre_days"]
        crisis_end   = params["pre_days"] + params["crisis_days"]
        for method, res in all_results.items():
            a = next((x for x in ["SPY","QQQ"] if x in res), list(res.keys())[0])
            if a not in res or "lower" not in res[a]: continue
            d = res[a]
            # Use the MEDIAN interval width as synthetic quantile proxy
            widths = d["upper"] - d["lower"]
            q_norm   = float(np.median(widths)) / 2
            q_crisis = float(np.percentile(widths, 90)) / 2
            lo_arr=[]; hi_arr=[]
            for i in range(len(syn)):
                in_c = crisis_start <= i < crisis_end
                q = q_crisis if in_c else q_norm
                lo_arr.append(-q); hi_arr.append(q)
            cov = coverage_at(np.array(lo_arr), np.array(hi_arr), syn)
            c_syn = syn[crisis_start:crisis_end]
            c_lo  = np.array(lo_arr[crisis_start:crisis_end])
            c_hi  = np.array(hi_arr[crisis_start:crisis_end])
            cov_c = coverage_at(c_lo, c_hi, c_syn)
            rows.append({"scenario":scenario,"method":method,
                         "overall_cov":cov,"crisis_cov":cov_c,
                         "needs_recal": cov_c < 0.80})
    return pd.DataFrame(rows)

def print_summary(metrics: dict):
    methods = list(metrics.keys())
    w = max(10, max(len(m) for m in methods))
    H = "="*95
    print("\n"+H)
    print("  V4 ONLINE-CONFORMAL PREDICTION – RESULTS")
    print(H)
    header = f"{'Metric':<40}" + "".join(f"{m:>{w}}" for m in methods)
    print(header)
    print("-"*95)
    def row(label, key, fmt="{:.1%}", inv=False):
        vals="".join(f"{fmt.format(metrics[m].get(key,float('nan'))):>{w}}" for m in methods)
        print(f"  {label:<38}{vals}")
    row("Marginal Coverage",           "marginal_cov")
    row("Worst-Case Coverage (20d)", "worst_cov")
    for cname in CRISIS_PERIODS:
        row(f"  Crisis: {cname}", f"crisis_cov_{cname}")
    row("VaR Violation Rate",         "var_viol_rate")
    row("Avg Interval Width",         "avg_width",    fmt="{:.4f}")
    row("Pinball Loss",               "pinball",      fmt="{:.4f}")
    row("Portfolio Sharpe",           "sharpe",       fmt="{:.4f}")
    row("Max Drawdown",               "max_dd")
    for cname in CRISIS_PERIODS:
        row(f"  Sharpe: {cname}",  f"sharpe_{cname}", fmt="{:.4f}")
    print("-"*95)
    print("\n── Kupiec POF Test (VaR @ 10%) ─────────────────────────────────────────")
    for m in methods:
        stat = metrics[m].get("kupiec_stat",float("nan"))
        pval = metrics[m].get("kupiec_p",   float("nan"))
        flag = "✓ PASS" if pval > 0.05 else "✗ FAIL"
        print(f"  {m:<22} stat={stat:+.3f}  p={pval:.4f}  {flag}")
    print("\n── Christoffersen CC ───────────────────────────────────────────────────")
    for m in methods:
        stat = metrics[m].get("cc_stat", float("nan"))
        pval = metrics[m].get("cc_p",    float("nan"))
        flag = "✓ PASS" if pval > 0.05 else "✗ FAIL"
        print(f"  {m:<22} stat={stat:+.3f}  p={pval:.4f}  {flag}")
    print(H+"\n")

def _save(name):
    p = OUT / name; plt.savefig(p, dpi=120, bbox_inches="tight",
                                facecolor="#0d1117"); plt.close()
    log.info("Saved %s", name)

def _asset_sample(all_res, methods, asset_pref=["SPY","QQQ","EEM"]):
    for a in asset_pref:
        for m in methods:
            if a in all_res.get(m,{}): return m, a
    m = methods[0]; a = list(all_res[m].keys())[0]; return m, a

def plot1_intervals(all_res, crisis_s, vix):
    methods = list(all_res.keys())
    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 4*len(methods)),
                              sharex=True, facecolor="#0d1117")
    if len(methods)==1: axes=[axes]
    _, a = _asset_sample(all_res, methods)
    for ax, m in zip(axes, methods):
        if a not in all_res[m]: continue
        d=all_res[m][a]; lo=d["lower"]; hi=d["upper"]
        dates=pd.DatetimeIndex(d["dates"]); act=d["actual"]
        ax.fill_between(dates, lo, hi, alpha=0.25, color=PALETTE.get(m,"#888"))
        ax.plot(dates, act, "w", lw=0.7, label="Actual")
        ax.plot(dates, d["point"], "--", color=PALETTE.get(m,"#888"), lw=0.8, label="Point")
        for cname,(cs,ce) in CRISIS_PERIODS.items():
            ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), color="red",alpha=0.12)
        ax.set_title(f"{m} – {a}", color="white", fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("V4 Prediction Intervals – All Methods", color="white", fontsize=14, y=1.01)
    plt.tight_layout(); _save("plot1_intervals.png")

def plot2_rolling_coverage(all_res):
    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for m, res in all_res.items():
        cov_ts, dt_ts = [], []
        for a in res:
            d=res[a]; lo=d["lower"]; hi=d["upper"]; act=d["actual"]
            dates=pd.DatetimeIndex(d["dates"])
            covered=((lo<=act)&(act<=hi)).astype(float)
            roll=pd.Series(covered, index=dates).rolling(20).mean().dropna()
            if len(roll)>len(cov_ts): cov_ts=list(roll); dt_ts=list(roll.index)
            break
        if dt_ts: ax.plot(dt_ts, cov_ts, color=PALETTE.get(m,"#888"), lw=1.5, label=m)
    ax.axhline(COVERAGE_TGT, color="white", ls="--", lw=1.5, label=f"{COVERAGE_TGT:.0%} target")
    ax.axhline(0.75, color="red", ls=":", lw=1, label="75% retrain threshold")
    for cname,(cs,ce) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), color="red", alpha=0.12, label=cname)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rolling 20d Coverage"); ax.legend(fontsize=8)
    ax.set_title("Rolling Coverage (20-day window)", color="white")
    plt.tight_layout(); _save("plot2_rolling_coverage.png")

def plot3_crisis_zoom(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#0d1117")
    _, a = _asset_sample(all_res, list(all_res.keys()))
    for ax, (cname, (cs, ce)) in zip(axes, CRISIS_PERIODS.items()):
        for m, res in all_res.items():
            if a not in res: continue
            d=res[a]; dates=pd.DatetimeIndex(d["dates"])
            mask=(dates>=cs)&(dates<=ce)
            if mask.sum()==0: continue
            lo=d["lower"][mask]; hi=d["upper"][mask]; act=d["actual"][mask]
            cov=float(np.mean((lo<=act)&(act<=hi)))
            ax.fill_between(dates[mask], lo, hi, alpha=0.2, color=PALETTE.get(m,"#888"))
            ax.plot(dates[mask], act, "w", lw=0.9)
            ax.plot([], [], color=PALETTE.get(m,"#888"), label=f"{m} ({cov:.0%})")
        ax.set_title(f"Crisis Zoom: {cname}", color="white"); ax.legend(fontsize=8)
    plt.tight_layout(); _save("plot3_crisis_zoom.png")

def plot4_violin_widths(all_res, crisis_s):
    methods=list(all_res.keys()); _, a = _asset_sample(all_res, methods)
    df_rows=[]
    for m in methods:
        if a not in all_res[m]: continue
        d=all_res[m][a]; dates=pd.DatetimeIndex(d["dates"])
        w=d["upper"]-d["lower"]
        in_c=[(crisis_s.loc[dt] if dt in crisis_s.index else False) for dt in dates]
        for i,wd in enumerate(w):
            df_rows.append({"Method":m,"Width":float(wd),"Period":"Crisis" if in_c[i] else "Normal"})
    df=pd.DataFrame(df_rows)
    fig, ax = plt.subplots(figsize=(14,6), facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    sns.violinplot(data=df, x="Method", y="Width", hue="Period",
                   palette={"Normal":"#2196F3","Crisis":"#F44336"}, ax=ax, split=True, inner="box")
    ax.set_title("Interval Width Distribution: Normal vs Crisis", color="white")
    plt.tight_layout(); _save("plot4_violin_widths.png")

def plot5_portfolio_weights(portfolios):
    methods=[m for m in portfolios if "weights" in portfolios[m]]
    if not methods: return
    n=len(methods); fig, axes = plt.subplots(1,n,figsize=(5*n,5),facecolor="#0d1117")
    if n==1: axes=[axes]
    for ax, m in zip(axes,methods):
        w=portfolios[m]["weights"]; ax.set_facecolor("#0d1117")
        ax.imshow(w.T, aspect="auto", cmap="RdYlGn")
        ax.set_title(f"{m} weights", color="white"); ax.set_ylabel("Assets")
    plt.tight_layout(); _save("plot5_portfolio_weights.png")

def plot6_var_backtest(all_res):
    fig, ax = plt.subplots(figsize=(16,5), facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    _, a = _asset_sample(all_res, list(all_res.keys()))
    date_ref = None
    for m, res in all_res.items():
        if a not in res: continue
        d=res[a]; vl=d.get("var_lo"); act=d["actual"]
        if vl is None: continue
        n = min(len(vl), len(act))
        vl_t = np.array(vl)[-n:]; act_t = np.array(act)[-n:]
        dates=pd.DatetimeIndex(d["dates"])[-n:]
        viol = act_t < vl_t
        ax.plot(dates, vl_t, color=PALETTE.get(m,"#888"), lw=0.9, label=f"{m} VaR")
        ax.scatter(dates[viol], act_t[viol], s=12, color="red", zorder=5, alpha=0.6)
        if date_ref is None: date_ref = (dates, act_t)
    if date_ref is not None:
        ax.plot(date_ref[0], date_ref[1], "w", lw=0.7, label="Actual", alpha=0.7)
    ax.set_title("VaR Back-test (dots = violations)", color="white")
    ax.legend(fontsize=8); plt.tight_layout(); _save("plot6_var_backtest.png")

def plot7_tradeoff(metrics):
    fig, ax = plt.subplots(figsize=(9,7), facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    for m, met in metrics.items():
        cov=met.get("marginal_cov",float("nan"))
        wid=met.get("avg_width",float("nan"))
        ax.scatter(wid, cov, s=200, color=PALETTE.get(m,"#888"), zorder=5)
        ax.annotate(m,(wid,cov), textcoords="offset points", xytext=(6,4), fontsize=9,color="white")
    ax.axhline(COVERAGE_TGT, color="white", ls="--", lw=1)
    ax.set_xlabel("Avg Interval Width"); ax.set_ylabel("Marginal Coverage")
    ax.set_title("Coverage–Width Trade-off", color="white")
    plt.tight_layout(); _save("plot7_tradeoff.png")

def plot8_dashboard(metrics):
    keys=["marginal_cov","worst_cov","var_viol_rate","sharpe","pinball","avg_width"]
    labels=["Marginal Cov","Worst Cov","VaR Viol","Sharpe","Pinball","Width"]
    methods=list(metrics.keys()); nk=len(keys); nm=len(methods)
    fig,axes=plt.subplots(1,nk,figsize=(20,6),facecolor="#0d1117")
    colors=[PALETTE.get(m,"#888") for m in methods]
    for ax,(k,lab) in zip(axes,zip(keys,labels)):
        vals=[metrics[m].get(k,float("nan")) for m in methods]
        ax.set_facecolor("#0d1117")
        ys=[v for v in vals if np.isfinite(v)]
        ax.barh(methods,vals,color=colors)
        ax.set_title(lab,color="white",fontsize=10); ax.tick_params(colors="white",labelsize=8)
    plt.tight_layout(); _save("plot8_dashboard.png")

def plot9_crisis_degradation(metrics):
    methods=list(metrics.keys())
    names=list(CRISIS_PERIODS.keys()); n=len(names)
    fig,ax=plt.subplots(figsize=(10,6),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    x=np.arange(n); w=0.8/len(methods)
    for i,m in enumerate(methods):
        covs=[metrics[m].get(f"crisis_cov_{c}",float("nan")) for c in names]
        bars=ax.bar(x+i*w,covs,w,label=m,color=PALETTE.get(m,"#888"))
    ax.axhline(COVERAGE_TGT,color="white",ls="--",lw=1)
    ax.set_xticks(x+w*len(methods)/2); ax.set_xticklabels(names,color="white")
    ax.set_ylabel("Crisis Coverage"); ax.set_ylim(0,1.05)
    ax.legend(fontsize=8); ax.set_title("Coverage Degradation Per Crisis",color="white")
    plt.tight_layout(); _save("plot9_crisis_degradation.png")

def plot10_violation_clustering(all_res):
    _, a = _asset_sample(all_res, list(all_res.keys()))
    m0=list(all_res.keys())[0]; d=all_res[m0][a]
    lo=d["lower"]; hi=d["upper"]; act=d["actual"]; dates=pd.DatetimeIndex(d["dates"])
    viol=(act<lo)|(act>hi)
    fig,ax=plt.subplots(figsize=(16,3),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    ax.stem(dates, viol.astype(float), linefmt="r-", markerfmt="ro", basefmt="gray")
    for cname,(cs,ce) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(cs),pd.Timestamp(ce),color="yellow",alpha=0.12)
    ax.set_title(f"Coverage Violation Clustering ({m0})", color="white")
    plt.tight_layout(); _save("plot10_violation_clustering.png")

def plot11_aci_alpha(all_res, crisis_s):
    methods=[m for m in all_res if "aci_alpha" in list(all_res[m].values())[0]]
    _, a = _asset_sample(all_res, methods)
    fig,ax=plt.subplots(figsize=(16,5),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    for m in methods:
        d=all_res[m].get(a,{}); aci_a=d.get("aci_alpha")
        if aci_a is None: continue
        dates=pd.DatetimeIndex(d["dates"])
        ax.plot(dates, aci_a, color=PALETTE.get(m,"#888"), lw=1.2, label=m)
    ax.axhline(ALPHA, color="white", ls="--", lw=1)
    for cname,(cs,ce) in CRISIS_PERIODS.items():
        ax.axvspan(pd.Timestamp(cs),pd.Timestamp(ce),color="red",alpha=0.12)
    ax.set_ylabel("ACI Alpha"); ax.set_title("Adaptive Alpha Over Time", color="white")
    ax.legend(fontsize=8); plt.tight_layout(); _save("plot11_aci_alpha.png")

def plot12_multi_crisis(metrics):
    data = {m: {c: metrics[m].get(f"crisis_cov_{c}", float("nan"))
                for c in CRISIS_PERIODS} for m in metrics}
    df=pd.DataFrame(data).T
    fig,ax=plt.subplots(figsize=(10,5),facecolor="#0d1117"); ax.set_facecolor("#0d1117")
    cmap=plt.cm.RdYlGn; im=ax.imshow(df.values, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns,color="white",rotation=30)
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index, color="white")
    for i in range(len(df)): 
        for j in range(len(df.columns)):
            v=df.values[i,j]
            ax.text(j,i,f"{v:.0%}" if np.isfinite(v) else "—", ha="center",va="center",
                    color="black",fontsize=9)
    plt.colorbar(im,ax=ax,label="Coverage"); ax.set_title("Multi-Crisis Coverage Heatmap",color="white")
    plt.tight_layout(); _save("plot12_multi_crisis.png")

def plot13_radar(metrics):
    dims=["marginal_cov","worst_cov","var_viol_rate","pinball","max_dd"]
    labels=["Marg.Cov","Worst.Cov","VaR Viol↓","Pinball↓","MaxDD↓"]
    methods=list(metrics.keys()); N=len(dims)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(1,1,figsize=(8,8),subplot_kw=dict(polar=True),facecolor="#0d1117")
    ax.set_facecolor("#0d1117"); ax.spines["polar"].set_color("#30363d")
    ax.tick_params(colors="white")
    def norm_val(m,k):
        v=metrics[m].get(k,0); return float(np.clip(v,0,1))
    for m in methods:
        vals=[norm_val(m,d) for d in dims]; vals+=vals[:1]
        ax.plot(angles,vals,color=PALETTE.get(m,"#888"),lw=2,label=m)
        ax.fill(angles,vals,color=PALETTE.get(m,"#888"),alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]),labels,color="white")
    ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.1),fontsize=9)
    ax.set_title("Method Comparison Radar",color="white",pad=20)
    plt.tight_layout(); _save("plot13_radar.png")

def plot14_risk_dashboard(all_res, vix, stress_df, metrics, crisis_s):
    """4-panel REAL-TIME RISK DASHBOARD (Upgrade 7)."""
    fig = plt.figure(figsize=(20,14), facecolor="#0d1117")
    gs  = fig.add_gridspec(2,2,hspace=0.4,wspace=0.3)
    axA = fig.add_subplot(gs[0,0]); axB = fig.add_subplot(gs[0,1])
    axC = fig.add_subplot(gs[1,0]); axD = fig.add_subplot(gs[1,1])
    for ax in [axA,axB,axC,axD]:
        ax.set_facecolor("#0d1117"); ax.tick_params(colors="white",labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d")

    _, a = _asset_sample(all_res, list(all_res.keys()))

    axA.set_title("A  Rolling VaR Violation Rate (20d)", color="white", fontsize=12)
    for m, res in all_res.items():
        if a not in res or "var_lo" not in res[a]: continue
        d=res[a]; vl=d["var_lo"]; act=d["actual"]
        dates=pd.DatetimeIndex(d["dates"]); n=min(len(vl),len(act))
        viol=pd.Series((act[:n]<vl[:n]).astype(float), index=dates[:n])
        roll=viol.rolling(20).mean().dropna()
        axA.plot(roll.index, roll.values, color=PALETTE.get(m,"#888"), lw=1.3, label=m)
    axA.axhline(ALPHA, color="white", ls="--", lw=1.5, label=f"{ALPHA:.0%} target")
    axA.axhline(0.03, color="green", ls=":", lw=1, alpha=0.7)
    axA.set_ylim(0, 0.35); axA.set_ylabel("Violation Rate", color="white")
    axA.legend(fontsize=7); axA.yaxis.label.set_color("white")
    for cname,(cs,ce) in CRISIS_PERIODS.items():
        axA.axvspan(pd.Timestamp(cs),pd.Timestamp(ce), color="red", alpha=0.10)

    axB.set_title("B  Adaptive Quantile Width Over Time", color="white", fontsize=12)
    for m, res in all_res.items():
        if a not in res: continue
        d=res[a]; dates=pd.DatetimeIndex(d["dates"])
        wid=(d["upper"]-d["lower"])/2
        axB.plot(dates, wid, color=PALETTE.get(m,"#888"), lw=1, label=m, alpha=0.9)
    axB.set_ylabel("Quantile q(t)", color="white"); axB.legend(fontsize=7)
    axB.yaxis.label.set_color("white")
    for cname,(cs,ce) in CRISIS_PERIODS.items():
        axB.axvspan(pd.Timestamp(cs),pd.Timestamp(ce), color="red", alpha=0.10)

    axC.set_title("C  VIX Regime Timeline", color="white", fontsize=12)
    vix_rng = vix.reindex(pd.DatetimeIndex(all_res[list(all_res.keys())[0]][a]["dates"])).ffill().bfill()
    regime  = np.digitize(vix_rng.values.flatten(), [15., 25., 40.])
    date_rng= pd.DatetimeIndex(all_res[list(all_res.keys())[0]][a]["dates"])
    cmap_reg= plt.cm.RdYlGn_r
    for i, (dt, reg) in enumerate(zip(date_rng, regime)):
        axC.axvspan(dt, date_rng[min(i+1, len(date_rng)-1)],
                    color=cmap_reg(reg/3.5), alpha=0.7, linewidth=0)
    axC.plot(vix_rng.index, vix_rng.values.flatten(), "w", lw=0.9, label="VIX")
    labels_r=["Calm (<15)","Normal (15-25)","Elevated (25-40)","Crisis (>40)"]
    patches  =[mpatches.Patch(color=cmap_reg(i/3.5), label=l)
               for i,l in enumerate(labels_r)]
    axC.legend(handles=patches, fontsize=7, loc="upper left")
    axC.set_ylabel("VIX Level", color="white"); axC.yaxis.label.set_color("white")

    axD.set_title("D  Method Scorecard", color="white", fontsize=12)
    score_keys=["marginal_cov","worst_cov",
                "crisis_cov_COVID_2020","crisis_cov_Inflation_2022",
                "kupiec_p","var_viol_rate"]
    score_lbl=["Marginal Cov","Worst Cov","COVID Cov","Infl Cov","Kupiec p","VaR Viol"]
    methods_s=list(metrics.keys())
    mat=np.zeros((len(score_keys), len(methods_s)))
    for j,m in enumerate(methods_s):
        for i,k in enumerate(score_keys):
            v=metrics[m].get(k, float("nan"))
            if np.isfinite(v): mat[i,j] = float(np.clip(v,0,1))
    im=axD.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axD.set_xticks(range(len(methods_s))); axD.set_xticklabels(methods_s,color="white",rotation=45,ha="right",fontsize=8)
    axD.set_yticks(range(len(score_keys))); axD.set_yticklabels(score_lbl,color="white",fontsize=8)
    for i in range(len(score_keys)):
        for j in range(len(methods_s)):
            v=mat[i,j]
            axD.text(j,i,f"{v:.2f}",ha="center",va="center",color="black",fontsize=8)
    plt.colorbar(im, ax=axD, label="Score (higher=better)")

    fig.suptitle("V4 – Real-Time Risk Dashboard", color="white", fontsize=16, y=1.01)
    plt.tight_layout()
    _save("plot14_risk_dashboard.png")

def main():
    log.info("="*60)
    log.info(" CPPS V4 – Online Conformal + Multi-Period + Fixed VaR")
    log.info("="*60)

    returns = fetch_data(ASSETS, DATA_START, DATA_END)
    vix     = fetch_vix(returns)
    fd      = engineer_features(returns, vix)
    train, cal, test = split_indices(returns.index)
    log.info("Crisis days flagged in test: %d", (detect_crisis_regime(returns)[test]).sum())
    crisis_s = detect_crisis_regime(returns)
    crisis_s = crisis_s[crisis_s.index[test]]

    cpps_m  = CPPSModel(ALPHA);     cpps_m.fit(returns, fd, train, cal, vix)
    bayes_m = BayesianModel(ALPHA); bayes_m.fit(returns, fd, train, cal)
    qr_m    = QRModel(ALPHA);       qr_m.fit(returns, fd, train, cal)
    acqr_m  = AdaptiveCQR(ALPHA);   acqr_m.fit(returns, fd, train, cal)

    # VaRCalibrator – fixes Kupiec failure for ALL methods
    var_cal = VaRCalibrator(ALPHA); var_cal.fit(returns, fd, train, cal)

    # MultiPeriodConformal – crisis bucket from VIX regimes
    cal_vix_arr = vix.reindex(vix.index[cal]).ffill().bfill().values.flatten()
    multi_period = MultiPeriodConformal(ALPHA)
    multi_period.fit(cpps_m.cal_raw_, cal_vix_arr)

    # OnlineCPPS – wraps CPPS with online calibration
    online_cpps = OnlineCPPSModel(cpps_m, ALPHA)

    # ScoreBlendedHybrid
    sbh_m = ScoreBlendedHybrid(cpps_m, acqr_m, ALPHA)

    log.info("Generating predictions …")
    res_online = online_cpps.predict_intervals(
        returns, fd, test, crisis_s, var_cal, multi_period, vix)
    res_cpps   = cpps_m.predict_intervals(returns, fd, test, crisis_s)
    res_bayes  = bayes_m.predict_intervals(returns, fd, test, crisis_s, var_cal)
    res_qr     = qr_m.predict_intervals(returns, fd, test, crisis_s, var_cal)
    res_acqr   = acqr_m.predict_intervals(returns, fd, test, crisis_s, var_cal)
    res_sbh    = sbh_m.predict_intervals(returns, fd, test, crisis_s, var_cal)

    # Override var_lo in CPPS with VaRCalibrator results
    var_res = var_cal.predict_var(returns, fd, test)
    for a in res_cpps:
        if a in var_res: res_cpps[a]["var_lo"] = var_res[a]["var_lo"]

    base_results = {
        "OnlineCPPS": res_online,
        "CPPS":       res_cpps,
        "Bayesian":   res_bayes,
        "QR":         res_qr,
        "AdaptCQR":   res_acqr,
        "ScoreHybrid":res_sbh,
    }

    log.info("Building ConformalEnsemble …")
    ens = ConformalEnsemble()
    res_ens = ens.predict_intervals(base_results, crisis_s)
    all_results = dict(base_results)
    all_results["ConfEnsemble"] = res_ens

    log.info("Building portfolios …")
    portfolios = build_portfolio(all_results, returns)
    log.info("Computing metrics …")
    metrics = compute_metrics(all_results, portfolios, crisis_s, var_cal)

    log.info("Running statistical tests …")
    print_summary(metrics)

    stress_df = stress_test_all(all_results)
    print("\n── Stress Test Results ──────────────────────────────────────────────────")
    print(stress_df.to_string(index=False))
    stress_df.to_csv(OUT / "stress_test.csv", index=False)

    log.info("Generating 14 plots …")
    plot1_intervals(all_results, crisis_s, vix)
    plot2_rolling_coverage(all_results)
    plot3_crisis_zoom(all_results)
    plot4_violin_widths(all_results, crisis_s)
    plot5_portfolio_weights(portfolios)
    plot6_var_backtest(all_results)
    plot7_tradeoff(metrics)
    plot8_dashboard(metrics)
    plot9_crisis_degradation(metrics)
    plot10_violation_clustering(all_results)
    plot11_aci_alpha(all_results, crisis_s)
    plot12_multi_crisis(metrics)
    plot13_radar(metrics)
    plot14_risk_dashboard(all_results, vix, stress_df, metrics, crisis_s)

    log.info("Saving CSVs …")
    pd.DataFrame(metrics).T.to_csv(OUT / "metrics_v4.csv")
    for method, res in all_results.items():
        rows = []
        for a in res:
            d=res[a]; n=len(d["actual"])
            for i in range(n):
                rows.append({"method":method,"asset":a,
                             "date":str(d["dates"][i]) if i<len(d["dates"]) else "",
                             "actual":float(d["actual"][i]),
                             "lower":float(d["lower"][i]) if i<len(d["lower"]) else float("nan"),
                             "upper":float(d["upper"][i]) if i<len(d["upper"]) else float("nan"),
                             "var_lo":float(d["var_lo"][i]) if "var_lo" in d and i<len(d["var_lo"]) else float("nan")})
        if rows:
            pd.DataFrame(rows).to_csv(OUT / f"predictions_{method}.csv", index=False)
    log.info("Done. 14 plots + CSVs in %s/", OUT)

if __name__ == "__main__":
    main()
