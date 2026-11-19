"""ICONN-2027 / Topic 2
Uncertainty-aware ensemble learning for reliable materials-property (band gap) prediction.

Data (real, public, unmodified): Matbench v0.1 `matbench_expt_gap` (Dunn et al., arXiv:2005.00707),
4,604 experimentally measured band gaps (Zhuo 2018 / Kingsbury 2022 curation).
Evaluation follows the official Matbench protocol: KFold(n_splits=5, shuffle=True, random_state=18012019).

Uncertainty methods compared
  M1 Random-forest tree variance                                   (arXiv:2107.07997, 2211.02235)
  M2 Bootstrap XGBoost ensemble                                    (arXiv:2310.13136, 2504.12627 shallow ensembles)
  M3 LightGBM quantile regression (5/50/95 %)                      (arXiv:2310.13136)
  M4 Gaussian-process regression on PCA descriptors                (arXiv:2211.02235)
  M5 MLP deep ensemble (5 networks)                                (arXiv:2311.02495, 2504.12627)
Enhancements from the literature
  E1 post-hoc sigma recalibration  sigma' = a*sigma + b (min. NLL)  (arXiv:2107.07997, Palmer et al.)
  E2 normalised split-conformal intervals / CQR for M3              (arXiv:2406.15650, conformal refs)
  E3 domain guidance: kNN distance-to-training as in-domain flag    (arXiv:2406.15650)
  E4 out-of-distribution leave-cluster-out evaluation               (arXiv:2401.08032)
  E5 uncertainty-aware screening for a photovoltaic window          (arXiv:2610.06985)
No synthetic data, oversampling or target imputation is used.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(TOPIC))
from common.features import (DATA_RAW, composition_features, regression_metrics,  # noqa: E402
                             setup_plots)

RES = os.path.join(TOPIC, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
ALPHA = 0.10  # target 90 % intervals
Z = norm.ppf(1 - ALPHA / 2)
plt = setup_plots()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================ UQ models: fit(X,y) -> predict(X) -> (mu, sigma)
class RFUQ:
    name = "M1_RandomForest"

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(n_estimators=500, max_features=0.33, min_samples_leaf=1,
                                       n_jobs=-1, random_state=SEED).fit(X, y)
        return self

    def predict(self, X):
        allp = np.stack([t.predict(X) for t in self.m.estimators_])
        return allp.mean(0), allp.std(0) + 1e-6


class XGBEnsembleUQ:
    name = "M2_XGB_BootstrapEnsemble"

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        from xgboost import XGBRegressor
        rng = np.random.default_rng(SEED)
        self.ms = []
        for i in range(self.k):
            idx = rng.integers(0, len(y), len(y))
            self.ms.append(XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                        colsample_bytree=0.6, n_jobs=-1, random_state=SEED + i)
                           .fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        allp = np.stack([m.predict(X) for m in self.ms])
        return allp.mean(0), allp.std(0) + 1e-6


class QuantileUQ:
    name = "M3_LGBM_Quantile"

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        kw = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, colsample_bytree=0.6,
                  subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)
        self.lo = LGBMRegressor(objective="quantile", alpha=ALPHA / 2, **kw).fit(X, y)
        self.md = LGBMRegressor(objective="quantile", alpha=0.5, **kw).fit(X, y)
        self.hi = LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, **kw).fit(X, y)
        return self

    def predict_q(self, X):
        lo, md, hi = self.lo.predict(X), self.md.predict(X), self.hi.predict(X)
        return np.minimum(lo, md), md, np.maximum(hi, md)

    def predict(self, X):
        lo, md, hi = self.predict_q(X)
        return md, np.maximum((hi - lo) / (2 * Z), 1e-3)


class GPUQ:
    name = "M4_GaussianProcess"

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(n_components=30, random_state=SEED).fit(self.sc.transform(X))
        Z_ = self.pca.transform(self.sc.transform(X))
        self.ym, self.ys = y.mean(), y.std()
        k = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.1)
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=False, random_state=SEED,
                                           n_restarts_optimizer=0).fit(Z_, (y - self.ym) / self.ys)
        return self

    def predict(self, X):
        Z_ = self.pca.transform(self.sc.transform(X))
        mu, sd = self.gp.predict(Z_, return_std=True)
        return mu * self.ys + self.ym, sd * self.ys + 1e-6


class MLPEnsembleUQ:
    name = "M5_MLP_DeepEnsemble"

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import RobustScaler
        # robust scaling + clipping prevents rare extreme descriptors from making a network diverge
        self.sc = RobustScaler(quantile_range=(5, 95)).fit(X)
        Xs = np.clip(self.sc.transform(X), -5, 5)
        self.ylo, self.yhi = float(y.min()), float(y.max())
        self.ms = [MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3,
                                max_iter=400, early_stopping=True, random_state=SEED + i).fit(Xs, y)
                   for i in range(5)]
        return self

    def predict(self, X):
        Xs = np.clip(self.sc.transform(X), -5, 5)
        allp = np.clip(np.stack([m.predict(Xs) for m in self.ms]), self.ylo, self.yhi)
        return allp.mean(0), allp.std(0) + 1e-6


METHODS = [RFUQ, XGBEnsembleUQ, QuantileUQ, GPUQ, MLPEnsembleUQ]


# ============================================================ calibration helpers
def nll(y, mu, s):
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + (y - mu) ** 2 / (2 * s ** 2)))


def fit_recal(y, mu, s):
    """E1: sigma' = a*sigma + b minimising Gaussian NLL on the calibration split."""
    f = lambda p: nll(y, mu, np.abs(p[0]) * s + np.abs(p[1]) + 1e-6)
    r = minimize(f, x0=[1.0, 0.05], method="Nelder-Mead")
    return abs(r.x[0]), abs(r.x[1])


def miscal_area(y, mu, s, n=50):
    ps = np.linspace(0.01, 0.99, n)
    obs = [np.mean(np.abs(y - mu) <= norm.ppf(0.5 + p / 2) * s) for p in ps]
    return float(np.trapezoid(np.abs(np.array(obs) - ps), ps)), ps, np.array(obs)


def uq_metrics(y, mu, s, lo=None, hi=None):
    if lo is None:
        lo, hi = mu - Z * s, mu + Z * s
    err = np.abs(y - mu)
    ma, _, _ = miscal_area(y, mu, s)
    return {"MAE": float(err.mean()), "NLL": nll(y, mu, s),
            "coverage90": float(np.mean((y >= lo) & (y <= hi))),
            "mean_width90": float(np.mean(hi - lo)),
            "spearman_err_sigma": float(spearmanr(err, s)[0]),
            "miscalibration_area": ma}


def knn_dist(Xref, Xq, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(Xref)
    return nn.kneighbors(Xq)[0].mean(1)


# ============================================================ main
def main():
    from matminer.datasets import load_dataset
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    out = {}
    df = load_dataset("matbench_expt_gap", data_home=DATA_RAW)
    df["formula"] = df["composition"].astype(str)
    X = composition_features(df["formula"], cache_name="t2_matbench_expt_gap_feats")
    X = X.loc[:, (X.notna().mean() > 0.9) & (X.std() > 0)]
    X = X.fillna(X.median())  # descriptor-level fill only (very few NaNs); targets untouched
    Xv, y = X.values.astype(float), df["gap expt"].values.astype(float)
    out["data"] = {"n": int(len(y)), "n_features": int(X.shape[1]), "metals": int((y == 0).sum()),
                   "gap_mean": float(y.mean()), "gap_max": float(y.max())}
    log("data", out["data"])

    kf = KFold(n_splits=5, shuffle=True, random_state=18012019)  # official Matbench folds
    rows, oof = [], {M.name: {"mu": np.zeros(len(y)), "s_raw": np.zeros(len(y)), "s_cal": np.zeros(len(y)),
                             "lo_conf": np.zeros(len(y)), "hi_conf": np.zeros(len(y)), "dist": np.zeros(len(y))}
                     for M in METHODS}
    for fold, (tr, te) in enumerate(kf.split(Xv)):
        # 80/20 proper-train / calibration split inside each training fold (split-conformal)
        ptr, cal = train_test_split(tr, test_size=0.2, random_state=SEED + fold)
        sc = StandardScaler().fit(Xv[ptr])
        d_cal = knn_dist(sc.transform(Xv[ptr]), sc.transform(Xv[cal]))
        d_te = knn_dist(sc.transform(Xv[ptr]), sc.transform(Xv[te]))
        for M in METHODS:
            t0 = time.time()
            m = M().fit(Xv[ptr], y[ptr])
            mu_c, s_c = m.predict(Xv[cal])
            mu_t, s_t = m.predict(Xv[te])
            mu_c, mu_t = np.clip(mu_c, 0, None), np.clip(mu_t, 0, None)
            a, b = fit_recal(y[cal], mu_c, s_c)                         # E1
            s_cal = a * s_t + b
            if isinstance(m, QuantileUQ):                               # E2 (CQR)
                lo_c, _, hi_c = m.predict_q(Xv[cal])
                sc_ = np.maximum(lo_c - y[cal], y[cal] - hi_c)
                q = np.quantile(sc_, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal))
                lo_t, _, hi_t = m.predict_q(Xv[te])
                lo_conf, hi_conf = lo_t - q, hi_t + q
            else:                                                        # E2 normalised conformal
                r = np.abs(y[cal] - mu_c) / s_c
                q = np.quantile(r, min(1, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal)))
                lo_conf, hi_conf = mu_t - q * s_t, mu_t + q * s_t
            lo_conf = np.clip(lo_conf, 0, None)
            o = oof[M.name]
            o["mu"][te], o["s_raw"][te], o["s_cal"][te] = mu_t, s_t, s_cal
            o["lo_conf"][te], o["hi_conf"][te], o["dist"][te] = lo_conf, hi_conf, d_te
            r_raw = uq_metrics(y[te], mu_t, s_t)
            r_cal = uq_metrics(y[te], mu_t, s_cal)
            r_conf = uq_metrics(y[te], mu_t, s_t, lo_conf, hi_conf)
            dthr = np.quantile(d_cal, 0.95)                               # E3 domain flag
            ind = d_te <= dthr
            rows.append({"fold": fold, "method": M.name, "fit_s": round(time.time() - t0, 1),
                         **{f"raw_{k}": v for k, v in r_raw.items()},
                         **{f"recal_{k}": v for k, v in r_cal.items() if k != "MAE"},
                         "conformal_coverage90": r_conf["coverage90"],
                         "conformal_width90": r_conf["mean_width90"],
                         "MAE_in_domain": float(np.abs(y[te] - mu_t)[ind].mean()),
                         "MAE_out_domain": float(np.abs(y[te] - mu_t)[~ind].mean()) if (~ind).any() else np.nan,
                         "frac_out_domain": float((~ind).mean()), "recal_a": a, "recal_b": b})
            log(f"fold {fold}", M.name, f"MAE={r_raw['MAE']:.3f} cov_raw={r_raw['coverage90']:.2f} "
                f"cov_cal={r_cal['coverage90']:.2f} cov_conf={r_conf['coverage90']:.2f}")
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(RES, "table_per_fold.csv"), index=False)
    summ = per_fold.drop(columns="fold").groupby("method").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(RES, "table_summary_mean_std.csv"))
    out["summary_mean_over_folds"] = per_fold.drop(columns="fold").groupby("method").mean().round(4).to_dict(orient="index")

    # ---------------- figures: calibration curves, confidence curves, error vs sigma
    cols = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b7791f"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for c, M in zip(cols, METHODS):
        o = oof[M.name]
        for ax, key, ttl in [(axes[0], "s_raw", "Before recalibration"), (axes[1], "s_cal", "After σ recalibration (E1)")]:
            a_, ps, obs = miscal_area(y, o["mu"], o[key])
            ax.plot(ps, obs, color=c, label=f"{M.name.split('_',1)[1]} ({a_:.3f})")
            ax.set_title(ttl, fontsize=10)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Expected confidence"); ax.set_ylabel("Observed fraction")
        ax.legend(fontsize=7, title="Miscalibration area", title_fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_curves.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    fr = np.linspace(0.1, 1.0, 19)
    conf = {}
    for c, M in zip(cols, METHODS):
        o = oof[M.name]; err = np.abs(y - o["mu"]); order = np.argsort(o["s_cal"])
        maes = [err[order[:int(f * len(y))]].mean() for f in fr]
        conf[M.name] = maes
        ax.plot(fr * 100, maes, color=c, label=M.name.split("_", 1)[1])
    oracle = np.sort(np.abs(y - oof[METHODS[1].name]["mu"]))
    ax.plot(fr * 100, [oracle[:int(f * len(y))].mean() for f in fr], "k:", label="Oracle (XGB ens.)")
    ax.set_xlabel("% of most-confident predictions kept"); ax.set_ylabel("MAE (eV)")
    ax.set_title("Confidence curves (5-fold Matbench OOF)", fontsize=10); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "confidence_curves.png")); plt.close(fig)
    pd.DataFrame(conf, index=np.round(fr, 2)).to_csv(os.path.join(RES, "table_confidence_curve_mae.csv"))

    # error-based reliability diagram (RMSE vs RMV) for the ensemble
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for c, M in zip(cols, METHODS):
        o = oof[M.name]; q = pd.qcut(o["s_cal"], 10, duplicates="drop")
        g = pd.DataFrame({"q": q, "se": (y - o["mu"]) ** 2, "v": o["s_cal"] ** 2}).groupby("q", observed=True).mean()
        ax.plot(np.sqrt(g.v), np.sqrt(g.se), "o-", ms=3, color=c, label=M.name.split("_", 1)[1])
    lim = ax.get_xlim()[1]; ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Root mean predicted variance (eV)"); ax.set_ylabel("Observed RMSE (eV)")
    ax.set_title("Error-based calibration (recalibrated)", fontsize=10); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "reliability_rmse_vs_rmv.png")); plt.close(fig)

    # E3: error vs distance to training data
    o = oof["M2_XGB_BootstrapEnsemble"]
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    q = pd.qcut(o["dist"], 10)
    g = pd.DataFrame({"q": q, "e": np.abs(y - o["mu"]), "s": o["s_cal"], "d": o["dist"]}).groupby("q", observed=True).mean()
    ax.plot(g.d, g.e, "o-", label="MAE"); ax.plot(g.d, g.s, "s--", label="mean σ (recal.)")
    ax.set_xlabel("Mean 5-NN distance to training set (scaled descriptors)"); ax.set_ylabel("eV")
    ax.set_title("Domain guidance (E3)", fontsize=10); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "domain_distance_vs_error.png")); plt.close(fig)

    # ---------------- E4 leave-cluster-out OOD (KMeans on scaled descriptors)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    Zs = PCA(n_components=20, random_state=SEED).fit_transform(StandardScaler().fit_transform(Xv))
    cl = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(Zs)
    ood = []
    for k in range(5):
        tr, te = np.where(cl != k)[0], np.where(cl == k)[0]
        ptr, cal = train_test_split(tr, test_size=0.2, random_state=SEED)
        for M in [RFUQ, XGBEnsembleUQ, QuantileUQ]:
            m = M().fit(Xv[ptr], y[ptr])
            mu_c, s_c = m.predict(Xv[cal]); mu_t, s_t = m.predict(Xv[te])
            mu_c, mu_t = np.clip(mu_c, 0, None), np.clip(mu_t, 0, None)
            a, b = fit_recal(y[cal], mu_c, s_c)
            r_ = np.abs(y[cal] - mu_c) / s_c
            qq = np.quantile(r_, np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal))
            cov = float(np.mean(np.abs(y[te] - mu_t) <= qq * s_t))
            ood.append({"held_out_cluster": k, "n_test": int(len(te)), "method": M.name,
                        "MAE": float(np.abs(y[te] - mu_t).mean()),
                        "conformal_coverage90": cov,
                        "recal_coverage90": float(np.mean(np.abs(y[te] - mu_t) <= Z * (a * s_t + b))),
                        "mean_sigma_recal": float((a * s_t + b).mean())})
            log("OOD cluster", k, M.name, round(ood[-1]["MAE"], 3), round(cov, 3))
    ood = pd.DataFrame(ood)
    ood.to_csv(os.path.join(RES, "table_ood_leave_cluster_out.csv"), index=False)
    out["ood_leave_cluster_out_mean"] = ood.groupby("method")[["MAE", "conformal_coverage90",
                                                               "recal_coverage90"]].mean().round(4).to_dict(orient="index")

    # ---------------- E5 uncertainty-aware screening: PV window 1.1-1.8 eV
    lo_w, hi_w = 1.1, 1.8
    truth = (y >= lo_w) & (y <= hi_w)
    o = oof["M2_XGB_BootstrapEnsemble"]
    p_in = norm.cdf(hi_w, o["mu"], o["s_cal"]) - norm.cdf(lo_w, o["mu"], o["s_cal"])
    scr = {}
    for N in [50, 100, 200]:
        top_mu = np.argsort(np.abs(o["mu"] - (lo_w + hi_w) / 2))[:N]
        top_p = np.argsort(-p_in)[:N]
        scr[N] = {"precision_by_mean_closeness": float(truth[top_mu].mean()),
                  "precision_by_probability_in_window": float(truth[top_p].mean())}
    out["screening_PV_window_1.1-1.8eV"] = {"n_true_in_window": int(truth.sum()),
                                           "base_rate": float(truth.mean()), "topN": scr}
    pd.DataFrame({"formula": df.formula, "gap_expt": y,
                  **{f"{M.name}_mu": oof[M.name]["mu"] for M in METHODS},
                  **{f"{M.name}_sigma_recal": oof[M.name]["s_cal"] for M in METHODS},
                  "XGBens_P_in_PV_window": p_in}).to_csv(os.path.join(RES, "predictions_oof_with_uncertainty.csv"), index=False)

    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    log("done")


if __name__ == "__main__":
    main()
