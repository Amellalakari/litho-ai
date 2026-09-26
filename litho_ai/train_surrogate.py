"""Train ML surrogates of the lithography simulator.

Models
  * print/fail classifier   : does the line survive development? (XGBoost + MLP)
  * CD regressor            : critical dimension of printed lines (XGBoost + MLP)
  * LER regressor           : line-edge-roughness proxy, fitted in log space (MLP)

Models are trained on the NOISY measured CD (what a fab sees) and evaluated
against both the measured and the noise-free true CD.
The small MLPs are exported to JSON so the dashboard can run inference in the browser.
"""
import json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from litho_ai.generate_dataset import FEATURES, RANGES
from litho_ai.simulator import LithoSimulator

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def export_mlp(model, scaler, path, y_mean=0.0, y_std=1.0, kind="regressor"):
    json.dump({
        "kind": kind, "features": FEATURES,
        "x_mean": scaler.mean_.tolist(), "x_std": scaler.scale_.tolist(),
        "y_mean": y_mean, "y_std": y_std, "activation": model.activation,
        "weights": [w.tolist() for w in model.coefs_],
        "biases": [b.tolist() for b in model.intercepts_],
    }, open(path, "w"))


def main():
    df = pd.read_csv("data/litho_dataset.csv")
    df["prints"] = (df.status == 0).astype(int)
    tr, te = train_test_split(df, test_size=0.2, random_state=0, stratify=df.prints)
    sx = StandardScaler().fit(tr[FEATURES])
    Xtr, Xte = sx.transform(tr[FEATURES]), sx.transform(te[FEATURES])
    res = {}

    # ---------------- classifier
    xgc = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9).fit(Xtr, tr.prints)
    mlpc = MLPClassifier(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=2000, alpha=1e-4,
                         early_stopping=True, random_state=0).fit(Xtr, tr.prints)
    for name, m in [("xgb", xgc), ("mlp", mlpc)]:
        p = m.predict_proba(Xte)[:, 1]
        res[f"clf_{name}"] = {"accuracy": accuracy_score(te.prints, p > 0.5),
                              "roc_auc": roc_auc_score(te.prints, p)}

    # ---------------- CD regression (printed fields only)
    trp, tep = tr[tr.prints == 1], te[(te.prints == 1) & te.CD_true.notna()]
    Xtrp, Xtep = sx.transform(trp[FEATURES]), sx.transform(tep[FEATURES])
    xgr = XGBRegressor(n_estimators=4000, max_depth=3, min_child_weight=10, learning_rate=0.03,
                       subsample=0.8, reg_lambda=5).fit(Xtrp, trp.CD)
    cd_mu, cd_sd = trp.CD.mean(), trp.CD.std()
    mlpr = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="tanh", max_iter=4000, alpha=1e-4,
                        early_stopping=True, n_iter_no_change=50, random_state=0
                        ).fit(Xtrp, (trp.CD - cd_mu) / cd_sd)
    preds = {"xgb": xgr.predict(Xtep), "mlp": mlpr.predict(Xtep) * cd_sd + cd_mu}
    # in-spec region (target 90 nm +/-10%) is what process engineers care about
    spec = tep.CD_true.between(70, 110).values
    for name, p in preds.items():
        res[f"cd_{name}"] = {
            "rmse_vs_measured": rmse(tep.CD, p), "rmse_vs_true": rmse(tep.CD_true, p),
            "r2_vs_measured": r2_score(tep.CD, p),
            "rmse_vs_true_in_70_110nm": rmse(tep.CD_true[spec], p[spec])}
    res["cd_noise_floor_rmse(measured_vs_true)"] = rmse(tep.CD, tep.CD_true)
    res["cd_noise_floor_rmse_in_70_110nm"] = rmse(tep.CD[spec], tep.CD_true[spec])

    # ---------------- LER regression (log space)
    trl = trp[trp.LER.between(1, 60)]
    tel = tep[tep.LER.between(1, 60)]
    ly_mu, ly_sd = np.log(trl.LER).mean(), np.log(trl.LER).std()
    mlpl = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=4000, alpha=1e-4,
                        early_stopping=True, n_iter_no_change=50, random_state=0
                        ).fit(sx.transform(trl[FEATURES]), (np.log(trl.LER) - ly_mu) / ly_sd)
    pl = np.exp(mlpl.predict(sx.transform(tel[FEATURES])) * ly_sd + ly_mu)
    res["ler_mlp"] = {"rmse": rmse(tel.LER, pl), "r2": r2_score(tel.LER, pl)}

    # ---------------- speed: simulator vs surrogate
    sim = LithoSimulator()
    Xs = te[FEATURES].values[:200]
    t0 = time.perf_counter()
    for r in Xs:
        sim.simulate(**dict(zip(FEATURES, r)))
    t_sim = (time.perf_counter() - t0) / len(Xs)
    big = np.repeat(Xte, 50, axis=0)
    t0 = time.perf_counter(); xgr.predict(big); t_xgb = (time.perf_counter() - t0) / len(big)
    t0 = time.perf_counter(); mlpr.predict(big); t_mlp = (time.perf_counter() - t0) / len(big)
    res["speed"] = {"sim_ms_per_field": t_sim * 1e3, "xgb_us_per_field": t_xgb * 1e6,
                    "mlp_us_per_field": t_mlp * 1e6,
                    "speedup_xgb": t_sim / t_xgb, "speedup_mlp": t_sim / t_mlp}
    res["n_train"], res["n_test"] = len(tr), len(te)

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (name, p) in zip(ax[:2], preds.items()):
        a.scatter(tep.CD_true, p, s=4, alpha=0.4, color="#2f6fb0")
        lim = [0, 200]; a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
        a.axhspan(81, 99, color="#e8a33d", alpha=0.12)
        a.set_xlabel("Simulated true CD (nm)"); a.set_ylabel(f"{name.upper()} predicted CD (nm)")
        a.set_title(f"{name.upper()}: RMSE vs true = {res[f'cd_{name}']['rmse_vs_true']:.2f} nm")
    ax[2].scatter(tel.LER, pl, s=4, alpha=0.4, color="#7a4fb5")
    ax[2].plot([0, 40], [0, 40], "k--", lw=1); ax[2].set_xlim(0, 40); ax[2].set_ylim(0, 40)
    ax[2].set_xlabel("Measured LER proxy (nm)"); ax[2].set_ylabel("MLP predicted LER (nm)")
    ax[2].set_title(f"LER: R² = {res['ler_mlp']['r2']:.3f}")
    fig.tight_layout(); fig.savefig("figures/surrogate_parity.png", dpi=150)

    imp = pd.Series(xgr.feature_importances_, FEATURES).sort_values()
    fig, a = plt.subplots(figsize=(6, 3.5)); imp.plot.barh(ax=a, color="#2f6fb0")
    a.set_title("XGBoost feature importance (CD)"); fig.tight_layout()
    fig.savefig("figures/feature_importance.png", dpi=150)
    res["feature_importance"] = imp.round(4).to_dict()

    # ---------------- export
    export_mlp(mlpr, sx, "models/mlp_cd.json", cd_mu, cd_sd)
    export_mlp(mlpc, sx, "models/mlp_print.json", kind="classifier")
    export_mlp(mlpl, sx, "models/mlp_ler_log.json", ly_mu, ly_sd)
    xgr.save_model("models/xgb_cd.json"); xgc.save_model("models/xgb_print.json")
    json.dump({"ranges": RANGES, **res}, open("models/metrics.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
