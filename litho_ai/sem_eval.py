"""Evaluate the SEM U-Net: metrology (CD, LER) vs a classical CD-SEM algorithm, defect
classification, and a closed loop back to Phase 1 (Bossung curves measured from images)."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from litho_ai.sem import render, SEMParams, CLASSES, PX
from litho_ai.sem_unet import UNet, predict
from litho_ai.sem_metrology import measure_unet, measure_classical, classify_unet, defect_boxes
from litho_ai.simulator import LithoSimulator

FRAMES = [1, 2, 4, 8, 16, 32]
BLUE, RED, INK, AMBER = "#2f6fb0", "#c0504d", "#222", "#e8a33d"
DEF_COL = {"particle": "#3c8d5a", "bridge": "#d64541", "break": "#8e44ad"}


def load_model(path="models/sem_unet.pt"):
    m = UNet(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); return m


def run_set(model, name):
    d = np.load(f"data/sem_{name}.npz"); meta = pd.read_csv(f"data/sem_{name}_meta.csv")
    P = predict(model, d["imgs"])
    rows = []
    for i in range(len(meta)):
        cu, lu, nu = measure_unet(P[i]); cc, lc, nc = measure_classical(d["imgs"][i] / 255.0)
        rows.append(dict(cd_unet=cu, ler_unet=lu, rows_unet=nu, cd_cl=cc, ler_cl=lc))
    return d, meta.join(pd.DataFrame(rows)), P


def tune_thresholds(P, meta):
    grid = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48]
    thr = {"particle": 8, "bridge": 8, "break": 8}
    def acc(t): return np.mean([classify_unet(p, t) == d for p, d in zip(P, meta.defect)])
    for _ in range(2):
        for k in thr:
            thr[k] = max(grid, key=lambda g: acc({**thr, k: g}))
    return thr, acc(thr)


def main():
    model = load_model()
    _, val, Pv = run_set(model, "val")
    thr, val_acc = tune_thresholds(Pv, val)
    # constant CD offset calibration on the validation set (standard CD-SEM practice vs reference metrology)
    off_u = float(np.nanmedian(val.cd_unet - val.cd_true)); off_c = float(np.nanmedian(val.cd_cl - val.cd_true))

    dte, te, Pt = run_set(model, "test")
    te["cd_unet"] -= off_u; te["cd_cl"] -= off_c
    te["pred"] = [classify_unet(p, thr) for p in Pt]
    for m in ["unet", "cl"]:
        te[f"e_cd_{m}"] = te[f"cd_{m}"] - te.cd_true; te[f"e_ler_{m}"] = te[f"ler_{m}"] - te.ler_true

    res = {"area_thresholds_px": thr, "val_defect_accuracy": val_acc,
           "cd_offset_calibration_nm": {"unet": off_u, "classical": off_c}}
    by = {}
    for f in FRAMES:
        s = te[te.frames == f]; clean = s[s.defect == "none"]
        by[f] = {
            "cd_rmse_unet": float(np.sqrt(np.nanmean(s.e_cd_unet ** 2))),
            "cd_rmse_classical": float(np.sqrt(np.nanmean(s.e_cd_cl ** 2))),
            "cd_rmse_unet_clean": float(np.sqrt(np.nanmean(clean.e_cd_unet ** 2))),
            "cd_rmse_classical_clean": float(np.sqrt(np.nanmean(clean.e_cd_cl ** 2))),
            "ler_bias_unet": float(np.nanmean(s.e_ler_unet)), "ler_bias_classical": float(np.nanmean(s.e_ler_cl)),
            "ler_rmse_unet": float(np.sqrt(np.nanmean(s.e_ler_unet ** 2))),
            "ler_rmse_classical": float(np.sqrt(np.nanmean(s.e_ler_cl ** 2))),
            "defect_accuracy": float((s.pred == s.defect).mean()),
            "unet_failed": int(s.cd_unet.isna().sum()), "classical_failed": int(s.cd_cl.isna().sum())}
    res["by_frames"] = by
    res["defect_accuracy_overall"] = float((te.pred == te.defect).mean())
    cm = pd.crosstab(te.defect, te.pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    res["confusion"] = cm.to_dict()
    te.to_csv("data/sem_test_results.csv", index=False)

    # ---------------- figures
    fr = np.array(FRAMES)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(fr, [by[f]["cd_rmse_unet"] for f in FRAMES], "o-", c=BLUE, label="U-Net, all images")
    ax[0].plot(fr, [by[f]["cd_rmse_classical"] for f in FRAMES], "s-", c=RED, label="Classical, all images")
    ax[0].plot(fr, [by[f]["cd_rmse_unet_clean"] for f in FRAMES], "o--", c=BLUE, alpha=.5, label="U-Net, defect-free")
    ax[0].plot(fr, [by[f]["cd_rmse_classical_clean"] for f in FRAMES], "s--", c=RED, alpha=.5, label="Classical, defect-free")
    ax[0].set(xscale="log", xticks=fr, xticklabels=fr, xlabel="SEM frames averaged (dose on wafer)",
              ylabel="CD error, RMSE (nm)", title="CD accuracy vs SEM noise")
    ax[1].plot(fr, [by[f]["ler_bias_unet"] for f in FRAMES], "o-", c=BLUE, label="U-Net")
    ax[1].plot(fr, [by[f]["ler_bias_classical"] for f in FRAMES], "s-", c=RED, label="Classical")
    ax[1].axhline(0, c=INK, lw=.8)
    ax[1].set(xscale="log", xticks=fr, xticklabels=fr, xlabel="SEM frames averaged",
              ylabel="LER bias, measured − true (nm)", title="Noise-induced LER bias")
    ax[2].plot(fr, [100 * by[f]["defect_accuracy"] for f in FRAMES], "o-", c=BLUE)
    ax[2].set(xscale="log", xticks=fr, xticklabels=fr, ylim=(50, 101), xlabel="SEM frames averaged",
              ylabel="Correct defect call (%)", title="Defect classification (4 classes)")
    for a in ax: a.grid(alpha=.3);
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/sem_metrology_vs_noise.png", dpi=150)

    fig, a = plt.subplots(figsize=(4.8, 4.2))
    M = cm.values; a.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): a.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M[i].sum() * .5 else INK)
    a.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="True",
          title=f"Defect calls, test set ({100 * res['defect_accuracy_overall']:.1f}% correct)")
    fig.tight_layout(); fig.savefig("figures/sem_confusion.png", dpi=150)

    # gallery
    cmap = ListedColormap(["#00000000", "#5b9bd5", DEF_COL["particle"], DEF_COL["bridge"], DEF_COL["break"]])
    picks = []
    for dname in CLASSES:
        for f in (8, 2):
            s = te[(te.defect == dname) & (te.frames == f) & (te.pred == dname)]
            if len(s): picks.append(s.index[0])
    fig, ax = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    for c, i in enumerate(picks):
        ax[0, c].imshow(dte["imgs"][i], cmap="gray", vmin=0, vmax=255)
        ax[0, c].set_title(f"{te.defect[i]}, {te.frames[i]} fr\nCD {te.cd_true[i]:.1f} nm", fontsize=9)
        ax[1, c].imshow(dte["imgs"][i], cmap="gray", vmin=0, vmax=255)
        ax[1, c].imshow(np.ma.masked_equal(Pt[i].argmax(0), 0), cmap=cmap, vmin=0, vmax=4, alpha=.45, interpolation="nearest")
        for x0, y0, x1, y1, n in defect_boxes(Pt[i]):
            ax[1, c].add_patch(plt.Rectangle((x0 - 3, y0 - 3), x1 - x0 + 6, y1 - y0 + 6, fill=False, ec=DEF_COL[n], lw=1.5))
        ax[1, c].set_title(f"U-Net: {te.pred[i]}\nCD {te.cd_unet[i]:.1f} nm", fontsize=9)
        for a in ax[:, c]: a.axis("off")
    fig.tight_layout(); fig.savefig("figures/sem_gallery.png", dpi=130)

    # ---------------- closed loop: virtual FEM measured by SEM
    res["fem_loop"] = fem_loop(model, thr, off_u, off_c)
    json.dump(res, open("models/sem_results.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("confusion",)}, indent=1, default=float))


def fem_loop(model, thr, off_u, off_c, frames=4, seed=7):
    """Expose a dose x focus matrix (Phase-1 simulator), image every field with a fast low-dose
    SEM scan, measure CD with both algorithms and call defects. Defect likelihood follows the
    process: bridging when lines print too wide, breaks when they print too narrow."""
    sim, rng = LithoSimulator(), np.random.default_rng(seed)
    layer = dict(pitch=180.0, thickness=120.0, peb_temp=110.0, peb_time=60.0)
    doses = [28.0, 29.5, 31.0, 32.5, 34.0, 35.5]; focs = np.arange(-200, 201, 25.0)
    sig = lambda z: 1 / (1 + np.exp(-z))
    rows = []
    for d in doses:
        for f in focs:
            r = sim.simulate(d, f, noise=True, rng=rng, **layer)
            row = dict(dose=d, focus=f, status=r["status"], cd_true=np.nan, defect="fail")
            if r["status"] == 0:
                cd = r["CD"]; ler = float(np.clip(r["LER"], 3, 16))
                pb, pk = 0.85 * sig((cd - 106) / 3), 0.85 * sig((68 - cd) / 3)
                u = rng.uniform()
                defect = "bridge" if u < pb else "break" if u < pb + pk else "particle" if u > 0.97 else "none"
                img, lab, meta = render(SEMParams(cd=cd, pitch=layer["pitch"], ler=ler, defect=defect, frames=frames,
                                                  beam_sigma=2.0, corr_len=22.0), rng)
                u8 = np.round(img * 255).astype(np.uint8)
                P = predict(model, u8[None])[0]
                cu, lu, _ = measure_unet(P); cc, lc, _ = measure_classical(u8 / 255.0)
                row.update(cd_true=meta["cd_true"], defect=defect, cd_unet=cu - off_u, cd_cl=cc - off_c,
                           pred=classify_unet(P, thr))
            rows.append(row)
    df = pd.DataFrame(rows); df.to_csv("data/sem_fem_loop.csv", index=False)
    ok = df[df.status == 0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fine = np.arange(-220, 221, 5.0)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(doses)))
    for c, d in zip(cols, doses):
        true = [sim.simulate(d, f, **layer) for f in fine]
        y = [t["CD"] if t["status"] == 0 else np.nan for t in true]
        ax[0].plot(fine, y, c=c, lw=1.4, label=f"{d} mJ/cm²")
        s = ok[ok.dose == d]
        ax[0].scatter(s.focus, s.cd_unet, color=c, s=26, edgecolor="k", lw=.5, zorder=3)
        ax[0].scatter(s.focus, s.cd_cl, color=c, s=26, marker="x", lw=1.2, zorder=3)
    ax[0].axhspan(81, 99, color=AMBER, alpha=.15)
    ax[0].set(xlim=(-220, 220), ylim=(30, 150), xlabel="Focus (nm)", ylabel="CD (nm)",
              title=f"Bossung curves measured from {frames}-frame SEM images\nlines: true CD   ●: U-Net   ×: classical")
    ax[0].legend(fontsize=7, ncol=2, loc="upper center")
    code = {"none": 0, "particle": 1, "bridge": 2, "break": 3}
    colors = ["#e9ecef", DEF_COL["particle"], DEF_COL["bridge"], DEF_COL["break"]]
    for _, r in df.iterrows():
        x, y = r.focus, r.dose
        if r.status != 0:
            ax[1].add_patch(plt.Rectangle((x - 12, y - .7), 24, 1.4, color="#bfc5ca")); continue
        ax[1].add_patch(plt.Rectangle((x - 12, y - .7), 24, 1.4, color=colors[code[r.pred]]))
        if r.pred != r.defect:
            ax[1].plot(x, y, "k*", ms=8)
    ax[1].set(xlim=(-215, 215), ylim=(27, 36.5), xlabel="Focus (nm)", ylabel="Dose (mJ/cm²)",
              title="Automatic defect map of the FEM wafer (U-Net calls)")
    for n, c in [("clean", colors[0]), ("particle", colors[1]), ("bridge", colors[2]), ("break", colors[3]),
                 ("pattern failed", "#bfc5ca")]:
        ax[1].add_patch(plt.Rectangle((0, 0), 0, 0, color=c, label=n))
    ax[1].plot([], [], "k*", label="call ≠ truth")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=6, frameon=False)
    fig.tight_layout(); fig.savefig("figures/sem_fem_loop.png", dpi=150)

    clean = ok[ok.defect == "none"]
    return {"fields": len(df), "imaged": len(ok),
            "cd_rmse_unet": float(np.sqrt(np.nanmean((ok.cd_unet - ok.cd_true) ** 2))),
            "cd_rmse_classical": float(np.sqrt(np.nanmean((ok.cd_cl - ok.cd_true) ** 2))),
            "cd_rmse_unet_clean": float(np.sqrt(np.nanmean((clean.cd_unet - clean.cd_true) ** 2))),
            "cd_rmse_classical_clean": float(np.sqrt(np.nanmean((clean.cd_cl - clean.cd_true) ** 2))),
            "defect_accuracy": float((ok.pred == ok.defect).mean()),
            "defect_counts_true": ok.defect.value_counts().to_dict(),
            "defect_counts_pred": ok.pred.value_counts().to_dict()}


if __name__ == "__main__":
    main()
