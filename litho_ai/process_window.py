"""Process-window analysis and Bayesian-optimised process centring.

Layer: 90 nm lines, 180 nm pitch (dense, k1 ~ 0.43 on dry ArF NA 0.93), spec CD = 90 nm +/- 10 %.
Fixed: 120 nm resist, PEB 110 C / 60 s. Knobs being centred: dose and focus.

1. Ground truth   noise-free FEM on a fine grid -> Bossung curves, ED window, EL-vs-DOF
2. Surrogate      the MLP reproduces the same window at a fraction of the cost
3. Centring       "experiment" = 3 exposure fields at (dose, focus-D), (dose, focus), (dose, focus+D),
                  measured with noise. Objective J = RMS CD error over the 3 fields (robust centring).
                  BO (Gaussian process + expected improvement) vs the classical route:
                  a full focus-exposure matrix (FEM) + polynomial Bossung fit.
                  Recipe quality = true DOF at 5 % exposure latitude around the recommended recipe.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from litho_ai.simulator import LithoSimulator
from litho_ai.bo_centring import CentringBO

TARGET, TOL = 90.0, 9.0
LAYER = dict(pitch=180.0, thickness=120.0, peb_temp=110.0, peb_time=60.0)
DOSE_RANGE, FOCUS_RANGE = (22.0, 44.0), (-200.0, 200.0)
DELTA_F = 100.0           # focus offset of the side fields in a robust experiment
EL = 0.05                 # exposure latitude used to quote DOF
FAIL_LOW, FAIL_HIGH = 0.0, 180.0   # CD assigned to line loss / bridging
sim = LithoSimulator()


# ------------------------------------------------------------------ helpers
def cd_or_fail(r):
    return {0: r["CD"], 1: FAIL_LOW, 2: FAIL_HIGH}[r["status"]]


def true_grid():
    doses = np.arange(18.0, 50.01, 0.25)
    focs = np.arange(-300.0, 300.01, 5.0)
    G = np.array([[cd_or_fail(sim.simulate(d, f, **LAYER)) for f in focs] for d in doses])
    return doses, focs, G


def dof_at(cd_fn, dose, focus, el=EL, step=2.0, fmax=300):
    """Usable DOF of a recipe: 2 x the focus margin to the nearest window edge, where the window
    is the contiguous focus range in spec at dose*(1 +/- el/2). Penalises off-centre recipes."""
    def ok(f):
        return all(abs(cd_fn(dose * k, f) - TARGET) <= TOL for k in (1 - el / 2, 1, 1 + el / 2))
    if not ok(focus):
        return 0.0
    lo = hi = focus
    while hi + step <= fmax and ok(hi + step): hi += step
    while lo - step >= -fmax and ok(lo - step): lo -= step
    return 2 * min(hi - focus, focus - lo)


def ed_curve(cd_map, doses, focs):
    """Max DOF achievable for each exposure latitude (rectangle method)."""
    inspec = np.abs(cd_map - TARGET) <= TOL
    els, dofs = np.linspace(0, 0.25, 51), []
    for el in els:
        best = 0.0
        for i, d in enumerate(doses):
            j = np.searchsorted(doses, d * (1 + el) - 1e-9)
            if j >= len(doses): break
            col_ok = inspec[i:j + 1].all(axis=0)            # every dose in band in-spec
            run = best_run = 0
            for v in col_ok:
                run = run + 1 if v else 0; best_run = max(best_run, run)
            best = max(best, (best_run - 1) * (focs[1] - focs[0]) if best_run else 0)
        dofs.append(best)
    return els, np.array(dofs)


def mlp_predict(path, X):
    m = json.load(open(path))
    h = (X - np.array(m["x_mean"])) / np.array(m["x_std"])
    for k, (W, b) in enumerate(zip(m["weights"], m["biases"])):
        h = h @ np.array(W) + np.array(b)
        if k < len(m["weights"]) - 1: h = np.tanh(h)
    if m["kind"] == "classifier": return 1 / (1 + np.exp(-h[:, 0]))
    return h[:, 0] * m["y_std"] + m["y_mean"]


def surrogate_grid(doses, focs):
    D, F = np.meshgrid(doses, focs, indexing="ij")
    n = D.size
    X = np.column_stack([D.ravel(), F.ravel(), np.full(n, LAYER["pitch"]), np.full(n, LAYER["thickness"]),
                         np.full(n, LAYER["peb_temp"]), np.full(n, LAYER["peb_time"])])
    cd = mlp_predict("models/mlp_cd.json", X)
    p = mlp_predict("models/mlp_print.json", X)
    cd = np.where(p < 0.5, FAIL_LOW, cd)
    return cd.reshape(D.shape)


# ------------------------------------------------------------------ experiments
def experiment(dose, focus, rng):
    """One robust-centring experiment = 3 noisy fields. Returns J and the 3 CDs."""
    cds = [cd_or_fail(sim.simulate(dose, focus + o, noise=True, rng=rng, **LAYER))
           for o in (-DELTA_F, 0.0, DELTA_F)]
    return float(np.sqrt(np.mean((np.array(cds) - TARGET) ** 2))), cds


def J_of(cd_fn, d, f):
    return np.sqrt(np.mean([(cd_fn(d, f + o) - TARGET) ** 2 for o in (-DELTA_F, 0, DELTA_F)]))


def to_unit(d, f):
    return np.column_stack([(np.asarray(d) - DOSE_RANGE[0]) / np.ptp(DOSE_RANGE),
                            (np.asarray(f) - FOCUS_RANGE[0]) / np.ptp(FOCUS_RANGE)])


CAND_D, CAND_F = np.meshgrid(np.linspace(*DOSE_RANGE, 89), np.linspace(*FOCUS_RANGE, 81), indexing="ij")
CAND = to_unit(CAND_D.ravel(), CAND_F.ravel())


def run_bo_naive(n_init=5, n_iter=20, seed=0):
    rng = np.random.default_rng(seed)
    U = qmc.LatinHypercube(d=2, seed=seed).random(n_init)
    X = [(DOSE_RANGE[0] + u[0] * np.ptp(DOSE_RANGE), FOCUS_RANGE[0] + u[1] * np.ptp(FOCUS_RANGE)) for u in U]
    Y = [experiment(d, f, rng)[0] for d, f in X]
    history = []
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.2, 0.3], nu=2.5) + WhiteKernel(0.05)
    for it in range(n_iter + 1):
        y = np.log(np.array(Y) + 1.0)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=seed).fit(to_unit(*np.array(X).T), y)
        mu, sd = gp.predict(CAND, return_std=True)
        k = int(np.argmin(mu))                           # current recommendation
        history.append((3 * len(X), CAND_D.ravel()[k], CAND_F.ravel()[k]))
        if it == n_iter: break
        best = y.min()
        z = (best - mu - 0.01) / np.maximum(sd, 1e-9)
        ei = (best - mu - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
        j = int(np.argmax(ei))
        d, f = CAND_D.ravel()[j], CAND_F.ravel()[j]
        X.append((d, f)); Y.append(experiment(d, f, rng)[0])
    return history, np.array(X)


def run_fem(n_dose, n_focus, seed=0):
    """Classical route: full FEM, per-dose Bossung fits (polynomial in focus), linear
    interpolation across dose, then centre on the fitted CD surface with the same objective."""
    rng = np.random.default_rng(10_000 + seed)
    ds, fs = np.linspace(*DOSE_RANGE, n_dose), np.linspace(*FOCUS_RANGE, n_focus)
    fine_f = np.linspace(FOCUS_RANGE[0] - DELTA_F, FOCUS_RANGE[1] + DELTA_F, 241)
    surf = np.full((n_dose, fine_f.size), FAIL_LOW)
    for i, d in enumerate(ds):
        cds = np.array([cd_or_fail(sim.simulate(d, f, noise=True, rng=rng, **LAYER)) for f in fs])
        ok = (cds > FAIL_LOW) & (cds < FAIL_HIGH)
        if ok.sum() >= 3:
            deg = 4 if ok.sum() >= 7 else 2
            c = np.polyfit(fs[ok], cds[ok], deg)
            inside = (fine_f >= fs[ok].min()) & (fine_f <= fs[ok].max())
            surf[i, inside] = np.polyval(c, fine_f[inside])
        bridged = cds >= FAIL_HIGH
        if bridged.any():                       # regions that bridged stay "fail high"
            near = np.abs(fine_f[:, None] - fs[bridged][None]).min(axis=1) < np.ptp(fs) / (n_focus - 1) / 2
            surf[i, near] = FAIL_HIGH
    fit = RegularGridInterpolator((ds, fine_f), surf, bounds_error=False, fill_value=FAIL_LOW)
    cd_fit = lambda d, f: fit(np.column_stack([d, f]))
    Jc = np.sqrt(np.mean([(cd_fit(CAND_D.ravel(), CAND_F.ravel() + o) - TARGET) ** 2
                          for o in (-DELTA_F, 0, DELTA_F)], axis=0))
    k = int(np.argmin(Jc))
    return n_dose * n_focus, CAND_D.ravel()[k], CAND_F.ravel()[k]


# ------------------------------------------------------------------ main
def main(n_seeds=12):
    out = {}
    doses, focs, G = true_grid()
    interp = RegularGridInterpolator((doses, focs), G, bounds_error=False, fill_value=FAIL_LOW)
    cd_true = lambda d, f: float(interp([[d, f]])[0])

    # optimum recipe by exhaustive noise-free search (reference only; not available in a fab)
    best = max(((dof_at(cd_true, d, f), d, f) for d in np.arange(24, 42, 0.25)
                for f in np.arange(-150, 151, 5)), key=lambda t: t[0])
    out["optimum"] = {"dof_nm": best[0], "dose": best[1], "focus": best[2]}

    # --- process window: truth vs surrogate
    S = surrogate_grid(doses, focs)
    els, dof_t = ed_curve(G, doses, focs)
    _, dof_s = ed_curve(S, doses, focs)
    i5 = int(np.argmin(abs(els - EL)))
    out["window"] = {"true_DOF_at_5pctEL_nm": float(dof_t[i5]), "mlp_DOF_at_5pctEL_nm": float(dof_s[i5]),
                     "true_max_EL": float(els[dof_t > 0].max()), "mlp_max_EL": float(els[dof_s > 0].max())}

    # --- model-based BO (main method), naive scalar-score BO, and FEM
    mb, bo_pts = [], None
    for s in range(n_seeds):
        opt = CentringBO({"dose": DOSE_RANGE, "focus": FOCUS_RANGE},
                         {"peb_temp": LAYER["peb_temp"], "thickness": LAYER["thickness"]}, seed=s)
        res_s = opt.run(n_init=4, n_exp=25, checkpoints=range(12, 76, 9))
        mb.append([(n, dof_at(cd_true, r[0], r[1]), r[0], r[1]) for n, r in res_s.items()])
        if s == 0: bo_pts = np.array(opt.fields)[1::3][:, :2]   # centre field of each experiment
    mb = np.array(mb)
    bo_curves = []
    for s in range(n_seeds):
        hist, _ = run_bo_naive(seed=s)
        bo_curves.append([(n, dof_at(cd_true, d, f), d, f) for n, d, f in hist])
    bo = np.array(bo_curves)                              # seeds x iters x 4
    fem_sizes = [(3, 3), (4, 5), (5, 7), (7, 9), (9, 11), (11, 13)]
    fem = np.array([[(lambda n, d, f: (n, dof_at(cd_true, d, f), d, f))(*run_fem(a, b, s))
                     for (a, b) in fem_sizes] for s in range(n_seeds)])
    frac = 0.9 * best[0]
    bo_med = np.median(bo[:, :, 1], axis=0); fem_med = np.median(fem[:, :, 1], axis=0)
    mb_med = np.median(mb[:, :, 1], axis=0)
    n_bo = bo[0, :, 0]; n_fem = fem[0, :, 0]; n_mb = mb[0, :, 0]
    first = lambda n, m: float(n[np.argmax(m >= frac)]) if (m >= frac).any() else None
    out["centring"] = {
        "fields_to_reach_90pct_opt_DOF": {"model_based_BO": first(n_mb, mb_med),
                                          "naive_BO": first(n_bo, bo_med), "full_FEM": first(n_fem, fem_med)},
        "model_based_BO_median_DOF_by_fields": dict(zip(map(int, n_mb), mb_med.round(1).tolist())),
        "naive_BO_median_DOF_by_fields": dict(zip(map(int, n_bo), bo_med.round(1).tolist())),
        "fem_median_DOF_by_fields": dict(zip(map(int, n_fem), fem_med.round(1).tolist())),
        "model_based_BO_final_recipe_median": {"dose": float(np.median(mb[:, -1, 2])), "focus": float(np.median(mb[:, -1, 3]))},
        "n_seeds": n_seeds}

    # --- figures -----------------------------------------------------------
    col = ["#1f4e79", "#2f6fb0", "#5b9bd5", "#e8a33d", "#c0504d", "#7a4fb5", "#3c8d5a"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for c, d in zip(col, [27, 29, 31, 33, 35, 37, 39]):
        i = int(np.argmin(abs(doses - d)))
        y = np.where(G[i] > FAIL_LOW, G[i], np.nan)
        ax.plot(focs, y, color=c, lw=1.8, label=f"{d} mJ/cm²")
    ax.axhspan(TARGET - TOL, TARGET + TOL, color="#e8a33d", alpha=0.15, label="spec ±10%")
    ax.set(xlim=(-250, 250), ylim=(30, 150), xlabel="Focus (nm)", ylabel="CD (nm)",
           title="Bossung curves — 90 nm L/S, 180 nm pitch")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig("figures/bossung.png", dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, M, t in [(ax[0], G, "Physics simulator (truth)"), (ax[1], S, "MLP surrogate")]:
        Mm = np.ma.masked_where((M <= FAIL_LOW) | (M >= FAIL_HIGH), M)   # failures shown grey
        a.set_facecolor("#d9d9d9")
        cs = a.contourf(focs, doses, Mm, levels=np.arange(20, 181, 10), cmap="viridis", extend="both")
        a.contour(focs, doses, Mm, levels=[TARGET - TOL, TARGET + TOL], colors="w", linewidths=1.6)
        a.set(xlim=(-250, 250), ylim=DOSE_RANGE, xlabel="Focus (nm)", title=t)
        a.text(0.98, 0.02, "grey = pattern fails\n(line lost / bridged)", transform=a.transAxes,
               ha="right", va="bottom", fontsize=8, color="#333")
    ax[0].set_ylabel("Dose (mJ/cm²)")
    ax[0].scatter(bo_pts[:, 1], bo_pts[:, 0], c=np.arange(len(bo_pts)), cmap="autumn", s=28,
                  edgecolor="k", lw=0.5, label="BO experiment centres (seed 0)")
    ax[0].plot(best[2], best[1], "w*", ms=16, mec="k", label="true optimum")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.colorbar(cs, ax=ax, label="CD (nm)"); fig.savefig("figures/process_window_map.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(dof_t, els * 100, color="#1f4e79", lw=2, label="Physics simulator")
    ax.plot(dof_s, els * 100, "--", color="#e8a33d", lw=2, label="MLP surrogate")
    ax.set(xlabel="Depth of focus (nm)", ylabel="Exposure latitude (%)", title="Exposure–defocus window")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("figures/ed_window.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    q = lambda M, p: np.percentile(M, p, axis=0)
    ax.fill_between(n_mb, q(mb[:, :, 1], 25), q(mb[:, :, 1], 75), color="#2f6fb0", alpha=0.2)
    ax.plot(n_mb, mb_med, "o-", color="#2f6fb0", label="Model-based BO (GP of CD)")
    ax.plot(n_bo, bo_med, "--", color="#7f7f7f", lw=1.3, label="Naive BO on a scalar score")
    ax.fill_between(n_fem, q(fem[:, :, 1], 25), q(fem[:, :, 1], 75), color="#c0504d", alpha=0.2)
    ax.plot(n_fem, fem_med, "s-", color="#c0504d", label="Full FEM + Bossung fit")
    ax.axhline(best[0], color="k", ls=":", label=f"true optimum ({best[0]:.0f} nm)")
    ax.set(xlabel="Exposure fields used", ylabel="True DOF @ 5% EL of chosen recipe (nm)",
           title=f"Process centring efficiency (median, IQR over {n_seeds} runs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("figures/bo_vs_fem.png", dpi=150)

    json.dump(out, open("models/process_window_results.json", "w"), indent=2, default=float)
    np.savez("data/true_fem_grid.npz", doses=doses, focs=focs, G=G)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
