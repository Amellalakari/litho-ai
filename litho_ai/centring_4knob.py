"""4-knob process centring: dose, focus, PEB temperature, resist thickness.

In 2D (dose x focus) a coarse FEM is already efficient. Real recipe development has more
knobs, and a grid of FEMs grows multiplicatively. This study compares:
  * BO   : Gaussian process + expected improvement over all 4 knobs (3 fields per experiment)
  * Grid : the classical route - a small FEM at every PEB-temperature x thickness combination,
           per-dose Bossung fits, pick the best-looking combination
Recipe quality = true centred DOF at 5 % EL (noise-free simulator), as in process_window.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from litho_ai.bo_centring import CentringBO
from litho_ai.process_window import (sim, cd_or_fail, dof_at, TARGET, DELTA_F, FAIL_LOW, FAIL_HIGH)

PITCH, PEB_TIME = 180.0, 60.0
KNOBS = {"dose": (20.0, 50.0), "focus": (-150.0, 150.0), "peb_temp": (100.0, 125.0), "thickness": (100.0, 160.0)}
LO = np.array([v[0] for v in KNOBS.values()]); HI = np.array([v[1] for v in KNOBS.values()])


def layer(T, th):
    return dict(pitch=PITCH, peb_time=PEB_TIME, peb_temp=T, thickness=th)


def true_dof(d, f, T, th):
    L = layer(T, th)
    return dof_at(lambda dd, ff: cd_or_fail(sim.simulate(dd, ff, **L)), d, f, step=5.0)


def experiment(x, rng):
    d, f, T, th = x
    cds = np.array([cd_or_fail(sim.simulate(d, f + o, noise=True, rng=rng, **layer(T, th)))
                    for o in (-DELTA_F, 0.0, DELTA_F)])
    return float(np.sqrt(np.mean((cds - TARGET) ** 2)))


def run_bo_naive(seed, n_init=8, n_iter=32, checkpoints=(24, 45, 60, 75, 90, 120)):
    rng = np.random.default_rng(seed)
    cand = qmc.Sobol(4, seed=seed).random(4096)
    U = list(qmc.LatinHypercube(d=4, seed=seed).random(n_init))
    Y = [experiment(LO + u * (HI - LO), rng) for u in U]
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.2, 0.3, 0.4, 0.3], nu=2.5) + WhiteKernel(0.05)
    out = {}
    while True:
        y = np.log(np.array(Y) + 1.0)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                      random_state=seed).fit(np.array(U), y)
        mu, sd = gp.predict(cand, return_std=True)
        n_fields = 3 * len(U)
        if n_fields in checkpoints:
            x = LO + cand[int(np.argmin(mu))] * (HI - LO)
            out[n_fields] = (true_dof(*x), *x)
        if len(U) >= n_init + n_iter: break
        z = (y.min() - mu - 0.01) / np.maximum(sd, 1e-9)
        ei = (y.min() - mu - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
        u = cand[int(np.argmax(ei))]
        U.append(u); Y.append(experiment(LO + u * (HI - LO), rng))
    return out


def fem_fit(ds, fs, T, th, rng):
    """One FEM at fixed (T, th) -> best (dose, focus) on a per-dose Bossung fit and its fitted J."""
    fine_f = np.linspace(fs[0] - DELTA_F, fs[-1] + DELTA_F, 201)
    surf = np.full((len(ds), fine_f.size), FAIL_LOW)
    for i, d in enumerate(ds):
        cds = np.array([cd_or_fail(sim.simulate(d, f, noise=True, rng=rng, **layer(T, th))) for f in fs])
        ok = (cds > FAIL_LOW) & (cds < FAIL_HIGH)
        if ok.sum() >= 3:
            c = np.polyfit(fs[ok], cds[ok], 4 if ok.sum() >= 7 else 2)
            inside = (fine_f >= fs[ok].min()) & (fine_f <= fs[ok].max())
            surf[i, inside] = np.polyval(c, fine_f[inside])
        if (cds >= FAIL_HIGH).any():
            near = np.abs(fine_f[:, None] - fs[cds >= FAIL_HIGH][None]).min(axis=1) < np.ptp(fs) / (len(fs) - 1) / 2
            surf[i, near] = FAIL_HIGH
    fit = RegularGridInterpolator((ds, fine_f), surf, bounds_error=False, fill_value=FAIL_LOW)
    D, F = np.meshgrid(np.linspace(ds[0], ds[-1], 61), np.linspace(fs[0], fs[-1], 61), indexing="ij")
    J = np.sqrt(np.mean([(fit(np.column_stack([D.ravel(), F.ravel() + o])) - TARGET) ** 2
                         for o in (-DELTA_F, 0, DELTA_F)], axis=0))
    k = int(np.argmin(J))
    return J[k], D.ravel()[k], F.ravel()[k]


def run_grid(seed, n_T, n_th, n_d, n_f):
    rng = np.random.default_rng(20_000 + seed)
    ds, fs = np.linspace(*KNOBS["dose"], n_d), np.linspace(*KNOBS["focus"], n_f)
    best = None
    for T in np.linspace(*KNOBS["peb_temp"], n_T):
        for th in np.linspace(*KNOBS["thickness"], n_th):
            J, d, f = fem_fit(ds, fs, T, th, rng)
            if best is None or J < best[0]: best = (J, d, f, T, th)
    return n_T * n_th * n_d * n_f, (true_dof(*best[1:]), *best[1:])


def main(n_seeds=8):
    bo = []
    for s in range(n_seeds):
        opt = CentringBO(KNOBS, {}, pitch=PITCH, peb_time=PEB_TIME, seed=s, n_cand=8192)
        rec = opt.run(n_init=12, n_exp=50, checkpoints=(36, 60, 90, 120, 150))
        bo.append({n: (true_dof(*x), *x) for n, x in rec.items()})
    ck = sorted(bo[0])
    bo_dof = np.array([[r[c][0] for c in ck] for r in bo])
    naive = [run_bo_naive(s) for s in range(n_seeds)]
    ck_n = sorted(naive[0])
    naive_dof = np.array([[r[c][0] for c in ck_n] for r in naive])
    grid_cfgs = [(2, 2, 5, 7), (3, 3, 5, 7), (3, 3, 7, 9), (4, 4, 7, 9)]
    grid = [[run_grid(s, *g) for g in grid_cfgs] for s in range(n_seeds)]
    n_grid = [g[0] for g in grid[0]]
    grid_dof = np.array([[g[1][0] for g in row] for row in grid])

    # reference optimum: best recipe seen by any method, refined by a local noise-free search
    allx = [r[c] for r in bo for c in ck] + [g[1] for row in grid for g in row]
    ref = max(allx, key=lambda t: t[0])
    d0, f0, T0, th0 = ref[1:]
    for d in d0 + np.arange(-1.0, 1.01, 0.25):
        for f in f0 + np.arange(-30, 31, 10):
            v = true_dof(d, f, T0, th0)
            if v > ref[0]: ref = (v, d, f, T0, th0)

    res = {"reference_optimum": dict(zip(["dof_nm", "dose", "focus", "peb_temp", "thickness"], map(float, ref))),
           "model_based_bo_median_dof_by_fields": dict(zip(ck, np.median(bo_dof, 0).tolist())),
           "naive_bo_median_dof_by_fields": dict(zip(ck_n, np.median(naive_dof, 0).tolist())),
           "grid_median_dof_by_fields": dict(zip(n_grid, np.median(grid_dof, 0).tolist())),
           "bo_final_recipes": [dict(zip(["dof", "dose", "focus", "peb_temp", "thickness"],
                                         np.round(r[ck[-1]], 2).tolist())) for r in bo],
           "n_seeds": n_seeds}

    fig, ax = plt.subplots(figsize=(7, 4.6))
    q = lambda M, p: np.percentile(M, p, axis=0)
    ax.fill_between(ck, q(bo_dof, 25), q(bo_dof, 75), color="#2f6fb0", alpha=0.2)
    ax.plot(ck, np.median(bo_dof, 0), "o-", color="#2f6fb0", label="Model-based BO (4 knobs)")
    ax.plot(ck_n, np.median(naive_dof, 0), "--", color="#7f7f7f", lw=1.3, label="Naive BO on a scalar score")
    ax.fill_between(n_grid, q(grid_dof, 25), q(grid_dof, 75), color="#c0504d", alpha=0.2)
    ax.plot(n_grid, np.median(grid_dof, 0), "s-", color="#c0504d", label="Grid of FEMs (PEB × thickness)")
    ax.axhline(ref[0], color="k", ls=":", label=f"best found ({ref[0]:.0f} nm)")
    ax.set_xscale("log"); ax.set(xlabel="Exposure fields used (log)", ylabel="True DOF @ 5% EL (nm)",
                                 title=f"4-knob recipe centring (median, IQR over {n_seeds} runs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); fig.tight_layout()
    fig.savefig("figures/bo_vs_grid_4knob.png", dpi=150)
    json.dump(res, open("models/centring_4knob_results.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "bo_final_recipes"}, indent=2, default=float))


if __name__ == "__main__":
    main()
