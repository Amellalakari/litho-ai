"""Model-based Bayesian optimisation for process-window centring.

Why not plain BO on a scalar score?  A scalar score such as "RMS CD error over 3 fields" throws
away most of what each field tells you, and its noise is about the same size as the signal
you are optimising (a 0.5 mJ/cm^2 dose error ~ 1.7 nm CD, CD-SEM + scanner noise ~ 1-2 nm).
Plain BO on that score stalls (see results).

Model-based BO instead:
  1. Gaussian-process model of CD over the knobs (every measured field is a training point),
     plus a second GP for the probability that the pattern prints at all.
  2. The process window (DOF @ 5 % EL) is computed from the GP mean - exactly what an engineer
     does with a Bossung fit, but in any number of dimensions.
  3. Next experiment = the recipe with the best optimistic (lower-confidence-bound) centring score,
     so the scanner time goes where the model is both promising and uncertain.
"""
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from litho_ai.simulator import LithoSimulator

TARGET, TOL, EL, DELTA_F = 90.0, 9.0, 0.05, 100.0
FAIL_LOW, FAIL_HIGH = 0.0, 180.0
ALL_KNOBS = ["dose", "focus", "peb_temp", "thickness"]
BOUNDS = {"dose": (18.0, 52.0), "focus": (-320.0, 320.0), "peb_temp": (98.0, 127.0), "thickness": (95.0, 165.0)}
sim = LithoSimulator()


def _unit(X):
    lo = np.array([BOUNDS[k][0] for k in ALL_KNOBS]); hi = np.array([BOUNDS[k][1] for k in ALL_KNOBS])
    return (X - lo) / (hi - lo)


class CentringBO:
    def __init__(self, search, fixed, pitch=180.0, peb_time=60.0, seed=0, kappa=3.0, n_cand=2048):
        """search: {knob: (lo, hi)} knobs being optimised; fixed: {knob: value} for the rest."""
        self.search, self.fixed, self.seed, self.kappa = search, fixed, seed, kappa
        self.layer = dict(pitch=pitch, peb_time=peb_time)
        self.rng = np.random.default_rng(seed)
        self.names = list(search)
        self.fields, self.cds = [], []           # full 4-knob field settings and measured CD
        self.cand = self._scale(qmc.Sobol(len(self.names), seed=seed).random(n_cand))

    # ---------------- recipe <-> 4-knob vector
    def _scale(self, U):
        lo = np.array([self.search[k][0] for k in self.names]); hi = np.array([self.search[k][1] for k in self.names])
        return lo + U * (hi - lo)

    def _full(self, R):
        R = np.atleast_2d(R); out = np.zeros((len(R), 4))
        for j, k in enumerate(ALL_KNOBS):
            out[:, j] = R[:, self.names.index(k)] if k in self.names else self.fixed[k]
        return out

    # ---------------- experiments
    def expose(self, recipe):
        """3 fields at focus-D, focus, focus+D (a mini FEM row), measured with noise."""
        d, f, T, th = self._full(recipe)[0]
        for o in (-DELTA_F, 0.0, DELTA_F):
            r = sim.simulate(d, f + o, thickness=th, peb_temp=T, noise=True, rng=self.rng, **self.layer)
            self.fields.append([d, f + o, T, th])
            self.cds.append({0: r["CD"], 1: FAIL_LOW, 2: FAIL_HIGH}[r["status"]])

    # ---------------- models
    def fit(self):
        X, y = _unit(np.array(self.fields)), np.array(self.cds)
        ok = (y > FAIL_LOW) & (y < FAIL_HIGH)
        ls = [0.15, 0.25, 0.5, 0.4]
        k_cd = ConstantKernel(1.0) * Matern(length_scale=ls, nu=2.5, length_scale_bounds=(0.02, 10)) \
            + WhiteKernel(0.01, (1e-4, 1.0))
        k_p = ConstantKernel(1.0) * Matern(length_scale=ls, nu=1.5, length_scale_bounds=(0.02, 10)) \
            + WhiteKernel(0.05, (1e-3, 1.0))
        self.gp_cd = GaussianProcessRegressor(k_cd, normalize_y=True, random_state=self.seed).fit(X[ok], y[ok]) \
            if ok.sum() >= 3 else None
        self.gp_p = GaussianProcessRegressor(k_p, normalize_y=False, random_state=self.seed).fit(X, ok.astype(float) - 0.5)

    def predict_cd(self, F, return_std=False):
        """F: (n, 4) field settings. Failed-print predictions are returned as FAIL_LOW (out of spec)."""
        Xu = _unit(F)
        prints = self.gp_p.predict(Xu) > 0.0
        if self.gp_cd is None:
            mu, sd = np.full(len(F), FAIL_LOW), np.full(len(F), 50.0)
        else:
            mu, sd = self.gp_cd.predict(Xu, return_std=True)
        mu = np.where(prints, mu, FAIL_LOW)
        return (mu, sd) if return_std else mu

    def centring_score(self, R, optimistic=False):
        """RMS CD error at focus-D, focus, focus+D predicted by the model (lower is better)."""
        full = self._full(R); errs, sds = [], []
        for o in (-DELTA_F, 0.0, DELTA_F):
            F = full.copy(); F[:, 1] += o
            mu, sd = self.predict_cd(F, return_std=True)
            errs.append((mu - TARGET) ** 2); sds.append(sd)
        J = np.sqrt(np.mean(errs, axis=0))
        return J - self.kappa * np.mean(sds, axis=0) if optimistic else J

    def predicted_dof(self, R, fgrid=np.arange(-300.0, 300.01, 5.0)):
        """DOF @ 5% EL around each recipe, computed on the model (same definition as the truth metric)."""
        full = self._full(R); out = []
        for row in full:
            ok = np.ones_like(fgrid, bool)
            for k in (1 - EL / 2, 1, 1 + EL / 2):
                F = np.tile(row, (fgrid.size, 1)); F[:, 0] *= k; F[:, 1] = fgrid
                ok &= np.abs(self.predict_cd(F) - TARGET) <= TOL
            i = int(np.argmin(abs(fgrid - row[1])))
            if not ok[i]: out.append(0.0); continue
            hi = i
            while hi + 1 < fgrid.size and ok[hi + 1]: hi += 1
            lo = i
            while lo - 1 >= 0 and ok[lo - 1]: lo -= 1
            out.append(2 * min(fgrid[hi] - row[1], row[1] - fgrid[lo]))
        return np.array(out)

    def recommend(self, top=40):
        J = self.centring_score(self.cand)
        idx = np.argsort(J)[:top]
        dof = self.predicted_dof(self.cand[idx])
        best = idx[np.lexsort((J[idx], -dof))[0]]
        return self.cand[best]

    # ---------------- loop
    def run(self, n_init, n_exp, checkpoints):
        U = qmc.LatinHypercube(d=len(self.names), seed=self.seed).random(n_init)
        for r in self._scale(U): self.expose(r)
        out = {}
        while True:
            self.fit()
            n_fields = len(self.fields)
            if n_fields in checkpoints:
                out[n_fields] = self._full(self.recommend())[0]
            if n_fields >= 3 * n_exp: break
            self.expose(self.cand[int(np.argmin(self.centring_score(self.cand, optimistic=True)))])
        return out
