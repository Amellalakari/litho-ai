"""Generate a synthetic focus-exposure / process dataset with the physics simulator.

Each row = one exposure field. We store both the noisy "measured" CD (what a fab
would actually see from CD-SEM) and the noise-free "true" CD (only available
because this is a simulation) so model error can be compared to the noise floor.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import qmc
from multiprocessing import Pool
from litho_ai.simulator import LithoSimulator

RANGES = {                       # process knob: (low, high)
    "dose":      (18.0, 55.0),   # mJ/cm^2
    "focus":     (-180.0, 180.0),  # nm
    "pitch":     (180.0, 400.0),   # nm
    "thickness": (100.0, 160.0),   # nm resist
    "peb_temp":  (100.0, 130.0),   # deg C
    "peb_time":  (45.0, 90.0),     # s
}
FEATURES = list(RANGES)

_sim = LithoSimulator()


def _run(args):
    i, row = args
    rng = np.random.default_rng(1000 + i)
    kw = dict(zip(FEATURES, row))
    meas = _sim.simulate(noise=True, rng=rng, **kw)
    true = _sim.simulate(noise=False, **kw)
    return {**kw, "CD": meas["CD"], "LER": meas["LER"], "status": meas["status"],
            "CD_true": true["CD"], "status_true": true["status"]}


def generate(n=8000, seed=0, out="data/litho_dataset.csv"):
    sampler = qmc.LatinHypercube(d=len(FEATURES), seed=seed)
    lo = np.array([RANGES[f][0] for f in FEATURES])
    hi = np.array([RANGES[f][1] for f in FEATURES])
    X = qmc.scale(sampler.random(n), lo, hi)
    with Pool() as pool:
        rows = pool.map(_run, list(enumerate(X)), chunksize=50)
    df = pd.DataFrame(rows)
    df.loc[df.status != 0, ["CD", "LER"]] = np.nan
    df.loc[df.status_true != 0, "CD_true"] = np.nan
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    df = generate(n)
    print(df.describe().round(2).T)
    print("status counts:", df.status.value_counts().to_dict())
