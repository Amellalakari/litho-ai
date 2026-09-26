"""Build the SEM image dataset. CD, pitch and LER come from Phase-1 simulator fields."""
import numpy as np, pandas as pd
from litho_ai.sem import render, SEMParams, CLASSES

FRAMES = [1, 2, 4, 8, 16, 32]


def recipes(seed=0):
    df = pd.read_csv("data/litho_dataset.csv")
    df = df[(df.status == 0) & (df.pitch <= 280) & df.CD_true.between(18, 155) & df.LER.between(3, 16)]
    df = df[df.CD_true < df.pitch - 20]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def build(n, seed, frames=None, out=None, class_p=(0.4, 0.2, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    rec = recipes(seed)
    imgs = np.zeros((n, 192, 192), np.uint8); labs = np.zeros((n, 192, 192), np.uint8); metas = []
    for i in range(n):
        r = rec.iloc[i % len(rec)]
        p = SEMParams(cd=float(r.CD_true), pitch=float(r.pitch), ler=float(r.LER),
                      defect=CLASSES[rng.choice(4, p=class_p)],
                      frames=int(frames[i] if frames is not None else rng.choice(FRAMES)),
                      beam_sigma=rng.uniform(1.2, 3.0), corr_len=rng.uniform(12, 35))
        img, lab, meta = render(p, rng)
        imgs[i] = np.round(img * 255); labs[i] = lab
        metas.append({**meta, "cd_nominal": p.cd, "ler_nominal": p.ler, "beam_sigma": p.beam_sigma,
                      "dose": r.dose, "focus": r.focus})
    meta = pd.DataFrame(metas)
    if out:
        np.savez_compressed(out, imgs=imgs, labs=labs); meta.to_csv(out.replace(".npz", "_meta.csv"), index=False)
    return imgs, labs, meta


if __name__ == "__main__":
    build(2400, 1, out="data/sem_train.npz")
    build(300, 2, out="data/sem_val.npz")
    build(900, 3, frames=np.repeat(FRAMES, 150)[np.random.default_rng(3).permutation(900)], out="data/sem_test.npz")
    m = pd.read_csv("data/sem_test_meta.csv"); print(m.defect.value_counts(), m.frames.value_counts().sort_index())
    print(m[["cd_true", "ler_true", "pitch", "n_lines"]].describe().round(2))
