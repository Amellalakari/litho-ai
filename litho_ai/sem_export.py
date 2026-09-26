"""Export sample images, predictions and results into the CD-SEM Inspector page."""
import base64, io, json
from litho_ai.page import full_page
import numpy as np
import pandas as pd
from PIL import Image
from litho_ai.sem import render, SEMParams
from litho_ai.sem_unet import predict
from litho_ai.sem_eval import load_model
from litho_ai.sem_metrology import measure_unet, measure_classical, classify_unet, defect_boxes

OVERLAY = {1: (91, 155, 213, 70), 2: (60, 141, 90, 200), 3: (214, 69, 65, 210), 4: (142, 68, 173, 210)}


def png(arr, mode):
    b = io.BytesIO(); Image.fromarray(arr, mode).save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def overlay(prob):
    lab = prob.argmax(0); rgba = np.zeros(lab.shape + (4,), np.uint8)
    for k, c in OVERLAY.items(): rgba[lab == k] = c
    return png(rgba, "RGBA")


def item(u8, prob, truth, thr, off):
    cu, lu, _ = measure_unet(prob); cc, lc, _ = measure_classical(u8 / 255.0)
    r = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 1)
    return {"img": png(u8, "L"), "ov": overlay(prob),
            "boxes": [[int(a), int(b), int(c), int(d), n] for a, b, c, d, n in defect_boxes(prob)],
            "truth": {k: (r(v) if isinstance(v, (float, np.floating)) else v) for k, v in truth.items()},
            "unet": {"cd": r(cu - off["unet"]), "ler": r(lu), "defect": classify_unet(prob, thr)},
            "classical": {"cd": r(cc - off["classical"]), "ler": r(lc)}}


def build():
    res = json.load(open("models/sem_results.json"))
    thr, off = res["area_thresholds_px"], res["cd_offset_calibration_nm"]
    model = load_model()
    d = np.load("data/sem_test.npz"); te = pd.read_csv("data/sem_test_results.csv")
    picks = []
    for dname in ["none", "bridge", "break", "particle"]:
        for f in (2, 8, 32):
            s = te[(te.defect == dname) & (te.frames == f) & te.cd_true.between(70, 110)]
            if len(s): picks.append(int(s.index[0]))
    P = predict(model, d["imgs"][picks])
    gallery = [item(d["imgs"][i], P[k], {"cd": te.cd_true[i], "ler": te.ler_true[i], "defect": te.defect[i],
                                         "frames": int(te.frames[i]), "pitch": round(float(te.pitch[i]))}, thr, off)
               for k, i in enumerate(picks)]

    explorer = []
    for f in [1, 2, 4, 8, 16, 32]:
        img, lab, meta = render(SEMParams(cd=88.0, pitch=180.0, ler=7.5, defect="bridge", frames=f, beam_sigma=2.0,
                                          corr_len=22.0), np.random.default_rng(42), noise_rng=np.random.default_rng(100 + f))
        u8 = np.round(img * 255).astype(np.uint8); prob = predict(model, u8[None])[0]
        explorer.append(item(u8, prob, {"cd": meta["cd_true"], "ler": meta["ler_true"], "defect": "bridge",
                                        "frames": f, "pitch": 180}, thr, off))

    fem = base64.b64encode(open("figures/sem_fem_loop.png", "rb").read()).decode()
    data = {"gallery": gallery, "explorer": explorer, "results": res, "fem_png": "data:image/png;base64," + fem}
    html = open("dashboard/sem_inspector_template.html").read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    open("docs/cd-sem-inspector.html", "w").write(full_page(html))
    print(f"docs/cd-sem-inspector.html {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    build()
