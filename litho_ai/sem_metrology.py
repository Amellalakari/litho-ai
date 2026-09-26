"""CD / LER extraction from SEM images: U-Net-based and classical CD-SEM algorithm."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d, binary_dilation
from litho_ai.sem import PX, CLASSES

DEFECT_CLS = {1: "bridge", 2: "break", 3: "particle"}   # CLASSES index -> name
PIX_OF = {"particle": 2, "bridge": 3, "break": 4}       # pixel label of each defect


def _line_centres(profile, thr, margin=3):
    """Centres of above-threshold runs that lie fully inside the field."""
    above = profile > thr
    runs, i, n = [], 0, len(profile)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]: j += 1
            if i > margin and j < n - 1 - margin: runs.append((i + j) / 2)
            i = j + 1
        else:
            i += 1
    return runs


def _ler(edge_rows):
    """3-sigma LER from a list of (rows, positions) per edge, after removing a linear tilt."""
    res = []
    for r, e in edge_rows:
        if len(r) > 20:
            res.append(e - np.polyval(np.polyfit(r, e, 1), r))
    return 3 * np.sqrt(np.mean(np.concatenate(res) ** 2)) * PX if res else np.nan


# --------------------------------------------------------------------------- U-Net based
def measure_unet(prob):
    """prob: (5, H, W) softmax. Returns CD (nm), LER (nm), number of rows used."""
    r = prob[1] + prob[3]                               # resist (line or bridge)
    lab = prob.argmax(0)
    defect = binary_dilation(np.isin(lab, [2, 3, 4]), iterations=3)
    H, W = r.shape
    centres = _line_centres(r.mean(0), 0.5)
    widths, edges = [], []
    for c in centres:
        c0 = int(round(c)); rows, xl_list, xr_list = [], [], []
        for y in range(H):
            row = r[y]
            if row[c0] < 0.5: continue
            i = c0
            while i > 0 and row[i - 1] >= 0.5: i -= 1
            j = c0
            while j < W - 1 and row[j + 1] >= 0.5: j += 1
            if i == 0 or j == W - 1: continue
            if defect[y, max(i - 2, 0):min(j + 3, W)].any(): continue
            xl = i - 1 + (0.5 - row[i - 1]) / (row[i] - row[i - 1])        # sub-pixel 0.5 crossing
            xr = j + (row[j] - 0.5) / (row[j] - row[j + 1])
            rows.append(y); xl_list.append(xl); xr_list.append(xr)
        if len(rows) > 20:
            rows = np.array(rows); xl_a, xr_a = np.array(xl_list), np.array(xr_list)
            widths.append(xr_a - xl_a); edges += [(rows, xl_a), (rows, xr_a)]
    if not widths:
        return np.nan, np.nan, 0
    w = np.concatenate(widths)
    return float(np.mean(w) * PX), float(_ler(edges)), int(len(w))


def classify_unet(prob, area_thr):
    """Image-level defect call from defect-pixel areas. area_thr: {name: min pixels}."""
    lab = prob.argmax(0)
    scores = {n: (lab == PIX_OF[n]).sum() / area_thr[n] for n in PIX_OF}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "none"


def defect_boxes(prob, min_px=4):
    """Bounding boxes (x0, y0, x1, y1, name) of predicted defect regions (for display)."""
    from scipy.ndimage import label, find_objects
    lab = prob.argmax(0); out = []
    for name, v in PIX_OF.items():
        cc, n = label(lab == v)
        for k, sl in enumerate(find_objects(cc)):
            if (cc[sl] == k + 1).sum() >= min_px:
                out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, name))
    return out


# --------------------------------------------------------------------------- classical CD-SEM
def measure_classical(img, band=5):
    """Line-scan algorithm: smooth, find the bright edge peaks around each line, sub-pixel
    peak fit, median/MAD outlier rejection. No knowledge of defects."""
    im = gaussian_filter1d(img.astype(float), 1.0, axis=1)
    im = uniform_filter1d(im, band, axis=0)                   # average `band` scan lines
    H, W = im.shape
    prof = gaussian_filter1d(img.mean(0), 3)                 # resist is brighter than space
    thr = 0.5 * (np.percentile(prof, 20) + np.percentile(prof, 80))
    centres = _line_centres(prof, thr)
    if not centres:
        return np.nan, np.nan, 0
    half_pitch = (np.diff(centres).mean() / 2) if len(centres) > 1 else 40.0
    widths, edges = [], []
    for c in centres:
        c0 = int(round(c)); rows, xl, xr = [], [], []
        lo_win = max(int(c0 - half_pitch * 0.95), 1); hi_win = min(int(c0 + half_pitch * 0.95), W - 2)
        for y in range(H):
            row = im[y]
            li = lo_win + np.argmax(row[lo_win:c0]) if c0 > lo_win else None      # left edge peak
            ri = c0 + np.argmax(row[c0:hi_win + 1]) if hi_win > c0 else None       # right edge peak
            if li is None or ri is None or li <= 0 or ri >= W - 1: continue
            def sub(i):                                                            # parabolic peak
                a, b, cc = row[i - 1], row[i], row[i + 1]; d = a - 2 * b + cc
                return i + (0.5 * (a - cc) / d if d != 0 else 0.0)
            rows.append(y); xl.append(sub(li)); xr.append(sub(ri))
        if len(rows) > 20:
            rows, xl, xr = np.array(rows), np.array(xl), np.array(xr)
            wdt = xr - xl; med = np.median(wdt); mad = np.median(np.abs(wdt - med)) * 1.4826 + 1e-6
            keep = np.abs(wdt - med) < 3 * mad
            widths.append(wdt[keep]); edges += [(rows[keep], xl[keep]), (rows[keep], xr[keep])]
    if not widths:
        return np.nan, np.nan, 0
    w = np.concatenate(widths)
    return float(np.mean(w) * PX), float(_ler(edges)), int(len(w))
