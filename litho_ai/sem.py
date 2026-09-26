"""
Synthetic top-down CD-SEM images of line/space resist patterns.

Geometry (continuous, rendered at 2x super-sampling):
  * vertical resist lines at the recipe's pitch, width = CD from the Phase-1 simulator
  * each edge wanders with a correlated Gaussian process: sigma = LER/3, correlation length xi
  * local CD non-uniformity between lines
  * optional defect: bridge (resist across a space), break (gap in a line, rounded ends),
    particle (foreign blob, irregular outline)

SEM signal model (secondary electrons):
  * material contrast: substrate < resist top < particle
  * edge brightening proportional to the topography gradient (the bright edge bands that
    CD-SEM edge detection relies on)
  * charging: darkening that trails each edge along the scan direction (x)
  * beam blur (Gaussian, 1.2-3 nm), slow brightness drift
  * shot noise: Poisson electron counts, scaled by the number of averaged frames
  * scan-line noise: per-row gain and sub-pixel x jitter

Pixel labels: 0 space, 1 resist line, 2 particle, 3 bridge, 4 break (missing line segment).
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

PX = 3.0            # nm per pixel
H = W = 192         # image size (576 nm field of view)
CLASSES = ["none", "bridge", "break", "particle"]
LABELS = {"space": 0, "line": 1, "particle": 2, "bridge": 3, "break": 4}


@dataclass
class SEMParams:
    cd: float                 # nm, mean line CD
    pitch: float              # nm
    ler: float                # nm, 3-sigma line-edge roughness
    defect: str = "none"      # one of CLASSES
    frames: int = 8           # frames averaged (more frames -> less shot noise)
    beam_sigma: float = 2.0   # nm
    corr_len: float = 22.0    # nm, edge-roughness correlation length
    lcdu: float = 1.0         # nm, line-to-line CD variation (1 sigma)


def _rough_edge(n, dy, sigma, xi, rng):
    """Correlated Gaussian edge displacement along y (Gaussian autocorrelation, length xi)."""
    if sigma <= 0:
        return np.zeros(n)
    white = rng.normal(size=n * 2)
    e = gaussian_filter1d(white, xi / dy / np.sqrt(2), mode="wrap")[:n]
    return (e - e.mean()) / (e.std() + 1e-12) * sigma


def render(p: SEMParams, rng=None, electrons_per_frame=10.0, noise_rng=None):
    """noise_rng: separate generator for detector noise, so one geometry can be imaged at several doses."""
    rng = rng or np.random.default_rng()
    nrng = noise_rng or rng
    S = 2                                   # super-sampling factor
    hs, ws, ps = H * S, W * S, PX / S
    y = (np.arange(hs) + 0.5) * ps
    x = (np.arange(ws) + 0.5) * ps
    X = x[None, :]

    # ---------------- lines
    x0 = rng.uniform(0, p.pitch)
    centres = np.arange(x0 - p.pitch, W * PX + p.pitch, p.pitch)
    edges = []                               # (xL(y), xR(y)) per line, super-sampled rows
    resist = np.zeros((hs, ws), bool)
    sig = p.ler / 3.0
    for c in centres:
        cd_k = p.cd + rng.normal(0, p.lcdu)
        xl = c - cd_k / 2 + _rough_edge(hs, ps, sig, p.corr_len, rng)
        xr = c + cd_k / 2 + _rough_edge(hs, ps, sig, p.corr_len, rng)
        edges.append((xl, xr))
        resist |= (X >= xl[:, None]) & (X < xr[:, None])

    full = [k for k, (xl, xr) in enumerate(edges) if xl.min() > 3 * PX and xr.max() < (W - 3) * PX]
    bridge = np.zeros_like(resist); brk = np.zeros_like(resist); particle = np.zeros_like(resist)
    Y = y[:, None]
    yc = rng.uniform(0.2, 0.8) * H * PX

    # ---------------- defects
    if p.defect == "bridge":
        inside = [k for k in range(len(edges) - 1)
                  if edges[k][1].min() > 4 * PX and edges[k + 1][0].max() < (W - 4) * PX]
        k = rng.choice(inside or [0])
        xl_gap, xr_gap = edges[k][1], edges[k + 1][0]           # space between line k and k+1
        hb = rng.uniform(4, 40)
        mid = 0.5 * (xl_gap + xr_gap)[:, None]; half = 0.5 * (xr_gap - xl_gap)[:, None]
        u = np.clip((X - mid) / np.maximum(half, 1e-3), -1, 1)
        waist = rng.uniform(0.2, 1.0)                          # thin "micro-bridge" .. full bridge
        h_local = hb / 2 * (waist + (1 - waist) * u ** 2)
        bridge = (np.abs(Y - yc) < h_local) & (X >= xl_gap[:, None]) & (X < xr_gap[:, None])
        resist |= bridge
    elif p.defect == "break":
        k = rng.choice(full or [0])
        xl, xr = edges[k]
        g = rng.uniform(3, 35)
        cx = 0.5 * (xl + xr)[:, None]; hw = 0.5 * (xr - xl)[:, None]
        v = np.clip((X - cx) / np.maximum(hw, 1e-3), -1, 1)
        gap = g / 2 + rng.uniform(2, 12) * v ** 2                  # rounded line ends
        brk = (np.abs(Y - yc) < gap) & (X >= xl[:, None]) & (X < xr[:, None])
        resist &= ~brk
    elif p.defect == "particle":
        r0 = rng.uniform(4, 26)
        pxc, pyc = rng.uniform(0.15, 0.85) * W * PX, yc
        th = np.arctan2(Y - pyc, X - pxc)
        r = r0 * (1 + 0.18 * np.sin(2 * th + rng.uniform(0, 6)) + 0.12 * np.sin(3 * th + rng.uniform(0, 6)))
        particle = np.hypot(X - pxc, Y - pyc) < r

    # ---------------- SEM signal (super-sampled, then binned)
    topo = resist.astype(float) + 1.4 * particle
    topo_s = gaussian_filter(topo, 1.5 / ps)                    # sidewall rounding ~1.5 nm
    gy, gx = np.gradient(topo_s, ps)
    edge = np.hypot(gx, gy)
    edge /= 0.25                                                # ~1 at a resist sidewall
    base = 0.26 + 0.10 * resist + 0.30 * particle
    signal = base + 0.42 * np.clip(edge, 0, 2.5)
    # charging: darkening that trails bright edges along the scan (+x)
    k = np.exp(-np.arange(0, 40) * ps / rng.uniform(10, 25)); k /= k.sum()
    trail = np.apply_along_axis(lambda r: np.convolve(r, k)[: r.size], 1, np.clip(edge, 0, 2.5))
    signal -= rng.uniform(0.05, 0.15) * trail
    signal = gaussian_filter(signal, p.beam_sigma / ps)
    sig1 = signal.reshape(H, S, W, S).mean(axis=(1, 3))
    yy, xx = np.mgrid[0:H, 0:W] / H
    sig1 *= 1 + rng.uniform(-0.06, 0.06) * (xx - 0.5) + rng.uniform(-0.06, 0.06) * (yy - 0.5)
    sig1 = np.clip(sig1, 0.01, None)

    # shot noise + scan-line noise
    n_e = electrons_per_frame * p.frames
    img = nrng.poisson(sig1 * n_e) / n_e
    img *= nrng.normal(1.0, 0.015, size=(H, 1))
    jitter = nrng.normal(0, 0.15, size=H)
    cols = np.arange(W)
    img = np.array([np.interp(cols + j, cols, row) for row, j in zip(img, jitter)])
    img = np.clip(img / 1.2, 0, 1)

    # ---------------- labels (1x)
    def bin_(m): return m.reshape(H, S, W, S).mean(axis=(1, 3)) > 0.5
    lab = np.zeros((H, W), np.uint8)
    lab[bin_(resist)] = LABELS["line"]
    lab[bin_(bridge)] = LABELS["bridge"]
    lab[bin_(brk)] = LABELS["break"]
    lab[bin_(particle)] = LABELS["particle"]

    # ---------------- ground-truth metrology on full lines (1x rows = mean of super rows)
    widths, resid = [], []
    for kk in full:
        xl = edges[kk][0].reshape(H, S).mean(1); xr = edges[kk][1].reshape(H, S).mean(1)
        widths.append(xr - xl)
        for e in (xl, xr):
            resid.append(e - np.polyval(np.polyfit(np.arange(H), e, 1), np.arange(H)))
    cd_true = float(np.mean(widths)) if widths else np.nan
    ler_true = float(3 * np.sqrt(np.mean(np.square(resid)))) if resid else np.nan
    meta = dict(cd_true=cd_true, ler_true=ler_true, n_lines=len(full), defect=p.defect,
                frames=p.frames, pitch=p.pitch)
    return img.astype(np.float32), lab, meta
