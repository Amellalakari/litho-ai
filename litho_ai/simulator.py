"""
Physics-based 1D lithography simulator for line/space patterns.

Chain modelled (all lengths in nm, dose in mJ/cm^2, temperature in deg C):
  1. Mask            binary chrome line of fixed width on a periodic grating (pitch p)
  2. Illumination    conventional disk source, partial coherence sigma (Abbe source summation)
  3. Projection      pupil cut-off NA/lambda, defocus phase, small spherical aberration (Z9)
  4. Resist film     thickness-averaged aerial image (focus varies through the film),
                     swing-curve modulation of the coupled dose (thin-film interference)
  5. Exposure        photo-acid generation  a = 1 - exp(-C * E * I)
  6. PEB             acid diffusion (Gaussian, length L = sqrt(2 D(T) t), Arrhenius D),
                     base-quencher neutralisation, catalytic deprotection (Arrhenius rate)
  7. Development     threshold on deprotection -> resist line CD
  8. Metrology       CD, LER proxy (from edge gradient + shot noise), failure modes
  9. Noise           field-to-field dose/focus jitter + CD-SEM metrology noise

This is a teaching / portfolio-grade model: every step is standard textbook physics
(Mack, "Fundamental Principles of Optical Lithography"), with simplified 1D geometry.
It is NOT a calibrated production model.
"""
from dataclasses import dataclass, field
import numpy as np

K_B = 8.617e-5  # Boltzmann constant, eV/K


@dataclass
class ToolConfig:
    wavelength: float = 193.0     # ArF, nm
    NA: float = 0.93              # dry ArF scanner (XT:1400-class)
    sigma: float = 0.75           # partial coherence (disk)
    z9_waves: float = 0.02        # spherical aberration, waves (shifts best focus vs pitch)
    n_source: int = 15            # source grid points across the diameter
    n_x: int = 400                # samples per pitch
    n_depth: int = 5              # focal planes averaged through the resist


@dataclass
class ResistConfig:
    n_resist: float = 1.70        # refractive index at 193 nm
    swing_amp: float = 0.12       # swing-curve amplitude on coupled dose
    swing_phase: float = 0.6      # rad
    C_dill: float = 0.030         # cm^2/mJ, acid generation efficiency
    L_ref: float = 14.0           # nm diffusion length at 110 C, 60 s
    Ea_diff: float = 0.45         # eV, activation energy of acid diffusion
    k_ref: float = 0.009          # 1/s deprotection rate at 110 C
    Ea_rxn: float = 0.60          # eV, activation energy of deprotection
    quencher: float = 0.12        # base quencher loading (normalised acid units)
    threshold: float = 0.50       # deprotection level that develops away
    ler_k: float = 0.090          # LER proxy scale
    T_ref: float = 110.0
    t_ref: float = 60.0


@dataclass
class NoiseConfig:
    dose_jitter: float = 0.005    # relative, 1 sigma
    focus_jitter: float = 6.0     # nm, 1 sigma
    cd_metrology: float = 0.7     # nm, 1 sigma (CD-SEM repeatability)
    ler_metrology: float = 0.25   # nm, 1 sigma


MASK_LINE = 90.0  # nm (1x) chrome line width, target wafer CD = 90 nm


class LithoSimulator:
    def __init__(self, tool=None, resist=None, noise=None, mask_line=MASK_LINE):
        self.tool = tool or ToolConfig()
        self.resist = resist or ResistConfig()
        self.noise = noise or NoiseConfig()
        self.mask_line = mask_line
        self._build_source()

    # ------------------------------------------------------------------ optics
    def _build_source(self):
        t = self.tool
        g = np.linspace(-1, 1, t.n_source)
        sx, sy = np.meshgrid(g, g)
        inside = sx**2 + sy**2 <= 1.0 + 1e-9
        fc = t.NA / t.wavelength
        self._sx = sx[inside] * t.sigma * fc   # source points as spatial-frequency tilts
        self._sy = sy[inside] * t.sigma * fc
        self._w = np.full(self._sx.size, 1.0 / self._sx.size)

    def aerial_image(self, pitch, focus, thickness):
        """Normalised intensity I(x) over one pitch (clear field = 1), averaged through film."""
        t, r = self.tool, self.resist
        lam, fc = t.wavelength, t.NA / t.wavelength
        x = (np.arange(t.n_x) / t.n_x - 0.5) * pitch
        n_max = int(np.ceil((1 + t.sigma) * fc * pitch)) + 1
        n = np.arange(-n_max, n_max + 1)
        duty = self.mask_line / pitch
        c = -duty * np.sinc(n * duty)
        c[n == 0] += 1.0                                   # t(x) = 1 - rect(line)

        fx = n[None, :] / pitch + self._sx[:, None]        # (S, N)
        fy = np.broadcast_to(self._sy[:, None], fx.shape)
        rho2 = (fx**2 + fy**2) / fc**2
        passed = rho2 <= 1.0
        # defocus + spherical phase (paraxial-exact defocus term)
        kz = np.sqrt(np.clip(1 - (lam**2) * (fx**2 + fy**2), 0, None))
        z9 = 2 * np.pi * t.z9_waves * (6 * rho2**2 - 6 * rho2 + 1)

        depths = np.linspace(-0.5, 0.5, t.n_depth) * thickness / r.n_resist
        phase_x = np.exp(2j * np.pi * np.outer(n / pitch, x))     # (N, X)
        # tilt term exp(i2pi sx x) is a common phase per source point -> drops out of |E|^2
        I = np.zeros_like(x)
        for d in depths:
            z = focus + d
            pupil = passed * np.exp(1j * (2 * np.pi * z / lam * (kz - 1) + z9))
            E = (c[None, :] * pupil) @ phase_x                     # (S, X)
            I += self._w @ (np.abs(E) ** 2)
        return x, I / t.n_depth

    # ------------------------------------------------------------------ resist
    def swing_factor(self, thickness):
        r = self.resist
        return 1.0 - r.swing_amp * np.cos(
            4 * np.pi * r.n_resist * thickness / self.tool.wavelength + r.swing_phase)

    def diffusion_length(self, peb_temp, peb_time):
        r = self.resist
        T, Tr = peb_temp + 273.15, r.T_ref + 273.15
        D = (r.L_ref**2 / (2 * r.t_ref)) * np.exp(r.Ea_diff / K_B * (1 / Tr - 1 / T))
        return np.sqrt(2 * D * peb_time)

    def deprotection(self, x, I, dose, thickness, peb_temp, peb_time):
        r = self.resist
        E = dose * self.swing_factor(thickness)
        acid = 1 - np.exp(-r.C_dill * E * I)
        L = self.diffusion_length(peb_temp, peb_time)
        pitch = (x[1] - x[0]) * x.size
        f = np.fft.fftfreq(x.size, d=x[1] - x[0])
        acid = np.real(np.fft.ifft(np.fft.fft(acid) * np.exp(-2 * (np.pi * f * L) ** 2)))
        acid = np.clip(acid - r.quencher, 0, None)
        T, Tr = peb_temp + 273.15, r.T_ref + 273.15
        k = r.k_ref * np.exp(r.Ea_rxn / K_B * (1 / Tr - 1 / T))
        return 1 - np.exp(-k * peb_time * acid / 0.1)   # acid normalised to ~0.1 scale

    # ------------------------------------------------------------------ metrology
    def measure(self, x, h, dose):
        """Return (CD, LER, status). status: 0 ok, 1 line lost, 2 space bridged."""
        th = self.resist.threshold
        centre = x.size // 2
        if h[centre] >= th:
            return 0.0, np.nan, 1
        if h.max() < th:
            return float(x[-1] - x[0]), np.nan, 2
        right = h[centre:]
        i = np.argmax(right >= th)
        x0, x1 = x[centre + i - 1], x[centre + i]
        h0, h1 = right[i - 1], right[i]
        x_edge = x0 + (th - h0) * (x1 - x0) / (h1 - h0)
        slope = (h1 - h0) / (x1 - x0)
        cd = 2 * x_edge
        ler = self.resist.ler_k / slope / np.sqrt(dose / 30.0)
        return float(cd), float(ler), 0

    # ------------------------------------------------------------------ public API
    def simulate(self, dose, focus, pitch=180.0, thickness=120.0,
                 peb_temp=110.0, peb_time=60.0, noise=False, rng=None):
        """One exposure field. Returns dict with CD (nm), LER (nm, 3-sigma proxy), status."""
        if noise:
            rng = rng or np.random.default_rng()
            dose = dose * (1 + rng.normal(0, self.noise.dose_jitter))
            focus = focus + rng.normal(0, self.noise.focus_jitter)
        x, I = self.aerial_image(pitch, focus, thickness)
        h = self.deprotection(x, I, dose, thickness, peb_temp, peb_time)
        cd, ler, status = self.measure(x, h, dose)
        if noise and status == 0:
            cd += rng.normal(0, self.noise.cd_metrology)
            ler += rng.normal(0, self.noise.ler_metrology)
        return {"CD": cd, "LER": ler, "status": status}

    def profile(self, dose, focus, pitch=180.0, thickness=120.0, peb_temp=110.0, peb_time=60.0):
        """Intermediate fields for plotting."""
        x, I = self.aerial_image(pitch, focus, thickness)
        h = self.deprotection(x, I, dose, thickness, peb_temp, peb_time)
        return x, I, h
