"""
desm_v2.py
==========
DESM V2 — Dynamic Economic State Model, Version 2.

Computational implementation of the mathematical structure derived in:
  M04 (Construction)  — state-space equations T1–T3, parameter law P1
  M05 (Emergent)      — five-regime partition, IRF, regime transitions
  M06 (Identification) — estimators for θ = (φ₁,σ_ε²,α₊,α₋,ρ_I,σ_ν²,μ₀,β_I,γ,σ_ζ²)

State vector:  s_{it} = (x1_{it},  g_{it},  I_{it})
  x1_{it} = log(real GDP per capita)               [A1]
  g_{it}  = Δx1_{it} − μ_i,  demeaned growth       [A2, A3]
  I_{it}  = gross fixed capital formation (% GDP)   [A4]

Transition laws (M04, Theorem 4.2):
  T1:  x1_{i,t+1} = x1_{it} + μ_i + g_{i,t+1}
  T2:  g_{i,t+1}  = φ₁ g_{it} + ε_{i,t+1}
  T3:  I_{i,t+1}  = ρ_I I_{it} + (1−ρ_I) Ī_i + ν_{it}

V1 bugs corrected:
  - kappa_0 attribute was never initialised in DESMPOMDPEnv, causing NaN at
    episode ~1600 of RL training (WP-6). V2 eliminates the RL environment
    entirely: the model is a pure state-space system, not a POMDP.
  - GMM/Arellano–Bond design mismatch (F < 10, AR(2) p=0.0056) corrected by
    the two-tier direct OLS strategy of M06.

Known scope limitation (disclosed in the manuscript, Remark rem:impl_status,
§Computational Implementation): this module implements the first-generation
single-lag AR(1) specification (estimate_phi1(), DESMTransition, psi_inf =
1/(1-φ₁)), including single-lag defaults in DESMParameters (e.g. φ₁=0.263).
The joint AR(2) Yule-Walker calibration actually reported throughout the
manuscript (φ₁=0.194, φ₂=0.062, ψ∞≈1.344) is a two-lag extension of the same
estimator, exercised in the companion scripts rc_irf_reconciliation.py,
rc_lp_finite_sample_mc.py and rc_pooled_heterogeneity_mc.py, not yet
back-ported here as first-class DESMParameters/DESMTransition fields. No
number in the manuscript is computed from this module's single-lag legacy
defaults; treat phi1=0.263 and psi_inf below as the single-lag predecessor
quantities, not the calibration used for the paper's reported results.

Module layout:
  1.  DESMParameters      — calibrated parameter vector θ with derived quantities
  2.  DESMTransition      — forward step (T1–T3) and IRF
  3.  build_panel()       — panel construction from raw WB data
  4.  estimate_phi1()     — Group I: φ₁ (pooled within-country OLS)
  5.  estimate_sigma_eps()— Group I: σ_ε
  6.  hill_estimator()    — Group I: tail indices α₊, α₋ (Hill 1975)
  7.  estimate_tail_indices() — Group I: calls hill_estimator on residuals
  8.  estimate_rhoI()     — Group II: ρ_I
  9.  estimate_sigma_nu() — Group II: σ_ν
  10. estimate_betaI_ols()— Group III: β_I^OLS, μ₀  (upper bound on β_I)
  11. estimate_wgi_panel()— Group IV: β_I, γ, σ_ζ (WGI-augmented panel)
  12. wild_bootstrap()    — within-country bootstrap for φ₁, ρ_I
  13. pairs_bootstrap()   — cross-sectional bootstrap for β_I, γ
  14. RegimeClassifier    — five-regime partition R1–R5 (M05, Theorem 5.2)
  15. DESMSimulator       — forward Monte Carlo simulation
  16. DESMDiagnostics     — derived quantities and model diagnostics
  17. run_full_estimation()— complete two-panel pipeline
  18. main()              — entry point: loads data, runs pipeline, prints report
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETER VECTOR θ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DESMParameters:
    """
    Calibrated free parameter vector θ of DESM V2.

    Values are taken from M06, Table 6.1 (primary panel calibration).
    Theoretical constraints from M04–M05:
      φ₁ ∈ (0,1),  ρ_I ∈ (0,1),  β_I > 0,  γ > 0,  α₊ ∈ (2,3),  α₋ ∈ (2,4).

    Group I  (Θ_I)   — transition dynamics, identified from within-country AR(1)
    Group II (Θ_II)  — investment persistence, identified from within-country AR(1)
    Group III(Θ_III) — drift slope, OLS upper bound on true β_I
    Group IV (Θ_IV)  — institutional coupling, requires WGI-augmented panel
    """

    # --- Group I ---
    phi1: float = 0.263           # AR(1) persistence of demeaned growth  [FC-9]
    sigma_eps: float = 0.0530     # Innovation s.d. (log-units / year)    [FC-3]
    alpha_plus: float = 2.2       # Right tail index of ε_{it}            [FC-4]
    alpha_minus: float = 2.9      # Left tail index  of ε_{it}            [FC-4]

    # --- Group II ---
    rho_I: float = 0.83           # Investment AR(1) coefficient           [FC-5]
    sigma_nu: float = 0.054       # Investment shock s.d.                  [FC-5]

    # --- Group III ---
    beta_I_ols: float = 0.00127   # Investment multiplier, OLS upper bound [FC-5]
    mu_bar: float = 0.020         # Cross-country mean drift               [FC-10]
    sigma_mu: float = 0.0155      # Cross-country s.d. of μ_i             [FC-10]
    I_bar_world: float = 0.225    # World-mean investment rate             [FC-5]

    # --- Group IV (pending WGI panel) ---
    gamma: Optional[float] = None
    sigma_zeta: Optional[float] = None

    # ── Derived quantities (properties, no free parameters) ──────────────────

    @property
    def mu0(self) -> float:
        """μ₀ = μ̄ − β_I · Ī  (M06, Proposition 6.3(ii))."""
        return self.mu_bar - self.beta_I_ols * self.I_bar_world

    @property
    def sigma_g2(self) -> float:
        """Long-run growth variance: σ_g² = σ_ε² / (1 − φ₁²)  (M04, Def. 4.3)."""
        return self.sigma_eps ** 2 / (1.0 - self.phi1 ** 2)

    @property
    def sigma_g(self) -> float:
        """Long-run growth s.d."""
        return float(np.sqrt(self.sigma_g2))

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio: SNR = σ_μ² / σ_g²  (M04, Def. 4.3)."""
        return self.sigma_mu ** 2 / self.sigma_g2

    @property
    def r2_max_eff(self) -> float:
        """Practical predictability ceiling: R²_max = SNR/(1+SNR)  (M04, Thm. 4.4)."""
        return self.snr / (1.0 + self.snr)

    @property
    def r2_max_theoretical(self) -> float:
        """Theoretical ceiling with known μ_i: (σ_μ²+φ₁²σ_g²)/(σ_μ²+σ_g²)."""
        return (self.sigma_mu ** 2 + self.phi1 ** 2 * self.sigma_g2) / (
            self.sigma_mu ** 2 + self.sigma_g2
        )

    @property
    def psi_inf(self) -> float:
        """
        Permanent IRF multiplier: ψ_∞ = 1/(1−φ₁)  (M05, Thm. 5.4).

        Single-lag formula (see module docstring); the two-lag ψ∞≈1.344
        actually reported in the manuscript uses 1/(1-φ₁-φ₂) instead.
        """
        return 1.0 / (1.0 - self.phi1)

    @property
    def investment_halflife(self) -> float:
        """Investment shock half-life: k_{1/2} = log2 / |log ρ_I|  (M04, Cor. 4.7)."""
        return np.log(2.0) / abs(np.log(self.rho_I))

    @property
    def sigma_lr2(self) -> float:
        """Long-run noise: σ_LR² = σ_g²(1+φ₁)/(1−φ₁)  (M04, Cor. 4.5)."""
        return self.sigma_g2 * (1.0 + self.phi1) / (1.0 - self.phi1)

    @property
    def I_max(self) -> float:
        """Maximum feasible investment rate (50 % of GDP, M04, Thm. 4.8)."""
        return 0.50


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE TRANSITION EQUATIONS T1–T3
# ─────────────────────────────────────────────────────────────────────────────

class DESMTransition:
    """
    One-period forward step for the DESM V2 state-space.

    Implements T1–T3 of M04, Theorem 4.2 exactly.
    Innovations ε_{it} are drawn from a Student-t(α₊) distribution, which
    belongs to the Lévy class with matching tail index and finite variance
    (α₊ > 2), consistent with A3.
    """

    def __init__(self, params: DESMParameters) -> None:
        self.p = params

    def sample_eps(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw ε ~ F_ε: Student-t(df=α₊) scaled to σ_ε.

        Student-t with ν = α₊ ∈ (2,3) has:
          E[ε] = 0,  Var(ε) = ν/(ν−2)·σ² → scaled to match σ_ε².
          Tail index = α₊ ∈ (2,3)  ✓ (A3).
          Fourth moment infinite for α₊ < 4  ✓ (M05, Thm. 5.8).
        """
        nu = self.p.alpha_plus
        scale = self.p.sigma_eps * np.sqrt((nu - 2.0) / nu)  # correct σ_ε
        return scale * rng.standard_t(df=nu, size=n)

    def step(
        self,
        x1: np.ndarray,
        g: np.ndarray,
        I: np.ndarray,
        mu_i: np.ndarray,
        I_bar_i: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-period forward step for N countries simultaneously.

        T1:  x1_{t+1} = x1_t + μ_i + g_{t+1}
        T2:  g_{t+1}  = φ₁ g_t + ε_{t+1}
        T3:  I_{t+1}  = ρ_I I_t + (1−ρ_I) Ī_i + ν_{t+1}

        Parameters
        ----------
        x1, g, I   : (N,) current state
        mu_i        : (N,) country drifts
        I_bar_i     : (N,) country long-run investment means
        rng         : numpy Generator for reproducibility
        """
        n = len(x1)
        p = self.p

        # T2
        eps = self.sample_eps(n, rng)
        g_next = p.phi1 * g + eps

        # T1
        x1_next = x1 + mu_i + g_next

        # T3
        nu = p.sigma_nu * rng.standard_normal(n)
        I_next = p.rho_I * I + (1.0 - p.rho_I) * I_bar_i + nu

        return x1_next, g_next, I_next

    def irf(self, horizon: int) -> np.ndarray:
        """
        Impulse response function ψ_k = (1−φ₁^k)/(1−φ₁),  k = 1,…,horizon.

        M05, Theorem 5.4.  ψ_∞ = 1/(1−φ₁) ≈ 1.357 at φ₁ = 0.263.
        """
        k = np.arange(1, horizon + 1)
        return (1.0 - self.p.phi1 ** k) / (1.0 - self.p.phi1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PANEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the estimation panel from raw World Bank data.

    Expected input columns:
      'Country Name'  — country identifier
      'year'          — integer year
      'gdp_pc'        — real GDP per capita (constant USD)
      'gfcf_gdp'      — gross fixed capital formation (% of GDP)

    Output adds per-country-year:
      x1       = log(gdp_pc)
      delta_x1 = first difference of x1
      mu_hat   = country-mean of delta_x1  (estimator of μ_i)
      g_hat    = delta_x1 − mu_hat          (demeaned growth)
      I        = gfcf_gdp / 100             (investment rate, 0–1 scale)
      I_bar    = country-mean of I          (estimator of Ī_i)
      I_tilde  = I − I_bar                 (within-country deviation)
    """
    df = df.copy().rename(columns={'Country Name': 'country'})
    df = df.sort_values(['country', 'year']).reset_index(drop=True)

    df['x1'] = np.log(df['gdp_pc'].astype(float))
    df['delta_x1'] = df.groupby('country')['x1'].diff()

    # Country-mean drift μ̂_i  (M06, eq. 6.6)
    mu_hat = (
        df.dropna(subset=['delta_x1'])
        .groupby('country')['delta_x1']
        .mean()
        .rename('mu_hat')
    )
    df = df.join(mu_hat, on='country')
    df['g_hat'] = df['delta_x1'] - df['mu_hat']

    # Investment (convert % → fraction)
    df['I'] = df['gfcf_gdp'].astype(float) / 100.0
    I_bar = df.groupby('country')['I'].mean().rename('I_bar')
    df = df.join(I_bar, on='country')
    df['I_tilde'] = df['I'] - df['I_bar']

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4–5. GROUP I: (φ₁, σ_ε)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_phi1(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Pooled within-country OLS estimator for φ₁  (M06, eq. 6.8).

      φ̂₁ = ΣΣ ĝ_{i,t−1} ĝ_{it}  /  ΣΣ ĝ²_{i,t−1}

    Nickell bias correction (M06, Proposition 6.2):
      φ̂₁^BC = φ̂₁ + (1 + φ̂₁) / (T̄ − 1),   bias ≈ −0.021 at T̄ = 60.

    Returns
    -------
    (phi1_raw, phi1_bc)  — raw OLS and bias-corrected estimates.
    """
    panel = df.dropna(subset=['g_hat']).copy()
    panel['g_lag'] = panel.groupby('country')['g_hat'].shift(1)
    panel = panel.dropna(subset=['g_lag'])

    num = (panel['g_lag'] * panel['g_hat']).sum()
    den = (panel['g_lag'] ** 2).sum()
    phi1_raw = float(num / den)

    T_mean = panel.groupby('country')['g_hat'].count().mean()
    phi1_bc = phi1_raw + (1.0 + phi1_raw) / (T_mean - 1.0)

    return phi1_raw, float(phi1_bc)


def estimate_sigma_eps(df: pd.DataFrame, phi1: float) -> float:
    """
    Pooled innovation standard deviation  (M06, eq. 6.9).

      σ̂_ε = sqrt[ ΣΣ ε̂²_{it} / (NT − N − 1) ]
      ε̂_{it} = ĝ_{it} − φ̂₁ ĝ_{i,t−1}

    Consistent for σ_ε; bootstrap required for inference (M05, Thm. 5.8(i)).
    """
    panel = df.dropna(subset=['g_hat']).copy()
    panel['g_lag'] = panel.groupby('country')['g_hat'].shift(1)
    panel = panel.dropna(subset=['g_lag'])
    panel['eps_hat'] = panel['g_hat'] - phi1 * panel['g_lag']

    NT = len(panel)
    N = panel['country'].nunique()
    sigma2 = (panel['eps_hat'] ** 2).sum() / (NT - N - 1)
    return float(np.sqrt(sigma2))


# ─────────────────────────────────────────────────────────────────────────────
# 6. HILL ESTIMATOR FOR TAIL INDICES
# ─────────────────────────────────────────────────────────────────────────────

def hill_estimator(
    x: np.ndarray,
    k: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Hill (1975) tail-index estimator for a single tail  (M06, Def. 6.5).

      α̂_k = [ k⁻¹ Σ_{j=1}^k  log( X_{(n−j+1)} / X_{(n−k)} ) ]⁻¹

    The estimator is consistent as n→∞, k→∞, k/n→0 with asymptotic
    distribution √k (α̂_k − α) →^d N(0, α²)  (M06, Proposition 6.4).

    Parameters
    ----------
    x : positive observations from one tail.
    k : upper-order statistics to use; if None, selected as ⌊n^0.6⌋
        (stable-region heuristic from Hill plots).

    Returns
    -------
    (alpha_hat, se)  where se = α̂ / √k.
    """
    x_pos = x[x > 0]
    n = len(x_pos)
    if n < 10:
        raise ValueError(f"Too few positive observations ({n}) for Hill estimator.")

    if k is None:
        k = max(10, int(round(n ** 0.6)))
    k = min(k, n - 2)

    x_sorted = np.sort(x_pos)
    threshold = x_sorted[n - k - 1]
    log_ratios = np.log(x_sorted[n - k:] / threshold)
    alpha_hat = float(1.0 / log_ratios.mean())
    se = alpha_hat / np.sqrt(k)
    return alpha_hat, float(se)


def estimate_tail_indices(
    df: pd.DataFrame,
    phi1: float,
    k: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Estimate α₊ and α₋ from filtered AR(1) residuals  (M06, Step 4).

    By Mikosch & Stărică (2000), the filtered process ε̂_{it} inherits the
    power-law tail index of ε_{it} when |φ₁| < 1.

    Returns
    -------
    (alpha_plus, se_plus, alpha_minus, se_minus)
    """
    panel = df.dropna(subset=['g_hat']).copy()
    panel['g_lag'] = panel.groupby('country')['g_hat'].shift(1)
    panel = panel.dropna(subset=['g_lag'])
    eps = (panel['g_hat'] - phi1 * panel['g_lag']).values

    alpha_p, se_p = hill_estimator(eps[eps > 0], k)
    alpha_m, se_m = hill_estimator(np.abs(eps[eps < 0]), k)
    return alpha_p, se_p, alpha_m, se_m


# ─────────────────────────────────────────────────────────────────────────────
# 8–9. GROUP II: (ρ_I, σ_ν)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rhoI(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Pooled within-country OLS for ρ_I  (M06, Def. 6.6, eq. 6.10).

      ρ̂_I = ΣΣ Ĩ_{i,t−1} Ĩ_{it}  /  ΣΣ Ĩ²_{i,t−1}

    Nickell bias correction analogous to φ₁ (M06 §6.3).

    Returns
    -------
    (rhoI_raw, rhoI_bc)
    """
    panel = df.dropna(subset=['I_tilde']).copy()
    panel['I_lag'] = panel.groupby('country')['I_tilde'].shift(1)
    panel = panel.dropna(subset=['I_lag'])

    num = (panel['I_lag'] * panel['I_tilde']).sum()
    den = (panel['I_lag'] ** 2).sum()
    rhoI_raw = float(num / den)

    T_mean = panel.groupby('country')['I_tilde'].count().mean()
    rhoI_bc = rhoI_raw + (1.0 + rhoI_raw) / (T_mean - 1.0)

    return rhoI_raw, float(rhoI_bc)


def estimate_sigma_nu(df: pd.DataFrame, rhoI: float) -> float:
    """
    Investment shock standard deviation  (M06, eq. 6.11).

      σ̂_ν = sqrt[ ΣΣ ν̂²_{it} / (NT − N − 1) ]
      ν̂_{it} = Ĩ_{it} − ρ̂_I Ĩ_{i,t−1}
    """
    panel = df.dropna(subset=['I_tilde']).copy()
    panel['I_lag'] = panel.groupby('country')['I_tilde'].shift(1)
    panel = panel.dropna(subset=['I_lag'])
    panel['nu_hat'] = panel['I_tilde'] - rhoI * panel['I_lag']

    NT = len(panel)
    N = panel['country'].nunique()
    sigma2 = (panel['nu_hat'] ** 2).sum() / (NT - N - 1)
    return float(np.sqrt(sigma2))


# ─────────────────────────────────────────────────────────────────────────────
# 10. GROUP III: (β_I^OLS, μ₀)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_betaI_ols(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Cross-country OLS for β_I^OLS and μ₀  (M06, eq. 6.12).

      β̂_I^OLS = Σ_i (Ī_i − Ī̄)(μ̂_i − μ̄̂)  /  Σ_i (Ī_i − Ī̄)²
      μ̂₀      = μ̄̂ − β̂_I^OLS · Ī̄

    β_I^OLS is a consistent estimator of β_I + γδ/Var(Ī_i), not of β_I.
    It is an upper bound on the true causal multiplier  (M06, Thm. 6.3).

    Returns
    -------
    (beta_I_ols, mu0_hat)
    """
    cs = (
        df.dropna(subset=['delta_x1', 'I'])
        .groupby('country')
        .agg(mu_hat=('delta_x1', 'mean'), I_bar=('I', 'mean'))
        .dropna()
    )

    I_mean = cs['I_bar'].mean()
    mu_mean = cs['mu_hat'].mean()

    num = ((cs['I_bar'] - I_mean) * (cs['mu_hat'] - mu_mean)).sum()
    den = ((cs['I_bar'] - I_mean) ** 2).sum()

    beta_I_ols = float(num / den)
    mu0_hat = float(mu_mean - beta_I_ols * I_mean)

    return beta_I_ols, mu0_hat


# ─────────────────────────────────────────────────────────────────────────────
# 11. GROUP IV: (β_I, γ, σ_ζ) — WGI-AUGMENTED PANEL
# ─────────────────────────────────────────────────────────────────────────────

def estimate_wgi_panel(
    df: pd.DataFrame,
    wgi: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Joint OLS for (β_I, γ) from the WGI-augmented panel  (M06, eq. 6.14).

    Composite governance index (M06, eq. 6.13):
      q̂_i = (1/6) Σ_{k=1}^6 WGĪ_{i,k}

    Standardised:
      q̂_i^std = (q̂_i − q̄) / σ_q

    Cross-country OLS:
      μ̂_i = μ₀ + β_I Ī_i + γ q̂_i^std + e_i

    Identification requires Gram matrix G to be full rank (M06, Thm. 6.4):
      det(G) ≠ 0  ↔  Ī_i and q̂_i not perfectly collinear.

    Expected wgi columns: 'country' + six WGI dimension columns
    (any numeric columns other than 'country' and 'year').

    Returns
    -------
    (beta_I_wgi, gamma_hat, mu0_wgi, sigma_zeta)
    """
    wgi_cols = [c for c in wgi.columns if c not in ('country', 'year', 'Country Name')]
    q_hat = (
        wgi.rename(columns={'Country Name': 'country'})
        .groupby('country')[wgi_cols]
        .mean()
        .mean(axis=1)
        .rename('q_hat')
    )

    cs = (
        df.dropna(subset=['delta_x1', 'I'])
        .groupby('country')
        .agg(mu_hat=('delta_x1', 'mean'), I_bar=('I', 'mean'))
        .join(q_hat)
        .dropna()
    )

    if len(cs) < 5:
        raise ValueError("Fewer than 5 countries with matched WGI data; cannot estimate.")

    q_mean = cs['q_hat'].mean()
    q_std_val = cs['q_hat'].std(ddof=1)
    cs['q_std'] = (cs['q_hat'] - q_mean) / q_std_val

    # OLS: μ̂_i = μ₀ + β_I Ī_i + γ q̂_i^std + e_i
    X = np.column_stack([np.ones(len(cs)), cs['I_bar'].values, cs['q_std'].values])
    y = cs['mu_hat'].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    mu0_wgi, beta_I_wgi, gamma_hat = coeffs

    residuals = y - X @ coeffs
    N = len(cs)
    sigma_zeta = float(np.sqrt(np.maximum(0.0, (residuals ** 2).sum() / (N - 3))))

    return float(beta_I_wgi), float(gamma_hat), float(mu0_wgi), sigma_zeta


# ─────────────────────────────────────────────────────────────────────────────
# 12. WILD BOOTSTRAP (within-country parameters)
# ─────────────────────────────────────────────────────────────────────────────

def wild_bootstrap(
    df: pd.DataFrame,
    phi1: float,
    B: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Wild bootstrap CI for φ₁  (M06, Def. 6.8).

    Rademacher multipliers w_{it} ~ Uniform{−1, +1}.
    Pseudo-sample: ĝ*_{it} = φ̂₁ ĝ_{i,t−1} + w_{it} ε̂_{it}.
    Re-estimate φ̂₁^(b) on each pseudo-sample.

    Valid without finite fourth-moment assumption (Liu 1988); required
    because α₊ ∈ (2,3) implies infinite fourth moment  (M05, Thm. 5.8(i)).

    Returns
    -------
    (phi1_hat, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng()

    panel = df.dropna(subset=['g_hat']).copy()
    panel['g_lag'] = panel.groupby('country')['g_hat'].shift(1)
    panel = panel.dropna(subset=['g_lag'])
    panel['eps_hat'] = panel['g_hat'] - phi1 * panel['g_lag']

    g_lag = panel['g_lag'].values
    eps_hat = panel['eps_hat'].values
    n = len(g_lag)
    den = (g_lag ** 2).sum()

    boot = np.empty(B)
    for b in range(B):
        w = rng.choice(np.array([-1.0, 1.0]), size=n)
        g_star = phi1 * g_lag + w * eps_hat
        boot[b] = (g_lag * g_star).sum() / den

    ci_lo = float(np.quantile(boot, alpha / 2.0))
    ci_hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return phi1, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# 13. PAIRS BOOTSTRAP (cross-sectional parameters)
# ─────────────────────────────────────────────────────────────────────────────

def pairs_bootstrap(
    df: pd.DataFrame,
    estimator_fn,
    B: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairs bootstrap CI for cross-sectional estimators  (M06, Def. 6.7).

    Resamples country tuples (μ̂_i, Ī_i) with replacement B times and
    re-computes estimator_fn(df_boot) on each bootstrap dataset.

    Valid without finite fourth-moment assumption (Theorem M05, Thm. 5.8(ii)).

    Parameters
    ----------
    estimator_fn : callable df → np.ndarray of scalar estimates.

    Returns
    -------
    (point, ci_lower, ci_upper)  — all shape (k,) where k = len(estimator_fn(df)).
    """
    if rng is None:
        rng = np.random.default_rng()

    countries = df['country'].unique()
    N = len(countries)
    point = np.atleast_1d(np.asarray(estimator_fn(df), dtype=float))
    k = len(point)
    boot = np.full((B, k), np.nan)

    for b in range(B):
        sampled = rng.choice(countries, size=N, replace=True)
        frames = [df[df['country'] == c] for c in sampled]
        df_b = pd.concat(frames, ignore_index=True)
        try:
            boot[b] = np.atleast_1d(np.asarray(estimator_fn(df_b), dtype=float))
        except Exception:
            pass  # leave as NaN; excluded from quantile

    ci_lo = np.nanquantile(boot, alpha / 2.0, axis=0)
    ci_hi = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)
    return point, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# 14. REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Five-regime partition of the parameter space  (M05, Theorem 5.2).

    Regimes are defined by the conditional expected drift:
      m(q, Ī) = μ₀ + β_I Ī + γ q                     (M05, Def. 5.1)

    Ordered thresholds (M05, Def. 5.2, Lemma 5.3):
      q† < q* < q̃ < q_h

    Regime assignment:
      R5:  m(q_i, I_max) < 0          institutional trap (no escape via investment)
      R4:  m(q_i, I_max) ≥ 0,
           m(q_i, Ī_i) < 0           decline, investment-rescuable
      R3:  0 ≤ m(q_i, Ī_i) < μ̄       stagnation (below-world-average growth)
      R2:  μ̄ ≤ m(q_i, Ī_i) ≤ μ̄+σ_μ  normal growth
      R1:  m(q_i, Ī_i) > μ̄+σ_μ      growth miracle

    If governance data q_i are unavailable, classify using μ̂_i directly
    against the distributional thresholds (μ̄, μ̄±σ_μ).
    """

    def __init__(self, params: DESMParameters) -> None:
        self.p = params

    def m(self, q: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Conditional expected drift m(q, Ī) = μ₀ + β_I·Ī + γ·q."""
        if self.p.gamma is None:
            raise ValueError("γ not calibrated; run WGI panel estimation first.")
        return self.p.mu0 + self.p.beta_I_ols * I + self.p.gamma * q

    def classify_by_mu(self, mu_hat: np.ndarray) -> np.ndarray:
        """
        Distributional regime assignment using μ̂_i and (μ̄, σ_μ) thresholds.

        Used when q_i is unavailable (Panel A only).
        """
        p = self.p
        regimes = np.zeros(len(mu_hat), dtype=int)
        regimes[mu_hat > p.mu_bar + p.sigma_mu] = 1
        regimes[(mu_hat >= p.mu_bar) & (mu_hat <= p.mu_bar + p.sigma_mu)] = 2
        regimes[(mu_hat >= 0.0) & (mu_hat < p.mu_bar)] = 3
        regimes[(mu_hat < 0.0)] = 4   # cannot distinguish R4/R5 without q_i
        return regimes

    def classify(
        self,
        mu_hat: np.ndarray,
        I_bar: np.ndarray,
        q_hat: np.ndarray,
    ) -> np.ndarray:
        """
        Structural regime assignment using m(q_i, Ī_i) and I_max.

        Requires γ to be calibrated (Panel B / WGI panel).
        """
        p = self.p
        n = len(mu_hat)
        m_val = self.m(q_hat, I_bar)
        m_at_Imax = self.m(q_hat, np.full(n, p.I_max))

        regimes = np.zeros(n, dtype=int)
        regimes[m_val > p.mu_bar + p.sigma_mu] = 1
        regimes[(m_val >= p.mu_bar) & (m_val <= p.mu_bar + p.sigma_mu)] = 2
        regimes[(m_val >= 0.0) & (m_val < p.mu_bar)] = 3
        regimes[(m_val < 0.0) & (m_at_Imax >= 0.0)] = 4
        regimes[m_at_Imax < 0.0] = 5
        return regimes

    def trap_threshold(self) -> float:
        """
        Structural trap threshold  (M04, Thm. 4.8):
          q* = −(μ₀ + β_I·Ī̄) / γ

        Countries with q_i < q* have E[μ_i | Ī_i=Ī̄] < 0.
        """
        if self.p.gamma is None:
            raise ValueError("γ not calibrated.")
        return -(self.p.mu0 + self.p.beta_I_ols * self.p.I_bar_world) / self.p.gamma

    def deep_trap_threshold(self) -> float:
        """
        Deep trap threshold (M04, Thm. 4.8):
          q† = −(μ₀ + β_I·I_max) / γ

        No feasible ΔĪ escapes for countries with q_i < q†.
        """
        if self.p.gamma is None:
            raise ValueError("γ not calibrated.")
        return -(self.p.mu0 + self.p.beta_I_ols * self.p.I_max) / self.p.gamma

    def rescue_investment(
        self, q: np.ndarray, I_bar: np.ndarray
    ) -> np.ndarray:
        """
        Minimum ΔĪ to reach E[μ_i]=0 for R4 countries  (M05, Thm. 5.7):
          ΔĪ_rescue(q) = |m(q, Ī)| / β_I
          ∂ΔĪ_rescue/∂q = −γ/β_I < 0  (lower governance → more investment required).
        """
        m_val = self.m(q, I_bar)
        return np.abs(m_val) / self.p.beta_I_ols

    def regime_summary(
        self, regimes: np.ndarray, country_names: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Return count and fraction of countries per regime."""
        labels = {1: 'R1 (miracle)', 2: 'R2 (normal)', 3: 'R3 (stagnation)',
                  4: 'R4 (decline)', 5: 'R5 (trap)'}
        rows = []
        N = len(regimes)
        for r in [1, 2, 3, 4, 5]:
            mask = regimes == r
            rows.append({
                'Regime': labels.get(r, f'R{r}'),
                'Count': int(mask.sum()),
                'Fraction': float(mask.sum()) / N,
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 15. SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DESMSimulator:
    """
    Monte Carlo forward simulation of the DESM V2 state-space.

    Implements T1–T3 of M04, Theorem 4.2 for N countries over T periods.
    Useful for:
      — generating synthetic panels to validate moment conditions
      — computing predictive distributions for x1_{i,T}
      — counterfactual policy experiments (change Ī_i or q_i)
    """

    def __init__(
        self,
        params: DESMParameters,
        seed: Optional[int] = None,
    ) -> None:
        self.p = params
        self.transition = DESMTransition(params)
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        N: int,
        T: int,
        mu_i: np.ndarray,
        I_bar_i: np.ndarray,
        x1_0: Optional[np.ndarray] = None,
        g_0: Optional[np.ndarray] = None,
        I_0: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Forward simulation for N countries, T periods.

        Parameters
        ----------
        N, T       : panel dimensions
        mu_i       : (N,) country drifts
        I_bar_i    : (N,) country long-run investment rates
        x1_0       : (N,) initial log-GDP; default 0
        g_0        : (N,) initial demeaned growth; default stationary draw
        I_0        : (N,) initial investment; default I_bar_i

        Returns
        -------
        dict with keys 'x1', 'g', 'I' each of shape (N, T+1),
        plus 'mu_i' (N,).
        """
        p = self.p
        if x1_0 is None:
            x1_0 = np.zeros(N)
        if g_0 is None:
            sigma_g_stat = p.sigma_eps / np.sqrt(1.0 - p.phi1 ** 2)
            g_0 = sigma_g_stat * self.rng.standard_normal(N)
        if I_0 is None:
            I_0 = I_bar_i.copy()

        x1 = np.empty((N, T + 1))
        g  = np.empty((N, T + 1))
        I  = np.empty((N, T + 1))
        x1[:, 0], g[:, 0], I[:, 0] = x1_0, g_0, I_0

        for t in range(T):
            x1[:, t + 1], g[:, t + 1], I[:, t + 1] = self.transition.step(
                x1[:, t], g[:, t], I[:, t], mu_i, I_bar_i, self.rng
            )

        return {'x1': x1, 'g': g, 'I': I, 'mu_i': mu_i}

    def simulate_from_data(
        self, df: pd.DataFrame, T_fwd: int
    ) -> Dict[str, np.ndarray]:
        """
        Initialise from observed last values and simulate T_fwd periods forward.
        """
        cs = (
            df.dropna(subset=['x1', 'g_hat', 'I'])
            .groupby('country')
            .agg(
                mu_hat=('delta_x1', 'mean'),
                I_bar=('I', 'mean'),
                x1_last=('x1', 'last'),
                g_last=('g_hat', 'last'),
                I_last=('I', 'last'),
            )
            .dropna()
        )
        return self.simulate(
            N=len(cs),
            T=T_fwd,
            mu_i=cs['mu_hat'].values,
            I_bar_i=cs['I_bar'].values,
            x1_0=cs['x1_last'].values,
            g_0=cs['g_last'].values,
            I_0=cs['I_last'].values,
        )

    def cross_section_variance(
        self, results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Cross-sectional variance of log-GDP at each period.

        M04, Theorem 4.9 predicts: Var(x1_i(T)) = V₀ + T²σ_μ² + T·σ_LR² + O(T⁻¹).
        """
        return results['x1'].var(axis=0)  # shape (T+1,)


# ─────────────────────────────────────────────────────────────────────────────
# 16. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

class DESMDiagnostics:
    """
    Derived quantities and model diagnostics  (M06, Remark 6.8).

    All quantities are functions of θ; no free parameters beyond θ.
    """

    def __init__(self, params: DESMParameters) -> None:
        self.p = params

    def scalar_summary(self) -> Dict[str, float]:
        """All analytically derived scalars."""
        p = self.p
        return {
            'sigma_g2': p.sigma_g2,
            'sigma_g': p.sigma_g,
            'SNR': p.snr,
            'R2_max_practical': p.r2_max_eff,
            'R2_max_theoretical': p.r2_max_theoretical,
            'psi_inf': p.psi_inf,
            'investment_halflife_yr': p.investment_halflife,
            'sigma_LR2': p.sigma_lr2,
            'mu0': p.mu0,
            'BW_asymptotic_limit': 3.0,
        }

    def between_within(
        self, df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Empirical between/within variance ratio  (M04, Theorem 4.6).

        Theoretical prediction: B/W → 3 as T → ∞.
        Observed B/W = 12.56 at T = 61 (initial-spread dominated).
        """
        cs_mean = df.groupby('country')['x1'].mean()
        var_B = float(cs_mean.var(ddof=1))
        within = df['x1'] - df.groupby('country')['x1'].transform('mean')
        var_W = float(within.var(ddof=1))
        return var_B, var_W, var_B / var_W

    def rank_stability(
        self, df: pd.DataFrame, t0_year: int, t1_year: int
    ) -> float:
        """
        Spearman rank correlation of x1_i between t0 and t1.

        FC-2 target: r_S(50 yr) ≈ 0.845.
        """
        snap0 = df[df['year'] == t0_year].set_index('country')['x1']
        snap1 = df[df['year'] == t1_year].set_index('country')['x1']
        common = snap0.index.intersection(snap1.index)
        if len(common) < 5:
            return float('nan')
        from scipy.stats import spearmanr
        corr, _ = spearmanr(snap0[common], snap1[common])
        return float(corr)

    def print_report(self, df: Optional[pd.DataFrame] = None) -> None:
        """Formatted console report of parameters and diagnostics."""
        p = self.p
        sep = "=" * 64
        print(sep)
        print("DESM V2 - Calibrated Parameter and Diagnostics Report")
        print(sep)
        print("\n  -- Group I (transition dynamics) --")
        print(f"    phi1            = {p.phi1:.4f}")
        print(f"    sigma_eps       = {p.sigma_eps:.4f}  (log-units / yr)")
        print(f"    alpha_plus      = {p.alpha_plus:.2f}  (right tail index)")
        print(f"    alpha_minus     = {p.alpha_minus:.2f}  (left tail index)")
        print("\n  -- Group II (investment persistence) --")
        print(f"    rho_I           = {p.rho_I:.4f}")
        print(f"    sigma_nu        = {p.sigma_nu:.4f}")
        print("\n  -- Group III (drift slope, OLS upper bound) --")
        print(f"    beta_I_OLS      = {p.beta_I_ols:.6f}")
        print(f"    mu0             = {p.mu0:.6f}")
        if p.gamma is not None:
            print("\n  -- Group IV (institutional coupling) --")
            print(f"    gamma           = {p.gamma:.6f}")
            if p.sigma_zeta is not None:
                print(f"    sigma_zeta      = {p.sigma_zeta:.6f}")
        print("\n  -- Derived Quantities --")
        for key, val in self.scalar_summary().items():
            print(f"    {key:<30} = {val:.6f}")
        if df is not None:
            var_B, var_W, bw = self.between_within(df)
            print("\n  -- Empirical B/W from Data --")
            print(f"    Var_B = {var_B:.4f},  Var_W = {var_W:.4f},  B/W = {bw:.2f}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 17. FULL ESTIMATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_full_estimation(
    df: pd.DataFrame,
    wgi: Optional[pd.DataFrame] = None,
    B_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[DESMParameters, Dict]:
    """
    Complete two-panel estimation pipeline for DESM V2.

    Panel A (primary):       estimates Θ_I, Θ_II, Θ_III.
    Panel B (WGI-augmented): estimates Θ_IV  (only if wgi is provided).

    Parameters
    ----------
    df            : primary estimation panel (output of build_panel).
    wgi           : WGI governance data frame with country + 6 dimension columns.
    B_bootstrap   : bootstrap replications (≥1000 as mandated by M06, Def. 6.7).
    seed          : random seed for reproducibility.

    Returns
    -------
    (DESMParameters, results_dict)

    results_dict contains for each parameter:
      {'estimate': float, 'ci': (lo, hi), ...}
    """
    rng = np.random.default_rng(seed)
    results: Dict = {}

    # ── Group I ──────────────────────────────────────────────────────────────
    phi1_raw, phi1_bc = estimate_phi1(df)
    sigma_eps = estimate_sigma_eps(df, phi1_bc)
    alpha_p, se_p, alpha_m, se_m = estimate_tail_indices(df, phi1_bc)

    _, phi1_lo, phi1_hi = wild_bootstrap(df, phi1_bc, B=B_bootstrap, rng=rng)

    results['phi1'] = {
        'estimate': phi1_bc, 'raw': phi1_raw,
        'ci': (phi1_lo, phi1_hi),
    }
    results['sigma_eps'] = {'estimate': sigma_eps}
    results['alpha_plus']  = {'estimate': alpha_p, 'se': se_p}
    results['alpha_minus'] = {'estimate': alpha_m, 'se': se_m}

    # ── Group II ─────────────────────────────────────────────────────────────
    rhoI_raw, rhoI_bc = estimate_rhoI(df)
    sigma_nu = estimate_sigma_nu(df, rhoI_bc)

    results['rhoI'] = {'estimate': rhoI_bc, 'raw': rhoI_raw}
    results['sigma_nu'] = {'estimate': sigma_nu}

    # ── Group III ────────────────────────────────────────────────────────────
    beta_I_ols, mu0_hat = estimate_betaI_ols(df)

    def _fn_betaI(d: pd.DataFrame) -> np.ndarray:
        b, m = estimate_betaI_ols(d)
        return np.array([b, m])

    _, b_lo, b_hi = pairs_bootstrap(df, _fn_betaI, B=B_bootstrap, rng=rng)
    results['beta_I_ols'] = {
        'estimate': beta_I_ols, 'ci': (float(b_lo[0]), float(b_hi[0])),
    }
    results['mu0'] = {
        'estimate': mu0_hat, 'ci': (float(b_lo[1]), float(b_hi[1])),
    }

    # ── Group IV (optional) ──────────────────────────────────────────────────
    gamma_hat: Optional[float] = None
    sigma_zeta: Optional[float] = None
    if wgi is not None:
        beta_I_wgi, gamma_hat, mu0_wgi, sigma_zeta = estimate_wgi_panel(df, wgi)
        results['beta_I_wgi'] = {'estimate': beta_I_wgi}
        results['gamma'] = {'estimate': gamma_hat}
        results['mu0_wgi'] = {'estimate': mu0_wgi}
        results['sigma_zeta'] = {'estimate': sigma_zeta}

    # ── Assemble DESMParameters ──────────────────────────────────────────────
    mu_bar_est = float(
        df.dropna(subset=['delta_x1']).groupby('country')['delta_x1'].mean().mean()
    )
    sigma_mu_est = float(
        df.dropna(subset=['delta_x1']).groupby('country')['delta_x1'].mean().std(ddof=1)
    )
    I_bar_world_est = float(
        df.dropna(subset=['I']).groupby('country')['I'].mean().mean()
    )

    params = DESMParameters(
        phi1=phi1_bc,
        sigma_eps=sigma_eps,
        alpha_plus=alpha_p,
        alpha_minus=alpha_m,
        rho_I=rhoI_bc,
        sigma_nu=sigma_nu,
        beta_I_ols=beta_I_ols,
        mu_bar=mu_bar_est,
        sigma_mu=sigma_mu_est,
        I_bar_world=I_bar_world_est,
        gamma=gamma_hat,
        sigma_zeta=sigma_zeta,
    )
    return params, results


# ─────────────────────────────────────────────────────────────────────────────
# 18. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Load primary panel, run estimation pipeline, print diagnostics.

    Data path: ../../data/panel_causal.csv
    (relative to this file's location in article/code/).
    """
    here = Path(__file__).parent
    data_path = here / '../../data/panel_causal.csv'

    print("Loading primary panel ...")
    raw = pd.read_csv(data_path)

    # Keep rows with both gdp_pc and gfcf_gdp
    raw = raw.dropna(subset=['gdp_pc', 'gfcf_gdp'])
    raw = raw[raw['gdp_pc'] > 0]

    print(f"  {len(raw)} country-year obs, {raw['Country Name'].nunique()} countries.")

    df = build_panel(raw)

    print("Running full estimation pipeline (B=500 for speed) ...")
    params, results = run_full_estimation(df, wgi=None, B_bootstrap=500, seed=42)

    diag = DESMDiagnostics(params)
    diag.print_report(df)

    # Regime classification (distributional, no WGI)
    classifier = RegimeClassifier(params)
    cs = df.dropna(subset=['delta_x1']).groupby('country')['delta_x1'].mean()
    regimes = classifier.classify_by_mu(cs.values)
    summary = classifier.regime_summary(regimes)
    print("\n  -- Regime Distribution (distributional, Panel A) --")
    print(summary.to_string(index=False))

    # Bootstrap CIs
    print("\n  -- Bootstrap 95% CIs --")
    for param in ['phi1', 'beta_I_ols']:
        if 'ci' in results.get(param, {}):
            lo, hi = results[param]['ci']
            est = results[param]['estimate']
            print(f"    {param:<15} = {est:.5f}  CI [{lo:.5f}, {hi:.5f}]")

    # Forward simulation check (10 countries, 30 periods)
    print("\nForward simulation check (N=10, T=30) ...")
    sim = DESMSimulator(params, seed=0)
    mu_test = np.array([params.mu_bar] * 10)
    I_bar_test = np.array([params.I_bar_world] * 10)
    sim_out = sim.simulate(N=10, T=30, mu_i=mu_test, I_bar_i=I_bar_test)
    final_x1 = sim_out['x1'][:, -1]
    print(f"  Mean final log-GDP: {final_x1.mean():.4f}  "
          f"(expected ~{30 * params.mu_bar:.4f})")

    print("\nDone.")


if __name__ == '__main__':
    main()
