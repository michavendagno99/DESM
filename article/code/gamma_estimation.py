"""
gamma_estimation.py
===================
Point identification of gamma (γ) — institutional coupling parameter of DESM V2.

Resolves [W1] / OQ-2 from F01_Doc_journal.tex:
  γ ∈ (0, 1.80] is a variance-decomposition bound; this script produces γ̂^SIMEX,
  the measurement-error-corrected OLS point estimate with 95% bootstrap CI.

Estimation strategy (M06, §6.5):
  Panel B: cross-country OLS of μ̂_i on (Ī_i, q̂_i^std)
  γ̂^EV  = OLS coefficient (attenuated by measurement error in q̂_i)
  γ̂^SIMEX = γ̂^EV / λ̂_q   (SIMEX correction; M06, Theorem 6.4)
  λ̂_q = reliability ratio = Var(q̂_i) / (Var(q̂_i) + σ̂²_u / T̄)

References: M06 §6.5, Algorithm alg:group4 (M07), Theorem 6.4 (M06).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
WGI_PATH   = DATA_DIR / "Data_Worldwide_Governance_Indicators_WB.xlsx"
PANEL_PATH = DATA_DIR / "panel_causal.csv"

# Calibrated Group III values (M08, T3)
BETA_I_OLS  = 0.00127    # % growth per % GDP investment (fraction scale)
MU0_HAT     = -0.00990   # baseline drift
SIGMA_MU    = 0.01755    # cross-country std of mu_i
I_BAR_WORLD = 0.225      # world mean investment rate

WGI_DIMS    = ['VA', 'PV', 'GE', 'RQ', 'RL', 'CC']
WGI_WINDOW  = (2000, 2021)   # M06 §6.5: estimation window

B_BOOTSTRAP = 2000
SEED        = 42

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND BUILD WGI COMPOSITE
# ──────────────────────────────────────────────────────────────────────────────

def load_wgi() -> pd.DataFrame:
    """
    Load all 6 WGI dimension sheets, filter to WGI_WINDOW, compute composite.

    Returns country-level DataFrame with:
      q_hat       — composite (mean of 6 dims, temporal mean 2000-2021)
      q_var_w     — within-country temporal variance of composite (noise proxy)
      T_wgi       — number of years of WGI coverage
    """
    frames = []
    for dim in WGI_DIMS:
        df = pd.read_excel(WGI_PATH, sheet_name=dim,
                           usecols=['econ_name', 'production_year', 'value'])
        df['dimension'] = dim
        frames.append(df)

    wgi_long = pd.concat(frames, ignore_index=True)
    wgi_long = wgi_long.rename(columns={'econ_name': 'country', 'production_year': 'year'})
    wgi_long = wgi_long.dropna(subset=['value'])
    wgi_long = wgi_long[
        (wgi_long['year'] >= WGI_WINDOW[0]) &
        (wgi_long['year'] <= WGI_WINDOW[1])
    ]

    # Composite per country-year: mean of available dimensions
    cy = (
        wgi_long.groupby(['country', 'year'])['value']
        .mean()
        .rename('q_cy')
        .reset_index()
    )

    # Country-level aggregates
    cs = cy.groupby('country').agg(
        q_hat=('q_cy', 'mean'),         # temporal mean composite
        q_var_w=('q_cy', 'var'),        # within-country variance (noise proxy)
        T_wgi=('q_cy', 'count'),        # years of coverage
    ).reset_index()

    cs = cs[cs['T_wgi'] >= 5]           # require at least 5 years
    return cs


# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD PRIMARY PANEL AND COMPUTE μ̂_i, Ī_i
# ──────────────────────────────────────────────────────────────────────────────

def load_primary_panel() -> pd.DataFrame:
    """
    Compute cross-country summary (μ̂_i, Ī_i) from annual panel.

    panel_causal.csv columns include: 'Country Name', 'year', 'gdp_pc', 'gfcf_gdp'
    """
    df = pd.read_csv(PANEL_PATH)
    df = df.rename(columns={'Country Name': 'country'})

    # Need annual gdp_pc and gfcf_gdp; filter rows with valid data
    df = df.dropna(subset=['gdp_pc', 'gfcf_gdp', 'year'])
    df = df.sort_values(['country', 'year'])

    # Log GDP per capita
    df['x1'] = np.log(df['gdp_pc'].astype(float))
    # First-difference of log GDP
    df['delta_x1'] = df.groupby('country')['x1'].diff()
    # Investment rate (fraction of GDP)
    df['I'] = df['gfcf_gdp'].astype(float) / 100.0

    # Country-level summaries
    cs = (
        df.dropna(subset=['delta_x1', 'I'])
        .groupby('country')
        .agg(
            mu_hat=('delta_x1', 'mean'),
            I_bar=('I', 'mean'),
            T_obs=('delta_x1', 'count'),
        )
        .reset_index()
    )
    # Require at least 20 years
    cs = cs[cs['T_obs'] >= 20]
    return cs


# ──────────────────────────────────────────────────────────────────────────────
# 3. MERGE PANELS
# ──────────────────────────────────────────────────────────────────────────────

def merge_panels(cs_primary: pd.DataFrame, wgi_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join on country name.  Standardize q̂_i to z-score.
    """
    merged = cs_primary.merge(wgi_cs, on='country', how='inner')
    merged = merged.dropna(subset=['mu_hat', 'I_bar', 'q_hat'])

    q_mean = merged['q_hat'].mean()
    q_std  = merged['q_hat'].std(ddof=1)
    merged['q_std'] = (merged['q_hat'] - q_mean) / q_std

    # Store standardization for interpretation
    merged.attrs['q_mean'] = q_mean
    merged.attrs['q_std_val'] = q_std

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 4. OLS ESTIMATION  (γ̂^EV)
# ──────────────────────────────────────────────────────────────────────────────

def ols_estimate(merged: pd.DataFrame):
    """
    Cross-country OLS:  μ̂_i = μ₀ + β_I·Ī_i + γ·q̂_i^std + e_i

    Returns (mu0, beta_I, gamma_EV, residuals, X, y)
    """
    X = np.column_stack([
        np.ones(len(merged)),
        merged['I_bar'].values,
        merged['q_std'].values,
    ])
    y = merged['mu_hat'].values

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    mu0, beta_I, gamma_EV = coeffs
    residuals = y - X @ coeffs
    return mu0, beta_I, gamma_EV, residuals, X, y


# ──────────────────────────────────────────────────────────────────────────────
# 5. SIMEX CORRECTION  (γ̂^SIMEX)
# ──────────────────────────────────────────────────────────────────────────────

def simex_correction(merged: pd.DataFrame, gamma_EV: float) -> tuple:
    """
    SIMEX correction for measurement error in q̂_i (M06, Theorem 6.4).

    q̂_i = q_i + u_i   where  Var(u_i) ≈ σ²_w / T̄_wgi

    σ²_w: within-country temporal variance of the composite governance index
          (average across countries)

    λ_q = Var(q̂_i^std) / (Var(q̂_i^std) + σ²_u_std / 1)
        = reliability ratio for the standardized composite

    Since q̂_i is already a temporal mean, the noise on the temporal mean is:
      Var(u_i) = Var(q_w_i,t) / T̄_i

    In standardized units:
      σ²_u_std = σ²_u / σ̂²_q  where σ̂²_q = Var(q̂_i)

    Returns (gamma_SIMEX, lambda_q, sigma2_u)
    """
    # Within-country variance of the composite (already temporal variance)
    q_var_w = merged['q_var_w'].values
    T_wgi   = merged['T_wgi'].values

    # Noise variance on temporal mean (σ²_u = σ²_w / T)
    sigma2_u_vec = q_var_w / T_wgi
    sigma2_u_mean = float(np.nanmean(sigma2_u_vec))   # average across countries

    # Cross-country variance of q̂_i (in original 0-1 units)
    var_q_hat = float(merged['q_hat'].var(ddof=1))

    # Reliability ratio  λ_q = (Var(q̂) - σ²_u) / Var(q̂)
    lambda_q = max(0.01, (var_q_hat - sigma2_u_mean) / var_q_hat)

    gamma_SIMEX = gamma_EV / lambda_q

    return gamma_SIMEX, lambda_q, sigma2_u_mean


# ──────────────────────────────────────────────────────────────────────────────
# 6. BOOTSTRAP CONFIDENCE INTERVALS
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(merged: pd.DataFrame, B: int = B_BOOTSTRAP, seed: int = SEED):
    """
    Pairs bootstrap for (beta_I, gamma_EV, gamma_SIMEX).

    Returns dict with keys 'beta_I', 'gamma_EV', 'gamma_SIMEX',
    each a tuple (estimate, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(merged)

    boot_beta  = np.empty(B)
    boot_gev   = np.empty(B)
    boot_gsimex = np.empty(B)

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        mb  = merged.iloc[idx].copy()

        # Re-standardize on bootstrap sample
        q_m = mb['q_hat'].mean()
        q_s = mb['q_hat'].std(ddof=1)
        if q_s < 1e-10:
            boot_beta[b] = boot_gev[b] = boot_gsimex[b] = np.nan
            continue
        mb['q_std'] = (mb['q_hat'] - q_m) / q_s

        try:
            _, beta_b, gev_b, _, _, _ = ols_estimate(mb)
            gsimex_b, _, _ = simex_correction(mb, gev_b)
            boot_beta[b]    = beta_b
            boot_gev[b]     = gev_b
            boot_gsimex[b]  = gsimex_b
        except Exception:
            boot_beta[b] = boot_gev[b] = boot_gsimex[b] = np.nan

    def ci(arr):
        return (float(np.nanquantile(arr, 0.025)),
                float(np.nanquantile(arr, 0.975)))

    # Point estimates on full merged dataset
    _, beta_pt, gev_pt, _, _, _ = ols_estimate(merged)
    gsimex_pt, lq_pt, su_pt = simex_correction(merged, gev_pt)

    return {
        'beta_I': (beta_pt, *ci(boot_beta)),
        'gamma_EV': (gev_pt, *ci(boot_gev)),
        'gamma_SIMEX': (gsimex_pt, *ci(boot_gsimex)),
        'lambda_q': lq_pt,
        'sigma2_u': su_pt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. DIAGNOSTICS: GRAM MATRIX AND COLLINEARITY
# ──────────────────────────────────────────────────────────────────────────────

def gram_diagnostics(merged: pd.DataFrame) -> dict:
    I_dev = merged['I_bar'] - merged['I_bar'].mean()
    q_dev = merged['q_std'] - merged['q_std'].mean()

    cov_IQ = float(np.mean(I_dev * q_dev))
    var_I  = float(np.var(merged['I_bar'], ddof=1))
    var_q  = float(np.var(merged['q_std'], ddof=1))

    G = np.array([[var_I, cov_IQ], [cov_IQ, var_q]])
    det_G = float(np.linalg.det(G))
    spearman_r = float(merged[['I_bar','q_hat']].corr(method='spearman').iloc[0,1])
    vif = 1.0 / (1.0 - spearman_r**2) if abs(spearman_r) < 1 else np.inf

    return {
        'cov_I_q': cov_IQ,
        'spearman_rho': spearman_r,
        'det_G': det_G,
        'VIF': vif,
        'N_countries': len(merged),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. VARIANCE DECOMPOSITION CHECK
# ──────────────────────────────────────────────────────────────────────────────

def variance_decomp(merged: pd.DataFrame, beta_I: float, gamma: float) -> dict:
    """
    Check γ against variance-decomposition bound:
      γ²·Var(q̂^std) ≤ σ²_μ   (M06, Remark 6.7)

    γ_max = σ_μ / √Var(q̂^std) = σ_μ  (since Var(q̂^std) = 1 by construction)
    """
    sigma_mu2 = SIGMA_MU**2
    var_I  = float(merged['I_bar'].var(ddof=1))
    var_q  = float(merged['q_std'].var(ddof=1))    # ≈ 1 by construction
    cov_IQ = float(np.cov(merged['I_bar'], merged['q_std'])[0,1])

    gamma_max = SIGMA_MU / np.sqrt(var_q)  # bound from Var(γ²·q̂²) ≤ σ²_μ

    explained = (beta_I**2 * var_I
                 + gamma**2 * var_q
                 + 2 * beta_I * gamma * cov_IQ)

    return {
        'sigma2_mu': sigma_mu2,
        'explained_fraction': explained / sigma_mu2,
        'gamma_max_vd': gamma_max,
        'gamma_exceeds_bound': gamma > gamma_max,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9. REGIME THRESHOLDS (require gamma to be calibrated)
# ──────────────────────────────────────────────────────────────────────────────

def regime_thresholds(gamma: float, q_mean: float, q_std_val: float) -> dict:
    """
    Trap threshold q* and deep-trap threshold q†, expressed in:
    (a) standardized units (z-scores)
    (b) raw WGI 0-1 units
    (c) WGI percentile
    """
    mu0 = MU0_HAT

    # In standardized units
    q_star_std  = -(mu0 + BETA_I_OLS * I_BAR_WORLD) / gamma
    q_dag_std   = -(mu0 + BETA_I_OLS * 0.40) / gamma  # I_max = 40% (M06)

    # Convert to raw 0-1 WGI units
    q_star_raw  = q_star_std * q_std_val + q_mean
    q_dag_raw   = q_dag_std  * q_std_val + q_mean

    return {
        'q_star_std': q_star_std,
        'q_dag_std':  q_dag_std,
        'q_star_raw': q_star_raw,
        'q_dag_raw':  q_dag_raw,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DESM V2 — Group IV: Point Identification of γ")
    print("OQ-2 Resolution: WGI Panel Merge with SIMEX Correction")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading WGI composite (2000-2021, 6 dimensions)...")
    wgi_cs = load_wgi()
    print(f"    WGI coverage: {len(wgi_cs)} countries")

    print("[2] Loading primary panel...")
    cs_primary = load_primary_panel()
    print(f"    Primary panel: {len(cs_primary)} countries with ≥20 years")

    print("[3] Merging panels...")
    merged = merge_panels(cs_primary, wgi_cs)
    q_mean   = merged.attrs['q_mean']
    q_std_v  = merged.attrs['q_std_val']
    print(f"    Merged sample: N = {len(merged)} countries")

    # 2. Gram matrix diagnostics
    print("\n[4] Gram matrix and collinearity diagnostics...")
    gram = gram_diagnostics(merged)
    print(f"    N countries in Panel B     : {gram['N_countries']}")
    print(f"    Cov(Ī_i, q̂_i)             : {gram['cov_I_q']:.5f}")
    print(f"    Spearman ρ(Ī, q̂)          : {gram['spearman_rho']:.3f}")
    print(f"    det(G) [rank condition]    : {gram['det_G']:.6e}")
    print(f"    VIF                        : {gram['VIF']:.3f}")

    # 3. OLS point estimate
    print("\n[5] OLS estimation of (β_I, γ^EV)...")
    mu0, beta_I, gamma_EV, residuals, X, y = ols_estimate(merged)
    N = len(merged)
    sigma_zeta = float(np.sqrt((residuals**2).sum() / (N - 3)))
    r2 = 1 - (residuals**2).sum() / ((y - y.mean())**2).sum()
    print(f"    μ̂₀          = {mu0*100:.3f} %/yr")
    print(f"    β̂_I^WGI     = {beta_I*100:.4f} %/yr per pp investment")
    print(f"    γ̂^EV        = {gamma_EV*100:.4f} %/yr per z-score WGI")
    print(f"    σ̂_ζ         = {sigma_zeta*100:.4f} %/yr")
    print(f"    R² (cross-section) = {r2:.3f}")

    # 4. SIMEX correction
    print("\n[6] SIMEX correction for measurement error in q̂_i...")
    gamma_SIMEX, lambda_q, sigma2_u = simex_correction(merged, gamma_EV)
    print(f"    σ̂²_u (noise variance on temporal mean)  : {sigma2_u:.6f}")
    print(f"    Var(q̂_i) (cross-country, 0-1 scale)    : {merged['q_hat'].var(ddof=1):.6f}")
    print(f"    Reliability ratio λ̂_q                   : {lambda_q:.4f}")
    print(f"    γ̂^SIMEX = γ̂^EV / λ̂_q                   : {gamma_SIMEX*100:.4f} %/yr")

    # 5. Bootstrap CI
    print(f"\n[7] Bootstrap inference (B = {B_BOOTSTRAP} pairs bootstrap)...")
    results = bootstrap_ci(merged, B=B_BOOTSTRAP, seed=SEED)
    beta_pt, beta_lo, beta_hi = results['beta_I']
    gev_pt, gev_lo, gev_hi   = results['gamma_EV']
    gs_pt, gs_lo, gs_hi      = results['gamma_SIMEX']

    print(f"    β̂_I^WGI: {beta_pt*100:.4f}  [95% CI: {beta_lo*100:.4f}, {beta_hi*100:.4f}] %/yr per pp")
    print(f"    γ̂^EV   : {gev_pt*100:.4f}  [95% CI: {gev_lo*100:.4f}, {gev_hi*100:.4f}] %/yr per z-score")
    print(f"    γ̂^SIMEX: {gs_pt*100:.4f}  [95% CI: {gs_lo*100:.4f}, {gs_hi*100:.4f}] %/yr per z-score")

    # 6. Variance decomposition
    print("\n[8] Variance decomposition check...")
    vd = variance_decomp(merged, beta_pt, gs_pt)
    print(f"    σ²_μ                           : {vd['sigma2_mu']:.6f}")
    print(f"    Fraction explained by (I,q)    : {vd['explained_fraction']:.3f}")
    print(f"    γ_max (variance bound)         : {vd['gamma_max_vd']*100:.4f} %/yr")
    print(f"    γ̂^SIMEX exceeds bound?         : {vd['gamma_exceeds_bound']}")

    # 7. Regime thresholds
    print("\n[9] Regime thresholds at γ̂^SIMEX...")
    thresholds = regime_thresholds(gs_pt, q_mean, q_std_v)
    print(f"    q* (trap threshold, z-score)   : {thresholds['q_star_std']:.3f}")
    print(f"    q* (raw WGI 0-1 scale)         : {thresholds['q_star_raw']:.3f}")
    print(f"    q† (deep trap, z-score)        : {thresholds['q_dag_std']:.3f}")
    print(f"    q† (raw WGI 0-1 scale)         : {thresholds['q_dag_raw']:.3f}")

    # 8. MRS policy frontier
    if beta_pt > 0 and gs_pt > 0:
        mrs = gs_pt / beta_pt
        print(f"\n[10] Policy frontier MRS = γ/β_I : {mrs:.2f} pp investment per z-score WGI")

    # 9. Country-level governance residuals
    merged['mu_hat_inv'] = mu0 + beta_pt * merged['I_bar']
    merged['gamma_term']  = gs_pt * merged['q_std']
    merged['mu_hat_full'] = merged['mu_hat_inv'] + merged['gamma_term']
    merged['zeta_hat']    = merged['mu_hat'] - merged['mu_hat_full']

    # Check Korea and Venezuela
    print("\n[11] Natural experiment check (Korea vs Venezuela):")
    for country in ['Korea, Rep.', 'Korea, South', 'Republic of Korea',
                    'Venezuela, RB', 'Venezuela']:
        row = merged[merged['country'].str.contains(country.split(',')[0], case=False)]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"    {r['country'][:20]:20s}: μ̂={r['mu_hat']*100:.2f}%  "
                  f"full={r['mu_hat_full']*100:.2f}%  "
                  f"ζ̂={r['zeta_hat']*100:.2f}%  "
                  f"q̂_std={r['q_std']:.2f}")

    # 10. Summary for LaTeX update
    print("\n" + "=" * 70)
    print("SUMMARY FOR LaTeX DOCUMENTS")
    print("=" * 70)
    print(f"\n  γ̂^EV     = {gev_pt*100:.3f} %/yr per z-score WGI")
    print(f"            [95% CI: ({gev_lo*100:.3f}, {gev_hi*100:.3f})]")
    print(f"  λ̂_q      = {lambda_q:.4f}  (reliability ratio)")
    print(f"  γ̂^SIMEX  = {gs_pt*100:.3f} %/yr per z-score WGI")
    print(f"            [95% CI: ({gs_lo*100:.3f}, {gs_hi*100:.3f})]")
    print(f"  β̂_I^WGI  = {beta_pt*100:.4f} %/yr per pp investment")
    print(f"            [95% CI: ({beta_lo*100:.4f}, {beta_hi*100:.4f})]")
    print(f"  N         = {len(merged)} countries")
    print(f"  R²        = {r2:.3f}")
    print(f"  σ̂_ζ       = {sigma_zeta*100:.4f} %/yr")
    print(f"  Variance bound γ_max = {vd['gamma_max_vd']*100:.4f} %/yr")
    print(f"  q* (z-score)  = {thresholds['q_star_std']:.3f}")
    print(f"  q† (z-score)  = {thresholds['q_dag_std']:.3f}")
    print(f"  MRS = γ/β_I   = {gs_pt/beta_pt:.2f} pp investment per z-score WGI")
    print("=" * 70)

    return merged, results, gram, vd, thresholds


if __name__ == '__main__':
    merged, results, gram, vd, thresholds = main()
