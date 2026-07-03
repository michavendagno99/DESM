"""
gamma_estimation_v2.py
======================
Point identification of gamma (gamma) — institutional coupling parameter DESM V2.
Resolves [W1] / OQ-2: WGI panel merge + SIMEX correction.

Data sources:
  Primary panel : panel_causal.csv  (WDI annual, 1960-2022)
  Governance    : qog_std_cs_jan26.xlsx  (QoG, WGI estimates ~2022)
  SIMEX proxy   : Data_WGI raw Excel (temporal variation 2000-2021)

Estimation strategy (M06 sec:ident_group4):
  Step 1. Build cross-country summary (mu_hat_i, I_bar_i) from primary panel.
  Step 2. Attach WGI composite q_hat_i from QoG (6 dimensions, standard scale).
  Step 3. Standardize q_hat_i to z-scores.
  Step 4. Cross-country OLS: mu_hat = mu0 + beta_I * I_bar + gamma_EV * q_std + e
  Step 5. Estimate reliability ratio lambda_q from WGI temporal variation (SIMEX).
  Step 6. SIMEX correction: gamma_SIMEX = gamma_EV / lambda_q.
  Step 7. Pairs bootstrap for 95% CI.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS AND CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
# Resolved relative to this file's location (article/code/), three levels
# up to the repository root, so the script runs unmodified on any
# machine/clone.
DATA_DIR  = Path(__file__).parent.parent.parent / "data"
WGI_PATH  = DATA_DIR / "Data_Worldwide_Governance_Indicators_WB.xlsx"
QOG_PATH  = DATA_DIR / "qog_std_cs_jan26.xlsx"
PANEL_CSV = DATA_DIR / "panel_causal.csv"

# M08 T3 calibrated values (Group I-III)
BETA_I_OLS_BOOK = 0.127   # pp/yr per pp investment (text; raw coeff = 0.127)
MU0_BOOK        = -0.990  # %/yr = -0.00990 fraction
SIGMA_MU_BOOK   = 1.755   # %/yr = 0.01755 fraction
I_BAR_WORLD_PCT = 23.5    # %GDP (from model, I_bar_world = 0.235 fraction)

WGI_DIMS_EST  = ['wbgi_vae', 'wbgi_gee', 'wbgi_pve', 'wbgi_rqe', 'wbgi_rle', 'wbgi_cce']
WGI_SHEETS    = ['VA', 'PV', 'GE', 'RQ', 'RL', 'CC']
WGI_WINDOW    = (2000, 2021)    # estimation window (M06 sec:ident_group4)

# Non-country aggregates to exclude from panel
REGIONAL_KEYWORDS = [
    'World', 'income', 'East Asia', 'Europe', 'Latin America', 'Middle East',
    'North America', 'South Asia', 'Sub-Saharan', 'OECD', 'Euro area',
    'Africa Eastern', 'Africa Western', 'Small states', 'Pacific island',
    'Fragile', 'Heavily indebted', 'Least developed', 'Low &', 'Lower middle',
    'Upper middle', 'High income', 'Low income', 'Arab World', 'Caribbean',
    'Central Europe', 'IDA', 'IBRD', 'not classified', 'Other small',
]

B_BOOTSTRAP = 2000
SEED        = 42
MIN_T_OBS   = 20     # minimum annual observations for mu_hat
MIN_T_WGI   = 5      # minimum WGI years for temporal variance


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRIMARY PANEL: mu_hat_i and I_bar_i
# ──────────────────────────────────────────────────────────────────────────────

def build_cross_section(csv_path: Path) -> pd.DataFrame:
    """
    Compute cross-country summary (mu_hat_i, I_bar_i) from annual panel.
    Excludes regional aggregates.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Country Name": "country"})
    df = df.dropna(subset=["gdp_pc", "gfcf_gdp", "year"])
    df = df.sort_values(["country", "year"])

    # Exclude regional aggregates
    mask = ~df["country"].str.contains("|".join(REGIONAL_KEYWORDS),
                                        case=False, regex=True, na=False)
    df = df[mask]

    df["x1"]      = np.log(df["gdp_pc"].astype(float))
    df["delta_x1"] = df.groupby("country")["x1"].diff()
    df["I"]        = df["gfcf_gdp"].astype(float) / 100.0   # fraction

    cs = (
        df.dropna(subset=["delta_x1", "I"])
        .groupby("country")
        .agg(mu_hat=("delta_x1", "mean"),
             I_bar=("I", "mean"),
             T_obs=("delta_x1", "count"))
        .reset_index()
    )
    cs = cs[cs["T_obs"] >= MIN_T_OBS]
    return cs


# ──────────────────────────────────────────────────────────────────────────────
# 2. GOVERNANCE: WGI composite from QoG (standard -2.5 to +2.5 scale)
# ──────────────────────────────────────────────────────────────────────────────

def load_qog_wgi(qog_path: Path) -> pd.DataFrame:
    """
    Load 6 WGI composite estimates from QoG Standard CS dataset.
    Returns country-level DataFrame with q_hat (mean of 6 dims).
    Scale: WGI standard normal units, approximately -2.5 to +2.5.
    """
    cols = ["cname", "ccodealp"] + WGI_DIMS_EST
    qog = pd.read_excel(qog_path, usecols=cols)
    qog = qog.dropna(subset=WGI_DIMS_EST, how="all")

    # Composite = mean of available WGI dimensions
    qog["q_hat"] = qog[WGI_DIMS_EST].mean(axis=1)
    qog["n_dims_available"] = qog[WGI_DIMS_EST].notna().sum(axis=1)

    # Keep countries with at least 4 of 6 dimensions
    qog = qog[qog["n_dims_available"] >= 4]

    return qog[["cname", "ccodealp", "q_hat", "n_dims_available"]]


# ──────────────────────────────────────────────────────────────────────────────
# 3. SIMEX: within-country temporal variance from WGI raw Excel
# ──────────────────────────────────────────────────────────────────────────────

def estimate_wgi_temporal_variance(wgi_path: Path) -> pd.DataFrame:
    """
    Estimate within-country temporal variance of governance composite.
    Used as σ²_w (noise proxy) for SIMEX correction.

    Method: For each dimension sheet, filter to 'average of all' indicator rows
    (source-level normalized composites, 0-1 scale), average across sources per
    (country, year), then compute temporal variance per country over 2000-2021.

    Returns: DataFrame with columns [country_approx, sigma2_w, T_wgi]
    """
    dim_series_list = []

    for dim, sheet in zip(['va', 'pv', 'ge', 'rq', 'rl', 'cc'], WGI_SHEETS):
        df = pd.read_excel(wgi_path, sheet_name=sheet,
                           usecols=['econ_name', 'production_year', 'indicator',
                                    'value', 'minimum', 'maximum'])
        # Filter to source-level composites on 0-1 scale
        df = df[
            (df['indicator'].str.contains('average of all', case=False, na=False)) &
            (df['minimum'] == 0) & (df['maximum'] == 1) &
            (df['production_year'] >= WGI_WINDOW[0]) &
            (df['production_year'] <= WGI_WINDOW[1])
        ]
        df = df.dropna(subset=['value'])
        if len(df) == 0:
            continue

        # Average across sources per (country, year)
        cy = (df.groupby(['econ_name', 'production_year'])['value']
              .mean()
              .rename(f'q_{dim}')
              .reset_index())
        dim_series_list.append(cy.rename(columns={'econ_name': 'country',
                                                   'production_year': 'year',
                                                   f'q_{dim}': 'q_dim'}))

    if not dim_series_list:
        return pd.DataFrame(columns=['country_wgi', 'sigma2_w', 'T_wgi'])

    # Combine all dimensions: average across dims per (country, year)
    all_dims = pd.concat(dim_series_list, ignore_index=True)
    cy_comp  = (all_dims.groupby(['country', 'year'])['q_dim']
                .mean().rename('q_composite').reset_index())

    # Within-country temporal variance
    cs_var = (
        cy_comp.groupby('country')
        .agg(sigma2_w=('q_composite', 'var'), T_wgi=('q_composite', 'count'))
        .reset_index()
    )
    cs_var = cs_var[cs_var['T_wgi'] >= MIN_T_WGI]
    cs_var = cs_var.rename(columns={'country': 'country_wgi'})
    return cs_var


# ──────────────────────────────────────────────────────────────────────────────
# 4. NAME MATCHING
# ──────────────────────────────────────────────────────────────────────────────

# Manual mapping for common mismatches between World Bank and QoG names
WB_TO_QOG = {
    "Korea, Rep.":            "Korea (the Republic of)",
    "Venezuela, RB":          "Venezuela (Bolivarian Republic of)",
    "Egypt, Arab Rep.":       "Egypt",
    "Iran, Islamic Rep.":     "Iran (Islamic Republic of)",
    "Syrian Arab Republic":   "Syrian Arab Republic",
    "Yemen, Rep.":            "Yemen",
    "Congo, Rep.":            "Congo",
    "Congo, Dem. Rep.":       "Congo (the Democratic Republic of the)",
    "Slovak Republic":        "Slovakia",
    "Kyrgyz Republic":        "Kyrgyzstan",
    "Lao PDR":                "Lao People's Democratic Republic (the)",
    "Micronesia, Fed. Sts.":  "Micronesia (Federated States of)",
    "Bolivia":                "Bolivia (Plurinational State of)",
    "Tanzania":               "Tanzania, the United Republic of",
    "Moldova":                "Moldova (the Republic of)",
    "North Macedonia":        "North Macedonia",
    "Czechia":                "Czechia",
    "Cote d'Ivoire":          "Cote d'Ivoire",
    "Bahamas, The":           "Bahamas (the)",
    "Gambia, The":            "Gambia (the)",
    "Korea, Dem. People's Rep.": "Korea (the Democratic People's Republic of)",
    "Brunei Darussalam":      "Brunei Darussalam",
    "Cabo Verde":             "Cabo Verde",
    "Sao Tome and Principe":  "Sao Tome and Principe",
    "Eswatini":               "Eswatini",
    "Hong Kong SAR, China":   "Hong Kong",
    "Macao SAR, China":       "Macao",
    "West Bank and Gaza":     "Palestine, State of",
    "Vietnam":                "Viet Nam",
    "Russia":                 "Russian Federation (the)",
    "Turkey":                 "Turkiye",
    "Netherlands":            "Netherlands (Kingdom of the)",
    "United States":          "United States of America (the)",
    "United Kingdom":         "United Kingdom of Great Britain and Northern Ireland (the)",
}


def match_countries(cs: pd.DataFrame, qog: pd.DataFrame) -> pd.DataFrame:
    """
    Merge primary panel with QoG governance scores.
    Uses manual mapping for common name mismatches.
    """
    qog_indexed = qog.set_index('cname')['q_hat'].to_dict()

    def lookup(name):
        # Direct match
        if name in qog_indexed:
            return qog_indexed[name]
        # Manual mapping
        mapped = WB_TO_QOG.get(name)
        if mapped and mapped in qog_indexed:
            return qog_indexed[mapped]
        return np.nan

    cs = cs.copy()
    cs['q_hat'] = cs['country'].apply(lookup)
    matched = cs.dropna(subset=['q_hat'])
    return matched


# ──────────────────────────────────────────────────────────────────────────────
# 5. OLS ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

def ols_estimate(merged: pd.DataFrame, q_col: str = 'q_std'):
    """OLS: mu_hat = mu0 + beta_I * I_bar + gamma * q_std + e."""
    X = np.column_stack([np.ones(len(merged)),
                         merged['I_bar'].values,
                         merged[q_col].values])
    y = merged['mu_hat'].values
    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    mu0, beta_I, gamma = coeffs
    resid = y - X @ coeffs
    N = len(merged)
    sigma_zeta = float(np.sqrt((resid**2).sum() / (N - 3)))
    r2 = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
    return {'mu0': mu0, 'beta_I': beta_I, 'gamma': gamma,
            'sigma_zeta': sigma_zeta, 'r2': r2, 'resid': resid}


# ──────────────────────────────────────────────────────────────────────────────
# 6. SIMEX CORRECTION
# ──────────────────────────────────────────────────────────────────────────────

def compute_lambda_q(merged: pd.DataFrame,
                     wgi_var_df: pd.DataFrame) -> dict:
    """
    Compute reliability ratio lambda_q for SIMEX correction.

    lambda_q = Var(q_i) / Var(q_hat_i)
             = (Var(q_hat_i) - sigma2_u) / Var(q_hat_i)

    sigma2_u: average within-country temporal variance of WGI composite,
    normalized to the same scale as q_hat.

    Since q_hat in merged is on the -2.5 to +2.5 standard WGI scale,
    but sigma2_w from raw WGI data is on the 0-1 source-normalized scale,
    we need a scale conversion.

    Scale conversion: WGI standard (-2.5 to +2.5) has typical sigma_q ~ 0.9,
    while the 0-1 source composite has typical sigma_q ~ 0.14.
    Ratio: sigma_WGI_scale / sigma_raw_scale ≈ 0.9 / 0.14 ≈ 6.4.
    sigma2_u_WGI_scale ≈ sigma2_u_raw_scale × (0.9/0.14)^2 ≈ sigma2_u_raw × 41.

    In standardized q_std units (mean=0, std=1):
    sigma2_u_std = sigma2_u_WGI / Var(q_hat_WGI)

    Returns dict with lambda_q, sigma2_u estimates.
    """
    # Within-country temporal variance on raw 0-1 scale
    # Average over countries available in wgi_var_df
    if len(wgi_var_df) == 0:
        # Fallback: use literature-based lambda_q (Kaufmann et al. 2010)
        lambda_q = 0.75
        sigma2_u_std = 0.25  # 1 - lambda_q
        sigma2_u_raw = np.nan
        note = "fallback (literature-based)"
    else:
        sigma2_w_mean = float(wgi_var_df['sigma2_w'].median())  # median robust to outliers

        # Cross-sectional variance of q_hat on WGI standard scale
        var_q_WGI = float(merged['q_hat'].var(ddof=1))

        # Scale conversion: raw 0-1 to WGI standard
        # We compute this empirically from the data cross-section std
        # sigma_q_WGI (standard scale) vs sigma_q_raw (0-1 scale)
        # Approximated from literature: WGI std dev across countries ~ 0.9
        sigma_q_WGI  = float(merged['q_hat'].std(ddof=1))
        sigma_q_raw  = 0.14   # typical for source-normalized 0-1 composite
        scale_ratio2 = (sigma_q_WGI / sigma_q_raw) ** 2

        # sigma2_u on WGI standard scale
        sigma2_u_WGI = sigma2_w_mean * scale_ratio2

        # Reliability ratio (must be in (0, 1))
        lambda_q = max(0.01, min(0.99,
                        (var_q_WGI - sigma2_u_WGI) / var_q_WGI))
        sigma2_u_raw = sigma2_w_mean
        sigma2_u_std = sigma2_u_WGI / var_q_WGI
        note = f"from WGI temporal variation (N={len(wgi_var_df)} countries)"

    return {'lambda_q': lambda_q,
            'sigma2_u_raw': sigma2_u_raw,
            'sigma2_u_std': sigma2_u_std,
            'note': note}


# ──────────────────────────────────────────────────────────────────────────────
# 7. BOOTSTRAP CI
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_gamma(merged: pd.DataFrame,
                    lambda_q: float,
                    B: int = B_BOOTSTRAP,
                    seed: int = SEED) -> dict:
    """Pairs bootstrap CI for (beta_I, gamma_EV, gamma_SIMEX)."""
    rng = np.random.default_rng(seed)
    n = len(merged)
    boot_beta = np.empty(B)
    boot_gev  = np.empty(B)
    boot_gs   = np.empty(B)

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        mb  = merged.iloc[idx].copy()
        q_m = mb['q_hat'].mean();  q_s = mb['q_hat'].std(ddof=1)
        if q_s < 1e-10:
            boot_beta[b] = boot_gev[b] = boot_gs[b] = np.nan
            continue
        mb['q_std'] = (mb['q_hat'] - q_m) / q_s
        try:
            res = ols_estimate(mb)
            boot_beta[b] = res['beta_I']
            boot_gev[b]  = res['gamma']
            # Re-estimate lambda_q would be ideal but is expensive;
            # use fixed lambda_q from full sample for SIMEX in bootstrap
            boot_gs[b]   = res['gamma'] / lambda_q
        except Exception:
            boot_beta[b] = boot_gev[b] = boot_gs[b] = np.nan

    ci = lambda a: (float(np.nanquantile(a, 0.025)),
                    float(np.nanquantile(a, 0.975)))

    pt = ols_estimate(merged)
    return {
        'beta_I':      (pt['beta_I'],  *ci(boot_beta)),
        'gamma_EV':    (pt['gamma'],   *ci(boot_gev)),
        'gamma_SIMEX': (pt['gamma'] / lambda_q, *ci(boot_gs)),
        'mu0':          pt['mu0'],
        'sigma_zeta':  pt['sigma_zeta'],
        'r2':           pt['r2'],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

def gram_check(merged: pd.DataFrame) -> dict:
    I  = merged['I_bar'].values
    q  = merged['q_std'].values
    G  = np.array([[np.var(I, ddof=1), np.cov(I, q)[0,1]],
                   [np.cov(I, q)[0,1], np.var(q, ddof=1)]])
    sp = float(merged[['I_bar','q_hat']].corr(method='spearman').iloc[0,1])
    return {
        'N':          len(merged),
        'det_G':      float(np.linalg.det(G)),
        'rho_S':      sp,
        'VIF':        1 / (1 - sp**2) if abs(sp) < 0.99 else np.inf,
    }


def variance_decomp(merged: pd.DataFrame, beta_I: float,
                    gamma: float) -> dict:
    sigma_mu = merged['mu_hat'].std(ddof=1)
    var_I  = merged['I_bar'].var(ddof=1)
    var_q  = merged['q_std'].var(ddof=1)   # ~1 by construction
    cov_IQ = np.cov(merged['I_bar'], merged['q_std'])[0,1]
    explained = (beta_I**2 * var_I + gamma**2 * var_q + 2*beta_I*gamma*cov_IQ)
    gamma_max = sigma_mu / np.sqrt(var_q)
    return {
        'sigma_mu_pct': sigma_mu * 100,
        'explained_fraction': explained / sigma_mu**2,
        'gamma_max_pct': gamma_max * 100,
        'exceeds_bound': abs(gamma) > gamma_max,
    }


def trap_thresholds(gamma: float, q_mean: float, q_std_val: float,
                    beta_I: float, mu0: float) -> dict:
    """Compute q* and q-dagger in standardized and raw WGI units."""
    # Trap threshold: drift = 0 at world avg investment
    I_world = I_BAR_WORLD_PCT / 100.0
    q_star_std  = -(mu0 + beta_I * I_world) / gamma if abs(gamma) > 1e-10 else np.nan
    q_dag_std   = -(mu0 + beta_I * 0.40) / gamma    if abs(gamma) > 1e-10 else np.nan
    q_star_raw  = q_star_std * q_std_val + q_mean
    q_dag_raw   = q_dag_std  * q_std_val + q_mean
    return {
        'q_star_std': q_star_std, 'q_star_raw': q_star_raw,
        'q_dag_std':  q_dag_std,  'q_dag_raw':  q_dag_raw,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 70

    print(SEP)
    print("DESM V2 -- Group IV: Point Identification of gamma")
    print("OQ-2 Resolution: WGI Panel Merge + SIMEX Correction")
    print(SEP)

    # 1. Primary panel
    print("\n[1] Building primary panel cross-section...")
    cs = build_cross_section(PANEL_CSV)
    print(f"    N = {len(cs)} countries (>= {MIN_T_OBS} obs)")
    print(f"    mu_hat: mean = {cs.mu_hat.mean()*100:.3f}%/yr, "
          f"std = {cs.mu_hat.std()*100:.3f}%/yr")
    print(f"    I_bar:  mean = {cs.I_bar.mean()*100:.2f}%, "
          f"std = {cs.I_bar.std()*100:.2f}%")

    # 2. OLS without governance (Panel A)
    X_a = np.column_stack([np.ones(len(cs)), cs['I_bar'].values])
    c_a, _, _, _ = lstsq(X_a, cs['mu_hat'].values, rcond=None)
    print(f"\n    Panel A OLS (no governance):")
    print(f"    mu0 = {c_a[0]*100:.3f}%/yr, beta_I = {c_a[1]:.4f} "
          f"(= {c_a[1]*100:.4f}%/yr per 1% investment)")
    print(f"    [Note: 'beta_I x 0.01 = {c_a[1]*0.01*100:.4f}%/yr per 1pp investment']")

    # 3. QoG governance
    print("\n[2] Loading WGI composite from QoG...")
    qog = load_qog_wgi(QOG_PATH)
    print(f"    N = {len(qog)} countries in QoG")
    print(f"    q_hat range: {qog.q_hat.min():.3f} to {qog.q_hat.max():.3f}")

    # 4. Merge
    print("\n[3] Merging primary panel with governance...")
    merged = match_countries(cs, qog)
    print(f"    N = {len(merged)} countries matched")

    # Standardize q_hat
    q_mean    = merged['q_hat'].mean()
    q_std_val = merged['q_hat'].std(ddof=1)
    merged['q_std'] = (merged['q_hat'] - q_mean) / q_std_val
    print(f"    q_hat on WGI scale: mean = {q_mean:.3f}, std = {q_std_val:.3f}")

    # 5. Gram matrix diagnostics
    print("\n[4] Gram matrix / collinearity diagnostics...")
    gram = gram_check(merged)
    print(f"    Spearman rho(I, q)  = {gram['rho_S']:.3f}")
    print(f"    det(G)              = {gram['det_G']:.4e}  (> 0 => identified)")
    print(f"    VIF                 = {gram['VIF']:.3f}  (< 2 => no collinearity)")

    # 6. OLS (Panel B)
    print("\n[5] OLS with governance (Panel B)...")
    pt = ols_estimate(merged)
    print(f"    mu0        = {pt['mu0']*100:.3f}%/yr")
    print(f"    beta_I^WGI = {pt['beta_I']:.4f}  (= {pt['beta_I']*0.01*100:.4f}%/yr per 1pp investment)")
    print(f"    gamma^EV   = {pt['gamma']*100:.4f}%/yr per z-score WGI")
    print(f"    sigma_zeta = {pt['sigma_zeta']*100:.4f}%/yr")
    print(f"    R^2        = {pt['r2']:.3f}")

    # 7. SIMEX: temporal variance from raw WGI
    print("\n[6] Estimating reliability ratio (SIMEX)...")
    wgi_var = estimate_wgi_temporal_variance(WGI_PATH)
    lq_info = compute_lambda_q(merged, wgi_var)
    lambda_q = lq_info['lambda_q']
    gamma_SIMEX = pt['gamma'] / lambda_q
    print(f"    WGI temporal variance countries: {len(wgi_var)}")
    if len(wgi_var) > 0:
        print(f"    Median within-country var (0-1 scale): {wgi_var['sigma2_w'].median():.5f}")
    print(f"    Reliability ratio lambda_q  = {lambda_q:.4f}  ({lq_info['note']})")
    print(f"    gamma^SIMEX = gamma^EV / lambda_q = {gamma_SIMEX*100:.4f}%/yr per z-score")

    # 8. Bootstrap CI
    print(f"\n[7] Pairs bootstrap (B = {B_BOOTSTRAP})...")
    boot = bootstrap_gamma(merged, lambda_q, B=B_BOOTSTRAP, seed=SEED)
    b_pt, b_lo, b_hi   = boot['beta_I']
    gev_pt, gev_lo, gev_hi = boot['gamma_EV']
    gs_pt, gs_lo, gs_hi    = boot['gamma_SIMEX']

    print(f"    beta_I^WGI  = {b_pt:.4f} [95% CI: {b_lo:.4f}, {b_hi:.4f}]")
    print(f"    (pp/yr per pp invest: {b_pt*0.01*100:.4f} [{b_lo*0.01*100:.4f}, {b_hi*0.01*100:.4f}])")
    print(f"    gamma^EV    = {gev_pt*100:.4f}%/yr  [95% CI: {gev_lo*100:.4f}, {gev_hi*100:.4f}]")
    print(f"    gamma^SIMEX = {gs_pt*100:.4f}%/yr  [95% CI: {gs_lo*100:.4f}, {gs_hi*100:.4f}]")

    # 9. Variance decomposition check
    print("\n[8] Variance decomposition...")
    vd = variance_decomp(merged, b_pt, gs_pt)
    print(f"    sigma_mu (data)  = {vd['sigma_mu_pct']:.3f}%/yr")
    print(f"    Fraction explained by (I, q): {vd['explained_fraction']:.3f}")
    print(f"    gamma_max (bound) = {vd['gamma_max_pct']:.4f}%/yr per z-score")
    print(f"    gamma^SIMEX exceeds bound: {vd['exceeds_bound']}")

    # 10. Regime thresholds
    print("\n[9] Regime thresholds at gamma^SIMEX...")
    th = trap_thresholds(gs_pt, q_mean, q_std_val, b_pt, boot['mu0'])
    print(f"    q* (trap threshold, WGI std z-score): {th['q_star_std']:.3f}")
    print(f"    q* (WGI standard scale):              {th['q_star_raw']:.3f}")
    print(f"    q-dagger (deep trap, z-score):        {th['q_dag_std']:.3f}")
    print(f"    q-dagger (WGI standard scale):        {th['q_dag_raw']:.3f}")

    # 11. Country-level check
    print("\n[10] Country-level verification (Korea, Venezuela):")
    merged['mu_hat_full'] = (boot['mu0'] + b_pt * merged['I_bar']
                             + gs_pt * merged['q_std'])
    merged['zeta_hat'] = merged['mu_hat'] - merged['mu_hat_full']
    target_countries = {
        'Korea, Rep.':    'KOR',
        'Venezuela, RB':  'VEN',
        'China':          'CHN',
        'United States':  'USA',
        'Japan':          'JPN',
        'Nigeria':        'NGA',
    }
    for wb_name, iso in target_countries.items():
        row = merged[merged['country'] == wb_name]
        if len(row) == 0:
            # Try by ISO mapping
            continue
        r = row.iloc[0]
        print(f"    {wb_name[:20]:20s} | mu={r.mu_hat*100:+.2f}%"
              f" | full={r.mu_hat_full*100:+.2f}%"
              f" | zeta={r.zeta_hat*100:+.2f}%"
              f" | q_WGI={r.q_hat:.2f}"
              f" | regime guess: {'R4' if r.mu_hat<0 else 'R1-3'}")

    # 12. MRS
    if abs(b_pt) > 1e-6:
        mrs_per_pp = gs_pt / b_pt  # z-score WGI per unit I = z-score WGI per 100pp invest
        mrs_per_1pp = mrs_per_pp * 0.01  # z-score per 1pp invest
        print(f"\n[11] Policy frontier MRS:")
        print(f"    gamma/beta_I = {gs_pt/b_pt:.4f} z-score per unit I")
        print(f"    = {1/mrs_per_1pp:.1f} pp investment per z-score WGI")
        print(f"    [Range at 95% CI: {1/(gs_hi/b_lo):.1f}--{1/(gs_lo/b_hi):.1f} pp per z-score]")

    # ── LATEX SUMMARY ──────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("LATEX UPDATE SUMMARY")
    print(SEP)
    print(f"\n  N (Panel B)       = {len(merged)}")
    print(f"  Spearman rho(I,q) = {gram['rho_S']:.3f}")
    print(f"  VIF               = {gram['VIF']:.3f}")
    print(f"  R^2               = {boot['r2']:.3f}")
    print()
    print(f"  mu0 (WGI panel)   = {boot['mu0']*100:.3f}%/yr")
    print(f"  beta_I^WGI        = {b_pt:.4f} fraction units"
          f"  = {b_pt*0.01*100:.4f}%/yr per 1pp investment")
    print(f"    95% CI: [{b_lo*0.01*100:.4f}, {b_hi*0.01*100:.4f}]%/yr per 1pp")
    print()
    print(f"  gamma^EV          = {gev_pt*100:.4f}%/yr per z-score WGI")
    print(f"    95% CI: [{gev_lo*100:.4f}, {gev_hi*100:.4f}]")
    print()
    print(f"  lambda_q          = {lambda_q:.4f}")
    print(f"  gamma^SIMEX       = {gs_pt*100:.4f}%/yr per z-score WGI")
    print(f"    95% CI: [{gs_lo*100:.4f}, {gs_hi*100:.4f}]")
    print()
    print(f"  sigma_zeta        = {boot['sigma_zeta']*100:.4f}%/yr")
    print(f"  gamma_max (bound) = {vd['gamma_max_pct']:.4f}%/yr")
    print(f"  q* (WGI std)      = {th['q_star_std']:.3f}")
    print(f"  q-dag (WGI std)   = {th['q_dag_std']:.3f}")
    if abs(b_pt) > 1e-6:
        mrs_display = gs_pt / (b_pt * 0.01)
        print(f"  MRS               = {mrs_display:.1f} pp investment per z-score WGI")
    print(SEP)

    return merged, boot, lq_info, vd, th


if __name__ == '__main__':
    merged, boot, lq_info, vd, th = main()
