"""
rc_irf_reconciliation.py -- Addresses MC-4: reconcile the LP-IRF magnitude gap
(psi_hat_20^MG = 0.311 vs theoretical psi_inf = 1.455) with a calibrated
random-coefficients (RC) AR(2) extension (OQ-1: g_it = phi1_i g_{t-1}
+ phi2 g_{t-2} + eps_it, phi2 common).

Protocol:
  1. Reuse the lp_irf_country.py sample (N=93 countries, T_i in [59,62]).
  2. Estimate phi1_i per country by OLS with phi2=0.05 fixed (no intercept,
     g already within-country demeaned): g_it - phi2*g_{t-2} = phi1_i*g_{t-1} + e.
  3. Decompose cross-country Var(phi1_i_hat) into true heterogeneity
     variance + average sampling-noise variance (method-of-moments / a
     noisy-measurement variance decomposition).
  4. Build the theoretical homogeneous AR(2) CIR_h path (phi1=0.263,phi2=0.05).
  5. Build the theoretical RC mean CIR path two ways:
       (A) plug-in: mean_i CIR_h(phi1_i_hat^BC)  [contaminated by estimation noise]
       (B) calibrated: E[CIR_h(phi1)] under phi1 ~ N(phi1_bar, sigma_true^2),
           truncated to the stationarity region phi1+phi2<1, Monte Carlo.
  6. Test (B) against the empirical MG path (psi_h^MG, se_h^MG) from
     lp_irf_country.py: pointwise z-test at h=20 and a reduced-horizon
     (h=5,10,15,20) joint chi-square concordance check.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Resolved relative to this file's location (article/code/), three levels
# up to the repository root, so the script runs unmodified on any
# machine/clone.
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "panel_causal.csv"
PHI1_POOLED_BC = 0.194   # bias-corrected pooled phi1 (M06 line 316; corrected per rem:phi1_gap)
PHI2_FIXED     = 0.062   # common phi2 (M06/RC-2c; corrected per rem:phi1_gap)
H_MAX  = 20
MIN_T  = 35
RNG    = np.random.default_rng(20260630)
N_MC   = 200_000

AGGREGATES = {
    "Africa Eastern and Southern", "Africa Western and Central",
    "Arab World", "Caribbean small states", "Central Europe and the Baltics",
    "Early-demographic dividend", "East Asia & Pacific",
    "East Asia & Pacific (IDA & IBRD countries)",
    "East Asia & Pacific (excluding high income)", "Europe & Central Asia",
    "Europe & Central Asia (IDA & IBRD countries)",
    "Europe & Central Asia (excluding high income)", "European Union",
    "Fragile and conflict affected situations",
    "Heavily indebted poor countries (HIPC)", "High income",
    "IBRD only", "IDA & IBRD total", "IDA blend", "IDA only", "IDA total",
    "Late-demographic dividend",
    "Latin America & Caribbean",
    "Latin America & Caribbean (excluding high income)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Least developed countries: UN classification",
    "Low & middle income", "Low income", "Lower middle income",
    "Middle East & North Africa",
    "Middle East & North Africa (IDA & IBRD countries)",
    "Middle East & North Africa (excluding high income)",
    "Middle income", "OECD members", "Other small states",
    "Pacific island small states", "Post-demographic dividend",
    "Pre-demographic dividend", "Small states", "South Asia",
    "South Asia (IDA & IBRD)", "Sub-Saharan Africa",
    "Sub-Saharan Africa (IDA & IBRD countries)",
    "Sub-Saharan Africa (excluding high income)", "Upper middle income",
    "World",
}

# ── 1. Load and clean (identical to lp_irf_country.py) ────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[~df["Country Name"].isin(AGGREGATES)].copy()
df = df.sort_values(["Country Name", "year"]).reset_index(drop=True)
df["log_y"]  = np.log(df["gdp_pc"])
df = df.dropna(subset=["log_y"]).copy()
df["dlog_y"] = df.groupby("Country Name")["log_y"].diff()
df["mu_i"]   = df.groupby("Country Name")["dlog_y"].transform("mean")
df["g_it"]   = df["dlog_y"] - df["mu_i"]

# ── 2. Country-level phi1_i (phi2 fixed) ──────────────────────────────────────
results = []
for cname, cd in df.groupby("Country Name"):
    cd = cd.sort_values("year").reset_index(drop=True)
    Ti = cd["log_y"].notna().sum()
    if Ti < MIN_T + H_MAX + 4:        # same inclusion rule as lp_irf_country.py
        continue
    g = cd["g_it"].values
    y_dep, x_reg = [], []
    for t in range(2, len(g)):
        if np.any(np.isnan([g[t], g[t-1], g[t-2]])):
            continue
        y_dep.append(g[t] - PHI2_FIXED * g[t-2])
        x_reg.append(g[t-1])
    y_dep, x_reg = np.array(y_dep), np.array(x_reg)
    n = len(y_dep)
    if n < 20:
        continue
    Sxx = np.sum(x_reg**2)
    phi1_hat = np.sum(x_reg * y_dep) / Sxx
    resid = y_dep - phi1_hat * x_reg
    sigma2 = np.sum(resid**2) / (n - 1)
    se_phi1 = np.sqrt(sigma2 / Sxx)
    phi1_bc = phi1_hat + (1 + phi1_hat) / (Ti - 1)   # Nickell-type correction, M06 l.313-315
    results.append(dict(country=cname, Ti=Ti, n=n, phi1_hat=phi1_hat,
                         se_phi1=se_phi1, phi1_bc=phi1_bc))

R = pd.DataFrame(results)
print(f"Countries with country-level phi1_i estimated: {len(R)}")

# ── 3. Variance decomposition ─────────────────────────────────────────────────
raw_mean   = R["phi1_hat"].mean()
raw_var    = R["phi1_hat"].var(ddof=1)
bc_mean    = R["phi1_bc"].mean()
bc_var     = R["phi1_bc"].var(ddof=1)
mean_samp_var = (R["se_phi1"]**2).mean()
true_het_var  = max(0.0, bc_var - mean_samp_var)
sigma_true    = np.sqrt(true_het_var)

print("\n" + "="*72)
print("CROSS-COUNTRY phi1_i DISTRIBUTION (phi2=0.05 fixed)")
print("="*72)
print(f"  N countries:                 {len(R)}")
print(f"  Mean phi1_hat (raw):         {raw_mean:.3f}")
print(f"  Mean phi1_bc (Nickell-corr): {bc_mean:.3f}   (pooled reference: {PHI1_POOLED_BC})")
print(f"  Cross-country SD, raw:       {np.sqrt(raw_var):.3f}")
print(f"  Cross-country SD, bc:        {np.sqrt(bc_var):.3f}")
print(f"  Mean sampling SD (noise):    {np.sqrt(mean_samp_var):.3f}")
print(f"  Implied TRUE het. SD sigma_phi1: {sigma_true:.3f}")
print(f"  5th/25th/50th/75th/95th pct (phi1_bc): "
      f"{np.percentile(R['phi1_bc'],[5,25,50,75,95]).round(3)}")

# ── 4. Theoretical CIR paths ───────────────────────────────────────────────────
def cir_path(phi1, phi2, hmax=H_MAX):
    psi = np.zeros(hmax)
    psi_prev2, psi_prev1 = None, None
    vals = [1.0]
    for j in range(1, hmax):
        if j == 1:
            v = phi1 * vals[-1]
        else:
            v = phi1 * vals[-1] + phi2 * vals[-2]
        vals.append(v)
    vals = np.array(vals)           # psi_0..psi_{hmax-1}
    cir = np.cumsum(vals)           # CIR_0..CIR_{hmax-1}  (CIR_{h-1} <-> horizon h)
    return cir

cir_homog = cir_path(PHI1_POOLED_BC, PHI2_FIXED)   # homogeneous benchmark
print("\n" + "="*72)
print("THEORETICAL HOMOGENEOUS AR(2) CIR PATH (phi1=0.263, phi2=0.05)")
print("="*72)
for h in range(1, H_MAX+1):
    print(f"  h={h:>2}: CIR_(h-1) = {cir_homog[h-1]:.4f}")
print(f"  Asymptote psi_inf = {1/(1-PHI1_POOLED_BC-PHI2_FIXED):.4f}")

# ── 4b. RC theoretical path: (A) plug-in mean of country-level estimates ───────
cir_plugin = np.mean([cir_path(p, PHI2_FIXED) for p in R["phi1_bc"]], axis=0)

# ── 4c. RC theoretical path: (B) calibrated Monte Carlo, noise-corrected sigma ─
draws = RNG.normal(bc_mean, sigma_true, size=N_MC)
draws = draws[draws + PHI2_FIXED < 0.995]   # stationarity truncation
draws = draws[draws > -0.99]
batch = 5000
acc2 = np.zeros(H_MAX)
for i in range(0, len(draws), batch):
    chunk = draws[i:i+batch]
    psis = np.ones((len(chunk), H_MAX))
    for j in range(1, H_MAX):
        if j == 1:
            psis[:, j] = chunk * psis[:, j-1]
        else:
            psis[:, j] = chunk * psis[:, j-1] + PHI2_FIXED * psis[:, j-2]
    cir_chunk = np.cumsum(psis, axis=1)
    acc2 += cir_chunk.sum(axis=0)
cir_calibrated = acc2 / len(draws)

print("\n" + "="*72)
print("RC THEORETICAL MEAN CIR PATH")
print("="*72)
print(f"{'h':>3}  {'Homog.':>8}  {'RC plug-in':>10}  {'RC calibrated':>14}")
for h in range(1, H_MAX+1):
    print(f"{h:>3}  {cir_homog[h-1]:>8.3f}  {cir_plugin[h-1]:>10.3f}  {cir_calibrated[h-1]:>14.3f}")

# ── 5. Empirical MG path (rerun lp_irf_country logic inline) ──────────────────
def ar2_resid(grp):
    g = grp["g_it"].values.copy()
    e = np.full(len(g), np.nan)
    for t in range(2, len(g)):
        if not any(np.isnan([g[t], g[t-1], g[t-2]])):
            e[t] = g[t] - PHI1_POOLED_BC*g[t-1] - PHI2_FIXED*g[t-2]
    return pd.Series(e, index=grp.index)

df["eps_hat"] = df.groupby("Country Name", group_keys=False).apply(ar2_resid)
country_list = df["Country Name"].unique()
psi_mat = {}
for cname in country_list:
    cd = df[df["Country Name"] == cname].copy().reset_index(drop=True)
    Ti = cd["log_y"].notna().sum()
    if Ti < MIN_T + H_MAX + 4:
        continue
    ly  = cd["log_y"].values
    eps = cd["eps_hat"].values
    dly = cd["dlog_y"].values
    psi_c = np.full(H_MAX, np.nan)
    for h in range(1, H_MAX + 1):
        rows_X, rows_y = [], []
        for t in range(2, len(ly) - h):
            if any(np.isnan([eps[t], dly[t-1], dly[t-2], ly[t], ly[t+h]])):
                continue
            rows_X.append([1.0, eps[t], dly[t-1], dly[t-2]])
            rows_y.append(ly[t+h] - ly[t])
        if len(rows_y) < 6:
            continue
        X, y = np.array(rows_X), np.array(rows_y)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        psi_c[h-1] = b[1]
    if not np.all(np.isnan(psi_c)):
        psi_mat[cname] = psi_c

M = np.array(list(psi_mat.values()))
mg_psi = np.nanmean(M, axis=0)
Nvalid = np.sum(~np.isnan(M), axis=0)
mg_se  = np.nanstd(M, axis=0) / np.sqrt(Nvalid)

# ── 6. Formal concordance test: empirical MG vs calibrated RC theory ──────────
print("\n" + "="*72)
print("EMPIRICAL MG vs RC-CALIBRATED THEORETICAL PATH")
print("="*72)
print(f"{'h':>3}  {'MG_emp':>8}  {'SE_MG':>7}  {'RC_theory':>9}  {'z=(MG-RC)/SE':>13}")
zvals = {}
for h in [1,5,10,15,20]:
    z = (mg_psi[h-1] - cir_calibrated[h-1]) / mg_se[h-1]
    zvals[h] = z
    print(f"{h:>3}  {mg_psi[h-1]:>8.3f}  {mg_se[h-1]:>7.3f}  {cir_calibrated[h-1]:>9.3f}  {z:>13.2f}")

chi2_stat = sum(zvals[h]**2 for h in [5,10,15,20])
p_chi2 = 1 - stats.chi2.cdf(chi2_stat, df=4)
print(f"\nJoint (h=5,10,15,20) chi2 concordance stat = {chi2_stat:.2f} (df=4), p={p_chi2:.3f}")
print("[NOTE: ignores cross-horizon correlation from overlapping LP windows;")
print(" reported as a directional concordance check, not a calibrated test.]")

z20 = zvals[20]
p20_two = 2*(1 - stats.norm.cdf(abs(z20)))
print(f"\nPointwise h=20 test: z = {z20:.2f}, two-sided p = {p20_two:.3f}")
print(f"  Empirical MG psi_20    = {mg_psi[19]:.3f} (SE={mg_se[19]:.3f})")
print(f"  RC-calibrated theory   = {cir_calibrated[19]:.3f}")
print(f"  Homogeneous theory     = {cir_homog[19]:.3f}")
print(f"  Gap closed by RC ext.: {(cir_calibrated[19]-1.0)/(cir_homog[19]-1.0)*100:.0f}% "
      "of distance from CIR_0=1 to homogeneous asymptote")
