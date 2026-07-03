"""
robustness_checks.py
=====================
W3 Resolution: Robustness checks for DESM V2.

Two checks:
  RC-1: Subsample parameter stability of beta_I (investment-growth coupling).
         Cross-country OLS mu_hat_i = mu0 + beta_I * I_bar_i + eta_i
         estimated for full sample and three sub-periods.

  RC-2: ARIMA(p,1,0) specification robustness.
         (a) IPS panel unit root test confirms I(1) in log GDP per capita.
         (b) Lag-order selection within ARIMA(p,1,0), p in {1,2,3},
             using AIC and BIC on the SAME dependent variable (differences).

Outputs: printed tables ready for LaTeX transcription.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq

warnings.filterwarnings("ignore")

# Resolved relative to this file's location (article/code/), three levels
# up to the repository root, so the script runs unmodified on any
# machine/clone.
DATA_DIR  = Path(__file__).parent.parent.parent / "data"
PANEL_CSV = DATA_DIR / "panel_causal.csv"

MIN_T_OBS = 20
MIN_T_AIC = 30

REGIONAL_KEYWORDS = [
    'World', 'income', 'East Asia', 'Europe', 'Latin America', 'Middle East',
    'North America', 'South Asia', 'Sub-Saharan', 'OECD', 'Euro area',
    'Africa Eastern', 'Africa Western', 'Small states', 'Pacific island',
    'Fragile', 'Heavily indebted', 'Least developed', 'Low &', 'Lower middle',
    'Upper middle', 'High income', 'Low income', 'Arab World', 'Caribbean',
    'Central Europe', 'IDA', 'IBRD', 'not classified', 'Other small',
]

SUB_PERIODS = [
    ('P1', 1960, 1980),
    ('P2', 1981, 2000),
    ('P3', 2001, 2021),
]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_panel():
    df = pd.read_csv(PANEL_CSV)
    df = df.rename(columns={"Country Name": "country"})
    df = df.dropna(subset=["gdp_pc", "gfcf_gdp", "year"])
    df = df.sort_values(["country", "year"])
    mask = ~df["country"].str.contains(
        "|".join(REGIONAL_KEYWORDS), case=False, regex=True, na=False
    )
    df = df[mask].copy()
    df["x1"]       = np.log(df["gdp_pc"].astype(float))
    df["delta_x1"] = df.groupby("country")["x1"].diff()
    df["I"]        = df["gfcf_gdp"].astype(float) / 100.0
    df["year"]     = df["year"].astype(int)
    return df


def cross_section(df, y_lo=1960, y_hi=2021, min_t=MIN_T_OBS):
    sub = df[(df["year"] >= y_lo) & (df["year"] <= y_hi)].copy()
    cs = (
        sub.dropna(subset=["delta_x1", "I"])
           .groupby("country")
           .agg(mu_hat=("delta_x1", "mean"),
                I_bar=("I", "mean"),
                T_obs=("delta_x1", "count"))
           .reset_index()
    )
    return cs[cs["T_obs"] >= min_t]


def ols_mu_vs_I(cs):
    X = np.column_stack([np.ones(len(cs)), cs["I_bar"].values])
    y = cs["mu_hat"].values
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k  = len(cs), 2
    s2    = (resid ** 2).sum() / (n - k)
    se    = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2    = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"mu0": b[0], "beta_I": b[1],
            "se_mu0": se[0], "se_beta": se[1],
            "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# RC-1: SUBSAMPLE PARAMETER STABILITY
# ---------------------------------------------------------------------------

def rc1_subsample_stability(df):
    SEP = "=" * 72
    print()
    print(SEP)
    print("RC-1: Subsample Parameter Stability of beta_I")
    print(SEP)

    cs_full  = cross_section(df)
    res_full = ols_mu_vs_I(cs_full)
    print(
        f"\nFull sample (1960-2021):  N={res_full['n']:3d}, "
        f"beta_I = {res_full['beta_I']:.4f}  SE={res_full['se_beta']:.4f}  "
        f"R2={res_full['r2']:.3f}"
    )

    results = []
    for lbl, y_lo, y_hi in SUB_PERIODS:
        cs = cross_section(df, y_lo, y_hi, min_t=15)
        if len(cs) < 10:
            print(f"  {lbl} ({y_lo}-{y_hi}): insufficient data (N={len(cs)})")
            continue
        res = ols_mu_vs_I(cs)
        results.append((lbl, y_lo, y_hi, res))
        t_stat = (res["beta_I"] - res_full["beta_I"]) / res["se_beta"]
        print(
            f"  {lbl} ({y_lo}-{y_hi}):  N={res['n']:3d}, "
            f"beta_I = {res['beta_I']:.4f}  SE={res['se_beta']:.4f}  "
            f"R2={res['r2']:.3f}  t(diff from full)={t_stat:+.2f}"
        )

    betas = [r["beta_I"] for _, _, _, r in results]
    print(f"\n  Range beta_I across sub-periods: [{min(betas):.4f}, {max(betas):.4f}]")
    print(
        f"  Max deviation from full-sample ({res_full['beta_I']:.4f}): "
        f"{max(abs(b - res_full['beta_I']) for b in betas):.4f}"
    )

    # Cross-section split by investment level
    cs_full2  = cross_section(df)
    med_I     = cs_full2["I_bar"].median()
    res_lo    = ols_mu_vs_I(cs_full2[cs_full2["I_bar"] <  med_I])
    res_hi    = ols_mu_vs_I(cs_full2[cs_full2["I_bar"] >= med_I])
    print(
        f"\n  Low-investment  (I_bar < {med_I*100:.1f}%):  "
        f"N={res_lo['n']:3d}, beta_I={res_lo['beta_I']:.4f}  SE={res_lo['se_beta']:.4f}"
    )
    print(
        f"  High-investment (I_bar >= {med_I*100:.1f}%): "
        f"N={res_hi['n']:3d}, beta_I={res_hi['beta_I']:.4f}  SE={res_hi['se_beta']:.4f}"
    )

    return results, res_full, med_I, res_lo, res_hi


# ---------------------------------------------------------------------------
# RC-2: ARIMA(p,1,0) SPECIFICATION ROBUSTNESS
# ---------------------------------------------------------------------------

def aic_crit(n, k, sse):
    s2 = sse / n
    return n * np.log(s2) + 2 * k if s2 > 0 else np.nan

def bic_crit(n, k, sse):
    s2 = sse / n
    return n * np.log(s2) + k * np.log(n) if s2 > 0 else np.nan


def fit_arima_p(series, p):
    """ARIMA(p,1,0): Dx_t = c + phi1*Dx_{t-1} + ... + phip*Dx_{t-p} + eps."""
    dx = np.diff(series)
    n  = len(dx)
    if n < p + 2:
        return np.nan, p + 1, 0
    y_reg = dx[p:]
    lags  = [dx[p - j - 1: n - j - 1] for j in range(p)]
    X_reg = np.column_stack([np.ones(n - p)] + lags)
    try:
        b, _, _, _ = lstsq(X_reg, y_reg, rcond=None)
        resid = y_reg - X_reg @ b
        sse   = float((resid ** 2).sum())
        return sse, p + 1, n - p
    except Exception:
        return np.nan, p + 1, 0


def rc2_spec_robustness(df):
    SEP = "=" * 72
    print()
    print(SEP)
    print("RC-2: ARIMA(p,1,0) Specification Robustness")
    print("  (a) IPS panel unit root   (b) Lag-order AIC/BIC in differences")
    print(SEP)

    records    = []
    adf_tstats = []

    for country, grp in df.groupby("country"):
        grp = grp.sort_values("year")
        x   = grp["x1"].dropna().values
        T   = len(x)
        if T < MIN_T_AIC:
            continue

        # (b) AIC/BIC for ARIMA(1,1,0), ARIMA(2,1,0), ARIMA(3,1,0)
        row = {"country": country, "T": T}
        fits = {}
        for p in [1, 2, 3]:
            sse, k, n_eff = fit_arima_p(x, p)
            if not np.isnan(sse) and n_eff > 0:
                fits[p] = {
                    "aic": aic_crit(n_eff, k, sse),
                    "bic": bic_crit(n_eff, k, sse),
                }
                row[f"aic_p{p}"] = fits[p]["aic"]
                row[f"bic_p{p}"] = fits[p]["bic"]

        if len(fits) < 2:
            continue
        row["best_aic_p"] = min(fits, key=lambda p: fits[p]["aic"])
        row["best_bic_p"] = min(fits, key=lambda p: fits[p]["bic"])
        records.append(row)

        # (a) ADF t-stat: Dx_t = alpha + rho*x_{t-1} + phi*Dx_{t-1} + eps
        dx = np.diff(x)
        if len(dx) >= 4:
            y_adf = dx[1:]
            X_adf = np.column_stack([np.ones(len(y_adf)), x[1:-1], dx[:-1]])
            try:
                b_adf, _, _, _ = lstsq(X_adf, y_adf, rcond=None)
                resid_adf = y_adf - X_adf @ b_adf
                n_adf     = len(y_adf)
                s2_adf    = (resid_adf ** 2).sum() / (n_adf - 3)
                XtX_inv   = np.linalg.inv(X_adf.T @ X_adf)
                se_rho    = np.sqrt(s2_adf * XtX_inv[1, 1])
                t_rho     = b_adf[1] / se_rho
                adf_tstats.append(t_rho)
            except Exception:
                pass

    recs = pd.DataFrame(records)
    N_c  = len(recs)
    print(f"\n  Countries with T >= {MIN_T_AIC}: {N_c}")

    # (a) IPS W-bar
    W_bar = np.nan
    if len(adf_tstats) >= 10:
        t_arr = np.array(adf_tstats)
        n_ips = len(t_arr)
        # IPS (2003) Table 1: ADF(1) with intercept, T~60: E[t]=-1.52, Var[t]=0.82
        E_t, Var_t = -1.52, 0.82
        W_bar  = (t_arr.mean() - E_t) / np.sqrt(Var_t / n_ips)
        reject = W_bar < -1.645
        print(f"\n  (a) IPS Panel Unit Root (Im-Pesaran-Shin 2003)")
        print(f"      N={n_ips}, mean ADF t={t_arr.mean():.3f}"
              f"  (E[t] under H0={E_t:.2f})")
        print(f"      W-bar={W_bar:.3f}  |  critical value (5%) = -1.645")
        print(f"      Decision: {'Reject H0' if reject else 'Fail to reject H0 (I(1) confirmed)'}")

    # (b) Lag-order selection
    if len(recs) > 0 and "best_aic_p" in recs.columns:
        for crit in ["aic", "bic"]:
            col = f"best_{crit}_p"
            counts = {p: (recs[col] == p).sum() for p in [1, 2, 3]}
            total  = sum(counts.values())
            print(f"\n  (b) Best {crit.upper()} order:  "
                  + "  ".join(
                      f"p={p}: {counts[p]} ({100*counts[p]/total:.1f}%)"
                      for p in [1, 2, 3]
                  ))
        # AIC/BIC gain of p=2 over p=1
        if "aic_p1" in recs.columns and "aic_p2" in recs.columns:
            gain_aic = (recs["aic_p1"] - recs["aic_p2"]).dropna()
            gain_bic = (recs["bic_p1"] - recs["bic_p2"]).dropna()
            print(
                f"\n  Median AIC gain of ARIMA(2,1,0) over ARIMA(1,1,0): "
                f"{gain_aic.median():.2f}  (n={len(gain_aic)})"
            )
            print(
                f"  Median BIC gain of ARIMA(2,1,0) over ARIMA(1,1,0): "
                f"{gain_bic.median():.2f}"
            )
            pct_p2_better_aic = 100 * (gain_aic > 0).mean()
            print(f"  Fraction countries where ARIMA(2,1,0) beats (1,1,0) by AIC: "
                  f"{pct_p2_better_aic:.1f}%")

    return recs, W_bar


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading panel...")
    df = load_panel()
    print(f"  {df['country'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}, "
          f"{len(df)} obs")

    results, res_full, med_I, res_lo, res_hi = rc1_subsample_stability(df)
    recs, W_bar = rc2_spec_robustness(df)

    # --- LaTeX-ready summary ---
    SEP = "=" * 72
    print()
    print(SEP)
    print("LATEX SUMMARY -- transcribe into M08 sec:val_robustness")
    print(SEP)
    print()
    print("RC-1 (beta_I subsample stability):")
    full_b = res_full['beta_I']
    print(f"  Full 1960-2021: N={res_full['n']}, "
          f"beta_I={full_b*100:.3f}%/yr per pp invest, "
          f"SE={res_full['se_beta']*100:.3f}, R2={res_full['r2']:.3f}")
    for lbl, y_lo, y_hi, res in results:
        print(f"  {lbl} ({y_lo}-{y_hi}): N={res['n']}, "
              f"beta_I={res['beta_I']*100:.3f}%/yr, "
              f"SE={res['se_beta']*100:.3f}, "
              f"|dev|={abs(res['beta_I']-full_b)*100:.3f}%/yr")
    print(f"  Low-I:  N={res_lo['n']}, beta_I={res_lo['beta_I']*100:.3f}%/yr")
    print(f"  High-I: N={res_hi['n']}, beta_I={res_hi['beta_I']*100:.3f}%/yr")

    print()
    print("RC-2:")
    print(f"  IPS W-bar = {W_bar:.3f}  (H0: I(1); fail to reject => ARIMA justified)")
    if len(recs) > 0 and "best_aic_p" in recs.columns:
        N_c = len(recs)
        n2_aic = (recs["best_aic_p"] == 2).sum()
        n2_bic = (recs["best_bic_p"] == 2).sum()
        print(f"  ARIMA(2,1,0) wins AIC in {n2_aic}/{N_c} ({100*n2_aic/N_c:.1f}%) countries")
        print(f"  ARIMA(2,1,0) wins BIC in {n2_bic}/{N_c} ({100*n2_bic/N_c:.1f}%) countries")
        if "aic_p1" in recs.columns and "aic_p2" in recs.columns:
            gain = (recs["aic_p1"] - recs["aic_p2"]).dropna()
            print(f"  Median AIC gain (2 over 1): {gain.median():.2f}")
            print(f"  Pct countries ARIMA(2,1,0)>ARIMA(1,1,0) by AIC: "
                  f"{100*(gain>0).mean():.1f}%")


if __name__ == "__main__":
    main()
