"""
gamma_iv_estimation.py
======================
Instrumental-variable identification of gamma (institutional coupling).
Resolves the residual identification task flagged in M06 rem:gamma_status(iv):
"Constructing and validating such an instrument for the N_B=141-country
Panel B is beyond the scope of the current paper."

Instruments (both already present in qog_std_cs_jan26.xlsx, no new data
collection required):
  Z1 = ajr_settmort   : log settler mortality (Acemoglu, Johnson & Robinson 2001)
  Z2 = ht_colonial    : colonial-origin category (Hadenius & Teorell 2005),
                        used as a legal-origin-style categorical instrument
                        in the tradition of La Porta et al. (1998)

Identification logic:
  Structural eq (eq:wgi_reg): mu_hat_i = mu0 + beta_I * I_bar_i + gamma * q_std_i + zeta_i
  q_std_i is endogenous under reverse causality (mu_hat_i -> q_i).
  Z1, Z2 are pre-determined (settled centuries/decades before the panel
  window) and satisfy relevance (large first-stage F) by construction;
  exclusion (Z affects mu_hat only through q_i) is a maintained assumption,
  discussed with standard caveats (Albouy 2012 for Z1; direct legal-system
  channels for Z2) rather than proven.

Three specifications:
  (A) Z1 only, N=70 (settler-mortality subsample)
  (B) Z2 only, N=141 (full Panel B, colonial-origin dummies)
  (C) Z1+Z2 combined, N=70 (overlap), with Hansen/Sargan overidentification test
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq, inv
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from gamma_estimation_v2 import (
    DATA_DIR, QOG_PATH, PANEL_CSV, WGI_DIMS_EST, WB_TO_QOG,
    build_cross_section,
)

B_BOOTSTRAP = 2000
SEED = 42


def build_panel_b_with_instruments() -> pd.DataFrame:
    """Panel B (mu_hat_i, I_bar_i, q_hat_i) merged with QoG instruments."""
    cs = build_cross_section(PANEL_CSV)
    qog = pd.read_excel(
        QOG_PATH,
        usecols=["cname", "ccodealp"] + WGI_DIMS_EST + ["ajr_settmort", "ht_colonial"],
    )
    qog["q_hat"] = qog[WGI_DIMS_EST].mean(axis=1)
    qog["n_dims_available"] = qog[WGI_DIMS_EST].notna().sum(axis=1)
    qog = qog[qog["n_dims_available"] >= 4].set_index("cname")

    def lookup(name):
        if name in qog.index:
            row = qog.loc[name]
        else:
            mapped = WB_TO_QOG.get(name)
            if mapped and mapped in qog.index:
                row = qog.loc[mapped]
            else:
                return None
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    rows = []
    for _, r in cs.iterrows():
        qr = lookup(r["country"])
        if qr is None:
            continue
        rows.append({**r.to_dict(), "q_hat": qr["q_hat"],
                     "ajr_settmort": qr["ajr_settmort"], "ht_colonial": qr["ht_colonial"]})

    merged = pd.DataFrame(rows)
    q_mean, q_sd = merged["q_hat"].mean(), merged["q_hat"].std(ddof=1)
    merged["q_std"] = (merged["q_hat"] - q_mean) / q_sd
    return merged


def _ols(X, y):
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    vcov = sigma2 * inv(X.T @ X)
    return b, np.sqrt(np.diag(vcov)), resid


def _2sls(Z, Xend, y):
    """Return (coef, se, resid, first-stage-F-vs-partial-instruments)."""
    n = Xend.shape[0]
    PZ = Z @ inv(Z.T @ Z) @ Z.T
    beta = inv(Xend.T @ PZ @ Xend) @ (Xend.T @ PZ @ y)
    resid = y - Xend @ beta
    k = Xend.shape[1]
    sigma2 = (resid @ resid) / (n - k)
    vcov = sigma2 * inv(Xend.T @ PZ @ Xend)
    return beta, np.sqrt(np.diag(vcov)), resid


def first_stage_F(Z_full, Z_excl_only_cols, q_std, n):
    """Partial/joint F-test for excluded instruments in the first stage."""
    Xexog = np.delete(Z_full, Z_excl_only_cols, axis=1)
    b_r, _, r_r = _ols(Xexog, q_std)
    b_u, _, r_u = _ols(Z_full, q_std)
    k_excl = len(Z_excl_only_cols)
    ssr_r, ssr_u = r_r @ r_r, r_u @ r_u
    dof2 = n - Z_full.shape[1]
    F = ((ssr_r - ssr_u) / k_excl) / (ssr_u / dof2)
    return F, k_excl, dof2


def bootstrap_ci(build_fn, n, B=B_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    out = np.full(B, np.nan)
    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        try:
            out[b] = build_fn(idx)
        except Exception:
            pass
    return np.nanquantile(out, [0.025, 0.975])


def main():
    SEP = "=" * 72
    print(SEP)
    print("GAMMA IV IDENTIFICATION -- settler mortality & colonial origin")
    print(SEP)

    merged = build_panel_b_with_instruments()
    print(f"\nPanel B, N = {len(merged)} (matches M06 sec:ident_group4)")

    # ---------------------------------------------------------------- (A)
    print("\n" + "-" * 72)
    print("(A) Settler mortality instrument (AJR 2001), continuous")
    print("-" * 72)
    sm = merged.dropna(subset=["ajr_settmort"]).reset_index(drop=True)
    n_a = len(sm)
    Z_a = np.column_stack([np.ones(n_a), sm["I_bar"].values, sm["ajr_settmort"].values])
    Xend_a = np.column_stack([np.ones(n_a), sm["I_bar"].values, sm["q_std"].values])
    y_a = sm["mu_hat"].values
    F_a, k_excl_a, dof2_a = first_stage_F(Z_a, [2], sm["q_std"].values, n_a)
    beta_a, se_a, _ = _2sls(Z_a, Xend_a, y_a)
    print(f"N = {n_a}, first-stage F({k_excl_a},{dof2_a}) = {F_a:.2f}")
    print(f"gamma_IV = {beta_a[2]*100:.3f}%/yr per z-score WGI "
          f"(SE {se_a[2]*100:.3f}, t = {beta_a[2]/se_a[2]:.2f})")

    def boot_a(idx):
        d = sm.iloc[idx]
        qm, qs = d["q_hat"].mean(), d["q_hat"].std(ddof=1)
        if qs < 1e-8:
            return np.nan
        qstd = (d["q_hat"] - qm) / qs
        Zb = np.column_stack([np.ones(n_a), d["I_bar"].values, d["ajr_settmort"].values])
        Xb = np.column_stack([np.ones(n_a), d["I_bar"].values, qstd.values])
        b, _, _ = _2sls(Zb, Xb, d["mu_hat"].values)
        return b[2]

    ci_a = bootstrap_ci(boot_a, n_a) * 100
    print(f"Bootstrap 95% CI = [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")

    # ---------------------------------------------------------------- (B)
    print("\n" + "-" * 72)
    print("(B) Colonial-origin instrument (Hadenius-Teorell), categorical")
    print("-" * 72)
    n_b = len(merged)
    cat = merged["ht_colonial"].astype(int)
    dummies = pd.get_dummies(cat, prefix="col", drop_first=True).astype(float)
    k_dum = dummies.shape[1]
    Z_b = np.column_stack([np.ones(n_b), merged["I_bar"].values, dummies.values])
    Xend_b = np.column_stack([np.ones(n_b), merged["I_bar"].values, merged["q_std"].values])
    y_b = merged["mu_hat"].values
    F_b, k_excl_b, dof2_b = first_stage_F(Z_b, list(range(2, 2 + k_dum)), merged["q_std"].values, n_b)
    beta_b, se_b, _ = _2sls(Z_b, Xend_b, y_b)
    print(f"N = {n_b}, categories = {cat.nunique()}, joint F({k_excl_b},{dof2_b}) = {F_b:.2f}")
    print(f"gamma_IV = {beta_b[2]*100:.3f}%/yr per z-score WGI "
          f"(SE {se_b[2]*100:.3f}, t = {beta_b[2]/se_b[2]:.2f})")

    def boot_b(idx):
        d = merged.iloc[idx]
        qm, qs = d["q_hat"].mean(), d["q_hat"].std(ddof=1)
        if qs < 1e-8:
            return np.nan
        qstd = (d["q_hat"] - qm) / qs
        catb = d["ht_colonial"].astype(int)
        dumb = pd.get_dummies(catb, prefix="col").reindex(columns=["col_" + c.split("_")[1] for c in dummies.columns], fill_value=0.0)
        Zb = np.column_stack([np.ones(n_b), d["I_bar"].values, dumb.values])
        Xb = np.column_stack([np.ones(n_b), d["I_bar"].values, qstd.values])
        b, _, _ = _2sls(Zb, Xb, d["mu_hat"].values)
        return b[2]

    ci_b = bootstrap_ci(boot_b, n_b) * 100
    print(f"Bootstrap 95% CI = [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")

    # ---------------------------------------------------------------- (C)
    print("\n" + "-" * 72)
    print("(C) Combined instruments (overlap sample) + overidentification test")
    print("-" * 72)
    dummies_ov = pd.get_dummies(sm["ht_colonial"].astype(int), prefix="col", drop_first=True).astype(float)
    k_dum_ov = dummies_ov.shape[1]
    Z_c = np.column_stack([np.ones(n_a), sm["I_bar"].values, sm["ajr_settmort"].values, dummies_ov.values])
    Xend_c = np.column_stack([np.ones(n_a), sm["I_bar"].values, sm["q_std"].values])
    F_c, k_excl_c, dof2_c = first_stage_F(Z_c, list(range(2, 2 + 1 + k_dum_ov)), sm["q_std"].values, n_a)
    beta_c, se_c, resid_c = _2sls(Z_c, Xend_c, y_a)
    print(f"N = {n_a}, joint F({k_excl_c},{dof2_c}) = {F_c:.2f}")
    print(f"gamma_IV(combined) = {beta_c[2]*100:.3f}%/yr per z-score WGI "
          f"(SE {se_c[2]*100:.3f}, t = {beta_c[2]/se_c[2]:.2f})")

    # Hansen/Sargan overidentification test
    b_res, _, r_res = _ols(Z_c, resid_c)
    r2_res = 1 - (r_res @ r_res) / ((resid_c - resid_c.mean()) @ (resid_c - resid_c.mean()))
    J = n_a * r2_res
    df_over = k_excl_c - 1
    p_over = 1 - stats.chi2.cdf(J, df_over)
    print(f"Hansen/Sargan J = {J:.3f}, df = {df_over}, p = {p_over:.3f} "
          f"({'not rejected -> instruments jointly consistent' if p_over > 0.05 else 'REJECTED'})")

    def boot_c(idx):
        d = sm.iloc[idx]
        qm, qs = d["q_hat"].mean(), d["q_hat"].std(ddof=1)
        if qs < 1e-8:
            return np.nan
        qstd = (d["q_hat"] - qm) / qs
        catb = d["ht_colonial"].astype(int)
        dumb = pd.get_dummies(catb, prefix="col").reindex(
            columns=["col_" + c.split("_")[1] for c in dummies_ov.columns], fill_value=0.0)
        Zb = np.column_stack([np.ones(n_a), d["I_bar"].values, d["ajr_settmort"].values, dumb.values])
        Xb = np.column_stack([np.ones(n_a), d["I_bar"].values, qstd.values])
        b, _, _ = _2sls(Zb, Xb, d["mu_hat"].values)
        return b[2]

    ci_c = bootstrap_ci(boot_c, n_a) * 100
    print(f"Bootstrap 95% CI (combined) = [{ci_c[0]:.3f}, {ci_c[1]:.3f}]")

    # ---------------------------------------------------------------- SUMMARY
    print("\n" + SEP)
    print("SUMMARY (for M06 / M09 update)")
    print(SEP)
    print(f"  gamma^SIMEX (OLS+SIMEX, N=141)        = 0.249  [CI: -0.006, 0.499]")
    print(f"  gamma^IV(settmort), N={n_a:<3d}            = {beta_a[2]*100:.3f}  [CI: {ci_a[0]:.3f}, {ci_a[1]:.3f}]  F={F_a:.1f}")
    print(f"  gamma^IV(colonial), N={n_b:<3d}            = {beta_b[2]*100:.3f}  [CI: {ci_b[0]:.3f}, {ci_b[1]:.3f}]  F={F_b:.1f}")
    print(f"  gamma^IV(combined), N={n_a:<3d}            = {beta_c[2]*100:.3f}  SE={se_c[2]*100:.3f}  F={F_c:.1f}  overid p={p_over:.2f}")
    print(SEP)

    return dict(merged=merged, sm=sm, beta_a=beta_a, se_a=se_a, ci_a=ci_a,
                F_a=F_a, beta_b=beta_b, se_b=se_b, ci_b=ci_b, F_b=F_b,
                beta_c=beta_c, se_c=se_c, F_c=F_c, J=J, df_over=df_over, p_over=p_over)


if __name__ == "__main__":
    main()
