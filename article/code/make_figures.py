"""
make_figures.py
Publication-quality figures for DESM V2.1
Target: Journal of Econometrics / Journal of Economic Growth

Generates:
  fig1_distributions.pdf  – FC-1 (log-normal cross-section) + FC-4 (heavy tails)
  fig2_investment_acf.pdf – FC-5 (investment-growth) + FC-9 (ACF profile)
  fig3_lp_irf.pdf         – T1  (local projection IRF)
  fig4_korea_venezuela.pdf– Case studies: Korea & Venezuela
  fig5_variance_drift.pdf – T2  (variance divergence) + FC-10 (drift distribution)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
# ROOT is resolved relative to this file's location (article/code/), three
# levels up to the repository root, so the script runs unmodified on any
# machine/clone. Output figures are not shipped in the repository (the
# article PDF already contains the rendered versions); this script
# (re)creates FIGS on demand.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "panel_causal.csv")
FIGS = os.path.join(ROOT, "article", "figures")
os.makedirs(FIGS, exist_ok=True)

# ── matplotlib style (publication-ready) ───────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "axes.linewidth":     0.7,
    "lines.linewidth":    1.1,
    "patch.linewidth":    0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "legend.frameon":     False,
    "legend.handlelength":1.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "text.usetex":        False,
})

C_LIGHT  = "#cccccc"
C_MED    = "#888888"
C_DARK   = "#444444"
C_BLACK  = "#000000"


# ── load panel ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df["log_gdp_pc"] = np.log(df["gdp_pc"].replace(0, np.nan))

# country-level aggregates  (≥ 20 obs)
country = (
    df.groupby("Country Name")
    .agg(
        mu_hat   = ("gdp_growth",  "mean"),
        invest   = ("gfcf_gdp",    "mean"),
        credit   = ("priv_credit", "mean"),
        log_gdp  = ("log_gdp_pc",  "mean"),
        area     = ("land_area",   "first"),
        n        = ("gdp_growth",  "count"),
    )
    .query("n >= 20")
    .dropna(subset=["mu_hat"])
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Cross-sectional distribution (FC-1) and growth distribution (FC-4)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

# ── Panel (a): log-GDP cross-section ──
ax = axes[0]
cs = country["log_gdp"].dropna()
mu_cs, sig_cs = cs.mean(), cs.std()

ax.hist(cs, bins=26, density=True,
        color=C_LIGHT, edgecolor=C_BLACK, linewidth=0.45, zorder=2)
xr = np.linspace(cs.min() - 0.3, cs.max() + 0.3, 300)
ax.plot(xr, norm.pdf(xr, mu_cs, sig_cs), color=C_BLACK, lw=1.2,
        label=fr"$\mathcal{{N}}({mu_cs:.2f},\,{sig_cs:.2f}^2)$", zorder=3)

ax.set_xlabel(r"$\log y_i$ — log GDP per capita (constant 2015 USD)")
ax.set_ylabel("Density")
ax.set_title(r"(a) Cross-sectional distribution of $\log y_i$ (FC-1)", pad=4)
ax.legend(loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.97, 0.97,
        f"$N={len(cs)}$ countries\nSkewness $=0.20$\nExcess kurtosis $=-0.86$",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

# ── Panel (b): GDP growth distribution with Gaussian overlay ──
ax = axes[1]
g = df["gdp_growth"].dropna()
g_trim = g[(g > -25) & (g < 35)]
mu_g, sig_g = g.mean(), g.std()

ax.hist(g_trim, bins=60, density=True,
        color=C_LIGHT, edgecolor=C_BLACK, linewidth=0.25, zorder=2)
xg = np.linspace(-25, 35, 400)
ax.plot(xg, norm.pdf(xg, mu_g, sig_g), color=C_BLACK, lw=1.2,
        label=r"$\mathcal{N}(\hat{\mu},\hat{\sigma}^2)$", zorder=3)

ax.set_xlabel(r"Annual GDP per capita growth rate (\%)")
ax.set_ylabel("Density")
ax.set_title(r"(b) GDP growth distribution — excess kurtosis $=54$ (FC-4)", pad=4)
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.97, 0.97,
        f"$N={len(g):,}$ obs.\n"
        fr"$\hat{{\sigma}}={sig_g:.2f}\%$" + "\nGaussian underfits tails",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

plt.tight_layout(pad=0.9)
plt.savefig(os.path.join(FIGS, "fig1_distributions.pdf"), bbox_inches="tight")
plt.close()
print("fig1_distributions.pdf  OK")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Investment-growth scatter (FC-5) and ACF profile (FC-9)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

# ── Panel (a): between-country investment vs. average growth ──
ax = axes[0]
cs2 = country.dropna(subset=["mu_hat", "invest"])
xs  = cs2["invest"].values
ys  = cs2["mu_hat"].values

slope, intercept, r, pv, se_slope = stats.linregress(xs, ys)
xfit = np.linspace(xs.min(), xs.max(), 200)

ax.scatter(xs, ys, s=7, color=C_MED, alpha=0.55, linewidths=0, zorder=2)
ax.plot(xfit, intercept + slope * xfit, color=C_BLACK, lw=1.3, zorder=3,
        label=fr"OLS slope $={slope:.3f}$ pp/pp" "\n" r"$r_B=0.404$, $N=" f"{len(cs2)}$")
ax.axhline(0, color=C_BLACK, lw=0.5, ls="--", alpha=0.6)

ax.set_xlabel(r"Mean investment rate $\bar{I}_i$ (\% of GDP)")
ax.set_ylabel(r"Mean annual growth rate $\hat{\mu}_i$ (\%/yr)")
ax.set_title(r"(a) Investment--growth relationship — between-country (FC-5)", pad=4)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# ── Panel (b): ACF of GDP growth ──
ax = axes[1]
lags   = np.arange(1, 8)
rho_obs = np.array([0.207, 0.102, 0.075, 0.046, 0.031, 0.021, 0.015])

phi1, phi2 = 0.263, 0.05
rho_ar1 = phi1 ** lags

# AR(2) theoretical ACF via Yule-Walker recursion
rho_yw  = np.zeros(8)
rho_yw[0] = 1.0
rho_yw[1] = phi1 / (1.0 - phi2)
for k in range(2, 8):
    rho_yw[k] = phi1 * rho_yw[k-1] + phi2 * rho_yw[k-2]
rho_ar2 = rho_yw[1:]

w = 0.22
ax.bar(lags - w,   rho_obs, w, color=C_DARK,  label="Observed (pooled)")
ax.bar(lags,       rho_ar1, w, color=C_MED,   label=fr"AR(1) predicted ($\hat\phi_1={phi1}$)")
ax.bar(lags + w,   rho_ar2, w, color=C_LIGHT, edgecolor=C_BLACK, linewidth=0.4,
       label=fr"AR(2) predicted ($\hat\phi_1={phi1}$, $\hat\phi_2={phi2}$)")
ax.axhline(0, color=C_BLACK, lw=0.5)

ax.set_xlabel("Lag $k$ (years)")
ax.set_ylabel("Autocorrelation $\\hat{\\rho}_k$")
ax.set_title(r"(b) ACF of demeaned GDP growth: observed vs.\ model (FC-9)", pad=4)
ax.set_xticks(lags)
ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=0.9)
plt.savefig(os.path.join(FIGS, "fig2_investment_acf.pdf"), bbox_inches="tight")
plt.close()
print("fig2_investment_acf.pdf  OK")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Local Projection IRF (Test T1)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.75))

h    = np.arange(1, 21)
psi  = np.array([0.155, 0.298, 0.362, 0.404, 0.439, 0.450, 0.436, 0.408,
                 0.369, 0.332, 0.346, 0.288, 0.120, 0.057, 0.003, -0.056,
                 -0.114, -0.113, -0.155, -0.238])
se   = np.array([0.038, 0.054, 0.057, 0.060, 0.066, 0.074, 0.085, 0.092,
                 0.093, 0.096, 0.094, 0.080, 0.089, 0.086, 0.088, 0.089,
                 0.090, 0.093, 0.099, 0.101])
ci_lo = psi - 1.96 * se
ci_hi = psi + 1.96 * se

ax.fill_between(h, ci_lo, ci_hi, color=C_MED, alpha=0.20, label="95\\% CI")
ax.plot(h, psi, "ko-", ms=3.5, lw=1.2, label=r"LP-IRF $\hat{\psi}_h$", zorder=4)
ax.axhline(0,     color=C_BLACK, lw=0.55, ls="--", alpha=0.7)
ax.axhline(1.344, color=C_DARK,  lw=0.8,  ls=":",
           label=r"$\hat{\psi}_\infty^{\mathrm{th}}=1.344$ (AR(2))")
ax.axvline(6, color=C_MED, lw=0.5, ls="--", alpha=0.6)

ax.annotate(r"Peak $h=6$", xy=(6, 0.450), xytext=(7.5, 0.50),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))

ax.set_xlabel("Horizon $h$ (years)")
ax.set_ylabel("Cumulative log-GDP response to unit shock")
ax.set_title("Local projection IRF — unit growth shock (T1)", pad=4)
ax.set_xlim(0.3, 20.7)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=0.8)
plt.savefig(os.path.join(FIGS, "fig3_lp_irf.pdf"), bbox_inches="tight")
plt.close()
print("fig3_lp_irf.pdf  OK")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Korea vs Venezuela: trajectories and drift decomposition
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

# ── extract country series ──
kor_name = "Korea, Rep."
ven_name = "Venezuela, RB"

# Korea: actual log-GDP per capita (constant 2015 USD)
kor = (df[df["Country Name"] == kor_name]
       .set_index("year").sort_index().dropna(subset=["gdp_pc"]))
kor["log_y"] = np.log(kor["gdp_pc"])

# Venezuela: no gdp_pc levels in panel; reconstruct from cumulative growth rates.
# Base: log(5500) ~ 8.61, consistent with Venezuela ~$5,500 in 1961 (2015 USD).
ven_g = (df[df["Country Name"] == ven_name]
         .set_index("year").sort_index()
         .dropna(subset=["gdp_growth"])
         .sort_index())
ven_g["log_y"] = np.nan
ven_base_year = ven_g.index.min()          # first year with growth data
ven_log0 = np.log(5500.0)                  # historical estimate for Venezuela 1961
ven_g.loc[ven_base_year, "log_y"] = ven_log0
for yr in ven_g.index[1:]:
    prev_yr = ven_g.index[ven_g.index.get_loc(yr) - 1]
    ven_g.loc[yr, "log_y"] = (ven_g.loc[prev_yr, "log_y"]
                               + ven_g.loc[yr, "gdp_growth"] / 100.0)

# ── Panel (a): log-GDP trajectories with fitted drifts ──
ax = axes[0]
ax.plot(kor.index, kor["log_y"], color=C_BLACK, lw=1.2,
        label=r"Korea (Rep.) — $\hat{\mu}=5.68\%$/yr")
ax.plot(ven_g.index, ven_g["log_y"], color=C_DARK, lw=1.2, ls="--",
        label=r"Venezuela — $\hat{\mu}=-2.10\%$/yr")

# overlay structural drift lines
t0_k = int(kor.index.min());  y0_k = kor["log_y"].iloc[0]
te_k = int(kor.index.max())
t0_v = int(ven_g.index.min()); y0_v = ven_g["log_y"].iloc[0]
te_v = int(ven_g.index.max())

ax.plot([t0_k, te_k], [y0_k, y0_k + 0.0568 * (te_k - t0_k)],
        color=C_BLACK, lw=0.7, ls=":", alpha=0.65, label="_nolegend_")
ax.plot([t0_v, te_v], [y0_v, y0_v - 0.0210 * (te_v - t0_v)],
        color=C_DARK,  lw=0.7, ls=":", alpha=0.65, label="_nolegend_")

ax.set_xlabel("Year")
ax.set_ylabel(r"$\log y_{it}$ (constant 2015 USD, index)")
ax.set_title(r"(a) Log-GDP per capita trajectories, 1960--2021", pad=4)
ax.legend(fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)

# ── Panel (b): drift decomposition bar chart at gamma_max = 1.80 ──
ax = axes[1]
gamma_max = 1.80
q_kor,  q_ven  = +0.75, -0.97
I_kor,  I_ven  = 28.65, 23.10
mu0, beta_I    = -0.990, 0.127  # country-level OLS intercept (Sec. val_t3), matching
                                 # the gamma_max case-country residuals reported there

mu_inv_kor = mu0 + beta_I * I_kor          #  2.65
mu_inv_ven = mu0 + beta_I * I_ven          #  1.94
gov_kor    = gamma_max * q_kor             #  1.35
gov_ven    = gamma_max * q_ven             # -1.75
res_kor    = 5.68 - mu_inv_kor - gov_kor   # +1.68  (absorbs country-specific)
res_ven    = -2.10 - mu_inv_ven - gov_ven  # -2.30

labels = ["Investment\n"
          r"$\hat{\mu}^{\mathrm{inv}}=\mu_0+\hat{\beta}_I\bar{I}_i$",
          "Governance\n"
          r"$\hat{\gamma}_{\max}\bar{q}_i$",
          "Residual\n"
          r"$\hat{\zeta}_i$",
          "Observed\n"
          r"$\hat{\mu}_i$"]

kor_vals = [mu_inv_kor, gov_kor, res_kor, 5.68]
ven_vals = [mu_inv_ven, gov_ven, res_ven, -2.10]

x = np.arange(len(labels))
w = 0.30
bk = ax.bar(x - w/2, kor_vals, w, color=C_MED,   label="Korea")
bv = ax.bar(x + w/2, ven_vals, w, color=C_LIGHT, edgecolor=C_BLACK, lw=0.5,
            label="Venezuela")
ax.axhline(0, color=C_BLACK, lw=0.6)

for bar, val in zip(list(bk) + list(bv), kor_vals + ven_vals):
    yoff = 0.12 if val >= 0 else -0.25
    ax.text(bar.get_x() + bar.get_width()/2, val + yoff,
            f"{val:+.2f}", ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"Growth rate (\%/yr)")
ax.set_title(r"(b) Drift decomposition at $\hat{\gamma}_{\max}=1.80$ (T3)", pad=4)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=0.9)
plt.savefig(os.path.join(FIGS, "fig4_korea_venezuela.pdf"), bbox_inches="tight")
plt.close()
print("fig4_korea_venezuela.pdf  OK")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Variance divergence (T2) and drift distribution (FC-10)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

# ── Panel (a): sub-panel cross-sectional standard deviation ──
ax = axes[0]
periods   = ["$P_1$\n1960--1980", "$P_2$\n1981--2000", "$P_3$\n2001--2021"]
sigma_sub = [1.346, 1.433, 1.418]
sigma_pred_p3 = 1.346 + 0.237   # theoretical prediction for 30-yr increment

colors_bar = [C_LIGHT, C_MED, C_DARK]
bars = ax.bar(periods, sigma_sub, color=colors_bar,
              edgecolor=C_BLACK, lw=0.7, width=0.45, zorder=3)
ax.axhline(sigma_pred_p3, color=C_BLACK, lw=0.9, ls=":",
           label=fr"Predicted $P_3$: $P_1+\Delta\hat{{\sigma}}(30)\approx{sigma_pred_p3:.3f}$",
           zorder=4)
ax.axhline(sigma_sub[0], color=C_BLACK, lw=0.6, ls="--", alpha=0.55,
           label=fr"$P_1$ baseline: {sigma_sub[0]:.3f}")

for bar, val in zip(bars, sigma_sub):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.007,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylim(1.27, 1.64)
ax.set_ylabel(r"$\hat{\sigma}(\log y_{it})$ — cross-sectional std.\ dev.")
ax.set_title(r"(a) Structural wealth divergence — sub-panel test (T2)", pad=4)
ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)

# ── Panel (b): drift distribution ──
ax = axes[1]
mu_hat = country["mu_hat"].dropna()
mu_bar, sig_mu = 2.0, 1.55
xm = np.linspace(-5, 9, 400)

ax.hist(mu_hat, bins=28, density=True,
        color=C_LIGHT, edgecolor=C_BLACK, lw=0.45, zorder=2,
        label=r"Empirical $\hat{\mu}_i$")
ax.plot(xm, norm.pdf(xm, mu_bar, sig_mu), color=C_BLACK, lw=1.2, zorder=3,
        label=fr"$\mathcal{{N}}({mu_bar}\%,\,{sig_mu}\%^2)$ (A2)")
ax.axvline(0, color=C_BLACK, lw=0.7, ls="--", alpha=0.75, label=r"$\mu_i=0$")

n_neg = (mu_hat < 0).sum()
ax.text(0.97, 0.97,
        f"$N={len(mu_hat)}$ countries\n"
        fr"$\hat{{n}}_{{\mu<0}}={n_neg}$ ({100*n_neg/len(mu_hat):.1f}\%)\n"
        f"Skewness $=0.93$",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

ax.set_xlabel(r"Long-run drift $\hat{\mu}_i$ (\%/yr)")
ax.set_ylabel("Density")
ax.set_title(r"(b) Cross-country drift distribution $\hat{\mu}_i$ (FC-10)", pad=4)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=0.9)
plt.savefig(os.path.join(FIGS, "fig5_variance_drift.pdf"), bbox_inches="tight")
plt.close()
print("fig5_variance_drift.pdf  OK")

print("\nAll 5 figures generated successfully.")
