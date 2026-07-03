"""
rc_lp_finite_sample_mc.py -- Monte Carlo check: does finite-T mean-group LP
estimation, applied to a KNOWN homogeneous AR(2) DGP with the real sample's
T_i and N, reproduce the empirical collapse from psi_inf=1.344 to
psi_hat_20^MG ~ 0.3, with zero specification error and zero true
heterogeneity?

If yes: the magnitude gap is a finite-sample LP-estimator artifact, not a
phi1-heterogeneity effect (which rc_irf_reconciliation.py already showed is
too small at the realistic calibration to matter) and not evidence against
A1/A3.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy import stats

PHI1, PHI2 = 0.194, 0.062
T_SIM   = 62          # matches the N=93 real sample (T_i in [59,62], modal 62)
N_COUNTRIES = 93
H_MAX   = 20
SIGMA_EPS = 0.046      # = sd(eps_hat) in the real N=93 sample (pooled AR(2) resid.)
N_REPS  = 500
RNG = np.random.default_rng(20260630)

def simulate_country(T, phi1, phi2, sigma, rng):
    g = np.zeros(T)
    burn = 50
    gb = np.zeros(T + burn)
    eps = rng.normal(0, sigma, T + burn)
    for t in range(2, T + burn):
        gb[t] = phi1*gb[t-1] + phi2*gb[t-2] + eps[t]
    return gb[burn:], eps[burn:]

def lp_country(g, eps_true, phi1, phi2, hmax):
    # Same generated-regressor LP as lp_irf_country.py: eps_hat constructed
    # from the (correctly specified, known) phi1, phi2 -- so eps_hat = eps_true
    # exactly here (zero specification error case).
    T = len(g)
    ly = np.cumsum(g)              # log-level analogue (x_t), arbitrary origin
    dly = np.r_[np.nan, np.diff(np.r_[0.0, ly])]  # dlog_y_t = g_t (by construction)
    dly = g.copy()
    eps_hat = eps_true.copy()
    psi_c = np.full(hmax, np.nan)
    for h in range(1, hmax+1):
        rows_X, rows_y = [], []
        for t in range(2, T - h):
            rows_X.append([1.0, eps_hat[t], dly[t-1], dly[t-2]])
            rows_y.append(ly[t+h] - ly[t])
        X, y = np.array(rows_X), np.array(rows_y)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        psi_c[h-1] = b[1]
    return psi_c

def cir_path(phi1, phi2, hmax):
    vals = [1.0]
    for j in range(1, hmax):
        v = phi1*vals[-1] + (phi2*vals[-2] if j >= 2 else 0.0)
        vals.append(v)
    return np.cumsum(np.array(vals))

true_cir = cir_path(PHI1, PHI2, H_MAX)
print(f"True theoretical CIR path (phi1={PHI1}, phi2={PHI2}):")
print(f"  CIR(h=1)={true_cir[0]:.3f}  CIR(h=10)={true_cir[9]:.3f}  "
      f"CIR(h=20)={true_cir[19]:.3f}  (asymptote {1/(1-PHI1-PHI2):.3f})")

mg_reps = np.full((N_REPS, H_MAX), np.nan)
for r in range(N_REPS):
    psi_mat = np.full((N_COUNTRIES, H_MAX), np.nan)
    for c in range(N_COUNTRIES):
        g, eps = simulate_country(T_SIM, PHI1, PHI2, SIGMA_EPS, RNG)
        psi_mat[c] = lp_country(g, eps, PHI1, PHI2, H_MAX)
    mg_reps[r] = np.nanmean(psi_mat, axis=0)

mg_mean = mg_reps.mean(axis=0)
mg_sd   = mg_reps.std(axis=0)

print("\n" + "="*78)
print(f"MONTE CARLO: {N_REPS} replications, N={N_COUNTRIES} countries, T={T_SIM}, "
      "ZERO heterogeneity, ZERO spec. error")
print("="*78)
print(f"{'h':>3}  {'True CIR':>9}  {'Simulated MG mean':>18}  {'MC sd':>7}  {'Attenuation %':>14}")
for h in range(1, H_MAX+1):
    atten = 100*(1 - mg_mean[h-1]/true_cir[h-1])
    print(f"{h:>3}  {true_cir[h-1]:>9.3f}  {mg_mean[h-1]:>18.3f}  {mg_sd[h-1]:>7.3f}  {atten:>13.1f}%")

print(f"\nReal-data empirical MG psi_20 = 0.311 (SE=0.108)")
print(f"Simulated finite-T MG psi_20 (homogeneous DGP, zero het.) "
      f"= {mg_mean[19]:.3f} (MC sd={mg_sd[19]:.3f})")
z = (0.311 - mg_mean[19]) / mg_sd[19]
print(f"z-score of real-data estimate vs simulated finite-T null: {z:.2f}")
