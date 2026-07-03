"""
rc_pooled_heterogeneity_mc.py -- Completes the MC-4 diagnostic chain.

rc_lp_finite_sample_mc.py showed that finite-T LP/MG estimation on a
HOMOGENEOUS, correctly-specified AR(2) DGP (N=93, T=62) already produces
severe attenuation (simulated MG psi_20 ~ -0.12), fully accounting for the
empirical MG estimate (0.311) without invoking heterogeneity.

This script asks the complementary question: does adding REALISTIC
phi1_i heterogeneity (sigma=0.148, calibrated in rc_irf_reconciliation.py)
specifically push the POOLED (entity-FE, single common coefficient)
estimator further negative than the MG estimator, as the Pesaran-Smith
(1995) mechanism predicts? This isolates the PS effect from the baseline
finite-T LP attenuation common to both estimators.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

PHI1_BAR, SIGMA_PHI1 = 0.227, 0.145   # calibrated cross-country phi1_i moments
PHI2 = 0.062
T_SIM, N_COUNTRIES, H_MAX = 62, 93, 20
SIGMA_EPS = 0.046
N_REPS = 300
RNG = np.random.default_rng(20260630)

def simulate_country(T, phi1, phi2, sigma, rng):
    burn = 50
    gb = np.zeros(T + burn)
    eps = rng.normal(0, sigma, T + burn)
    for t in range(2, T + burn):
        gb[t] = phi1*gb[t-1] + phi2*gb[t-2] + eps[t]
    return gb[burn:], eps[burn:]

def cir_path(phi1, phi2, hmax):
    vals = [1.0]
    for j in range(1, hmax):
        v = phi1*vals[-1] + (phi2*vals[-2] if j >= 2 else 0.0)
        vals.append(v)
    return np.cumsum(np.array(vals))

mg_reps  = np.full((N_REPS, H_MAX), np.nan)
pool_reps = np.full((N_REPS, H_MAX), np.nan)

for r in range(N_REPS):
    phi1_draws = RNG.normal(PHI1_BAR, SIGMA_PHI1, N_COUNTRIES)
    phi1_draws = np.clip(phi1_draws, -0.95, 0.94)   # stationarity safety margin

    g_all, eps_all = [], []
    for c in range(N_COUNTRIES):
        g, eps = simulate_country(T_SIM, phi1_draws[c], PHI2, SIGMA_EPS, RNG)
        g_all.append(g); eps_all.append(eps)

    # eps_hat constructed with the POOLED (mis-specified, common) phi -- exactly
    # as in the real exercise, using phi1=0.194 pooled estimate as if it applied
    # to every country (Pesaran-Smith mis-specification by construction).
    PHI1_POOLED_USED = 0.194
    psi_country = np.full((N_COUNTRIES, H_MAX), np.nan)
    pooled_rows = {h: ([], []) for h in range(1, H_MAX+1)}

    for c in range(N_COUNTRIES):
        g = g_all[c]
        ly = np.cumsum(g)
        dly = g.copy()
        eps_hat = np.full(T_SIM, np.nan)
        for t in range(2, T_SIM):
            eps_hat[t] = g[t] - PHI1_POOLED_USED*g[t-1] - PHI2*g[t-2]

        for h in range(1, H_MAX+1):
            rows_X, rows_y = [], []
            for t in range(2, T_SIM - h):
                rows_X.append([1.0, eps_hat[t], dly[t-1], dly[t-2]])
                rows_y.append(ly[t+h] - ly[t])
            X, y = np.array(rows_X), np.array(rows_y)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            psi_country[c, h-1] = b[1]

            # entity-FE pooled: demean (y, regressors) by country before pooling
            Xc = X.copy(); yc = y.copy()
            Xc[:, 1:] -= Xc[:, 1:].mean(axis=0)
            yc = yc - yc.mean()
            pooled_rows[h][0].append(Xc[:, 1:])
            pooled_rows[h][1].append(yc)

    mg_reps[r] = np.nanmean(psi_country, axis=0)

    for h in range(1, H_MAX+1):
        Xp = np.vstack(pooled_rows[h][0])
        yp = np.concatenate(pooled_rows[h][1])
        b = np.linalg.lstsq(Xp, yp, rcond=None)[0]
        pool_reps[r, h-1] = b[0]    # coefficient on eps_hat

mg_mean, mg_sd = mg_reps.mean(axis=0), mg_reps.std(axis=0)
pool_mean, pool_sd = pool_reps.mean(axis=0), pool_reps.std(axis=0)

print("="*86)
print(f"MC: {N_REPS} reps, N={N_COUNTRIES}, T={T_SIM}, REALISTIC phi1_i heterogeneity "
      f"(mean={PHI1_BAR}, sd={SIGMA_PHI1})")
print("Both MG and entity-FE POOLED estimators use the (mis-specified, common) "
      f"phi1={PHI1_POOLED_USED} to build eps_hat -- exactly as in the real exercise.")
print("="*86)
print(f"{'h':>3}  {'MG mean':>9}  {'MG sd':>7}  {'Pooled mean':>12}  {'Pooled sd':>10}  {'Pool-MG':>9}")
for h in range(1, H_MAX+1):
    print(f"{h:>3}  {mg_mean[h-1]:>9.3f}  {mg_sd[h-1]:>7.3f}  "
          f"{pool_mean[h-1]:>12.3f}  {pool_sd[h-1]:>10.3f}  {pool_mean[h-1]-mg_mean[h-1]:>9.3f}")

print(f"\nReal data:   MG psi_20 = 0.311 (SE=0.108);  Pooled psi_20 = -0.238 (t=-2.4)")
print(f"Simulated:   MG psi_20 = {mg_mean[19]:.3f} (sd={mg_sd[19]:.3f});  "
      f"Pooled psi_20 = {pool_mean[19]:.3f} (sd={pool_sd[19]:.3f})")
print(f"Simulated Pool-MG gap at h=20: {pool_mean[19]-mg_mean[19]:.3f}  "
      f"(real-data Pool-MG gap: {-0.238-0.311:.3f})")
