# Reference implementation

Python code implementing the estimation, simulation, and figure-generation
pipeline behind the article
[`F01_Doc_journal.pdf`](../F01_Doc_journal.pdf). Table 6.x ("Correspondence
between mathematical objects and Python identifiers") in the article's
*Computational Implementation* section maps specific classes and functions
in these modules to the theorems and equations they compute.

This is the only code in the repository — everything here is needed to
reproduce a result from the article; nothing here is exploratory or
process material.

## Modules

| File | Purpose |
|---|---|
| `desm_v2.py` | Core reference implementation: `DESMParameters`, the state-space transition and IRF (`DESMTransition`), panel construction, the Group I–III estimators (single-lag AR(1) drift, tail indices, investment persistence — see caveat below), bootstrap inference, the five-regime classifier (`RegimeClassifier`), forward simulation (`DESMSimulator`), and the full estimation pipeline (`run_full_estimation`). |
| `gamma_estimation.py` | Earlier point-identification pass for the institutional coupling parameter γ (OLS + SIMEX correction against measurement error in governance quality). |
| `gamma_estimation_v2.py` | Canonical γ estimation referenced by the article: merges the primary panel with the QoG/WGI governance panel, builds the standardized institutional-quality index, and produces γ̂^EV and the SIMEX-corrected γ̂^SIMEX with a pairs-bootstrap confidence interval. |
| `gamma_iv_estimation.py` | Instrumental-variables corroboration of γ using historical settler mortality and colonial origin as instruments (two-stage least squares, three specifications, overidentification test). Imports directly from `gamma_estimation_v2.py`. |
| `lp_irf_country.py` | Country-level local-projection impulse response (Pesaran–Smith mean-group estimator), used to cross-check the theoretical persistence multiplier ψ∞. |
| `rc_irf_reconciliation.py` | Random-coefficients AR(2) extension used to test whether cross-country heterogeneity in φ₁ explains the LP-IRF long-run magnitude gap. |
| `rc_lp_finite_sample_mc.py` | Monte Carlo check of whether finite-sample mean-group LP estimation, applied to a known homogeneous AR(2) data-generating process, reproduces the observed attenuation on its own. |
| `rc_pooled_heterogeneity_mc.py` | Companion Monte Carlo check isolating the Pesaran–Smith heterogeneity-bias mechanism from baseline finite-sample LP attenuation. |
| `robustness_checks.py` | Subsample stability of β_I (investment–growth coupling) and ARIMA(p,1,0) lag-order/unit-root robustness checks. |
| `make_figures.py` | Generates the five data-driven article figures (`fig1_distributions.pdf` … `fig5_variance_drift.pdf`) into `article/figures/` (created on demand; not shipped in the repository since the article PDF already contains the rendered versions). |

`fig6_dependency_graph` (the logical dependency diagram) is not
Python-generated; it is produced independently of this codebase.

**Single-lag vs. two-lag caveat.** `desm_v2.py` implements the
first-generation single-lag AR(1) specification (`estimate_phi1()`,
`DESMTransition`, `DESMParameters` defaults such as `phi1=0.263`). The
two-lag AR(2) Yule-Walker calibration actually reported throughout the
article (φ₁=0.194, φ₂=0.062, ψ∞≈1.344) is exercised in
`rc_irf_reconciliation.py`, `rc_lp_finite_sample_mc.py` and
`rc_pooled_heterogeneity_mc.py`, not yet back-ported into `desm_v2.py`'s
class hierarchy as first-class fields. This is a disclosed, intentional gap
— see the article's §*Computational Implementation* — not an inconsistency
to silently fix: no numerical result in the article is computed from
`desm_v2.py`'s single-lag defaults.

## Data dependencies

All scripts read from [`data/`](../../data/) at the repository root.
Every module resolves this path
relative to its own file location (`Path(__file__)`, three levels up to the
repository root), so no script needs editing to run after cloning — see the
`DATA_DIR` / `DATA_PATH` / `PANEL_PATH` / `ROOT` constant near the top of
each file. The underlying data files themselves are not distributed in this
repository (see [`../../data/README.md`](../../data/README.md));
scripts will raise a `FileNotFoundError` on the missing `.csv`/`.xlsx` until
they are supplied locally.

## Requirements

```
python >= 3.10
numpy
pandas
scipy
matplotlib
openpyxl        # required by pandas.read_excel for the .xlsx governance panels
```

No `requirements.txt` is pinned; the code was developed against current
(2026) releases of the above.

## Running

Each module is a standalone script with a `main()` entry point or top-level
execution block; run it directly, e.g.:

```bash
python desm_v2.py
python gamma_estimation_v2.py
python make_figures.py
```

There is no shared CLI or package entry point — this is a research
reference implementation, not a library, and modules are meant to be read
and rerun individually alongside the corresponding section of the
article.
