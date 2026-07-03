# Notation reference

> A quick-reference table for the symbols used throughout the article.
> This page is a navigation aid, not a definition source — every symbol's
> precise, formal definition is in the article
> (`article/F01_Doc_journal.pdf`); consult it
> before relying on any meaning given here.

## Core state variables

| Symbol | Meaning |
|---|---|
| `y(t)` | Real GDP per capita (constant 2015 USD) at time `t`. |
| `x₁(t) = log y(t)` | The model's fundamental state variable: log GDP per capita. |
| `i` | Country index. |
| `t` | Time index (years). |
| `T` | Panel horizon / a fixed forecast horizon, depending on context. |
| `g_it` | Within-country demeaned growth: `Δx₁,it − μᵢ`. |
| `I_it` | Investment rate (gross fixed capital formation, % of GDP) for country `i` at time `t`. |
| `Ī_i` | Country `i`'s long-run mean investment rate. |
| `C_it` | Private credit / GDP for country `i` at time `t`. |
| `T_it` | Trade openness (trade / GDP) for country `i` at time `t`. |
| `q_i` | Latent institutional-quality variable for country `i` (not directly observed in the primary panel; proxied via governance indices such as WGI). |

## Drift and growth process (A1–A3)

| Symbol | Meaning |
|---|---|
| `μᵢ` | Country `i`'s permanent, time-invariant growth drift. |
| `μ̄`, `σ_μ` | Mean and standard deviation of the cross-country drift distribution `F_μ`. |
| `φ₁, φ₂` | AR(2) autoregressive coefficients of the demeaned growth process. |
| `ε_it` | i.i.d. innovation (shock) term in the AR(2) growth equation. |
| `σ_ε²` | Variance of the innovation `ε_it`. |
| `α₊, α₋` | Right- and left-tail power-law indices of `ε_it` (heavier tail = smaller index). |
| `ψ_∞` | Persistence multiplier: the long-run amplification of a one-period growth shock on the permanent income level, `ψ_∞ = 1/(1−φ₁−φ₂)`. |

## Investment, institutions, and drift decomposition (A4, A6)

| Symbol | Meaning |
|---|---|
| `ρ_I` | AR(1) persistence coefficient of the investment process. |
| `σ_ν²` | Variance of the investment-process innovation `ν_it`. |
| `β_I` | Coupling coefficient: effect of investment on the growth drift `μᵢ`. |
| `μ₀` | Baseline (intercept) drift term. |
| `η_i` | Structural residual of the drift equation not explained by investment. |
| `γ` | Coupling coefficient: effect of institutional quality `qᵢ` on the drift residual. |
| `ζ_i` | Idiosyncratic residual after accounting for both investment and institutions. |
| `q*`, `q†`, `q̃`, `q_h` | Institutional-quality thresholds defining the boundaries between the five growth regimes (structural and distributional, respectively). |

## Parameter vector and estimation

| Symbol | Meaning |
|---|---|
| `θ` | The full 11-parameter structural vector: `(φ₁, φ₂, σ_ε², α₊, α₋, ρ_I, σ_ν², μ₀, β_I, γ, σ_ζ²)`. |
| `γ̂^EV` | OLS estimate of `γ` from the WGI-augmented cross-section, before measurement-error correction. |
| `γ̂^SIMEX` | SIMEX (simulation-extrapolation) measurement-error-corrected estimate of `γ`. |
| `γ̂^IV` | Instrumental-variables estimate of `γ` (settler mortality / colonial origin instruments). |
| `λ̂_q` | Reliability ratio used in the SIMEX correction for `q̂ᵢ`. |

## Regimes and policy

| Symbol | Meaning |
|---|---|
| `ℛ₁ … ℛ₅` | The five growth regimes (growth miracle, normal growth, below-average growth, shallow trap, deep trap). |
| `m(qᵢ, Īᵢ)` | The conditional expected drift function used to classify a country's regime. |
| `I_max` | The maximum feasible investment rate. |
| `λ` | Consumption-equivalent welfare gain from a drift improvement, `λ = exp(Δμ/ρ) − 1`. |
| `ρ` | Discount rate used in the welfare formula. |
| `MRS` | Marginal rate of substitution between investment and governance, `γ/β_I`. |

## Statistical / validation notation

| Symbol | Meaning |
|---|---|
| `FC-k` (`FC-1` … `FC-10`) | The ten empirical "stylized facts" / validation conditions the model is required to reproduce. |
| `Ak` (`A1` … `A6`) | Shorthand for axiom `k`. |
| `B/W` | Between-country to within-country variance ratio. |
| `SNR` | Signal-to-noise ratio, `σ_μ²/σ_ε²`. |
| `R²_max` | The theoretical predictability ceiling on within-country growth `R²`. |
| `r_S`, `r_B`, `r_W` | Spearman rank correlation, between-country correlation, within-country correlation, respectively. |

## Where to look for more

- Full formal definitions: §"Mathematical Construction" and §"Axiomatic
  Foundation" of the article.
- Calibrated numerical values of `θ`: article Table "Calibrated free
  parameter vector".
- Correspondence between these symbols and Python identifiers in the
  reference implementation: article Table "Correspondence between
  mathematical objects and Python identifiers", and
  [`../article/code/README.md`](../article/code/README.md).
