# Glossary

> Plain-language definitions of recurring terms. For symbols, see
> [`notation.md`](notation.md); for the axioms themselves, see
> [`axioms.md`](axioms.md). Formal definitions always take precedence over
> this page — see the article.

**Axiom** — In this article, a structural assumption that is
individually justified by specific empirical evidence and stated together
with an explicit condition under which data could refute it — as opposed
to an arbitrary modeling convenience. See §"Methodological Note: Why
'Axiom,' Not 'Structural Assumption'" in the article for the author's
own defense of this terminology.

**Calibration** — Assigning numerical values to a model's parameters from
data, as opposed to deriving the model's *structure* (which is what the
axioms do). DESM's parameter vector `θ` is calibrated in §"Identification
and Estimation."

**DESM** — Dynamic Economic State Model, the axiomatic framework this
article develops (version 2, referred to as "DESM V2" in some internal
documents). Not an acronym expanded in the final article text itself,
but used throughout as the model's name.

**Falsification condition** — A quantitative statement of what pattern in
the data would contradict a given axiom or theorem. Central to the
article's methodology: every axiom includes one.

**Growth regime** — One of the five categories (`ℛ₁`–`ℛ₅`) every country
falls into, determined jointly by its institutional quality and investment
rate. See [`model-explained.md`](model-explained.md).

**Institutional poverty trap** — The prediction that below a certain
institutional-quality threshold, no feasible increase in investment can
produce positive expected growth; only institutional improvement can. A
derived theorem, not an assumption.

**Mean-group (MG) estimator** — An estimator that fits a separate
regression per country/unit and averages the resulting coefficients,
contrasted with a "pooled" estimator that imposes one common coefficient
across all units. Used in the article's local-projection impulse
response checks (Pesaran–Smith 1995).

**Panel (data)** — A dataset that tracks the same units (here, countries)
repeatedly over time; distinguished from a pure cross-section (one
observation per country) or a pure time series (one country over time).
The article's primary panel spans 295 countries and 62 years,
unbalanced (not every country has data for every year).

**Persistence multiplier (`ψ_∞`)** — The long-run amplification factor by
which a one-period growth shock permanently raises the level of log income,
determined by the AR(2) coefficients of the growth process.

**SIMEX (simulation-extrapolation)** — A statistical technique for
correcting an estimate for the effect of measurement error in an
explanatory variable, used here to correct the raw OLS estimate of `γ`
(the institutional-quality coupling) for measurement error in the latent
institutional-quality proxy.

**Slow manifold** — A subset of the model's variables (credit depth, trade
openness) that move slowly and track a country's structural position
(income level, geography) without exerting a first-order effect on
short-run growth dynamics.

**Stylized fact (`FC-k`)** — One of ten quantitative empirical regularities
established directly from the panel data, prior to and independent of any
modeling choice, that the axiom system is required to reproduce as a
mathematical consequence.

**Structural parameter** — A parameter with a specific causal or mechanistic
interpretation within the model (e.g., `β_I`, the investment-drift
coupling), as opposed to a purely descriptive statistical coefficient.

**Tail index (`α₊`, `α₋`)** — A parameter describing how "heavy" (fat) the
tail of a probability distribution is; a smaller tail index means more
frequent extreme values. Values below 4 imply infinite kurtosis (the
fourth statistical moment is undefined), which is what the article finds
for growth shocks.

**Unit root** — A property of a time series where shocks have a permanent,
non-decaying effect (as opposed to "mean-reverting" series, where the
effect of a shock fades over time). Log GDP per capita is found to behave
as a near-unit-root process.

**Welfare-equivalent gain (`λ`)** — A way of translating a change in a
country's growth drift into a single number: the permanent percentage
increase in consumption that would make a household indifferent to the
drift change, computed as `λ = exp(Δμ/ρ) − 1`.
