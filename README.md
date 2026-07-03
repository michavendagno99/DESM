# A Minimal Axiomatic Theory of National Macroeconomic Dynamics with Emergent Regime Structure

**Author:** Michael Stephen Arias Avendaño ([ORCID: 0009-0009-2778-3790](https://orcid.org/0009-0009-2778-3790))
**Affiliation:** Pontificia Universidad Javeriana, Facultad de Ingeniería, Bogotá, D.C., Colombia
**Contact:** [mstephen-arias@javeriana.edu.co](mailto:mstephen-arias@javeriana.edu.co)

This repository accompanies the article *"A Minimal Axiomatic Theory of National Macroeconomic Dynamics with Emergent Regime Structure,"* which develops **DESM** (Dynamic Economic State Model): a compact mathematical theory of how national economies grow, stagnate, or fall into poverty traps — built from six axioms and tested against a 295-country, 62-year panel of World Bank data.

> This README sells the idea and orients you around the repository. It does not restate the article's proofs — for those, read the article itself (linked below). For an accessible walkthrough of the ideas, see [`docs/`](docs/).

---

## The 30-second version

Why do some countries stay rich, some catch up, and some fall into poverty traps they can't seem to escape? Growth economics has plenty of answers — but most of them work by picking equations that look reasonable, fitting them to data, and checking that the fit is decent. That makes it very hard for the data to actually *prove a growth model wrong*.

This project does it the other way around. It starts from ten patterns that are simply *true* of the last 62 years of world growth data — no one seriously disputes them — and asks: **what is the smallest set of independent rules that is *forced* to produce exactly those patterns, as mathematical theorems rather than modeling choices?** Six such rules (the axioms) turn out to be enough. The payoff is that the same six rules, with nothing added, also generate results nobody asked them to produce: a five-way classification of growth trajectories, a poverty trap with a computable escape threshold, a multiplier for how long growth shocks persist, and concrete investment/governance targets for real countries. Predicting more than you built the model to explain is what makes an axiom system worth taking seriously — that is the test this project holds itself to.

## Headline results

- **Ten empirical patterns, fixed as pass/fail targets before any model was written** — near-unit-root persistence in log GDP per capita, no cross-country income convergence, growth noise dominated by within-country shocks rather than global ones, heavy-tailed growth shocks, and six more. All ten are reproduced by the calibrated model.
- **A five-regime map of the world's growth trajectories** — from growth miracles to an institutional poverty trap — falls out of the same six axioms as a proven theorem, not a clustering algorithm applied after the fact.
- **A persistence multiplier of ≈1.34×** for growth shocks: a one-off shock to a country's growth rate raises its long-run income path by about 34% more than the shock's face value, because of how growth shocks echo through the dynamics. This number is derived analytically and then checked against the data.
- **The institutional-quality effect on growth survives its toughest test.** Does good governance *cause* growth, or does growth just make governance look better in hindsight (reverse causality)? The article checks this with two historical instruments — colonial-era settler mortality and colonial legal origin — that predate modern growth by decades. Both instruments agree with the original estimate on sign, and if anything push the effect **higher**, not lower.
- **Concrete, country-specific policy numbers**, not just qualitative direction, for eight real countries spanning all five regimes — e.g., Colombia needs about a 3-percentage-point increase in investment to move up a regime, worth roughly a 21% permanent increase in per-capita income at conventional discounting; Zimbabwe's escape from its current trap depends on investment far more than on governance reform, once the governance effect is estimated at its current best (IV-corroborated) range rather than an illustrative upper bound.
- **The theory owns its predecessor's failure.** An earlier version of this project ("DESM V1") is confronted with the data and found wanting — it under-predicts South Korea's miracle growth by more than half and gets Venezuela's collapse backwards in sign. That failure isn't hidden; it's used to motivate the structural corrections that produced the current framework.
- **The classic textbook models aren't rivals — they're special cases.** The Solow-Swan and AK growth models are shown to be limiting cases nested inside this framework, not competing theories that have to be argued down.

## Why this matters for economics

Macro-growth theory is regularly criticized — including by this article's own literature review — for models flexible enough to fit almost any outcome after the fact, which makes them hard to falsify and hard to trust for policy. This project is a direct answer to that criticism: every axiom comes with an explicit, quantitative condition under which the data would have refuted it, the axioms are shown to be mutually independent (none is a restatement of another) and jointly minimal (drop any one and something the model needs to explain stops being explained), and the theory is confronted with its own falsified predecessor in public rather than quietly discarding it. The result is not just an explanation of *why* the ten patterns hold, but a policy tool: for any country, the model produces a *specific number* — how many points of investment, or how much of a governance improvement — needed to escape decline, expressed in a common welfare-equivalent currency (permanent income gain) that lets policies be compared across very different countries.

## Limitations, honestly

- **The institutional-quality effect (`γ`) is corroborated, not proven, causally.** Two independent instruments confirm its sign and suggest the original estimate may be conservative, but neither instrument's exclusion restriction is beyond debate, and one of the two instrument specifications has a comparatively weak first stage. Treat the *direction* of the governance effect as solid and its *exact magnitude* as a corroborated range, not a settled point estimate.
- **The panel window is 1960–2021.** The theory is calibrated and validated entirely within this window; it makes no claim about pre-1960 growth regimes (e.g., the Kaldor-Kuznets facts of early industrialization) by design.
- **Welfare comparisons assume log utility and a conventional discount rate**, with sensitivity checked across a small range of alternative discount rates — a standard simplification in this literature, not a novel claim.
- **One open question is acknowledged in the article itself**: how the drift equation and the investment-transition law should be coupled for medium-run (as opposed to long-run) policy analysis. A related concern — a large apparent gap between the theoretical and empirically estimated size of the persistence multiplier — is investigated and resolved within the article via a dedicated Monte Carlo audit, rather than left open.
- **This is a reduced-form macro-panel theory, not a general-equilibrium model.** It does not model prices, trade balances, or cross-country spillovers explicitly; institutional quality and investment enter as the two channels the data supports, not as an exhaustive list of everything that matters for growth.

None of this is buried — the article devotes an entire validation section and a "what this model does not (yet) explain" discussion to it. See [`docs/faq.md`](docs/faq.md) for a longer, plain-language treatment.

## See it for yourself

The fastest way to trust (or challenge) any number above is to run the code that produced it. [`article/code/`](article/code/) is a small, single-purpose Python codebase — no shared framework to learn — where each script maps directly to one estimator or one robustness check described in the article's computational-implementation section. Start with [`code/README.md`](article/code/README.md) for what each file does, then read [`docs/overview.md`](docs/overview.md) or [`docs/model-explained.md`](docs/model-explained.md) if you want the ideas in plain language before the equations.

---

## Repository structure

```
.
├── README.md                    ← you are here
├── LICENSE
├── CITATION.cff                 ← machine-readable citation metadata
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── docs/                        ← complementary documentation (does not modify the article)
│   ├── overview.md              ← motivation, objectives, scientific framing
│   ├── model-explained.md       ← DESM explained in plain language
│   ├── axioms.md                ← conceptual summary of the six axioms
│   ├── notation.md              ← notation used throughout the article
│   ├── glossary.md              ← glossary of terms
│   ├── faq.md                   ← frequently asked questions
│   └── repository-guide.md      ← how this repository is organized, for collaborators
├── data/                        ← provenance notes only; raw datasets are not distributed in this repo
│   ├── README.md
│   └── sources.txt              ← source-by-source notes (World Bank, WGI, QoG, Barro-Lee, CEPII, ...)
└── article/                     ← the code behind the article
    └── code/                    ← ★ reference implementation (Python) backing every
                                      estimate, table, and figure in the article
```

This repository holds only what is needed to run the code behind the
article — no drafts, no intermediate results, no process documents.
Earlier exploratory work, including the project's falsified predecessor
("DESM V1," see [Headline results](#headline-results) above and
[`docs/faq.md`](docs/faq.md)), is not distributed here.

**The article is the sole authoritative source for the theory.** Everything under `docs/` is explanatory scaffolding around it and must never be treated as a substitute for, or extension of, its content.

## The article

The article itself (`F01_Doc_journal.pdf`) is **not distributed in this
repository** — it is currently under journal review, and the journal's
policy does not permit public posting while under submission (see
[`LICENSE`](LICENSE)). Contact the author for a copy (see contact details
above).

## Reference implementation

[`article/code/`](article/code/) contains the Python code that produces every parameter estimate, table, and figure reported in the article. See [`code/README.md`](article/code/README.md) for what each module does and how to run it.

## Data

The underlying panel data (World Bank World Development Indicators, Worldwide Governance Indicators, Quality of Government Standard dataset, Barro-Lee educational attainment, CEPII gravity/trade data, and the Global Financial Development Database) is **not distributed in this repository** — see [`data/README.md`](data/README.md) for why and where to obtain it. [`data/sources.txt`](data/sources.txt) documents source-by-source provenance for every dataset used.

## How to cite

See [`CITATION.cff`](CITATION.cff). In brief:

> Arias Avendaño, M. S. *A Minimal Axiomatic Theory of National Macroeconomic Dynamics with Emergent Regime Structure*. Article.

## License

See [`LICENSE`](LICENSE). All rights reserved; the article is currently under journal submission.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/repository-guide.md`](docs/repository-guide.md) for how this repository is organized and how to propose changes to the documentation or the reference implementation. **Contributions must not alter the article itself.**
