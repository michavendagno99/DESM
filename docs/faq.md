# Frequently asked questions

> Answers here paraphrase and point to the article; they do not add new
> claims. Where a number is quoted, verify it against the article
> (`article/F01_Doc_journal.pdf`) before citing it.

### What is DESM, in one sentence?

A minimal, six-axiom mathematical theory of how a country's log GDP per
capita evolves over time, calibrated and validated against a 295-country,
62-year panel. See [`model-explained.md`](model-explained.md).

### Is this a new dataset, or a new model on existing data?

A new model. The panel data used comes from established public sources —
World Bank World Development Indicators, Worldwide Governance Indicators,
the Quality of Government Standard dataset, Barro-Lee educational
attainment, CEPII gravity/trade data, and the Global Financial Development
Database. See [`../data/sources.txt`](../data/sources.txt)
for provenance details.

### Why "axioms" and not just "assumptions"?

The article addresses this directly (§"Methodological Note: Why
'Axiom,' Not 'Structural Assumption'"). In short: each of the six is
individually motivated by specific data patterns, comes with an explicit
falsification condition, and the set as a whole is shown to be both
mutually independent and minimal. That combination is stronger than a
typical "modeling assumption" made for convenience.

### Is there a "DESM V1"? What happened to it?

Yes — an earlier version of this project is discussed and explicitly
falsified in the article's introduction. It shared some qualitative
insights with the current framework but failed several quantitative
validation exercises (GMM estimation diagnostics, out-of-sample
regime-classification accuracy, and historical country counterfactuals for
South Korea and Venezuela). The article treats that failure as
informative — it's used to motivate three specific structural corrections
in the current version, rather than discarded silently. See the article's
introduction for the full account of what failed and why; this repository
holds only the current, validated framework (DESM V2) and does not
distribute the earlier version's exploratory code or intermediate results.

### Does the model claim causality for the institutional-quality effect?

The article is careful here: the coupling coefficient `γ` (institutional
quality → growth drift) is estimated two ways — a WGI-panel OLS/SIMEX
estimate and an instrumental-variables estimate using historical settler
mortality and colonial origin, specifically to address the concern that
governance scores could partly reflect past economic performance (reverse
causality) rather than cause future growth. Both approaches agree on the
sign and the IV estimates are, if anything, larger. See the article's
identification section and its explicit discussion of this axiom's
"epistemological status" relative to the other five.

### Does the model say poor countries are stuck?

Not uniformly. Five regimes are distinguished precisely because the
answer differs by regime: some declining countries can recover through
sustained investment alone; only the deepest institutional trap regime
cannot be escaped by investment and specifically requires institutional
improvement. See [`model-explained.md`](model-explained.md) and the
article's policy section for the regime-specific prescriptions.

### Why does the model predict growing inequality between countries?

Because two of its core, independently-justified ingredients — each
country has its own roughly permanent growth rate, and there's no
statistical pull toward convergence — mechanically imply that the income
gap between fast- and slow-growing countries widens over time. This is
presented as a structural mathematical consequence, not a value judgment
or a separate pessimistic assumption. See the article's discussion of
why the currently available 60-year data window may not yet show this
starkly.

### What does "all ten stylized facts pass" mean exactly?

It means that when the model is calibrated to the data, each of the ten
independently-established empirical patterns (`FC-1`–`FC-10`) is
reproduced by the model's implied statistics, within the article's
stated validation protocol and tolerance. The article documents this in
detail, including a "cascade diagnosis" for the one partial-pass case and
the reconciliation performed for it. See §"Empirical Validation."

### Is there anything the model does not (yet) explain?

Yes — the article is explicit about one identified open question:
coupling the drift equation with the investment transition law for
short-horizon (medium-run) policy analysis. A second issue that was
initially suspected as an open gap (a shortfall in the long-run magnitude
of a local-projection impulse-response check) is reported as resolved
within the article itself via a calibrated random-coefficients test and
a finite-sample estimator audit, rather than left open. See §"Conclusions"
→ "Open Questions and Minimal Extensions."

### Where's the code that produced the paper's numbers?

[`article/code/`](../article/code/), with a per-module description in that folder's
`README.md`.

### Can I reuse the article, data, or code?

Not yet without permission — see [`../LICENSE`](../LICENSE). The
article is under journal submission at the time of writing.

### How do I ask a question not answered here?

Open an issue in the repository, or contact the corresponding author
directly (see [`../README.md`](../README.md)).
