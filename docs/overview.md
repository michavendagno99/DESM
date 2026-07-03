# Project overview

> This page explains what the project is trying to do and why. It does not
> restate the article's results — for those, read the article itself
> (not distributed in this repository while under journal review — see
> [`../README.md`](../README.md)).

## The question

Why do some countries grow rich and others stay poor, or decline — and can
that process be described with the same kind of rigor physics uses to
describe motion?

Growth economics has no shortage of models: Solow-Swan, AK-type endogenous
growth models, conditional-convergence regressions, institutions-and-growth
frameworks. Most of them are built by choosing a functional form that seems
economically reasonable, fitting it to data, and checking whether the fit
is good. This project asks a different, narrower question: **starting only
from patterns that are unambiguously present in the data, what is the
smallest set of independent structural assumptions — axioms — that forces
those patterns to be true as a matter of mathematics, and what else does
that same set of assumptions force to be true?**

That second half is the point. An axiom system is only interesting if it
predicts more than it was built to explain. This project's standard of
success is that the five growth regimes, the institutional poverty trap,
the shock-persistence multiplier, and the policy prescriptions are not
extra ingredients added to make the model useful — they are theorems that
follow from the same six axioms used to explain the original ten empirical
patterns.

## Why this approach

Economics is often criticized — including in this article's own related
literature discussion — for models where free parameters absorb the gap
between theory and data, making the theory nearly unfalsifiable in
practice. The article's methodology is designed to close that gap
directly:

1. **Establish the empirical benchmark before proposing any model.** Ten
   stylized facts (`FC-1` … `FC-10`) are measured from a 178-country,
   62-year panel and stated as falsification targets first.
2. **Derive, don't assume.** Every axiom is justified by pointing to which
   specific stylized fact(s) require it and why a simpler alternative fails
   to reproduce them.
3. **Prove independence and minimality.** The article shows that no
   axiom can be derived from the other five, and that removing any single
   axiom leaves at least one stylized fact unexplained — i.e., the axiom
   system is neither redundant nor incomplete relative to its own goal.
4. **State falsification conditions up front.** Each axiom comes with an
   explicit, quantitative condition under which the data would refute it.
5. **Confront a prior version of the same project with the data, and
   report the failure.** An earlier framework ("DESM V1") was
   quantitatively rejected across several validation exercises. Rather than
   quietly discarding it, the article documents *why* it failed and uses
   that failure to motivate specific structural corrections in the current
   version.

## What "DESM" is, in one sentence

DESM (Dynamic Economic State Model) treats a country's log GDP per capita
as a stochastic process with a permanent, country-specific growth drift,
short-memory heavy-tailed noise, and a drift equation that decomposes into
an investment channel and an institutional-quality channel — six axioms
in total, described conceptually in [`axioms.md`](axioms.md) and narratively
in [`model-explained.md`](model-explained.md).

## Intended audience of this repository

- **Reviewers and readers of the article** who want the supporting code
  and data alongside the paper.
- **Researchers extending or stress-testing the framework**, who need to
  know where the reference implementation lives and how it maps to the
  article's equations and theorems (see
  [`../article/code/README.md`](../article/code/README.md)).
- **Newcomers to the project** orienting themselves — see
  [`repository-guide.md`](repository-guide.md).

## What this documentation is not

It is not a second copy of the article, an abstract, or a summary of
results intended to be cited in place of the paper. Numbers, thresholds,
and claims quoted anywhere in `docs/` are drawn directly from the
article and may lag behind it if the article is later revised; the
article itself is always the authoritative source.
