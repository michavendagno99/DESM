# DESM, explained without equations

> This is a plain-language walkthrough of the model's *ideas*, written for
> readers who want the intuition before (or instead of) the mathematics. It
> paraphrases the article's own non-technical summary
> (§"Intuitive Summary: Five Results Without Equations") and framing
> sections; it is not a substitute for them, and any number quoted below
> should be checked against the article before being relied upon. For
> the formal statements, see the article
> (`article/F01_Doc_journal.pdf`). For the axioms
> themselves, see [`axioms.md`](axioms.md).

## Start with one idea: a country's wealth is a path, not a point

Instead of asking "why is country X richer than country Y today," DESM asks
"what is the *process* that generated country X's whole trajectory of
income, from year to year, for six decades?" The object being modeled is
not a snapshot but a path — like tracking a specific hiker's GPS trace up a
mountain, rather than just noting how high up they currently are.

The variable DESM actually tracks is not GDP per capita itself but its
**logarithm**. This matters because income differences across countries
span two orders of magnitude, and growth is naturally a *percentage*
change, not an absolute one — "grew 3%" means something consistent whether
you started rich or poor, but "grew $1,000" does not. Working in logs turns
percentage growth into simple addition, which is what makes the rest of
the model tractable.

## Every country walks its own path, with its own average slope

Each country's log-income path has two ingredients:

1. **A personal, essentially permanent slope** — its long-run average
   growth rate. Some countries have a slope of +5% a year, some +2%, some
   negative. Crucially, this slope does *not* depend on how rich or poor
   the country started out: a poor country is not mathematically destined
   to grow faster to "catch up," and the data confirm there is no such
   pull (this is one of the ten stylized facts the model is built to
   match, and it directly contradicts a much older prediction called
   "convergence").
2. **Year-to-year noise around that slope** — good years and bad years,
   which average out over long horizons but dominate what you'd notice
   in any given five-year window. In fact, most of what looks like
   "growth variation" in short-run data is this noise, not a change in the
   country's underlying trajectory.

This already explains something important: because a country's slope is
close to permanent and the noise washes out over time, **the ranking of
countries by income is very stable across decades** — which is exactly
what's observed.

## What sets a country's slope?

The model doesn't leave the slope unexplained — it decomposes it into two
pieces policymakers can act on:

- **Investment.** How much of GDP a country plows back into capital
  formation. Investment itself moves slowly and persistently (a country
  that invests a lot this year tends to keep investing a lot for years),
  and higher sustained investment is associated with a higher growth
  slope.
- **Institutional quality.** A summary of how well a country's governance,
  rule of law, and public administration function. This channel exists
  because investment alone leaves most of the cross-country differences in
  growth rates unexplained — something else, slow-moving and
  institutional in nature, accounts for the rest.

A subtlety the model insists on: the raw correlation between "countries
that invest more" and "countries that grow faster" overstates investment's
own causal power, because good institutions and high investment tend to
travel together. Part of the correlation is really the institutional
channel showing up disguised as an investment effect. Untangling the two
is one of the model's central estimation problems.

## Shocks don't fade — they compound

Because income accumulates like a running total (think of a savings
account, not a thermostat that returns to a set point), a one-time shock to
growth doesn't just nudge the level of GDP once and disappear — it changes
the trajectory permanently, and because of *how* growth is serially
correlated from one year to the next, the eventual effect on the income
level ends up **larger** than the original shock. A one-year boost to
growth translates into a bigger-than-one-year-boost in the permanent income
level. This is the model's "persistence multiplier."

There's also an asymmetry: extreme good years are somewhat more common,
relative to their size, than extreme bad years of the same size — so the
long-run distribution of outcomes leans slightly positive, matching what's
observed.

## Fat tails: rare events aren't as rare as you'd think

If you plotted year-to-year GDP growth rates and compared them to a normal
bell curve, the extremes — both booms and busts — would happen far more
often than the bell curve predicts. This isn't treated as messy data to be
smoothed over; the model builds it in directly, which has a practical
consequence: ordinary statistical tools (standard errors, confidence
intervals, t-tests) that assume bell-curve-like behavior can be misleading
here, and the estimation code uses techniques designed for exactly this
situation (bootstrapping, robust tail estimators) instead.

## Five kinds of countries

Combining "each country has a slope," "the slope depends on investment and
institutions," and "there's a ceiling on how much you can invest," the
model mathematically forces every country into exactly one of five
categories — not as a classification someone chose, but as the inevitable
output of the math once you fix a country's institutional quality and
investment rate:

1. **Deep trap** — decline so severe that no feasible amount of investment
   can fix it; only institutional improvement can.
2. **Shallow trap** — decline, but investment alone, if sustained, can pull
   the country out.
3. **Below-average growth** — growing, but falling behind the world average
   in relative terms.
4. **Average growth** — keeping pace with the rest of the world.
5. **Growth miracle** — growing well above the world average.

This is arguably the model's signature result: it doesn't assume a
"poverty trap" exists and then go looking for one — the trap, and the
threshold that defines it, falls out of the same equations used to explain
ordinary growth.

## The uncomfortable prediction: the rich-poor gap widens, not narrows

If each country keeps its own roughly permanent slope, and there's no force
pulling poorer countries to grow faster, the mathematics says the *gap*
between the richest and poorest countries should keep growing over time,
not shrink — even while poor countries themselves get richer in absolute
terms. This isn't a pessimistic assumption bolted onto the model; it's a
direct logical consequence of the same two ingredients (permanent
country-specific slopes, no convergence pull) used to explain everything
else. The article argues the data don't yet show this dramatically only
because six decades is a short window relative to how spread out countries
already were when the data begin.

## Why bother with all this rigor?

Because the payoff is that the five regimes, the trap, the persistence
multiplier, and the policy prescriptions aren't independent claims tacked
onto a growth story — they're all forced to be true by the same six
axioms that were justified, one at a time, by specific empirical patterns.
If a future dataset contradicted one of those regimes or the trap, it
would mean one of the six axioms is wrong, not that a design choice needs
tweaking. That falsifiability is the entire point — see
[`axioms.md`](axioms.md) for what the six axioms actually say, and
[`faq.md`](faq.md) for common questions about what the model does and
doesn't claim.
