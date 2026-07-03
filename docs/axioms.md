# The six axioms — conceptual summary

> This page summarizes, in plain language, what each of the article's
> six axioms says and why it's there. It intentionally omits the formal
> statements, the derivations, and the falsification conditions — those
> live in §"Axiomatic Foundation" of the article
> (`article/F01_Doc_journal.pdf`), which is the
> only authoritative source for them. Numbers below are illustrative
> pointers to the article, not independent claims.

All six axioms exist to explain ten empirically observed patterns
("stylized facts," labeled `FC-1` through `FC-10` in the article) as
mathematical consequences, not as assumptions chosen to make the model fit.
The article proves the six are mutually independent (none follows from
the other five) and minimal (dropping any one leaves some pattern
unexplained).

## A1 — Logarithmic geometry of wealth

**In one line:** work with `log(GDP per capita)`, not GDP per capita
itself, as the basic quantity being modeled.

**Why:** income levels across countries are strongly skewed and don't fit
a simple two-parameter distribution, but *log* income is close to a
symmetric, normal-looking bell curve. Logs also turn multiplicative
growth into additive growth, which is the natural way to describe a
process with no fixed ceiling or floor.

## A2 — Permanent country-specific drift without convergence

**In one line:** each country has its own long-run average growth rate,
and that rate has nothing to do with how rich or poor the country started
out.

**Why:** the data show essentially zero correlation between a country's
initial income and its subsequent growth — a poor country is not
mathematically "due" to grow faster to catch up. This directly rules out
the "conditional convergence" idea used in many older growth models, which
the article shows was a specific cause of failure in an earlier version
of this project.

## A3 — Growth process with short memory and heavy-tailed innovations

**In one line:** year-to-year growth surprises are somewhat predictable
from the last two years, but mostly not predictable at all — and when
surprises happen, extreme ones (both good and bad) are far more common
than a normal bell curve would suggest, with big positive surprises
somewhat more common than equally big negative ones.

**Why:** growth has measurable but weak short-run autocorrelation, the
vast majority of year-to-year growth variation is noise rather than
signal, and the distribution of growth shocks has "fat tails" with a mild
upward skew. Any model that assumes smooth, thin-tailed, symmetric noise
will misdescribe how often booms and busts actually occur.

## A4 — Investment as a slow state variable with drift coupling

**In one line:** how much a country invests is itself a slow-moving,
persistent quantity, and it feeds into the country's long-run growth
slope — but investment doesn't affect growth directly or instantly; it
works through this slower channel.

**Why:** investment is the single strongest observable predictor of a
country's growth rate, but it's also highly persistent over time (a
country's investment rate this year strongly predicts next year's), which
is different from being a lever that can be pulled for instant results.

## A5 — Structural slow manifold (credit and trade)

**In one line:** a country's credit market depth and trade openness track
its income level and geography, but neither moves the needle on
year-to-year growth once investment and the country's growth slope are
already accounted for.

**Why:** credit deepens as countries get richer and trade openness is
largely explained by physical size and geography (small countries trade
more relative to GDP), but within a given country over time, quarter-to-
quarter swings in either don't predict growth surprises. They describe a
country's structural position, not its short-run dynamics.

## A6 — Institutional residual in country drift

**In one line:** after investment, there is still a large unexplained gap
in why countries grow at different rates — and part of that gap is
explained by a slow-moving, latent measure of institutional quality
(governance, rule of law, and similar).

**Why:** investment alone leaves most of the cross-country variation in
growth rates unaccounted for, and institutional-quality measures are
strongly associated with the kinds of outcomes (like sovereign default
risk) that reflect a country's overall growth trajectory. The article
is explicit that this axiom rests on different, less direct evidence than
the other five (it draws on directional evidence rather than being
measured in the primary panel directly), and discusses that difference in
evidentiary weight openly rather than treating all six axioms as
equally well-grounded.

This axiom is also the one that produces the model's institutional
poverty trap: below a certain threshold of institutional quality, no
amount of investment is enough to generate positive growth.

## How the six fit together

- A1 sets the coordinate system (log income).
- A2 says each country has its own permanent trend, unrelated to its
  starting point.
- A3 describes the noise around that trend.
- A4 and A6 together explain *what determines* the trend: an investment
  channel and an institutional channel.
- A5 rules in credit and trade as structurally informative but rules them
  out as direct drivers of short-run growth, keeping the model from
  overfitting on variables that are really just correlated symptoms of
  income and geography.

For the derived consequences of this system — the five growth regimes, the
poverty trap, the persistence multiplier, and structural divergence — see
[`model-explained.md`](model-explained.md), and for the formal statements,
proofs, and calibrated parameter values, see the article.
