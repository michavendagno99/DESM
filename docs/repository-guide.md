# Repository guide for new collaborators

This page is an orientation for someone opening this repository for the
first time: how it's organized, where things live, and how to work in it
without touching what shouldn't be touched.

## The one thing to internalize first

**The article is the fixed point of this repository. Everything else
orbits it and does not modify it.** If you take one rule away from this
page, take that one — see [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for
the full statement of what is and isn't in scope for a contribution.

## Where things live

```
Repository/                                  ← repository root
├── README.md, LICENSE, CITATION.cff, …      ← repo-level metadata (this layer)
├── docs/                                    ← you are here — explanatory documentation
├── data/                                    ← provenance/source notes only (no raw datasets)
└── article/
    └── code/                                ← ★ THE REFERENCE IMPLEMENTATION
```

The article itself is not distributed in this repository — it is under
journal review, and the journal's policy does not permit public posting
while under submission (see [`../README.md`](../README.md) for contact
details). `article/code/` carries a star above because it is the one
thing here that stands on its own for a reader; nothing in `docs/`
supersedes the article for "what does the paper actually say."

This repository intentionally holds only what is needed to run the code
behind the article — no drafts, no intermediate results, no process
documents. If you're looking for exploratory notebooks, working
drafts, or superseded framework versions, they don't belong here by
design; see [`faq.md`](faq.md#is-there-a-desm-v1-what-happened-to-it) for
where the project's falsified predecessor ("DESM V1") is discussed.

### `article/code/`

The Python reference implementation that produces the article's
estimates, tables, and figures. It lives alongside the article
specifically so that a repository visitor finds both together. See its
own [`README.md`](../article/code/README.md) for a
per-file description.

### `data/`

Provenance notes only — the raw and merged datasets themselves are not
distributed in this repository (see [`../data/README.md`](../data/README.md)
for why and where to obtain them). `data/sources.txt`
documents where each raw source used in the institutional-coupling (`γ`)
estimation comes from — read it before adding or
reinterpreting a data file.

## Naming conventions you'll notice

- **`FC-k`** (in prose/docs) — stylized fact / validation condition `k`.
- **`Ak`** (in prose/docs) — axiom `k`.
- **`rc_*.py`, `gamma_*.py`** — code files are generally named after the
  specific estimation or robustness check they perform, not after a
  generic "utils"-style role; see `article/code/README.md` for the map
  from filename to purpose.

## Recommended workflow for contributions

1. Read [`overview.md`](overview.md) and [`model-explained.md`](model-explained.md)
   first if you're new to the project's substance — they'll save you from
   misreading the article's terminology.
2. Identify whether your change belongs in `docs/` (explanatory content),
   `article/code/` (the reference implementation), or `data/`
   (provenance/source notes) — never a change to the article itself.
3. Open an issue for anything beyond a trivial fix, per
   [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
4. Cross-check any factual or numerical claim you add to `docs/` against
   the article section it describes, and prefer linking to that section
   over restating it.

## Questions

If something about the organization is unclear, that's a documentation gap
— open an issue rather than guessing, so this guide can be corrected.
