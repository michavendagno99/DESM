# Contributing

This repository accompanies an academic article that is currently under
journal submission. Contributions are welcome for the **documentation** and
the **reference implementation**, subject to the rule below.

## The one hard rule

**The article is not open for contribution.** This means
[`article/F01_Doc_journal.pdf`](article/F01_Doc_journal.pdf) itself.

Do not open pull requests that edit text, equations, figures, tables,
citations, or notation in the article. Scientific corrections belong in
correspondence with the author (see contact in [`README.md`](README.md)), not
in a pull request — the article's content is the author's sole
responsibility as the paper moves through peer review.

Everything else in the repository is fair game for improvement:

- **Documentation** (`docs/`, `README.md`, and this file): fixing errors,
  improving clarity, adding examples — as long as the documentation
  continues to *explain* the article rather than restate, extend, or
  reinterpret its scientific claims.
- **Reference implementation**
  (`article/code/`): bug fixes, reproducibility
  improvements, dependency pinning, and test coverage, provided they do not
  change the numerical results reported in the article. Any change that
  would alter an estimate, table, or figure value must be flagged explicitly
  and is out of scope for a routine contribution.
- **Data documentation** (`data/`): provenance notes, source corrections in
  `data/sources.txt`.

## How to propose a change

1. Open an issue describing the problem or improvement before submitting a
   pull request, unless the change is a small, obvious fix (typo, broken
   link).
2. Keep pull requests scoped to one concern.
3. For code changes, note which script(s) you tested and how.
4. For documentation changes, note which section(s) of the article the
   documentation is describing, so the claim can be checked against the
   source text.

See [`docs/repository-guide.md`](docs/repository-guide.md) for a fuller
orientation to how the repository is organized.

## Code of conduct

Participation in this repository is governed by
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
