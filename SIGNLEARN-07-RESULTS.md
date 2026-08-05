# SignLearn paper

The complete manuscript is provided in two editable formats:

- `signlearn_paper.md` is the most readable source for GitHub review.
- `signlearn_paper.tex` is a submission-oriented LaTeX source.
- `references.bib` contains the bibliography used by both versions.

Before submitting, add the authors and affiliations, choose the target venue's official template,
and adapt the manuscript to that venue's disclosure and formatting rules.

## Build the LaTeX version

With a TeX distribution and `latexmk` installed:

```bash
latexmk -pdf signlearn_paper.tex
```

Or use the traditional sequence:

```bash
pdflatex signlearn_paper.tex
bibtex signlearn_paper
pdflatex signlearn_paper.tex
pdflatex signlearn_paper.tex
```

The repository does not commit generated PDF or LaTeX auxiliary files. The Markdown manuscript is
the canonical narrative draft; keep the two sources synchronized after substantive edits.
