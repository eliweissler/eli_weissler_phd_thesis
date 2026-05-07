# CU Boulder PhD Thesis

This repository contains my PhD thesis, built using LaTeX. Every push to the main branch automatically compiles the latest version using GitHub Actions.

## View the Thesis

The latest compiled version of the thesis is automatically built and stored on the `pdf` branch. 

📄 **[View the latest Compiled PDF](../../raw/pdf/thesis.pdf)**

*(Note: If you are viewing this on GitHub, the link above will download the latest synced PDF.)*

## Acknowledgements

This thesis uses an improved LaTeX template based on the [CU Boulder Thesis Template](https://github.com/GiacoCorsiglia/cu-boulder-thesis-template) created by Giaco Corsiglia. It implements various improvements over the graduate school's original template, including better footnote typesetting and macro support.

## Building Locally

To compile this thesis locally, run `latexmk` from the repository root with shell-escape enabled:
```bash
latexmk -pdf -shell-escape thesis.tex
```

`-shell-escape` is required because:
- The `standalone` package (`mode=buildnew`) shells out to pre-compile each chapter's `figs/tikz.tex` into a PDF.
- `glossaries-extra` (`automake`) invokes `makeglossaries` automatically via `\write18`.

Building from a subdirectory (e.g. running `pdflatex` directly inside a `figs/` folder) also works — each `figs/tikz.tex` uses `\IfFileExists` to locate `custom_circuitikz.sty` either in the current directory or two levels up.

## Structure

- `thesis.tex`: The root LaTeX file that stitches everything together.
- `thesis.bib`: Reference files for bibliography.
- `chapter-*/`: The individual chapter contents.
  - `chapter-*/figs/`: Per-chapter standalone TikZ figures (e.g. `tikz.tex`), pre-compiled by the `standalone` package.
- `appendix-*/`: Appendix contents, with their own `figs/` directories.
- `macros.tex`: Custom LaTeX definitions.
- `custom_circuitikz.sty`: Extra `circuitikz` element definitions (Josephson junctions, SNAILs, SQUIDs, etc.). Definitions courtesy of Thomas Smith.
- `elements/`: Path/shape definitions for the custom circuit elements used by `custom_circuitikz.sty`.
