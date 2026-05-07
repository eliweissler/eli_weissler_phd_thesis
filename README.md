# CU Boulder Graduate Thesis

This repository contains my graduate thesis, built using LaTeX. Every push to the main branch automatically compiles the latest version using GitHub Actions.

## View the Thesis

The latest compiled version of the thesis is automatically built and stored on the `pdf` branch. 

📄 **[View the latest Compiled PDF](../../raw/pdf/thesis.pdf)**

*(Note: If you are viewing this on GitHub, the link above will download the latest synced PDF.)*

## Acknowledgements

This thesis uses an improved LaTeX template based on the [CU Boulder Thesis Template](https://github.com/GiacoCorsiglia/cu-boulder-thesis-template) created by Giaco Corsiglia. It implements various improvements over the graduate school's original template, including better footnote typesetting and macro support.

## Building Locally

To compile this thesis locally, simply use `latexmk` or your preferred LaTeX engine:
```bash
latexmk -pdf thesis.tex
```

## Structure

- `thesis.tex`: The root LaTeX file that stitches everything together.
- `thesis.bib`: Reference files for bibliography.
- `chapter-*/`: The individual chapter contents.
- `macros.tex`: Custom LaTeX definitions.
