# Peak-Matched Confirmation Windows

This repository contains the analysis code, environment specification, tests,
and aggregate outputs for the IEEE Signal Processing Letters study
"Peak-Matched Confirmation Windows for Time-Series Anomaly Scores: A
Tail-Location Design Criterion."

The release implements the fixed event-level protocol used in the paper. It
does not redistribute source datasets, score archives, the manuscript, or its
LaTeX source. Dataset acquisition instructions and the expected local layout
are in `REPRODUCING.md`.

## Contents

- `code/`: Python package, experiment scripts, and tests.
- `environment/`: Conda and container dependency declarations.
- `results/tables/`: verified aggregate CSV outputs.
- `results/figures/`: single-column vector figures used to inspect the result.

The primary protocol fixes a three-sample causal confirmation window, peak
matching caliper 0.2, alarm fraction 0.005, and a 20-pair primary evidence
gate. Rows below that gate remain explicitly exploratory.

## License

The code is released under the MIT License. Dataset terms remain those of the
original providers and are not changed by this repository.
