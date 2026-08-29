# Tail-Contrast Smoothing v1.0.3

This release contains the tested analysis code, environment specification,
aggregate outputs, figures, and checksums used for the IEEE Signal Processing
Letters reproducibility package. It excludes the manuscript, LaTeX sources,
credentials, and source datasets. Dataset acquisition and local placement are
documented in `REPRODUCING.md`.

This patch release adds the prespecified alarm-fraction sensitivity analysis
for all six datasets at fractions 0.005, 0.01, and 0.05. The primary operating
point remains 0.005; the two wider fractions are secondary checks and are not
selected by achieved F1. The release also adds validation for the complete
method-by-dataset-by-fraction grid and updates the reproduction entry point.
