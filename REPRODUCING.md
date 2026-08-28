# Reproducing the Aggregate Results

Run the commands below from this repository root. The public package uses
relative paths only.

## Environment

```powershell
conda env create --file .\environment\environment.yml
conda run -n paper47-spl python -m pip install -e ".\code[test]"
conda run -n paper47-spl python -m pytest .\code\tests -q
```

The analysis is CPU-compatible. A GPU is not required.

## Dataset acquisition

The release does not include source datasets or score archives. Obtain the
Time-Series-Library data from its official provider and place it under a local
path outside this repository. HAI 21.03 is obtained from its official source
repository under that provider's terms. Do not commit either dataset.

The corrected pipeline expects score artifacts with `scores`, `labels`, and
`training_normal_scores` fields. Segment boundaries must be preserved for
multi-file test sets. The exact source revisions and hashes used for the paper
are recorded in the authors' private audit records.

## Dry run and analysis stages

```powershell
conda run -n paper47-spl python .\code\scripts\reproduce_spl.py --dry-run
conda run -n paper47-spl python .\code\scripts\reproduce_spl.py `
  --score-root ..\local-score-artifacts\raw `
  --output-root .\results\reproduced
```

The dry run prints the stages without reading data. A full run writes only to
the selected output directory. Aggregate tables can be compared with the
tracked CSV files using a script or a spreadsheet; no values are copied into
the manuscript by hand.
