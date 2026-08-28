import hashlib
from pathlib import Path

from pypdf import PdfReader
from scripts.generate_release_manifest import canonical_bytes, public_paths


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DATASETS = {"SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI"}


def test_release_text_files_have_a_cross_platform_line_ending_contract():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes
    assert "*.pdf binary" in attributes

    entries = {}
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        entries[relative] = digest

    for relative_path in public_paths(ROOT):
        path = ROOT / relative_path
        relative = relative_path.as_posix()
        data = canonical_bytes(path)
        assert relative in entries, f"manifest entry missing for {relative}"
        assert hashlib.sha256(data).hexdigest() == entries[relative]


def test_aggregate_tables_have_the_fixed_six_dataset_case_set():
    import pandas as pd

    case = pd.read_csv(TABLES / "dataset_tail_contrast.csv")
    performance = pd.read_csv(TABLES / "delay_aware_performance.csv")
    assert set(case["dataset"]) == DATASETS
    assert set(performance["dataset"]) == DATASETS


def test_vector_figures_are_single_column_assets():
    for name in ("fig1_regime_alignment.pdf", "fig2_case_instances.pdf"):
        page = PdfReader(str(FIGURES / name)).pages[0]
        width_mm = float(page.mediabox.width) * 25.4 / 72.0
        assert 87.0 <= width_mm <= 89.0


def test_package_imports_and_dry_run_are_relative():
    from hbpc.spl_result_verifier import verify_spl_results
    from scripts.reproduce_spl import main

    assert callable(verify_spl_results)
    assert main(["--dry-run"]) == 0
