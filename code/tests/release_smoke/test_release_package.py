from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DATASETS = {"SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI"}


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
