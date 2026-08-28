import json
from pathlib import Path

from pypdf import PdfReader


MM_PER_POINT = 25.4 / 72.0


def _figure_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "submission-spl" / "figures"


def test_case_figure_is_single_column_vector_output():
    figure_dir = _figure_dir()
    pdf = figure_dir / "fig2_case_instances.pdf"
    svg = figure_dir / "fig2_case_instances.svg"
    assert pdf.exists() and svg.exists()
    page = PdfReader(str(pdf)).pages[0]
    width_mm = float(page.mediabox.width) * MM_PER_POINT
    assert 87.0 <= width_mm <= 89.0
    assert "<text" in svg.read_text(encoding="utf-8")


def test_case_figure_has_vertical_three_case_structure():
    metadata = json.loads((_figure_dir() / "figure_metadata.json").read_text(encoding="utf-8"))
    fig2 = metadata["fig2"]
    assert fig2["orientation"] == "vertical"
    assert fig2["panels"] == ["positive:SMD", "positive:PSM", "null-compatible:HAI", "reversal:MSL"]
    assert fig2["underpowered_rows"] == ["SMAP", "SWaT"]


def test_case_figure_uses_global_labels_and_shared_pair_scaling():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_spl_case_figure.py"
    ).read_text(encoding="utf-8")
    assert source.count("fig.supxlabel(") == 1
    assert "ax.set_xlabel(" not in source
    assert "pair_scale" in source
    assert "for values in (normal_arr, anomaly)" not in source
