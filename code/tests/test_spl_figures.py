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
    tiff = figure_dir / "fig2_case_instances.tiff"
    png = figure_dir / "fig2_case_instances.png"
    assert pdf.exists() and svg.exists() and tiff.exists() and png.exists()
    page = PdfReader(str(pdf)).pages[0]
    width_mm = float(page.mediabox.width) * MM_PER_POINT
    assert 87.0 <= width_mm <= 89.0
    assert "<text" in svg.read_text(encoding="utf-8")


def test_case_figure_has_vertical_three_case_structure():
    metadata = json.loads((_figure_dir() / "figure_metadata.json").read_text(encoding="utf-8"))
    fig2 = metadata["fig2"]
    assert fig2["orientation"] == "vertical"
    assert fig2["archetype"] == "schematic-led composite"
    assert fig2["case_order"] == ["positive", "null_compatible", "reversal"]
    assert fig2["components"] == {
        "positive": ["prediction", "empirical"],
        "null_compatible": ["prediction", "empirical"],
        "reversal": ["prediction", "empirical"],
    }
    assert fig2["datasets"] == {
        "positive": ["SMD", "PSM"],
        "null_compatible": ["HAI"],
        "reversal": ["MSL"],
    }
    upper, middle, lower = fig2["case_bounds"]
    assert upper[1] > middle[1] > lower[1]
    assert upper[0] == middle[0] == lower[0]
    assert fig2["underpowered_rows"] == ["SMAP", "SWaT"]
    assert fig2["minimum_font_pt"] >= 7.5
    assert fig2["minimum_line_width_pt"] >= 0.75
    assert fig2["minimum_marker_size_pt"] >= 3.5
    assert fig2["encodings"]["anomalous"] == ["blue", "solid", "filled"]
    assert fig2["encodings"]["normal"] == ["orange", "dashed", "open"]


def test_case_figure_uses_global_labels_and_shared_pair_scaling():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_spl_case_figure.py"
    ).read_text(encoding="utf-8")
    assert source.count("fig.supxlabel(") == 1
    assert "ax.set_xlabel(" not in source
    assert "pair_scale" in source
    assert "fig.legend(" not in source
    assert "closest matched pair to the dataset mean tail difference" in source
    assert "Model prediction" in source
    assert "Matched score tails" in source
