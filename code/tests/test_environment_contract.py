from pathlib import Path

import yaml


def test_conda_environment_has_fixed_name_and_python_series():
    spec = yaml.safe_load(
        Path("../environment/environment.yml").read_text(encoding="utf-8")
    )
    assert spec["name"] == "paper47-spl"
    assert "python=3.11" in spec["dependencies"]


def test_runtime_manifests_declare_scipy_and_figure_audit_dependencies():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("../environment/requirements.txt").read_text(encoding="utf-8")
    assert '"scipy>=1.10,<2"' in pyproject
    assert "scipy>=1.10,<2" in requirements
    assert "pillow" in requirements.lower()
    assert "pypdf" in requirements.lower()
