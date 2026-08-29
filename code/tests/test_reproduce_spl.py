import sys

from scripts.reproduce_spl import CODE_ROOT, main


def test_reproduce_spl_prioritizes_its_own_code_tree():
    assert sys.path[0] == str(CODE_ROOT)


def test_reproduce_spl_dry_run_lists_frozen_stages(capsys):
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "spl_experiments" in output
    assert "synthetic_regime_experiments" in output
    assert "swat_filter_experiments" in output
    assert "alarm_fraction_sensitivity" in output
    assert "spl_result_verifier" in output
