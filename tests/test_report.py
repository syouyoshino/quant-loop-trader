import re
from pathlib import Path

from quant_loop_trader.report import generate_report


def test_report_generated_with_sections(tmp_path):
    out = generate_report(out_dir=tmp_path)
    text = out.read_text()
    assert out.exists() and text
    for section in ["Experiment activity", "Belief state", "Recent lessons",
                    "Model registry", "Data health", "Research frontier"]:
        assert section in text
    assert "OBSERVATION" in text  # constitution reminder present
