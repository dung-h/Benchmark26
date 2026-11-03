"""Check stage accounting and input validation in the portable latency tools."""
import csv
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "experiments/latency/compose.py"


def run_compose(tmp_path, rows):
    source, output = tmp_path / "stages.csv", tmp_path / "totals.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = subprocess.run([sys.executable, str(COMPOSE), "--stages", str(source),
                             "--output", str(output)], capture_output=True, text=True)
    return result, output


def test_compilation_counted_once_and_groups_weighted_equally(tmp_path):
    rows = [dict(id=str(i), group_id=group, method="R0", parse_compile_ns=compile_ns,
                 method_stage_ns=2_000_000, inference_ns=3_000_000)
            for i, (group, compile_ns) in enumerate([("a", 10_000_000),
                                                    ("a", 20_000_000),
                                                    ("b", 100_000_000)])]
    result, output = run_compose(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    with output.open() as handle:
        totals = [float(r["composite_ms"]) for r in csv.DictReader(handle)]
    assert totals == [15, 25, 105]
    # Equal groups: mean(20, 105), rather than the row-weighted mean(15,25,105).
    assert "group_mean_ms 62.5" in result.stdout


@pytest.mark.parametrize("problem", ["duplicate", "negative", "nonfinite"])
def test_invalid_stage_input_does_not_create_output(tmp_path, problem):
    row = dict(id="a", group_id="g", method="QISKIT", parse_compile_ns=10,
               method_stage_ns=20, inference_ns=0)
    rows = [row]
    if problem == "duplicate":
        rows.append(dict(row))
    elif problem == "negative":
        row["parse_compile_ns"] = -1
    else:
        row["method_stage_ns"] = "nan"
    result, output = run_compose(tmp_path, rows)
    assert result.returncode != 0
    assert not output.exists()
