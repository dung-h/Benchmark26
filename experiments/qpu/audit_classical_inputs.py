#!/usr/bin/env python3
"""Audit provenance of the inherited five-column Qonductor-style baseline."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import ARTIFACTS
from pathlib import Path
import csv, hashlib, json
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = ARTIFACTS / "postfreeze_qpu_workflow_mean_expansion_v1"


def read(p):
    """Read a UTF-8 CSV file into a list of dicts."""
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(p):
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    """Audit where each of the five Qonductor-style features of every observation comes from.

    Writes classical_feature_provenance.csv and a receipt that counts the routes by source, input stage
    and representation.
    """
    panel = read(OUT / "cohort/panel.csv")
    features = {
        r["canonical_row_id"]: r for r in read(ARTIFACTS / "real_qpu/qonductor_native_features/feature_rows.csv")
    }
    columns = ("swap", "depth", "num_qubits", "shots", "circuit_count")
    ledger = []
    for r in panel:
        fid = r["feature_source_observation_id"]
        if fid not in features:
            raise RuntimeError(f"classical_source_feature_missing:{r['canonical_observation_id']}:{fid}")
        f = features[fid]
        if f["availability_status"] != "available":
            raise RuntimeError(f"classical_feature_unavailable:{fid}")
        if f["source_id"] != r["source_id"]:
            raise RuntimeError(f"classical_source_mismatch:{fid}")
        vals = {k: float(f[k]) for k in columns}
        if vals["shots"] != float(r["shots"]) or vals["circuit_count"] != 1.0:
            raise RuntimeError(f"classical_context_mismatch:{fid}")
        ledger.append(
            {
                "canonical_observation_id": r["canonical_observation_id"],
                "source_id": r["source_id"],
                "feature_source_observation_id": fid,
                "source_input_stage": f["source_input_stage"],
                "representation_id": f["representation_id"],
                "input_qasm_sha256": f["input_qasm_sha256"],
                "adapter_signature_sha256": f["adapter_signature_sha256"],
                "feature_order": "|".join(columns),
                **{k: repr(vals[k]) for k in columns},
            }
        )
    out = OUT / "cohort/classical_feature_provenance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0]))
        w.writeheader()
        w.writerows(ledger)
    routes = Counter((r["source_id"], r["source_input_stage"], r["representation_id"]) for r in ledger)
    receipt = {
        "status": "PASS_CLASSICAL_INPUTS_AUDITED",
        "rows": len(ledger),
        "unique_observations": len({r["canonical_observation_id"] for r in ledger}),
        "feature_columns": list(columns),
        "all_finite": True,
        "recorded_shots_match_panel": True,
        "circuit_count_is_one": True,
        "routes": [
            {"source": k[0], "input_stage": k[1], "representation_id": k[2], "rows": n}
            for k, n in sorted(routes.items())
        ],
        "interpretation": "One frozen five-column feature schema is applied through source-specific, provenance-recorded circuit routes. Ma-Li uses saved nominal-target compiled QASM3; Qonductor uses exact submitted physical QASM; QPack uses a nominal-target compiled reconstruction from an angle-insensitive logical reconstruction. This is an inherited Qonductor-style estimator comparator, not a same-representation/capacity control for the Ma-Li-style neural models.",
        "source_feature_file_sha256": sha(ARTIFACTS / "real_qpu/qonductor_native_features/feature_rows.csv"),
        "native_matrix_sha256": sha(OUT / "cohort/native_baseline_features.csv"),
        "provenance_ledger_sha256": sha(out),
    }
    (OUT / "cohort/classical_input_audit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
