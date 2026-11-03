"""Export identity-pinned archived graph/duration fixtures; no circuit execution."""

import argparse, csv, hashlib, importlib.metadata, json, pickle
from pathlib import Path
import numpy as np
from qiskit_ibm_runtime import fake_provider

ap = argparse.ArgumentParser()
ap.add_argument("--source-root", type=Path, required=True)
ap.add_argument("--output-root", type=Path, required=True)
a = ap.parse_args()
assert importlib.metadata.version("qiskit") == "2.5.2" and importlib.metadata.version("qiskit-ibm-runtime") == "0.49.0"
R = a.source_root
D = a.output_root / "artifacts/analytical_kernel_validation"
D.mkdir(parents=True, exist_ok=True)
(D / "fixtures").mkdir(exist_ok=True)
if (D / "manifest.json").exists():
    raise RuntimeError("Refuse fixture overwrite")
read = lambda p: list(csv.DictReader(p.open()))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
A = R / "artifacts/postfreeze_unified_analytical_completion_v1"
G = R / "artifacts/benchmark_v3/real_qpu/compiled_unified/graphs"
raw = read(A / "raw_features.csv")
reg = {
    r["snapshot_id"]: r
    for r in read(R / "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv")
}
chosen = {}
for r in raw:
    sid = r["snapshot_id"]
    sz = (G / r["graph_file"]).stat().st_size
    if sid not in chosen or sz < chosen[sid][0]:
        chosen[sid] = (sz, r)
entries = []
fields = [
    "depth_1q",
    "depth_2q",
    "unitary_depth",
    "paper_kappa1",
    "paper_kappa2",
    "notads_code",
    "notads_paper",
    "hyperq_full_seconds",
    "hyperq_unitary_seconds",
    "measurement_count",
]
for sid, (_, r) in sorted(chosen.items()):
    rec = reg[sid]
    pins = {}
    for pkey, hkey in [
        ("configuration_path", "configuration_sha256"),
        ("properties_path", "properties_sha256"),
        ("class_module_path", "class_module_sha256"),
    ]:
        p = Path(rec[pkey])
        assert sha(p) == rec[hkey]
        pins[p.name] = rec[hkey]
    p = G / r["graph_file"]
    assert sha(p) == r["graph_file_sha256"]
    g = pickle.load(p.open("rb"))
    backend = getattr(fake_provider, rec["fake_backend_class"])()
    q0 = np.frombuffer(g["q0"], dtype=np.int32)
    q1 = np.frombuffer(g["q1"], dtype=np.int32)
    arity = np.frombuffer(g["arity"], dtype=np.uint8)
    durations = np.array(
        [
            backend.target[name][(int(q0[i]),) if arity[i] == 1 else (int(q0[i]), int(q1[i]))].duration
            for i, name in enumerate(g["names"])
        ],
        dtype=np.float64,
    )
    assert np.isfinite(durations).all() and (durations >= 0).all()
    dest = D / "fixtures" / f"{sid.split('::')[1]}.npz"
    np.savez_compressed(
        dest,
        arity=arity,
        q0=q0,
        q1=q1,
        measure=np.array([x == "measure" for x in g["names"]], dtype=np.uint8),
        durations=durations,
        edge_src=np.frombuffer(g["edge_src"], dtype=np.int32),
        edge_dst=np.frombuffer(g["edge_dst"], dtype=np.int32),
        num_qubits=np.array(g["num_qubits"]),
    )
    entries.append(
        {
            "fixture": str(dest.relative_to(a.output_root)),
            "fixture_sha256": sha(dest),
            "snapshot_id": sid,
            "snapshot_asset_pins": pins,
            "original_graph": r["graph_file"],
            "original_graph_sha256": r["graph_file_sha256"],
            "operations": int(g["operation_count"]),
            "expected": {f: float(r[f]) for f in fields},
        }
    )
out = {
    "scope": "Frozen graph and nominal Target-duration fixtures; no labels or executions",
    "sampling_rule": "smallest serialized saved native graph for every snapshot present in unified raw_features.csv",
    "raw_features_sha256": sha(A / "raw_features.csv"),
    "qiskit": "2.5.2",
    "qiskit_ibm_runtime": "0.49.0",
    "fixtures": entries,
}
(D / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
print("Exported", len(entries), "fixtures; bytes", sum((a.output_root / e["fixture"]).stat().st_size for e in entries))
