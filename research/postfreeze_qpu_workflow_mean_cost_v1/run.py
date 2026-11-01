#!/usr/bin/env python3
"""Target-blind A/B inference cost campaign for the expanded 3,443-row study."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from pathlib import Path

os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})

import numpy as np
import torch

from common import ART, METHODS, ROOT, SRC, atomic_json, load_panel, load_split, rows, sha, stable, verify_plan_pins
from loaders import StudyInputs
from timing import check_prediction

GIB = 1024**3


def gpu_inventory():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True)
    foreign = []
    desktops = []
    desktop_paths = {"/usr/bin/ptyxis", "/usr/share/rustdesk/rustdesk"}
    for line in raw.splitlines():
        if not line.strip():
            continue
        pid, name = line.split(",", 1)
        if int(pid.strip()) == os.getpid():
            continue
        (desktops if name.strip() in desktop_paths else foreign).append(line)
    if foreign:
        raise RuntimeError(f"foreign_GPU_compute_process_present:{foreign}")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [x.strip() for x in query.splitlines()[0].split(",")]
    total, used = int(fields[1]) * 1024**2, int(fields[2]) * 1024**2
    if total - used < 4 * GIB:
        raise RuntimeError(f"less_than_4GiB_GPU_headroom:{total-used}")
    return {"compute_processes": raw, "desktop_contexts": desktops, "gpu_query": query,
            "gpu_total_bytes": total, "gpu_used_bytes": used, "gpu_headroom_bytes": total-used}


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_missing_no_silent_CPU_fallback")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 2:
        raise RuntimeError("fewer_than_two_allowed_CPUs")
    affinity = affinity[:2]
    os.sched_setaffinity(0, affinity)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits, threadpool_info
    threadpool_limits(limits=1)
    gpu = gpu_inventory()
    import sklearn, torch_geometric
    return {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cpu_affinity": affinity, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(), "threadpools": threadpool_info(),
        "load_average": os.getloadavg(), "gpu_inventory": gpu,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def atomic_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with tmp.open("w", encoding="utf-8") as f:
        for item in values:
            f.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def lock_manifest(environment):
    panel, panel_by_id, ledger = load_panel()
    all_hashes = sorted({r["graph_artifact"]: r["graph_artifact_sha256"] for r in ledger.values()}.items())
    if len(all_hashes) != 466:
        raise RuntimeError(f"unique_graph_identity_count_changed:{len(all_hashes)}")
    for rel, expected in all_hashes:
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"graph_pin_mismatch:{rel}")
    canary_ids = {}
    for ev in ("E1", "E2", "E3"):
        split = load_split(ev)
        _, pby, lby = load_panel()
        canary_ids[ev] = {}
        for case, ids in split.items():
            selected = set()
            by_source = {}
            for cid in ids:
                by_source.setdefault(pby[cid]["source_id"], []).append(cid)
            for source, values in by_source.items():
                ordered = sorted(values, key=lambda cid: (int(lby[cid]["graph_node_count"]), cid))
                for cid in (ordered[len(ordered)//2], ordered[-1]):
                    selected.add(cid)
            canary_ids[ev][str(case)] = sorted(selected)
    checkpoints = {}
    states = {}
    for ev in ("E1", "E2", "E3"):
        for case in split_cases(ev):
            for method in ("M0", "M1", "G0", "G1"):
                for seed in (42, 1234, 31415):
                    p = SRC / "training" / ev / f"fold_{case}" / method / f"seed_{seed}" / "checkpoint.pt"
                    checkpoints[str(p.relative_to(ROOT))] = sha(p)
            for method in ("R0", "C0"):
                folder = ART / ("recovered_classical" if method == "R0" else "../postfreeze_qpu_workflow_mean_expansion_v1/classical")
                if method == "R0":
                    p = folder / ev / f"fold_{case}" / "estimator.joblib"
                    receipt = p.parent / "receipt.json"
                    states[str(p.relative_to(ROOT))] = sha(p)
                    states[str(receipt.relative_to(ROOT))] = sha(receipt)
                else:
                    p = SRC / "classical" / ev / f"fold_{case}" / "constant_manifest.json"
                    states[str(p.relative_to(ROOT))] = sha(p)
    pinned = verify_plan_pins()["evidence_hashes"]
    code = sorted((p.relative_to(ROOT).as_posix(), sha(p)) for p in Path(__file__).parent.glob("*.py"))
    payload = {
        "schema_version": 1, "status": "LOCKED_PRE_TIMING_CANARY",
        "cohort": {"rows": len(panel), "unique_ids": len(panel_by_id), "groups": len({r["group_id"] for r in panel}),
                   "sources": {s: sum(r["source_id"] == s for r in panel) for s in sorted({r["source_id"] for r in panel})}},
        "source_study": "artifacts/postfreeze_qpu_workflow_mean_expansion_v1",
        "methods": list(METHODS), "model_seeds": [42, 1234, 31415],
        "target_read_by_timing_worker": False, "accuracy_source": "saved reduced predictions/metrics; targets never enter timing calls",
        "clocks": {
            "A": "CPU ready inputs; includes singleton batching, H2D, 3 seed forwards, median, host scalar and CUDA completion; excludes feature transforms",
            "B": "raw CPU globals/graph arrays/native5; includes saved transform application, graph expansion, tensorization, all A work; no normalized-cache reuse",
        },
        "repetitions": {"canary": {"warmups": 1, "timed": 2}, "full": {"sessions": 2, "warmups_per_method_request_session": 3, "timed_per_method_request_session": 10, "batch_size": 1}},
        "paired_canary_ids": canary_ids,
        "input_and_model_pins": {"source_evidence_hashes": pinned, "checkpoints": checkpoints,
                                  "serving_states": states, "unique_logical_graphs": dict(all_hashes)},
        "code_hashes": code,
        "runtime_environment": environment,
        "replay_tolerances": {"neural_abs": 1e-5, "neural_rel": 1e-6, "R0_abs_seconds": 1e-9, "C0_exact": True},
        "resource_guards": {"rss_bytes": 8*GIB, "cuda_allocated_or_reserved_bytes": 8*GIB,
                            "gpu_headroom_at_start_bytes": 4*GIB, "worker_no_progress_seconds": 600},
        "scientific_constraints": {"neural_training": False, "HPO": False, "QPU_jobs": False,
                                   "simulator_execution": False, "frozen_results_overwritten": False},
    }
    payload["fingerprint"] = stable(payload)
    run_root = ART / "v2"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "timing_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != payload:
            raise RuntimeError("existing_timing_manifest_differs_refuse_overwrite")
    else:
        atomic_json(manifest_path, payload)
    return payload


def split_cases(evaluation):
    if evaluation in {"E1", "E2"}:
        return range(5)
    split = rows(SRC / "splits/E3_outer.csv")
    return range(len(sorted({r["heldout_backend"] for r in split})))


@contextlib.contextmanager
def owner_lock():
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / ".inference_timing.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another_timing_worker_owns_lock") from exc
        yield


def resources():
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    record = {"process_peak_rss_bytes": rss, "cuda_max_allocated_bytes": allocated,
              "cuda_max_reserved_bytes": reserved, "cuda_current_allocated_bytes": torch.cuda.memory_allocated(),
              "cuda_current_reserved_bytes": torch.cuda.memory_reserved()}
    if rss > 8*GIB or max(allocated, reserved) > 8*GIB:
        raise RuntimeError(f"inference_resource_guard_exceeded:{record}")
    return record


def stage_predict(inputs, method, cid, clock):
    from timing import neural_predict, predict_r0, predict_c0
    request = inputs.raw_request(cid, method)
    if method in {"M0", "M1", "G0", "G1"}:
        graph = method.startswith("G")
        if clock == "A":
            ready = inputs.prepare_ready(method, request)
            return neural_predict(inputs.models[method], ready,
                                  lambda item, graph, device: __import__("run_mali_full_features")._batch_to_device([item], device, graph),
                                  graph=graph)
        return neural_predict(inputs.models[method], request,
                              lambda item, graph, device: __import__("run_mali_full_features")._batch_to_device([item], device, graph),
                              graph=graph, prepare_fn=lambda raw: inputs.prepare_raw_B(method, raw))
    if method == "R0":
        return (*predict_r0(inputs.estimators[method], request["native5"]), {})
    return (*predict_c0(inputs.constants[method]), {})


def measured_call(inputs, method, cid, clock):
    if method in {"R0", "C0"}:
        prediction, elapsed, seeds = stage_predict(inputs, method, cid, clock)
    else:
        prediction, seeds, elapsed = stage_predict(inputs, method, cid, clock)
    inputs.check(method, cid, prediction, seeds)
    return {"method": method, "elapsed_ns": int(elapsed), "prediction_seconds": float(prediction),
            "seed_predictions_seconds": seeds, "parity": "PASS"}


def selected_ids(inputs, phase, case, session):
    if phase == "canary":
        payload = json.loads((ART / "v2/timing_manifest.json").read_text())
        return payload["paired_canary_ids"][inputs.evaluation][str(case)]
    ids = inputs.test_by_case[case]
    return sorted(ids, reverse=session == 2)


def run_phase(phase, evaluation, clock=None, session=1, resume=False):
    with owner_lock():
        environment = setup()
        if phase == "preflight":
            payload = lock_manifest(environment)
            receipt = {"status": "PASS_TARGET_BLIND_PREFLIGHT", "fingerprint": payload["fingerprint"],
                       "environment": environment, "cohort_rows": payload["cohort"]["rows"],
                       "groups": payload["cohort"]["groups"], "canary_ids": sum(len(v) for cases in payload["paired_canary_ids"].values() for v in cases.values()),
                       "no_model_forward_timed": True, "no_targets_loaded": True}
            atomic_json(ART / "v2/preflight_receipt.json", receipt)
            return receipt
        run_root = ART / "v2"
        manifest = json.loads((run_root / "timing_manifest.json").read_text())
        if stable({k:v for k,v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
            raise RuntimeError("timing_manifest_fingerprint_invalid")
        if environment != manifest["runtime_environment"]:
            # Load average and instantaneous GPU inventory are expected to move. Compare the fixed serving contract only.
            for key in ("python", "numpy", "sklearn", "torch", "torch_geometric", "cuda_runtime", "gpu_name", "cpu_affinity"):
                if environment[key] != manifest["runtime_environment"][key]:
                    raise RuntimeError(f"timing_environment_changed:{key}")
        if phase == "measure":
            canary = run_root / "canary" / evaluation / f"clock_{clock}" / "receipt.json"
            if not canary.is_file() or json.loads(canary.read_text()).get("status") != "PASS_CANARY":
                raise RuntimeError(f"canary_missing_or_failed:{evaluation}:{clock}")
        if phase == "canary" and not (run_root / "preflight_receipt.json").is_file():
            raise RuntimeError("preflight_required")
        output_base = run_root / phase / evaluation / (f"clock_{clock}" if clock else "")
        if phase == "measure":
            output_base = output_base / f"session_{session}"
        output_base.mkdir(parents=True, exist_ok=True)
        inputs = StudyInputs(evaluation)
        canary = phase == "canary"
        methods = list(METHODS)
        repetitions = list(range(-1, 2)) if canary else list(range(-3, 10))
        assigned = 0; completed = 0; case_receipts = []
        for case in range(len(inputs.test_by_case)):
            ids = selected_ids(inputs, phase, case, session)
            inputs.load_case(case)
            case_dir = output_base / f"case_{case}"
            case_dir.mkdir(parents=True, exist_ok=True)
            for unit_index, cid in enumerate(ids):
                assigned += 1
                output = case_dir / f"{stable(cid)}.json"
                if output.exists():
                    if not resume:
                        raise RuntimeError(f"existing_unit_use_resume:{output}")
                    saved = json.loads(output.read_text())
                    if saved.get("fingerprint") != manifest["fingerprint"] or saved.get("canonical_observation_id") != cid or saved.get("clock") != clock:
                        raise RuntimeError(f"resume_unit_identity_mismatch:{output}")
                    completed += 1
                    continue
                records = []
                # Untimed replay checks happen before any recorded repetitions.
                for method in methods:
                    measured_call(inputs, method, cid, clock or "A")
                for repetition in repetitions:
                    order = methods[(unit_index + repetition + session) % len(methods):] + methods[:(unit_index + repetition + session) % len(methods)]
                    if session == 2:
                        order = list(reversed(order))
                    for position, method in enumerate(order):
                        row = measured_call(inputs, method, cid, clock or "A")
                        row.update({"repetition": repetition, "sample_type": "warmup" if repetition < 0 else "timed",
                                    "order_position": position, "case": case, "session": session})
                        records.append(row)
                payload = {"status": "PASS", "fingerprint": manifest["fingerprint"],
                           "evaluation": evaluation, "case": case, "clock": clock or "A+B",
                           "session": session, "canonical_observation_id": cid,
                           "group_id": inputs.panel_by_id[cid]["group_id"],
                           "source_id": inputs.panel_by_id[cid]["source_id"],
                           "backend": inputs.panel_by_id[cid]["backend"],
                           "graph_nodes": int(inputs.ledger[cid]["graph_node_count"]),
                           "records": records, "resources_after_unit": resources(),
                           "no_target_values_loaded": True}
                atomic_json(output, payload)
                completed += 1
                if completed % (5 if canary else 25) == 0 or canary:
                    print(json.dumps({"phase": phase, "evaluation": evaluation, "clock": clock,
                                      "case": case, "completed_units": completed, "assigned_units": assigned,
                                      "latest_id": cid}), flush=True)
        if canary:
            receipt = {"status": "PASS_CANARY", "fingerprint": manifest["fingerprint"],
                       "evaluation": evaluation, "clock": clock, "unit_count": completed,
                       "methods": methods, "diagnostic_repetitions": 2, "warmups": 1,
                       "all_predictions_replayed": True, "no_targets_loaded": True,
                       "resource_checks": True, "note": "Canary clocks A and B are run separately; for diagnostic calls, A-ready preparation is used only for the A route."}
            atomic_json(output_base / "receipt.json", receipt)
            return receipt
        receipt = {"status": "PASS_TIMING_SESSION", "fingerprint": manifest["fingerprint"],
                   "evaluation": evaluation, "clock": clock, "session": session,
                   "completed_units": completed, "assigned_units": assigned,
                   "methods": methods, "timed_repetitions": 10, "warmups": 3,
                   "no_target_values_loaded": True, "no_training_or_refit": True}
        atomic_json(output_base / f"session_{session}_receipt.json", receipt)
        return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("preflight", "canary", "measure"), required=True)
    ap.add_argument("--evaluation", choices=("E1", "E2", "E3"), required=True)
    ap.add_argument("--clock", choices=("A", "B"))
    ap.add_argument("--session", type=int, choices=(1, 2), default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.phase != "preflight" and not args.clock:
        ap.error("--clock A or B is required for canary/measure")
    receipt = run_phase(args.phase, args.evaluation, args.clock, args.session, args.resume)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
