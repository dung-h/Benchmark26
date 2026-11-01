#!/usr/bin/env python3
"""Measure CUDA batch throughput, then fit a separately identified Ma-Li adaptation.

Reuse the frozen extractor/model/split/checkpoint helpers; do not edit their
receipts. Tune with fitting inputs and synthetic labels only. Logs and profiles
are durable evidence for this execution context, not replacement raw results.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import time
import tempfile

import run_mali_full_features as base

ROOT = base.ROOT
CONTRACT = ROOT / "benchmark_v1/protocol/mali_batch_optimization.json"
DEFAULT_FEATURES = ROOT / "artifacts/real_qpu/mali_full_features_unified"
DEFAULT_OUTPUT = ROOT / "artifacts/real_qpu/mali_full_features_large_batch"
METHOD_IDS = {name: name + "_large_batch" for name in base.METHODS}


def physical_groups(ids, ledger, node_limit, graph):
    """Keep order and intact graphs; bound CPU staging as well as GPU batches."""
    group, count = [], 0
    for cid in ids:
        nodes = int(ledger[cid]["graph_node_count"]) if graph else 0
        if group and count + nodes > node_limit:
            yield group
            group, count = [], 0
        group.append(cid)
        count += nodes
        if graph and count >= node_limit:
            yield group
            group, count = [], 0
    if group:
        yield group


class ExampleCache:
    def __init__(self, context, transform, budget_bytes):
        self.context, self.transform, self.budget = context, transform, budget_bytes
        self.items = OrderedDict()
        self.bytes = 0
        self.verified = set()

    def get(self, cid, variant):
        key = (cid, variant)
        if key in self.items:
            value, size = self.items.pop(key)
            self.items[key] = (value, size)
            return value
        value = base._make_example(cid, variant, self.context["ledger"], None,
                                   self.transform, self.context["feature_root"],
                                   "cpu", False, self.verified)
        size = sum(t.numel() * t.element_size() for _, t in value if hasattr(t, "numel"))
        if size <= self.budget:
            while self.items and self.bytes + size > self.budget:
                _, (_, removed) = self.items.popitem(last=False)
                self.bytes -= removed
            self.items[key] = (value, size)
            self.bytes += size
        return value


def load_context(feature_root):
    protocol, protocol_sha, execution_sha = base.canonical_protocol_hashes()
    inp_path = feature_root / "input_manifest.json"
    inp = json.loads(inp_path.read_text())
    ledger_path = feature_root / "feature_status_ledger.csv"
    if (inp.get("full_panel_materialized") is not True
            or inp.get("canonical_identity_count") != 8767
            or inp.get("protocol_sha256") != protocol_sha
            or inp.get("feature_status_ledger_sha256") != base.sha256_file(ledger_path)):
        raise RuntimeError("materialization_identity_mismatch")
    base.validate_feature_implementation_pins(inp)
    environment = base.validate_neural_environment(inp, feature_root)
    technical = base.validate_technical_gate_receipt(base.TECHNICAL_GATE_RECEIPT_PATH, feature_root)
    ledger_rows = base.read_csv(ledger_path)
    canonical_rows = base.read_csv(base.CANONICAL)
    ledger = {row["canonical_row_id"]: row for row in ledger_rows}
    canonical = {row["canonical_row_id"]: row for row in canonical_rows}
    if len(ledger_rows) != 8767 or len(ledger) != 8767 or set(ledger) != set(canonical):
        raise RuntimeError("canonical_ledger_identity_mismatch")
    preflight = feature_root / "preflight/preflight_report.json"
    if technical["preflight_report_sha256"] != base.sha256_file(preflight):
        raise RuntimeError("parent_preflight_pin_mismatch")
    return {"feature_root": feature_root, "protocol": protocol, "ledger": ledger,
            "canonical": canonical, "outer": base.read_csv(base.OUTER),
            "inner": base.read_csv(base.INNER), "technical": technical,
            "pins": {"parent_protocol": protocol_sha, "parent_execution": execution_sha,
                     "input_manifest": base.sha256_file(inp_path), "feature_ledger": base.sha256_file(ledger_path),
                     "parent_runner": base.sha256_file(Path(base.__file__)),
                     "runner": base.sha256_file(Path(__file__)), "batch_contract": base.sha256_file(CONTRACT),
                     "technical_receipt": base.sha256_file(base.TECHNICAL_GATE_RECEIPT_PATH),
                     "neural_environment": environment}}


def partitions_and_transform(context, fold):
    parts = base.common_eligible_partitions(base.fold_partitions(context["outer"], context["inner"], fold), context["ledger"])
    saved = context["feature_root"] / "training" / f"fold_{fold}/transform.json"
    raw = {cid: [float(context["ledger"][cid][f"global_{i:02d}"]) for i in range(51)] for cid in parts["fit"]}
    glob = base.fit_transform_global(raw, parts["fit"])
    if saved.exists():
        transform = json.loads(saved.read_text())
        if (transform["fit_ids_sha256"] != base.stable_hash(parts["fit"])
                or transform["protocol_sha256"] != context["pins"]["parent_protocol"]
                or transform["input_manifest_sha256"] != context["pins"]["input_manifest"]
                or transform["global"] != glob
                or transform["node"]["fit_ids_sha256"] != base.stable_hash(parts["fit"])):
            raise RuntimeError("parent_transform_identity_mismatch")
    else:
        paths = [(context["feature_root"] / context["ledger"][cid]["graph_artifact"]).resolve() for cid in parts["fit"]]
        transform = {"global": glob, "node": base.fit_node_transform(paths, parts["fit"]),
                     "outer_fold": fold, "fit_ids_sha256": base.stable_hash(parts["fit"]),
                     "protocol_sha256": context["pins"]["parent_protocol"],
                     "input_manifest_sha256": context["pins"]["input_manifest"]}
    return parts, transform


def model_for(variant, transform, recompute, torch):
    cls, _ = base.load_pinned_model()
    model = base.make_model(cls, variant, len(transform["global"]["retained_indices"]), recompute).to("cuda:0")
    if variant == base.METHODS[0]:
        model.mask = model.mask.to("cuda:0")
    return model


def loss_step(model, optimizer, ids, variant, node_limit, cache, targets, torch):
    """Each observation has equal weight, including the final partial batch."""
    graph = variant == base.METHODS[0]
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for group in physical_groups(ids, cache.context["ledger"], node_limit, graph):
        batch = base._batch_to_device([cache.get(cid, variant) for cid in group], "cuda:0", graph)
        predicted = base._prediction(model, batch)
        actual = torch.tensor([targets[cid] for cid in group], dtype=torch.float32, device="cuda:0")
        loss = torch.nn.functional.mse_loss(predicted, actual, reduction="sum") / len(ids)
        loss.backward()
        total += float(loss.detach().cpu())
        del batch, predicted, actual, loss
    optimizer.step()
    return total


def gradient_gate(context, transform, cache, ids, torch):
    """Check full batch versus accumulated whole-graph microbatches on CUDA."""
    base.seed_all(42)
    a = model_for(base.METHODS[0], transform, True, torch)
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=0.0005, weight_decay=0.0001)
    ob = torch.optim.Adam(b.parameters(), lr=0.0005, weight_decay=0.0001)
    targets = {cid: float(index + 1) for index, cid in enumerate(ids)}
    la = loss_step(a, oa, ids, base.METHODS[0], 10**9, cache, targets, torch)
    lb = loss_step(b, ob, ids, base.METHODS[0], 1, cache, targets, torch)
    if not math.isclose(la, lb, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError("batch_loss_parity_failed")
    optimizer_max_difference = 0.0
    gradient_max_difference = 0.0
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        gradient_max_difference = max(gradient_max_difference, float((pa.grad-pb.grad).abs().max()))
        optimizer_max_difference = max(optimizer_max_difference, float((pa-pb).abs().max()))
        if not torch.allclose(pa.grad, pb.grad, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"batch_gradient_parity_failed:{name}")
        if not torch.allclose(pa, pb, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"batch_adam_step_parity_failed:{name}:max_abs={float((pa-pb).abs().max())}:max_grad_abs={float((pa.grad-pb.grad).abs().max())}")
    # A fresh execution context also verifies its optimizer and RNG resume path.
    with tempfile.TemporaryDirectory(prefix="qre-batch-resume-") as temporary:
        path = Path(temporary) / "checkpoint.pt"
        identity = {"fixture": "batch-optimization", "ids": ids}
        base.save_checkpoint(path, {"identity": identity, "model_state": a.state_dict(),
                                   "optimizer_state": oa.state_dict(), "rng_state": base.capture_rng_state()})
        loss_step(a, oa, ids, base.METHODS[0], 1, cache, targets, torch)
        saved = base.load_saved_checkpoint(path, identity, torch)
        b.load_state_dict(saved["model_state"])
        ob.load_state_dict(saved["optimizer_state"])
        base.restore_rng_state(saved["rng_state"])
        loss_step(b, ob, ids, base.METHODS[0], 1, cache, targets, torch)
        for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            if not torch.allclose(pa, pb, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"batch_checkpoint_resume_failed:{name}")
    del a, b, oa, ob
    torch.cuda.empty_cache()
    return {"status": "PASS", "loss_gradient_and_Adam_step": True, "checkpoint_resume": True, "fixture_ids": ids,
            "loss_gradient_rtol": 1e-5, "loss_gradient_atol": 1e-6, "Adam_step_atol": 1e-5,
            "gradient_max_abs_difference": gradient_max_difference,
            "optimizer_max_abs_difference": optimizer_max_difference, "resume_atol": 1e-6}


def tune(context, output, contract, torch, gpu):
    import numpy as np
    parts, transform = partitions_and_transform(context, 0)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "batch_profile.json").exists():
        raise RuntimeError("batch_profile_already_exists_use_train_or_new_output")
    cache = ExampleCache(context, transform, int(contract["host_cache_gib"] * 1024**3))
    small = sorted(parts["fit"], key=lambda cid: int(context["ledger"][cid]["graph_node_count"]))[:2]
    parity = gradient_gate(context, transform, cache, small, torch)
    order = np.random.default_rng(contract["probe_seed"]).permutation(len(parts["fit"]))
    sample = [parts["fit"][int(i)] for i in order[:contract["probe_sample_count"]]]
    largest = max(parts["fit"], key=lambda cid: int(context["ledger"][cid]["graph_node_count"]))
    if largest not in sample:
        sample[-1] = largest
    results = []
    configurations = [(base.METHODS[0], b, n) for b in contract["graph_batch_candidates"]
                      for n in contract["node_budget_candidates"] if b <= len(sample)]
    configurations += [(base.METHODS[1], b, 0) for b in contract["mlp_batch_candidates"]]
    for variant, effective, node_limit in configurations:
        ids = sample if variant == base.METHODS[0] else [parts["fit"][int(i)] for i in order[:4096]]
        targets = {cid: 0.0 for cid in ids}
        # Preload probe inputs: measured GPU throughput is separate from disk warmup.
        cold = time.monotonic()
        for cid in ids:
            cache.get(cid, variant)
        preload_seconds = time.monotonic() - cold
        base.seed_all(42)
        model = model_for(variant, transform, True, torch)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
        loss_step(model, optimizer, ids[:min(2, len(ids))], variant, node_limit, cache, targets, torch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        record = {"variant": variant, "effective_batch_size": effective, "node_limit": node_limit,
                  "observations": len(ids), "fit_ids_sha256": base.stable_hash(ids),
                  "preload_seconds": preload_seconds, "uses_observed_labels": False}
        try:
            losses = [loss_step(model, optimizer, ids[start:start + effective], variant,
                                node_limit, cache, targets, torch) for start in range(0, len(ids), effective)]
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            free, _ = torch.cuda.mem_get_info()
            if not all(math.isfinite(v) for v in losses):
                raise RuntimeError("nonfinite_synthetic_probe_loss")
            record.update(status="PASS" if free >= 4 * 1024**3 else "reserve_exceeded",
                          seconds=elapsed, observations_per_second=len(ids) / elapsed,
                          optimizer_steps=len(losses), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_after_bytes=free)
        except torch.cuda.OutOfMemoryError as exc:
            record.update(status="OOM", error=str(exc), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        finally:
            del model, optimizer
            torch.cuda.empty_cache()
        results.append(record)
        base.atomic_json(output / "batch_probe.json", {"status": "measuring", "results": results, "gradient_gate": parity})
        print(json.dumps(record), flush=True)
    graph = [r for r in results if r["variant"] == base.METHODS[0] and r["status"] == "PASS"]
    best = max(r["observations_per_second"] for r in graph)
    chosen = min((r for r in graph if r["observations_per_second"] >= best * .95),
                 key=lambda r: (r["effective_batch_size"], r["node_limit"]))
    full_nodes = sum(int(context["ledger"][cid]["graph_node_count"]) for cid in parts["fit"])
    estimates = []
    for size in (2048, 4096):
        ids = [parts["fit"][int(i)] for i in order[:size]]
        nodes = sum(int(context["ledger"][cid]["graph_node_count"]) for cid in ids)
        estimates.append({"requested_physical_batch": size, "nodes": nodes,
                          "dense_178_input_bytes_only": nodes * 178 * 4,
                          "whole_graph_microbatches_required": len(list(physical_groups(ids, context["ledger"], chosen["node_limit"], True))),
                          "status": "resource_estimate_not_measured_physical_batch"})
    profile = {"status": "PASS", "authorization": contract["authorization"], "pins": context["pins"],
               "claim_boundary": contract["claim_boundary"], "gpu": gpu, "gradient_gate": parity,
               "effective_batch_size": chosen["effective_batch_size"], "graph_node_limit": chosen["node_limit"],
               "host_cache_gib": contract["host_cache_gib"], "epochs": 500, "seeds": list(base.SEEDS),
               "selection_uses_observed_labels": False, "selection": contract["selection"],
               "fit_partition_hash": base.stable_hash(parts["fit"]), "fit_nodes": full_nodes,
               "measured_results": results, "large_graph_batch_estimates": estimates,
               "old_batch2_checkpoint": {"path": str(context["feature_root"] / "training/fold_0/mali_full_features_graph/seed_42/checkpoint.pt"),
                                         "sha256": base.sha256_file(context["feature_root"] / "training/fold_0/mali_full_features_graph/seed_42/checkpoint.pt")}}
    base.atomic_json(output / "batch_profile.json", profile)
    base.atomic_json(output / "batch_probe.json", {"status": "PASS", "results": results, "gradient_gate": parity})
    print(json.dumps({"selected_batch": chosen["effective_batch_size"], "node_limit": chosen["node_limit"],
                      "graph_observations_per_second": chosen["observations_per_second"]}), flush=True)


def fit_cell(context, output, parts, transform, variant, fold, seed, profile, torch):
    import numpy as np
    method = METHOD_IDS[variant]
    directory = output / "training" / f"fold_{fold}" / method / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    identity = {"method": method, "fold": fold, "seed": seed, "pins": context["pins"],
                "profile_sha256": base.sha256_file(output / "batch_profile.json"),
                "fit_ids_sha256": base.stable_hash(parts["fit"]),
                "validation_ids_sha256": base.stable_hash(parts["validation"]),
                "test_ids_sha256": base.stable_hash(parts["test"]), "transform_sha256": base.stable_hash(transform)}
    final = directory / "cell_manifest.json"
    if final.exists():
        receipt = json.loads(final.read_text())
        if receipt["identity"] != identity:
            raise RuntimeError("completed_cell_identity_mismatch")
        for name, sha in receipt.get("outputs", {}).items():
            if base.sha256_file(directory / name) != sha:
                raise RuntimeError("completed_cell_output_hash_mismatch")
        return receipt
    base.seed_all(seed)
    model = model_for(variant, transform, context["technical"]["activation_recomputation_enabled"], torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.)
    start, best_loss, best_epoch, best_state, prior_seconds = 0, math.inf, -1, None, 0.
    if checkpoint.exists():
        saved = base.load_saved_checkpoint(checkpoint, identity, torch)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        base.restore_rng_state(saved["rng_state"])
        start, best_loss, best_epoch = saved["next_epoch"], saved["best_validation_mse"], saved["best_epoch"]
        best_state, prior_seconds = saved["best_model_state"], saved["elapsed_seconds"]
    cache = ExampleCache(context, transform, int(profile["host_cache_gib"] * 1024**3))
    graph = variant == base.METHODS[0]
    batch_size, node_limit = profile["effective_batch_size"], profile["graph_node_limit"]
    # Only fit/validation labels are admitted until the final test-scoring pass.
    targets = {cid: float(context["canonical"][cid]["target_seconds"]) for cid in parts["fit"] + parts["validation"]}
    started = time.monotonic()
    try:
        for epoch in range(start, profile["epochs"]):
            epoch_started = time.monotonic()
            model.train()
            order = np.random.default_rng(int(base.stable_hash([fold, seed, epoch])[:16], 16)).permutation(len(parts["fit"]))
            ids = [parts["fit"][int(i)] for i in order]
            squared_error, steps = 0., 0
            for pos in range(0, len(ids), batch_size):
                group = ids[pos:pos + batch_size]
                value = loss_step(model, optimizer, group, variant, node_limit, cache, targets, torch)
                squared_error += value * len(group)
                steps += 1
            scheduler.step()
            model.eval()
            valid_error = 0.
            with torch.no_grad():
                for group in physical_groups(parts["validation"], context["ledger"], node_limit, graph):
                    batch = base._batch_to_device([cache.get(cid, variant) for cid in group], "cuda:0", graph)
                    predicted = base._prediction(model, batch)
                    actual = torch.tensor([targets[cid] for cid in group], dtype=torch.float32, device="cuda:0")
                    valid_error += float(torch.nn.functional.mse_loss(predicted, actual, reduction="sum").cpu())
                    del batch, predicted, actual
            valid_mse = valid_error / len(parts["validation"])
            if not math.isfinite(valid_mse) or not math.isfinite(squared_error):
                raise RuntimeError("nonfinite_fit_or_validation_loss")
            if valid_mse < best_loss:
                best_loss, best_epoch = valid_mse, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elapsed = prior_seconds + time.monotonic() - started
            payload = {"identity": identity, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                       "rng_state": base.capture_rng_state(), "next_epoch": epoch + 1, "completed_epoch": epoch + 1,
                       "best_validation_mse": best_loss, "best_epoch": best_epoch, "best_model_state": best_state,
                       "transform": transform, "elapsed_seconds": elapsed}
            base.save_checkpoint(checkpoint, payload)
            progress = {"status": "training", "method": method, "fold": fold, "seed": seed,
                        "completed_epoch": epoch + 1, "epochs": profile["epochs"], "effective_batch_size": batch_size,
                        "optimizer_steps_in_epoch": steps, "validation_mse": valid_mse, "best_epoch": best_epoch,
                        "elapsed_seconds": elapsed, "epoch_seconds": time.monotonic() - epoch_started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "host_cache_bytes": cache.bytes}
            base.atomic_json(directory / "cell_progress.json", progress)
            print(json.dumps(progress), flush=True)
        model.load_state_dict(best_state)
        model.eval()
        rows, failures = [], []
        with torch.no_grad():
            for cid in parts["test"]:
                if context["ledger"][cid]["global_status"] != "available" or (graph and context["ledger"][cid]["graph_status"] != "available"):
                    failures.append({"canonical_row_id": cid, "status": "representation_unavailable"})
                    continue
                values, errors = base._predict_test_groups(model, [[cache.get(cid, variant)]], "cuda:0", graph, torch)
                if errors:
                    failures.append({"canonical_row_id": cid, "status": "resource_failure"})
                    continue
                prediction = values[0][1]
                actual = float(context["canonical"][cid]["target_seconds"])
                rows.append({"canonical_row_id": cid, "outer_fold": fold, "method": method, "seed": seed,
                             "actual_seconds": actual, "predicted_seconds": prediction,
                             "absolute_error_seconds": abs(prediction - actual), "negative_prediction": int(prediction < 0)})
        base.write_csv_atomic(directory / "predictions.csv", rows, ["canonical_row_id", "outer_fold", "method", "seed", "actual_seconds", "predicted_seconds", "absolute_error_seconds", "negative_prediction"])
        base.write_csv_atomic(directory / "test_failures.csv", failures, ["canonical_row_id", "status"])
        result = {"status": "complete", "identity": identity, "n_fit": len(parts["fit"]),
                  "n_validation": len(parts["validation"]), "n_test_assigned": len(parts["test"]),
                  "n_test_predicted": len(rows), "n_test_failure": len(failures), "best_epoch": best_epoch,
                  "best_validation_mse": best_loss, "elapsed_seconds": prior_seconds + time.monotonic() - started,
                  "outputs": {name: base.sha256_file(directory / name) for name in ("checkpoint.pt", "predictions.csv", "test_failures.csv")}}
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status": "resource_failure", "identity": identity, "error": str(exc),
                  "n_test_assigned": len(parts["test"]), "checkpoint_sha256": base.sha256_file(checkpoint) if checkpoint.exists() else None}
    finally:
        del model, optimizer, scheduler, cache
        torch.cuda.empty_cache()
    base.atomic_json(final, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tune", "train"), required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.folds != [0]:
        raise SystemExit("Fold 0 optimization QA must complete before enabling folds 1-4.")
    args.feature_root = args.feature_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == args.feature_root or args.feature_root in args.output_dir.parents:
        raise SystemExit("Batch optimization needs a separate artifact context.")
    context = load_context(args.feature_root.resolve())
    contract = json.loads(CONTRACT.read_text())
    import torch
    torch.set_num_threads(2)
    from gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease("mali-full-features-batch-optimization")
    try:
        gpu = base.validate_gpu_budget(torch)
        if args.stage == "tune":
            tune(context, args.output_dir, contract, torch, gpu)
            return
        profile_path = args.output_dir / "batch_profile.json"
        profile = json.loads(profile_path.read_text())
        if profile["pins"] != context["pins"] or profile["status"] != "PASS":
            raise RuntimeError("batch_profile_code_data_or_contract_identity_mismatch")
        state = {"status": "running", "profile_sha256": base.sha256_file(profile_path),
                 "pins": context["pins"], "gpu": gpu, "cells": [], "expected_fold0_cells": 6}
        base.atomic_json(args.output_dir / "run_manifest.json", state)
        for fold in args.folds:
            parts, transform = partitions_and_transform(context, fold)
            for variant in base.METHODS:
                for seed in base.SEEDS:
                    result = fit_cell(context, args.output_dir, parts, transform, variant, fold, seed, profile, torch)
                    state["cells"].append(result)
                    base.atomic_json(args.output_dir / "run_manifest.json", state)
        state["status"] = "awaiting_fold0_optimization_QA"
        base.atomic_json(args.output_dir / "run_manifest.json", state)
    finally:
        lease.release()


if __name__ == "__main__":
    main()
