#!/usr/bin/env python3
"""Prespecified graph ablation, separate from historical training outputs."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from benchmark26.paths import recorded
from experiments.graph_ablation.transforms import transform_graph
from pathlib import Path

import argparse
import copy
import csv
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from benchmark26 import mali_training as base  # noqa: E402
from benchmark26 import mali_batching as batching  # noqa: E402
from benchmark26.shots import append_shots, expand_global_projection, fit_shot_transform, transform_shots  # noqa: E402
from core import (  # noqa: E402
    LOCK,
    NEW_RUN,
    OUT,
    PREPARE_HINT,
    RUN,
    atomic_csv,
    atomic_json,
    current_code_hashes,
    execution_identity,
    historical_code_differences,
    load_context,
    load_lock,
    load_targets,
    locked_input_problems,
    parts_for,
    rows,
    run_manifest_problems,
    sha,
    stable,
)

SEEDS = (42, 1234, 31415)
EPOCHS = 500
NODE_BUDGET = 250000
CACHE_GIB = 8
GPU_BUDGET_GIB = 12
GPU_RESERVE_GIB = 4
EFFECTIVE_BATCH = int(os.environ.get("QPU_EXPANSION_BATCH", "32"))
CPU_RSS_LIMIT_GIB = 12
METHODS = {
    "M0": {"method": "expansion_mlp_no_shots", "internal": "mali_full_features_mlp", "graph": False, "shots": False},
    "M1": {"method": "expansion_mlp_plus_shots", "internal": "mali_full_features_mlp", "graph": False, "shots": True},
    "G0": {
        "method": "expansion_graph_mlp_no_shots",
        "internal": "mali_full_features_graph",
        "graph": True,
        "shots": False,
    },
    "G1": {
        "method": "expansion_graph_mlp_plus_shots",
        "internal": "mali_full_features_graph",
        "graph": True,
        "shots": True,
    },
}


METHODS = {k: METHODS[k] for k in ("M0", "M1", "G0", "G1")}
METHODS["G1e"] = {**METHODS["G1"], "method": "graph_no_edges"}
METHODS["G1p"] = {**METHODS["G1"], "method": "graph_scrambled_features"}
PATIENCE = 50


class ExampleCache:
    """Cache transformed features by observation ID, retaining row-specific shots."""

    def __init__(self, context, transform, treatment, budget_bytes):
        from collections import OrderedDict

        self.context = context
        self.transform = transform
        self.treatment = treatment
        self.budget = budget_bytes
        self.items = OrderedDict()
        self.bytes = 0
        self.verified = set()
        self.hits = 0
        self.misses = 0

    def get(self, cid, variant=None):
        """Return the cached example for an observation, building and caching it if needed.

        The cache is limited by a byte budget and evicts the least recently used examples. Shot variants
        append the standardized shot count to the example.
        """
        if variant is not None and variant != METHODS[self.treatment]["internal"]:
            raise RuntimeError(f"cache_variant_mismatch:{self.treatment}:{variant}")
        key = (cid, self.treatment)
        if key in self.items:
            self.hits += 1
            value, size = self.items.pop(key)
            self.items[key] = (value, size)
            return value
        self.misses += 1
        spec = METHODS[self.treatment]
        tfm = {"global": self.transform["global"], "node": self.transform["node"]}
        value = base._make_example(
            cid,
            spec["internal"],
            self.context["ledger"],
            None,
            tfm,
            self.context["feature_root"],
            "cpu",
            False,
            self.verified,
        )
        if spec["shots"]:
            raw = float(self.context["panel_by_id"][cid]["shots"])
            value = append_shots(value, transform_shots(raw, self.transform["shots"]))
        if spec["graph"]:
            arm = {"G1e": "G1e", "G1p": "G1p"}.get(self.treatment, "G1r")
            value = transform_graph(value, arm, cid)
        size = sum(t.numel() * t.element_size() for _, t in value if hasattr(t, "numel"))
        if size <= self.budget:
            while self.items and self.bytes + size > self.budget:
                _, (_, removed) = self.items.popitem(last=False)
                self.bytes -= removed
            self.items[key] = (value, size)
            self.bytes += size
        return value


def environment_receipt(torch) -> dict:
    """Return the software, GPU and CPU environment of the run, with the git revision, as a dict."""
    import numpy
    import sklearn
    import torch_geometric
    from threadpoolctl import threadpool_info

    return {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_budget_gib": GPU_BUDGET_GIB,
        "gpu_reserve_gib": GPU_RESERVE_GIB,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "threadpools": threadpool_info(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def protocol_lock(require_code: bool = True) -> tuple[dict, str]:
    """Locked settings and the identity hash that fitted outputs record."""
    lock, identity_sha = execution_identity(require_code=require_code)
    if int(lock.get("neural", {}).get("effective_batch", -1)) != EFFECTIVE_BATCH:
        raise RuntimeError("effective_batch_does_not_match_locked_profile")
    return lock, identity_sha


def training_readiness() -> dict:
    """Report, without fitting or GPU computation, whether training can start."""
    lock = load_lock()
    inputs = locked_input_problems(lock)
    current = current_code_hashes(lock)
    differ = historical_code_differences(lock, current)
    blocking = []
    if inputs:
        blocking.append("bundle differs from the locked protocol, inputs or graphs")
    if int(lock.get("neural", {}).get("effective_batch", -1)) != EFFECTIVE_BATCH:
        blocking.append("QPU_EXPANSION_BATCH differs from the locked effective batch")
    report = {
        "historical_lock_sha256": sha(LOCK),
        "locked_inputs_and_graphs": "match" if not inputs else inputs[:10],
        "code_identical_to_historical_lock": len(current) - len(differ),
        "code_differing_from_historical_lock": differ,
        "output_directory": str(RUN),
        "mode": "new run" if NEW_RUN else "original study outputs",
    }
    if NEW_RUN:
        manifest = run_manifest_problems(lock, current)
        report["run_manifest"] = "valid" if not manifest else manifest
        blocking += manifest
    elif differ:
        blocking.append(f"code differs from the historical lock; {PREPARE_HINT}")
    relative = "model/transformer_model.py"
    try:
        blob = subprocess.run(
            ["git", "-C", str(base.MALI_SOURCE), "show", f"{base.PINNED_MODEL_REVISION}:{relative}"],
            capture_output=True,
            check=True,
        ).stdout
        report["pinned_model_source"] = "match" if base.sha256_bytes(blob) == base.SOURCE_PINS[relative] else "differs"
    except (OSError, subprocess.CalledProcessError):
        report["pinned_model_source"] = f"unavailable in {base.MALI_SOURCE}"
    if report["pinned_model_source"] != "match":
        blocking.append("pinned Ma-Li model source not found; set B26_MALI_SOURCE to the Ma-Li Git checkout")
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": props.name, "total_gib": round(props.total_memory / 2**30, 2)}
            if props.total_memory / 2**30 < GPU_BUDGET_GIB:
                blocking.append(f"GPU has less than the locked {GPU_BUDGET_GIB} GiB budget")
        else:
            blocking.append("CUDA GPU not available (no CPU fallback)")
    except ImportError:
        report["torch"] = None
        blocking.append("PyTorch is not installed")
    report["blocking"] = blocking
    report["ready_to_train"] = not blocking
    return report


def fit_transform(context, parts, fold):
    """Fit the feature transforms of a fold on its fit rows only: global features, node features and shots."""
    raw_global = {cid: [float(context["ledger"][cid][f"global_{j:02d}"]) for j in range(51)] for cid in parts["fit"]}
    global_transform = base.fit_transform_global(raw_global, parts["fit"])
    graph_paths = [recorded(context["ledger"][cid]["graph_artifact"]) for cid in parts["fit"]]
    node_transform = base.fit_node_transform(graph_paths, parts["fit"])
    shot_fit = [float(context["panel_by_id"][cid]["shots"]) for cid in parts["fit"]]
    return {
        "global": global_transform,
        "node": node_transform,
        "shots": fit_shot_transform(shot_fit),
        "fold": fold,
        "fit_ids_sha256": base.stable_hash(parts["fit"]),
    }


def make_model(treatment, transform, seed, torch):
    """Build the model of a treatment for a seed, with the shot column added when the treatment uses shots, and move it to the GPU."""
    spec = METHODS[treatment]
    base.seed_all(seed)
    model_class, model_sha = base.load_pinned_model()
    old_model = base.make_model(
        model_class, spec["internal"], len(transform["global"]["retained_indices"]), checkpoint_layers=False
    )
    if spec["graph"]:
        old_model.mask = old_model.mask.to("cuda:0")
    projection = None
    if spec["shots"]:
        constructor_rng = base.capture_rng_state()
        model = old_model
        projection = expand_global_projection(model, torch)
        base.restore_rng_state(constructor_rng)
    else:
        model = old_model
    model.to("cuda:0")
    return model, model_sha, projection


def fit_transform_for_cell(context, parts, fold, evaluation):
    """Return the transform of a fold, saving it on first use and checking it against the saved one afterwards."""
    path = RUN / "training" / evaluation / f"fold_{fold}" / "transform.json"
    fitted = fit_transform(context, parts, fold)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != fitted:
            raise RuntimeError(f"fold_transform_mismatch:{fold}")
    else:
        atomic_json(path, fitted)
    return fitted


def initialization_gate(context, parts, transform, treatment, seed, torch):
    """Check that the model starts from the same function as its control before training.

    Uses three fit rows (the lowest, median and highest shot count). For the shot variants it checks
    that the added shot column leaves the initial output unchanged and receives a gradient. No test
    labels are used. Returns the gate result.
    """
    spec = METHODS[treatment]
    cache = ExampleCache(context, transform, treatment, 256 * 1024**2)
    fit_sorted = sorted(parts["fit"], key=lambda cid: float(context["panel_by_id"][cid]["shots"]))
    fixture_ids = list(dict.fromkeys([fit_sorted[0], fit_sorted[len(fit_sorted) // 2], fit_sorted[-1]]))
    old_treatment = {"M1": "M0", "G1": "G0"}.get(treatment)
    if old_treatment:
        old, old_sha, _ = make_model(old_treatment, transform, seed, torch)
        new, new_sha, projection = make_model(treatment, transform, seed, torch)
        if old_sha != new_sha:
            raise RuntimeError("shot_model_source_mismatch")
        old.eval()
        new.eval()
        maxdiff = 0.0
        for cid in fixture_ids:
            old_ex = cache.get(cid)
            # Build a second baseline example with exactly the old global width.
            old_ex = base._make_example(
                cid,
                METHODS[old_treatment]["internal"],
                context["ledger"],
                None,
                {"global": transform["global"], "node": transform["node"]},
                context["feature_root"],
                "cpu",
                False,
                cache.verified,
            )
            new_ex = append_shots(old_ex, 0.0)
            b0 = base._batch_to_device([old_ex], "cuda:0", spec["graph"])
            b1 = base._batch_to_device([new_ex], "cuda:0", spec["graph"])
            with torch.no_grad():
                a = base._prediction(old, b0).reshape(-1)
                b = base._prediction(new, b1).reshape(-1)
            diff = float((a - b).abs().max().cpu())
            if not torch.allclose(a, b, rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"zero_shot_initialization_parity_failed:{treatment}:{diff}")
            maxdiff = max(maxdiff, diff)
            del old_ex, new_ex, b0, b1, a, b
        # The appended column must receive a usable gradient on real, fit-only
        # standardized shot values. Synthetic labels avoid using runtime labels.
        low, high = fixture_ids[0], fixture_ids[-1]
        examples = []
        for cid in (low, high):
            old_ex = base._make_example(
                cid,
                METHODS[old_treatment]["internal"],
                context["ledger"],
                None,
                {"global": transform["global"], "node": transform["node"]},
                context["feature_root"],
                "cpu",
                False,
                cache.verified,
            )
            examples.append(
                append_shots(old_ex, transform_shots(float(context["panel_by_id"][cid]["shots"]), transform["shots"]))
            )
        batch = base._batch_to_device(examples, "cuda:0", spec["graph"])
        new.train()
        new.zero_grad(set_to_none=True)
        output = base._prediction(new, batch).reshape(-1)
        loss = torch.nn.functional.mse_loss(output, output.detach() + 0.5)
        loss.backward()
        grad = new.gf_linear1.weight.grad[:, -1]
        grad_size = float(grad.abs().max().detach().cpu())
        if not math.isfinite(grad_size) or grad_size <= 0:
            raise RuntimeError(f"shots_column_gradient_failed:{treatment}:{grad_size}")
        old_count = sum(p.numel() for p in old.parameters())
        new_count = sum(p.numel() for p in new.parameters())
        if new_count - old_count != projection["added_trainable_parameters"]:
            raise RuntimeError("shot_model_parameter_delta_mismatch")
        result = {
            "status": "PASS",
            "treatment": treatment,
            "seed": seed,
            "fixture_fit_ids": fixture_ids,
            "same_pinned_model_source": True,
            "zero_shot_initialization_max_abs_diff": maxdiff,
            "new_shot_column_gradient_max_abs": grad_size,
            "baseline_parameters": old_count,
            "treatment_parameters": new_count,
            "added_parameters": new_count - old_count,
            "projection": projection,
            "uses_test_labels": False,
        }
        del old, new, examples, batch, output, loss, grad
    else:
        model, model_sha, _ = make_model(treatment, transform, seed, torch)
        examples = [cache.get(cid) for cid in fixture_ids[:2]]
        batch = base._batch_to_device(examples, "cuda:0", spec["graph"])
        model.train()
        model.zero_grad(set_to_none=True)
        output = base._prediction(model, batch).reshape(-1)
        loss = torch.nn.functional.mse_loss(output, output.detach() + 0.5)
        loss.backward()
        grad_count = sum(int(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if not math.isfinite(float(loss.detach().cpu())) or grad_count == 0:
            raise RuntimeError(f"baseline_forward_backward_gate_failed:{treatment}")
        result = {
            "status": "PASS",
            "treatment": treatment,
            "seed": seed,
            "fixture_fit_ids": fixture_ids[:2],
            "model_source_sha256": model_sha,
            "finite_synthetic_loss": float(loss.detach().cpu()),
            "parameters_with_finite_gradients": grad_count,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "uses_test_labels": False,
        }
        del model, examples, batch, output, loss
    torch.cuda.empty_cache()
    return result


def cell_paths(evaluation, treatment, fold, seed, canary=False):
    """Return the directory of a cell (evaluation, treatment, fold, seed)."""
    parent = RUN / "training" / ("canary_v2" if canary else evaluation)
    return parent / f"fold_{fold}" / treatment / f"seed_{seed}"


def train_cell(
    context,
    parts,
    train_targets,
    transform,
    treatment,
    fold,
    seed,
    torch,
    lock_sha,
    environment,
    max_epochs=EPOCHS,
    canary=False,
):
    """Train one cell (treatment, fold, seed) and return its result.

    Runs the initialization gate, then Adam (learning rate 0.0005, weight decay 0.0001) for the
    planned epochs, resuming from the checkpoint if there is one and returning the saved result if the
    cell is complete. Keeps the best-validation state, writes the epoch history, then predicts the test
    rows. The training function never receives test labels.
    """
    spec = METHODS[treatment]
    evaluation = context["evaluation"]
    folder = cell_paths(evaluation, treatment, fold, seed, canary=canary)
    folder.mkdir(parents=True, exist_ok=True)
    gate_path = folder / "initialization_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or gate.get("transform_sha256") != stable(transform):
            raise RuntimeError(f"initialization_gate_identity_mismatch:{treatment}:{fold}:{seed}")
    else:
        gate = initialization_gate(context, parts, transform, treatment, seed, torch)
        gate["transform_sha256"] = stable(transform)
        gate["fold"] = fold
        atomic_json(gate_path, gate)
    model, model_sha, projection = make_model(treatment, transform, seed, torch)
    identity = {
        "study": "postfreeze_qpu_workflow_mean_expansion_v1",
        "evaluation": evaluation,
        "heldout_backend": context.get("cases", [""])[fold] if evaluation == "E3" else "",
        "canary": bool(canary),
        "treatment": treatment,
        "method": spec["method"],
        "internal_model": spec["internal"],
        "fold": fold,
        "seed": seed,
        "fit_ids_sha256": base.stable_hash(parts["fit"]),
        "validation_ids_sha256": base.stable_hash(parts["validation"]),
        "outer_train_ids_sha256": base.stable_hash(parts["train"]),
        "test_ids_sha256": base.stable_hash(parts["test"]),
        "transform_sha256": stable(transform),
        "model_source_sha256": model_sha,
        "runner_sha256": sha(Path(__file__)),
        "core_sha256": sha(HERE / "core.py"),
        "shots_helper_sha256": sha(ROOT / "benchmark26/shots.py"),
        "panel_sha256": sha(OUT / "cohort/panel.csv"),
        "feature_ledger_sha256": sha(OUT / "cohort/feature_ledger.csv"),
        "outer_split_sha256": sha(OUT / "splits" / f"{evaluation}_outer.csv"),
        "inner_split_sha256": sha(OUT / "splits" / f"{evaluation}_inner.csv"),
        "protocol_lock_sha256": lock_sha,
        "environment": environment,
        "graph_node_budget": NODE_BUDGET,
        "effective_batch": EFFECTIVE_BATCH,
    }
    done = folder / "manifest.json"
    if done.exists():
        saved = json.loads(done.read_text(encoding="utf-8"))
        if saved.get("identity") != identity or any(
            sha(folder / name) != digest for name, digest in saved["outputs"].items()
        ):
            raise RuntimeError(f"completed_cell_identity_or_checksum_mismatch:{treatment}:{fold}:{seed}")
        return saved

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    cache = ExampleCache(context, transform, treatment, CACHE_GIB * 1024**3)
    start, best, best_epoch, best_state, elapsed_before = 0, float("inf"), -1, None, 0.0
    checkpoint = folder / "checkpoint.pt"
    history_path = folder / "epoch_history.csv"
    if checkpoint.exists():
        saved = base.load_saved_checkpoint(checkpoint, identity, torch)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        base.restore_rng_state(saved["rng_state"])
        start, best, best_epoch = saved["next_epoch"], saved["best_validation_mse"], saved["best_epoch"]
        best_state, elapsed_before = saved["best_model_state"], saved["elapsed_seconds"]
        history = rows(history_path)
        if start > 0 and len(history) != start:
            raise RuntimeError("checkpoint_history_epoch_mismatch")
    elif history_path.exists():
        raise RuntimeError("epoch_history_without_checkpoint")
    else:
        history = []
    if start > max_epochs:
        raise RuntimeError("checkpoint_beyond_requested_epoch")

    begin = time.monotonic()
    already_stopped = start > 0 and start - 1 - best_epoch >= PATIENCE
    for epoch in range(start, start if already_stopped else max_epochs):
        tick = time.monotonic()
        model.train()
        order = (
            __import__("numpy")
            .random.default_rng(int(base.stable_hash([fold, seed, epoch])[:16], 16))
            .permutation(len(parts["fit"]))
        )
        ordered_ids = [parts["fit"][int(j)] for j in order]
        losses = [
            batching.loss_step(
                model,
                optimizer,
                ordered_ids[j : j + EFFECTIVE_BATCH],
                spec["internal"],
                NODE_BUDGET if spec["graph"] else 0,
                cache,
                train_targets,
                torch,
            )
            for j in range(0, len(ordered_ids), EFFECTIVE_BATCH)
        ]
        model.eval()
        val_sum = 0.0
        validation_targets = {cid: train_targets[cid] for cid in parts["validation"]}
        with torch.no_grad():
            for group in batching.physical_groups(
                parts["validation"], context["ledger"], NODE_BUDGET if spec["graph"] else 0, spec["graph"]
            ):
                examples = [cache.get(cid) for cid in group]
                payload = base._batch_to_device(examples, "cuda:0", spec["graph"])
                prediction = base._prediction(model, payload)
                actual = torch.tensor([validation_targets[cid] for cid in group], dtype=torch.float32, device="cuda:0")
                val_sum += float(torch.nn.functional.mse_loss(prediction, actual, reduction="sum").cpu())
                del examples, payload, prediction, actual
        validation_mse = val_sum / len(parts["validation"])
        if not math.isfinite(validation_mse) or not all(math.isfinite(x) for x in losses):
            raise RuntimeError(f"nonfinite_training_or_validation_loss:{treatment}:{fold}:{seed}:{epoch + 1}")
        if validation_mse < best:
            best, best_epoch = validation_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.cuda.synchronize()
        elapsed = elapsed_before + time.monotonic() - begin
        base.save_checkpoint(
            checkpoint,
            {
                "identity": identity,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "rng_state": base.capture_rng_state(),
                "next_epoch": epoch + 1,
                "best_validation_mse": best,
                "best_epoch": best_epoch,
                "best_model_state": best_state,
                "elapsed_seconds": elapsed,
            },
        )
        record = {
            "epoch": epoch + 1,
            "train_mse": sum(losses),
            "validation_mse": validation_mse,
            "best_validation_mse": best,
            "best_epoch": best_epoch + 1,
            "epoch_seconds": time.monotonic() - tick,
            "elapsed_seconds": elapsed,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cache_bytes": cache.bytes,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
        }
        if record["peak_rss_kib"] * 1024 > CPU_RSS_LIMIT_GIB * 1024**3:
            atomic_json(
                folder / "resource_stop.json",
                {
                    "status": "STOP_CPU_RSS_LIMIT",
                    "limit_gib": CPU_RSS_LIMIT_GIB,
                    "epoch": epoch + 1,
                    "peak_rss_kib": record["peak_rss_kib"],
                    "identity": identity,
                },
            )
            raise RuntimeError(f"cpu_rss_limit_exceeded:{record['peak_rss_kib']}")
        history.append(record)
        atomic_csv(history_path, history)
        atomic_json(
            folder / "progress.json",
            {
                "status": "training",
                "completed_epochs": epoch + 1,
                "requested_epochs": max_epochs,
                "identity": identity,
                "last": record,
                "updated_unix": time.time(),
            },
        )
        if (epoch + 1) % 10 == 0 or epoch + 1 == max_epochs:
            print(
                json.dumps(
                    {"stage": "train", "treatment": treatment, "fold": fold, "seed": seed, **record},
                    separators=(",", ":"),
                ),
                flush=True,
            )
        if epoch - best_epoch >= PATIENCE:
            break
    if max_epochs < EPOCHS:
        final = {
            "status": "canary_checkpointed",
            "identity": identity,
            "completed_epochs": max_epochs,
            "checkpoint_sha256": sha(checkpoint),
            "last_epoch": history[-1] if history else None,
        }
        del model, optimizer, cache, best_state
        torch.cuda.empty_cache()
        return final
    if best_state is None or best_epoch < 0:
        raise RuntimeError("best_validation_checkpoint_missing")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for cid in parts["test"]:
            payload = base._batch_to_device([cache.get(cid)], "cuda:0", spec["graph"])
            value = float(base._prediction(model, payload).detach().cpu().reshape(-1)[0])
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite_test_prediction:{treatment}:{fold}:{seed}:{cid}")
            predictions.append(
                {
                    "canonical_observation_id": cid,
                    "outer_fold": fold,
                    "treatment": treatment,
                    "method_id": spec["method"],
                    "seed": seed,
                    "predicted_seconds": value,
                    "selected_epoch": best_epoch + 1,
                }
            )
            del payload
    atomic_csv(folder / "predictions.csv", predictions)
    result = {
        "status": "complete",
        "identity": identity,
        "epochs": len(history),
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "best_epoch": best_epoch + 1,
        "best_validation_mse": best,
        "elapsed_seconds": history[-1]["elapsed_seconds"],
        "fit_rows": len(parts["fit"]),
        "validation_rows": len(parts["validation"]),
        "test_rows": len(parts["test"]),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "shot_projection": projection,
        "cache": {"hits": cache.hits, "misses": cache.misses, "bytes": cache.bytes},
        "outputs": {name: sha(folder / name) for name in ("checkpoint.pt", "epoch_history.csv", "predictions.csv")},
        "test_labels_available_to_training_function": False,
    }
    atomic_json(done, result)
    atomic_json(
        folder / "progress.json",
        {"status": "complete", "identity": identity, "manifest_sha256": sha(done), "updated_unix": time.time()},
    )
    del model, optimizer, cache, best_state
    torch.cuda.empty_cache()
    return result


def baseline_sanity(context):
    """Predeclared safety heuristic, not a statistical equivalence test."""
    import collections
    import numpy as np

    labels = load_targets(context["all_ids"])
    history = rows(OUT / "analysis_v1/group_metrics.csv")
    comparison = [
        r
        for r in rows(OUT / "analysis_v1/paired_comparisons.csv")
        if r["evaluation"] == "E1" and r["scope"] == "all" and r["contrast"] == "G1-M1"
    ]
    if len(comparison) != 1:
        raise RuntimeError("historical_interval_missing")
    threshold = float(comparison[0]["adjusted_ci_upper_seconds"]) - float(comparison[0]["adjusted_ci_lower_seconds"])
    out = {}
    for treatment in ("M1", "G1"):
        by_id = collections.defaultdict(list)
        for fold in range(context["case_count"]):
            expected = set(parts_for(context, fold)["test"])
            for seed in SEEDS:
                records = rows(cell_paths("E1", treatment, fold, seed) / "predictions.csv")
                ids = [r["canonical_observation_id"] for r in records]
                if len(ids) != len(set(ids)) or set(ids) != expected:
                    raise RuntimeError("sanity_prediction_coverage_mismatch")
                for r in records:
                    by_id[r["canonical_observation_id"]].append(float(r["predicted_seconds"]))
        if set(by_id) != context["all_ids"] or any(len(v) != 3 for v in by_id.values()):
            raise RuntimeError("sanity_seed_coverage_mismatch")
        groups = collections.defaultdict(list)
        for cid, values in by_id.items():
            groups[context["panel_by_id"][cid]["group_id"]].append(abs(float(np.median(values)) - labels[cid]))
        observed = float(np.mean([np.mean(v) for v in groups.values()]))
        reference = [
            float(r["mae_seconds"]) for r in history if r["evaluation"] == "E1" and r["treatment"] == treatment
        ]
        if len(reference) != len(groups):
            raise RuntimeError("sanity_historical_group_count_mismatch")
        historical = float(np.mean(reference))
        out[treatment] = {
            "rerun_group_balanced_mae": observed,
            "historical_group_balanced_mae": historical,
            "absolute_difference": abs(observed - historical),
        }
    passed = all(r["absolute_difference"] <= threshold for r in out.values())
    atomic_json(
        RUN / "training/baseline_sanity.json",
        {
            "status": "PASS" if passed else "STOP",
            "threshold_seconds": threshold,
            "threshold_scope": "width of historical E1 adjusted G1-M1 interval; heuristic only",
            "results": out,
            "test_labels_used_only_for_this_predeclared_safety_check": True,
        },
    )
    if not passed:
        raise RuntimeError("predeclared_baseline_sanity_stop")


def main():
    """Run one stage (`preflight`, `canary`, `first_cell` or `train`) for an evaluation (E1, E2 or E3).

    `preflight` checks the data, the splits and the training readiness. The other stages require the
    lock and CUDA, take the GPU lease and train the cells that belong to the stage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "canary", "first_cell", "train"), required=True)
    parser.add_argument("--evaluation", choices=("E1", "E2", "E3"), default="E1")
    parser.add_argument("--require-ready", action="store_true", help="preflight: exit 2 unless training could start")
    args = parser.parse_args()
    context, _targets, split_receipt = load_context(args.evaluation, include_targets=False)
    if args.stage == "preflight":
        readiness = training_readiness()
        print(json.dumps({"data_and_splits": split_receipt, "training_readiness": readiness}, indent=2))
        return 0 if readiness["ready_to_train"] or not args.require_ready else 2
    lock, lock_sha = protocol_lock()
    ablation_lock = RUN / "ablation_lock.json"
    ablation_protocol = RUN / "ABLATION_PROTOCOL.md"
    if not ablation_lock.is_file():
        raise RuntimeError("ablation_protocol_not_locked")
    saved_ablation = json.loads(ablation_lock.read_text())
    if (
        saved_ablation.get("protocol_sha256") != sha(ablation_protocol)
        or saved_ablation.get("run_manifest_sha256") != lock_sha
    ):
        raise RuntimeError("ablation_lock_mismatch")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_required_no_CPU_fallback")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    from benchmark26.gpu_lease import acquire_gpu_lease

    lease = acquire_gpu_lease("postfreeze-qpu-workflow-mean-expansion-v1")
    try:
        gpu = base.validate_gpu_budget(torch, budget_gib=GPU_BUDGET_GIB, reserve_gib=GPU_RESERVE_GIB)
        environment = environment_receipt(torch)
        atomic_json(RUN / "training/execution_environment.json", {**environment, "gpu_validation": gpu})
        if args.stage == "canary":
            fold = 0
            parts = parts_for(context, fold)
            train_targets = load_targets(parts["train"])
            transform = fit_transform_for_cell(context, parts, fold, args.evaluation)
            gates = []
            for treatment in ("M1", "G1", "G1e", "G1p"):
                gate = initialization_gate(context, parts, transform, treatment, 42, torch)
                gate["transform_sha256"] = stable(transform)
                gate["fold"] = fold
                gate_path = cell_paths(args.evaluation, treatment, fold, 42, canary=True) / "initialization_gate.json"
                atomic_json(gate_path, gate)
                gates.append(gate)
            graph_counts = [
                g.get("treatment_parameters", g.get("parameter_count"))
                for g in gates
                if g["treatment"] in ("G1", "G1e", "G1p")
            ]
            if len(graph_counts) != 3 or len(set(graph_counts)) != 1:
                raise RuntimeError("graph_arm_parameter_count_mismatch")
            canary = train_cell(
                context,
                parts,
                train_targets,
                transform,
                "G1",
                0,
                42,
                torch,
                lock_sha,
                environment,
                max_epochs=2,
                canary=True,
            )
            receipt = {
                "status": "PASS" if canary["status"] == "canary_checkpointed" else "FAIL",
                "stage": "two_epoch_fit_only_canary",
                "fold": 0,
                "treatment": "G1",
                "seed": 42,
                "canary": canary,
                "initialization_gates": gates,
                "split_receipt": split_receipt,
                "environment": environment,
                "no_test_scoring": True,
                "test_labels_passed_to_training": False,
                "evaluation": args.evaluation,
                "lock_sha256": lock_sha,
            }
            atomic_json(RUN / "training/canary_receipt_v2.json", receipt)
            print(json.dumps(receipt, indent=2))
            return
        canary_path = RUN / "training/canary_receipt_v2.json"
        if not canary_path.is_file():
            raise RuntimeError("random_row_canary_missing")
        canary = json.loads(canary_path.read_text(encoding="utf-8"))
        if canary.get("status") != "PASS" or canary.get("lock_sha256") != lock_sha:
            raise RuntimeError("random_row_canary_lock_or_status_mismatch")
        if args.stage == "first_cell":
            evals = ["E1"]
            cell_filter = ("E1", 0, "M1", 42)
        else:
            evals = ["E1", "E2"]
            cell_filter = None
        total = 120
        completed = sum(
            1
            for p in (RUN / "training").glob("*/fold_*/[MG]*/seed_*/manifest.json")
            if json.loads(p.read_text()).get("status") == "complete"
        )
        for ev in evals:
            context, _targets, split_receipt = load_context(ev, include_targets=False)
            for treatment in ("M1", "G1", "G1e", "G1p"):
                for fold in range(context["case_count"]):
                    if cell_filter and (ev, fold) != cell_filter[:2]:
                        continue
                    parts = parts_for(context, fold)
                    train_targets = load_targets(parts["train"])
                    transform = fit_transform_for_cell(context, parts, fold, ev)
                    for seed in SEEDS:
                        if cell_filter and (ev, fold, treatment, seed) != cell_filter:
                            continue
                        result = train_cell(
                            context,
                            parts,
                            train_targets,
                            transform,
                            treatment,
                            fold,
                            seed,
                            torch,
                            lock_sha,
                            environment,
                        )
                        if result.get("status") != "complete":
                            raise RuntimeError(f"neural_cell_incomplete:{fold}:{treatment}:{seed}")
                        completed = sum(
                            1
                            for p in (RUN / "training").glob("*/fold_*/[MG]*/seed_*/manifest.json")
                            if json.loads(p.read_text()).get("status") == "complete"
                        )
                        atomic_json(
                            RUN / "training/run_progress.json",
                            {
                                "status": "running",
                                "completed_cells": completed,
                                "total_cells": total,
                                "last_cell": {
                                    "evaluation": ev,
                                    "fold": fold,
                                    "treatment": treatment,
                                    "seed": seed,
                                    "best_epoch": result["best_epoch"],
                                    "elapsed_seconds": result["elapsed_seconds"],
                                },
                                "updated_unix": time.time(),
                                "lock_sha256": lock_sha,
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "cell_status": "complete",
                                    "completed_cells": completed,
                                    "total_cells": total,
                                    "evaluation": ev,
                                    "fold": fold,
                                    "treatment": treatment,
                                    "seed": seed,
                                    "best_epoch": result["best_epoch"],
                                    "elapsed_seconds": result["elapsed_seconds"],
                                }
                            ),
                            flush=True,
                        )
                if ev == "E1" and treatment == "G1" and args.stage == "train":
                    baseline_sanity(context)
        if args.stage == "train":
            atomic_json(
                RUN / "training/run_progress.json",
                {
                    "status": "COMPLETE_120_OF_120",
                    "completed_cells": completed,
                    "total_cells": total,
                    "lock_sha256": lock_sha,
                    "updated_unix": time.time(),
                },
            )
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
