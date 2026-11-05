#!/usr/bin/env python3
"""Measure CUDA batch throughput, then fit a separately identified Ma-Li adaptation.

Reuse the frozen extractor/model/split/checkpoint helpers; do not edit their
receipts. Tune with fitting inputs and synthetic labels only. Logs and profiles
are durable evidence for this execution context, not replacement raw results.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from benchmark26 import mali_training as base
from benchmark26.paths import ARTIFACTS, PROTOCOLS

ROOT = base.ROOT
CONTRACT = PROTOCOLS / "mali_batch_optimization.json"
DEFAULT_FEATURES = ARTIFACTS / "real_qpu/mali_full_features_unified"
DEFAULT_OUTPUT = ARTIFACTS / "real_qpu/mali_full_features_large_batch"
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
        """Return the cached example for (observation, variant), building and caching it if needed.

        The cache is limited to `budget` bytes and evicts the least recently used examples.
        """
        key = (cid, variant)
        if key in self.items:
            value, size = self.items.pop(key)
            self.items[key] = (value, size)
            return value
        value = base._make_example(
            cid,
            variant,
            self.context["ledger"],
            None,
            self.transform,
            self.context["feature_root"],
            "cpu",
            False,
            self.verified,
        )
        size = sum(t.numel() * t.element_size() for _, t in value if hasattr(t, "numel"))
        if size <= self.budget:
            while self.items and self.bytes + size > self.budget:
                _, (_, removed) = self.items.popitem(last=False)
                self.bytes -= removed
            self.items[key] = (value, size)
            self.bytes += size
        return value


def load_context(feature_root):
    """Load and verify everything the batch tools need from a feature directory built earlier.

    Checks the input manifest against the protocol hash and the feature ledger hash
    (8,767 canonical identities), the feature and environment pins, the technical gate
    receipt and the preflight report. Returns the ledger, the canonical rows, the outer and
    inner splits and the pins.
    """
    protocol, protocol_sha, execution_sha = base.canonical_protocol_hashes()
    inp_path = feature_root / "input_manifest.json"
    inp = json.loads(inp_path.read_text())
    ledger_path = feature_root / "feature_status_ledger.csv"
    if (
        inp.get("full_panel_materialized") is not True
        or inp.get("canonical_identity_count") != 8767
        or inp.get("protocol_sha256") != protocol_sha
        or inp.get("feature_status_ledger_sha256") != base.sha256_file(ledger_path)
    ):
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
    return {
        "feature_root": feature_root,
        "protocol": protocol,
        "ledger": ledger,
        "canonical": canonical,
        "outer": base.read_csv(base.OUTER),
        "inner": base.read_csv(base.INNER),
        "technical": technical,
        "pins": {
            "parent_protocol": protocol_sha,
            "parent_execution": execution_sha,
            "input_manifest": base.sha256_file(inp_path),
            "feature_ledger": base.sha256_file(ledger_path),
            "parent_runner": base.sha256_file(Path(base.__file__)),
            "runner": base.sha256_file(Path(__file__)),
            "batch_contract": base.sha256_file(CONTRACT),
            "technical_receipt": base.sha256_file(base.TECHNICAL_GATE_RECEIPT_PATH),
            "neural_environment": environment,
        },
    }


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


if __name__ == "__main__":
    main()
