#!/usr/bin/env python3
"""Fold-scoped model/input loading; deliberately keeps labels out of inference."""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch_geometric.data import Data

from common import ART, METHODS, ROOT, SEEDS, SRC, load_native, load_panel, load_split, rows, sha

sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))
sys.path.insert(0, str(ROOT / "benchmark_v1/shots"))
import run_mali_full_features as base
from shots_core import append_shots, transform_shots


class GraphRecordCache:
    """Bounded cache of compressed-source graph arrays, keyed by frozen path/hash."""
    def __init__(self, max_bytes=1024**3):
        self.max_bytes = int(max_bytes)
        self.items = OrderedDict()
        self.bytes = 0

    def get(self, relative_path, expected_sha):
        key = (relative_path, expected_sha)
        if key in self.items:
            value, size = self.items.pop(key)
            self.items[key] = (value, size)
            return value
        path = ROOT / relative_path
        if sha(path) != expected_sha:
            raise RuntimeError(f"graph_hash_mismatch:{relative_path}")
        record = base._load_graph_npz(path)
        size = sum(int(x.nbytes) for x in record.values() if hasattr(x, "nbytes"))
        if size <= self.max_bytes:
            while self.items and self.bytes + size > self.max_bytes:
                _, (_, old_size) = self.items.popitem(last=False)
                self.bytes -= old_size
            self.items[key] = (record, size)
            self.bytes += size
        return record


class StudyInputs:
    def __init__(self, evaluation):
        if evaluation not in {"E1", "E2", "E3"}:
            raise ValueError(evaluation)
        self.evaluation = evaluation
        self.panel, self.panel_by_id, self.ledger = load_panel()
        self.native = load_native()
        self.test_by_case = load_split(evaluation)
        self.methods = METHODS
        self.graph_cache = GraphRecordCache()
        self.current_case = None
        self.expected = {}
        self.seed_expected = {}
        self.models = {}
        self.transforms = {}
        self.estimators = {}
        self.constants = {}
        self.base_model_class, self.model_source_sha = base.load_pinned_model()

    def load_case(self, case):
        if case == self.current_case:
            return
        self.models.clear(); self.transforms.clear(); self.expected.clear(); self.seed_expected.clear()
        self.estimators.clear(); self.constants.clear()
        fold = int(case)
        transform_path = SRC / "training" / self.evaluation / f"fold_{fold}" / "transform.json"
        transform = json.loads(transform_path.read_text())
        self.transforms["fold"] = transform
        for method in ("M0", "M1", "G0", "G1"):
            internal = "mali_full_features_graph" if method.startswith("G") else "mali_full_features_mlp"
            graph = method.startswith("G")
            shots = method.endswith("1")
            model_group = {}
            expected_seeds = {}
            retained = len(transform["global"]["retained_indices"])
            for seed in SEEDS:
                folder = SRC / "training" / self.evaluation / f"fold_{fold}" / method / f"seed_{seed}"
                checkpoint_path = folder / "checkpoint.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                manifest = json.loads((folder / "manifest.json").read_text())
                if checkpoint.get("identity") != manifest.get("identity") or int(manifest["best_epoch"]) < 1:
                    raise RuntimeError(f"checkpoint_manifest_mismatch:{self.evaluation}:{fold}:{method}:{seed}")
                base.seed_all(seed)
                model = base.make_model(self.base_model_class, internal, retained, checkpoint_layers=graph)
                if graph:
                    model.mask = model.mask.to("cuda:0")
                if shots:
                    constructor_rng = base.capture_rng_state()
                    from shots_core import expand_global_projection
                    expand_global_projection(model, torch)
                    base.restore_rng_state(constructor_rng)
                state = checkpoint["best_model_state"]
                model.load_state_dict(state, strict=True)
                model.to("cuda:0").eval()
                model_group[seed] = model
                seed_rows = rows(folder / "predictions.csv")
                seed_predictions = {r["canonical_observation_id"]: float(r["predicted_seconds"]) for r in seed_rows}
                if len(seed_predictions) != len(seed_rows):
                    raise RuntimeError("duplicate_saved_seed_prediction_id")
                expected_seeds[seed] = seed_predictions
            self.models[method] = model_group
            self.seed_expected[method] = expected_seeds
            for cid in self.test_by_case[case]:
                seed_values = {str(s): expected_seeds[s][cid] for s in SEEDS}
                self.expected.setdefault(method, {})[cid] = {
                    "prediction": float(np.median(list(seed_values.values()))), "seeds": seed_values,
                }
        for method in ("R0", "C0"):
            folder = SRC / "classical" / self.evaluation / f"fold_{fold}"
            r0_manifest = json.loads((folder / "regression_manifest.json").read_text())
            c0_manifest = json.loads((folder / "constant_manifest.json").read_text())
            if not c0_manifest.get("status") == "complete":
                raise RuntimeError("constant_manifest_incomplete")
            if method == "R0":
                state_path = ART / "recovered_classical" / self.evaluation / f"fold_{fold}" / "estimator.joblib"
                receipt = json.loads((state_path.parent / "receipt.json").read_text())
                if receipt.get("status") != "PASS_PREDICTION_PARITY_VALIDATED_RECONSTRUCTION" or sha(state_path) != receipt["state_sha256"]:
                    raise RuntimeError(f"R0_recovery_not_valid:{self.evaluation}:{fold}")
                self.estimators[method] = joblib.load(state_path)
                pred_rows = rows(folder / "regression_predictions.csv")
            else:
                self.constants[method] = float(c0_manifest["outer_train_median_seconds"])
                pred_rows = rows(folder / "constant_predictions.csv")
            pred_map = {r["canonical_observation_id"]: float(r["predicted_seconds"]) for r in pred_rows}
            if set(pred_map) != set(self.test_by_case[case]):
                raise RuntimeError(f"saved_prediction_identity_mismatch:{self.evaluation}:{case}:{method}")
            self.expected[method] = {cid: {"prediction": pred_map[cid]} for cid in pred_map}
        # Reference files above contain predictions only; the combined target/prediction
        # analysis table is never opened by this timing process.
        ids = self.test_by_case[case]
        for method in self.methods:
            for cid in ids:
                if cid not in self.expected.get(method, {}):
                    raise RuntimeError(f"missing_saved_prediction:{method}:{cid}")
        self.current_case = case

    def raw_record(self, cid):
        row = self.ledger[cid]
        return self.graph_cache.get(row["graph_artifact"], row["graph_artifact_sha256"])

    def raw_request(self, cid, method):
        panel = self.panel_by_id[cid]
        request = {"id": cid}
        if method in {"M0", "M1", "G0", "G1"}:
            row = self.ledger[cid]
            request["globals"] = np.asarray([float(row[f"global_{i:02d}"]) for i in range(51)], dtype=np.float32)
            if method.endswith("1"):
                request["shots"] = float(panel["shots"])
            if method.startswith("G"):
                request["graph_record"] = self.raw_record(cid)
                request["graph_path"] = row["graph_artifact"]
                request["graph_sha"] = row["graph_artifact_sha256"]
        elif method == "R0":
            request["native5"] = self.native[cid]
        return request

    def prepare_ready(self, method, request):
        transform = self.transforms["fold"]
        global_features = torch.as_tensor(base.transform_global(request["globals"].tolist(), transform["global"]), dtype=torch.float32).reshape(1, -1)
        if method.startswith("G"):
            x = base.expand_dense_x(request["graph_record"]).astype(np.float32, copy=True)
            mean = np.asarray(transform["node"]["mean"], dtype=np.float32)
            std = np.asarray(transform["node"]["std"], dtype=np.float32)
            variable = np.asarray(transform["node"]["variable_mask"], dtype=bool)
            x[:, variable] = (x[:, variable] - mean[variable]) / std[variable]
            x[:, ~variable] = 0.0
            data = Data(x=torch.as_tensor(x, dtype=torch.float32),
                        edge_index=torch.as_tensor(request["graph_record"]["edge_index"].astype(np.int64, copy=False), dtype=torch.long),
                        global_features=global_features)
        else:
            data = Data(global_features=global_features)
        if method.endswith("1"):
            data = append_shots(data, transform_shots(request["shots"], transform["shots"]))
        return data

    def prepare_raw_B(self, method, request):
        """Transform raw cached arrays inside B. No normalized feature reuse."""
        if method in {"R0", "C0"}:
            return request
        return self.prepare_ready(method, request)

    def predict_ready(self, method, cid, prepared):
        if method in {"M0", "M1", "G0", "G1"}:
            graph = method.startswith("G")
            pred, seed_preds, elapsed = __import__("timing").neural_predict(
                self.models[method], prepared,
                lambda examples, graph, device: base._batch_to_device([examples], device, graph),
                graph=graph,
            )
            return pred, seed_preds, elapsed
        if method == "R0":
            from timing import predict_r0
            pred, elapsed = predict_r0(self.estimators[method], prepared["native5"])
            return pred, {}, elapsed
        from timing import predict_c0
        pred, elapsed = predict_c0(self.constants[method])
        return pred, {}, elapsed

    def check(self, method, cid, prediction, seeds):
        from timing import check_prediction
        check_prediction(prediction, self.expected[method][cid], method, seeds)
