#!/usr/bin/env python3
"""Build a target-separated, hash-pinned model-input table for the expanded QPU study.

This is an amendment-local reader/materializer. Frozen source tables and feature
archives are read-only. The five GraphState-q3 rows are materialized from the
already functionally-qualified MQT 1.0.3 logical recipe and their pinned backend
snapshot; no runtime value is passed to feature extraction.
"""
from __future__ import annotations

import os as _b26_os
from pathlib import Path

import csv, hashlib, json, math, os, sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research/postfreeze_qpu_workflow_mean_expansion_v1"
OUT = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_expansion_v1'
PLAN_DIR = ROOT / "docs/final_completion/qpu_workflow_mean_expansion_v1"
PANEL_IN = PLAN_DIR / "proposed_observations.csv"
FEATURES_IN = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'real_qpu/logical_inputs/feature_attempts.csv'
NATIVE_IN = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'real_qpu/qonductor_native_features/feature_rows.csv'
GROUPS_IN = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'real_qpu/logical_inputs/model_input_groups.csv'
QPACK_IN = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v1/canonical_corpus_8767_20260926/qpack_mcp.csv'
REP_IN = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/real_qpu/compiled_unified/representation_rows.csv'
SNAPSHOT_REGISTRY = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv'
GRAPHSTATE_QASM = Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qonductor_generator_version_semantics_v1/generated/1.0.3/graphstate__q3.qasm'

sys.path.insert(0, str(ROOT / "benchmark_v1/scripts"))

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def read(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def write_csv(path: Path, rows: list[dict], fields=None):
    if not rows: raise RuntimeError(f"refuse_empty_output:{path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction="ignore");w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def write_json(path: Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with tmp.open("rb") as f: os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    if (OUT/"cohort/feature_ledger.csv").exists():
        raise RuntimeError("refuse_overwrite_existing_cohort; use a new amendment attempt directory")
    import numpy as np
    import mali_full_features as mf
    from qiskit import qasm2
    def canonical_instruction_digest(circuit, *, include_parameters=True):
        # Use the same ordered operation/operand signature as logical-input grouping.
        body={"qubits":circuit.num_qubits,"clbits":circuit.num_clbits,"instructions":[]}
        for inst in circuit.data:
            op=inst.operation; params=[]
            for value in getattr(op,"params",()):
                try: params.append(float(value))
                except (TypeError,ValueError): params.append(str(value))
            item={"name":str(op.name),"qargs":[circuit.find_bit(q).index for q in inst.qubits],
                  "cargs":[circuit.find_bit(b).index for b in inst.clbits],
                  "condition":str(getattr(op,"condition",None))}
            if include_parameters: item["params"]=params
            body["instructions"].append(item)
        return sha_bytes(json.dumps(body,sort_keys=True,separators=(",",":")).encode())

    proposed=read(PANEL_IN); attempts={r["canonical_observation_id"]:r for r in read(FEATURES_IN)}
    native={r["canonical_row_id"]:r for r in read(NATIVE_IN)}
    group_rows={r["canonical_observation_id"]:r for r in read(GROUPS_IN)}
    rep={r["canonical_row_id"]:r for r in read(REP_IN)}
    registry=mf.load_snapshot_registry()
    qpack_rows=read(QPACK_IN)
    qpack_by_workflow=defaultdict(list)
    for r in qpack_rows:
        raw=json.loads(r["raw_row_json"])
        wid=hashlib.sha256(raw["workflow_id"].encode()).hexdigest()
        qpack_by_workflow[wid].append(r)

    # Deterministic, target-blind representative for each declared workflow mean.
    qpack_representative={}
    for row in proposed:
        if row["source"]!="qpack_mcp": continue
        wid=row["observation_id"].split("|",1)[1]
        members=qpack_by_workflow.get(wid,[])
        if not members: raise RuntimeError(f"qpack_workflow_missing:{wid}")
        if len({(x["backend"],x["shots"]) for x in members})!=1:
            raise RuntimeError(f"qpack_workflow_context_not_constant:{wid}")
        ids=[x["canonical_row_id"] for x in members]
        feature_rows=[attempts.get(i) for i in ids]
        native_rows=[native.get(i) for i in ids]
        if any(x is None or x["status"]!="available" for x in feature_rows):
            raise RuntimeError(f"qpack_workflow_feature_missing:{wid}")
        if any(x is None or x["availability_status"]!="available" for x in native_rows):
            raise RuntimeError(f"qpack_workflow_native_feature_missing:{wid}")
        if len({x["graph_sha256"] for x in feature_rows})!=1 or len({tuple(x[f"g{i:02d}"] for i in range(51)) for x in feature_rows})!=1:
            raise RuntimeError(f"qpack_workflow_logical_features_vary:{wid}")
        ncols=("swap","depth","num_qubits","shots","circuit_count")
        if len({tuple(x[k] for k in ncols) for x in native_rows})!=1:
            raise RuntimeError(f"qpack_workflow_regression_features_vary:{wid}")
        qpack_representative[wid]=min(ids)

    # Extract the GraphState representation missing from the feature table.
    missing=[r for r in proposed if r["source"]=="qonductor_single_circuit_ibm"
             and attempts.get(r["observation_id"],{}).get("status")!="available"]
    repair_rows=[]; repair_receipt={"status":"NOT_NEEDED","observations":0}
    if missing:
        if len(missing)!=5 or {r["logical_signature"] for r in missing}!={"4ee5905a76da603df5b50df3c28f2e198bec4a68c1d784702b3b5577c43c0f72"}:
            raise RuntimeError(f"unexpected_missing_feature_set:{[(r['observation_id'],r['logical_signature']) for r in missing]}")
        qbytes=GRAPHSTATE_QASM.read_bytes(); qhash=sha_bytes(qbytes)
        circuit=qasm2.loads(qbytes.decode("utf-8"),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        sig=canonical_instruction_digest(circuit,include_parameters=False)
        if sig!="4ee5905a76da603df5b50df3c28f2e198bec4a68c1d784702b3b5577c43c0f72":
            raise RuntimeError("graphstate_q3_recipe_signature_mismatch")
        first=rep[missing[0]["observation_id"]]
        if any(rep[r["observation_id"]]["target_properties_sha256"]!=first["target_properties_sha256"] for r in missing):
            raise RuntimeError("graphstate_missing_rows_snapshot_not_constant")
        calibration, snap_info=mf.snapshot_t1_t2(first,registry)
        globals51=mf.extract_global_features(circuit,mf.load_pinned_global_extractor()[0])
        sparse,graph_info=mf.compact_graph_record(circuit,calibration)
        # Cache by QASM, backend snapshot and schema so different representations do not share a graph.
        gkey=sha_bytes((qhash+first["target_properties_sha256"]+"mali-51x178-v1").encode())
        graph_path=OUT/"features/graphs"/(gkey+".npz")
        graph_path.parent.mkdir(parents=True,exist_ok=True)
        if graph_path.exists(): raise RuntimeError("refuse_overwrite_graph_cache")
        np.savez_compressed(graph_path,**sparse)
        graph_hash=sha(graph_path); rel=str(graph_path.relative_to(ROOT))
        for r in missing:
            rec={"canonical_observation_id":r["observation_id"],"graph_file":rel,"graph_sha256":graph_hash,
                 "node_count":str(int(len(sparse["node_type"]))),"edge_count":str(int(sparse["edge_index"].shape[1])),
                 "source_qasm_sha256":qhash,"properties_sha256":first["target_properties_sha256"],
                 "snapshot_id":first["target_snapshot_id"],"logical_signature":sig,"status":"available_direct_mqt103_recipe"}
            rec.update({f"g{i:02d}":repr(float(v)) for i,v in enumerate(globals51)})
            repair_rows.append(rec)
        repair_receipt={"status":"PASS_DIRECT_MATERIALIZATION_FROM_FUNCTIONALLY_QUALIFIED_RECIPE",
          "observations":len(missing),"logical_signature":sig,"candidate_qasm_path":str(GRAPHSTATE_QASM.relative_to(ROOT)),
          "candidate_qasm_sha256":qhash,"snapshot_id":first["target_snapshot_id"],
          "snapshot_properties_sha256":first["target_properties_sha256"],"snapshot_metadata":snap_info,
          "node_count":len(sparse["node_type"]),"edge_count":int(sparse["edge_index"].shape[1]),
          "global_feature_width":len(globals51),"feature_graph_sha256":graph_hash,
          "feature_graph_path":rel,"graph_metadata":graph_info,
          "no_runtime_targets_used":True,"source_feature_artifacts_modified":False,
          "qualification_scope":"readout-state functional qualification from the saved Qonductor generator-version audit; not exact historical logical QASM recovery"}

    repair_by_id={r["canonical_observation_id"]:r for r in repair_rows}
    panel=[]; targets=[]; ledger=[]; native_rows_out=[]
    for r in proposed:
        cid=r["observation_id"]; source=r["source"]
        if source=="qpack_mcp": feature_id=qpack_representative[cid.split("|",1)[1]]
        else: feature_id=cid
        feat=attempts.get(feature_id)
        if feature_id in repair_by_id:
            f=repair_by_id[feature_id]
            frow={"graph_file":f["graph_file"],"graph_sha256":f["graph_sha256"],"node_count":f["node_count"],"edge_count":f["edge_count"],
                  "source_qasm_sha256":f["source_qasm_sha256"],"status":"available"}
            frow.update({f"g{i:02d}":f[f"g{i:02d}"] for i in range(51)})
        elif feat and feat["status"]=="available":
            frow=feat
        else:
            raise RuntimeError(f"no_feature_for_candidate:{cid}:{feature_id}:{feat.get('reason') if feat else 'missing'}")
        if frow.get("status")!="available": raise RuntimeError(f"feature_unavailable:{cid}")
        vals=[float(frow[f"g{i:02d}"]) for i in range(51)]
        if not all(math.isfinite(v) for v in vals): raise RuntimeError(f"nonfinite_globals:{cid}")
        graph_rel=frow["graph_file"] if frow["graph_file"].startswith("artifacts/") else str(Path("artifacts/real_qpu/logical_inputs")/frow["graph_file"])
        graph=(ROOT/graph_rel).resolve()
        if ROOT.resolve() not in graph.parents or not graph.is_file(): raise RuntimeError(f"graph_path_invalid:{cid}")
        actual_graph_sha=sha(graph)
        if actual_graph_sha!=frow["graph_sha256"]: raise RuntimeError(f"graph_sha_mismatch:{cid}")
        y=float(r["target_seconds"]); shots=float(r["shots"])
        if not math.isfinite(y) or y<=0 or not math.isfinite(shots) or shots<=0: raise RuntimeError(f"target_or_shots_invalid:{cid}")
        # QPack representative mapping is part of the input provenance, not model features.
        source_row=feature_id
        nrow=native.get(source_row)
        if nrow is None or nrow["availability_status"]!="available": raise RuntimeError(f"native_baseline_input_missing:{cid}:{source_row}")
        panel.append({"canonical_observation_id":cid,"source_id":source,"group_id":r["proposed_conservative_group"],
          "logical_signature":r["logical_signature"],"backend":r["backend"].replace("ibmq_","ibm_"),"shots":repr(shots),
          "scope":r["scope"],"aggregation":r["aggregation"],"constituent_count":r["constituent_count"],
          "provenance":r["provenance"],"feature_source_observation_id":feature_id,
          "feature_source_qasm_sha256":frow.get("source_qasm_sha256","")})
        targets.append({"canonical_observation_id":cid,"target_seconds":repr(y)})
        row={"canonical_observation_id":cid,"group_id":r["proposed_conservative_group"],
          "graph_artifact":graph_rel,"graph_artifact_sha256":frow["graph_sha256"],
          "graph_node_count":frow["node_count"],"graph_edge_count":frow["edge_count"],
          "graph_status":"available","global_status":"available",
          "source_qasm_sha256":frow.get("source_qasm_sha256","")}
        row.update({f"global_{i:02d}":frow[f"g{i:02d}"] for i in range(51)})
        ledger.append(row)
        native_rows_out.append({"canonical_observation_id":cid,**{k:nrow[k] for k in ("swap","depth","num_qubits","shots","circuit_count")}})

    if len(panel)!=3443 or len({r["canonical_observation_id"] for r in panel})!=3443:
        raise RuntimeError("candidate_observation_identity_mismatch")
    sources=Counter(r["source_id"] for r in panel); groups=Counter((r["source_id"],r["group_id"]) for r in panel)
    if sources!={"mali_real_qpu":340,"qonductor_single_circuit_ibm":3058,"qpack_mcp":45}:
        raise RuntimeError(f"candidate_source_counts_mismatch:{sources}")
    if len({r["group_id"] for r in panel})!=203: raise RuntimeError("candidate_declared_group_count_mismatch")

    # Feature-collision audit. Full-input identity uses every raw global value
    # and every sparse DAG array; global-only collisions are reported separately.
    graph_payload_cache={}; group_ids={r["canonical_observation_id"]:r["group_id"] for r in panel}
    global_keys=defaultdict(set); full_keys=defaultdict(set)
    feature_by_id={r["canonical_observation_id"]:r for r in ledger}
    collision_rows=[]
    for r in ledger:
        cid=r["canonical_observation_id"]; gid=r["group_id"]
        gkey=sha_bytes(json.dumps([r[f"global_{i:02d}"] for i in range(51)],separators=(",",":")).encode())
        global_keys[gkey].add(gid)
        graphpath=(ROOT/r["graph_artifact"]).resolve()
        if r["graph_artifact"] not in graph_payload_cache:
            with np.load(graphpath,allow_pickle=False) as a:
                canonical=[]
                for name in sorted(a.files):
                    arr=np.ascontiguousarray(a[name]);canonical.append([name,str(arr.dtype),list(arr.shape),sha_bytes(arr.tobytes())])
            graph_payload_cache[r["graph_artifact"]]=sha_bytes(json.dumps(canonical,separators=(",",":")).encode())
        fkey=sha_bytes((gkey+graph_payload_cache[r["graph_artifact"]]).encode())
        full_keys[fkey].add(gid)
    for kind, mapping in (("global51_only",global_keys),("complete_51plus_sparse_DAG",full_keys)):
        for key,gs in mapping.items():
            if len(gs)>1: collision_rows.append({"identity_kind":kind,"feature_sha256":key,"distinct_groups":len(gs),"group_ids":";".join(sorted(gs))})
    exact_aliases=[r for r in collision_rows if r["identity_kind"]=="complete_51plus_sparse_DAG"]
    if exact_aliases:
        raise RuntimeError(f"complete_feature_identity_crosses_declared_groups:{len(exact_aliases)}; inspect before splits")

    panel_fields=list(panel[0]); ledger_fields=list(ledger[0]); native_fields=list(native_rows_out[0])
    write_csv(OUT/"cohort/panel.csv",panel,panel_fields)
    write_csv(OUT/"cohort/targets.csv",targets)
    write_csv(OUT/"cohort/feature_ledger.csv",ledger,ledger_fields)
    write_csv(OUT/"cohort/native_baseline_features.csv",native_rows_out,native_fields)
    write_csv(OUT/"cohort/feature_collision_audit.csv",collision_rows or [{"identity_kind":"none","feature_sha256":"","distinct_groups":0,"group_ids":""}])
    write_csv(OUT/"cohort/qpack_feature_source_map.csv",[{"workflow_mean_id":r["canonical_observation_id"],"feature_source_observation_id":r["feature_source_observation_id"],"constituent_count":r["constituent_count"],"backend":r["backend"],"shots":r["shots"],"mapping_rule":"minimum canonical observation ID among same QPack workflow; feature constancy audited across all constituents"} for r in panel if r["source_id"]=="qpack_mcp"])
    inputs=[PANEL_IN,FEATURES_IN,NATIVE_IN,GROUPS_IN,QPACK_IN,REP_IN,SNAPSHOT_REGISTRY,GRAPHSTATE_QASM,
      Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'real_qpu/qualified_logical_panel/panel.csv',
      Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qonductor_generator_version_semantics_v1/semantics_v2_legacy_parser/observation_semantic_results.csv',
      ROOT/"benchmark_v1/scripts/mali_full_features.py",ROOT/"benchmark_v1/scripts/prepare_logical_benchmark.py"]
    receipt={"status":"PASS_TARGET_FREE_FEATURE_PREFLIGHT","observations":len(panel),"groups":len({r['group_id'] for r in panel}),
      "source_counts":dict(sources),"available_feature_rows":len(ledger),"feature_width":51,"graph_node_width":178,
      "group_source_counts":{s:len({r['group_id'] for r in panel if r['source_id']==s}) for s in sorted(sources)},
      "backends":len({r['backend'] for r in panel}),"shot_levels":len({r['shots'] for r in panel}),
      "graph_repair":repair_receipt,"qpack_workflow_means":len(qpack_representative),
      "qpack_constituent_rows":sum(int(r['constituent_count']) for r in panel if r['source_id']=='qpack_mcp'),
      "qpack_within_workflow_feature_constancy":"PASS_45_OF_45_LOGICAL_GLOBAL_DAG_AND_CLASSICAL_DESCRIPTOR_VECTORS",
      "global_only_collision_groups":sum(r['identity_kind']=='global51_only' for r in collision_rows),
      "complete_feature_cross_group_collisions":0,"target_values_used_for_feature_extraction":False,
      "frozen_source_artifacts_modified":False,"input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in inputs},
      "output_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'cohort/panel.csv',OUT/'cohort/targets.csv',OUT/'cohort/feature_ledger.csv',OUT/'cohort/native_baseline_features.csv',OUT/'cohort/feature_collision_audit.csv',OUT/'cohort/qpack_feature_source_map.csv']},
      "code_sha256":sha(Path(__file__))}
    write_json(OUT/"cohort/feature_preflight_receipt.json",receipt)
    print(json.dumps(receipt,indent=2))

if __name__=="__main__": main()
