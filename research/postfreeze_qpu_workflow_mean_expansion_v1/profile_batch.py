#!/usr/bin/env python3
"""Target-free CUDA batch profile for the expanded cohort (E1 fold-0 fit only)."""
from __future__ import annotations
import json, os, resource, statistics, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"artifacts/postfreeze_qpu_workflow_mean_expansion_v1"
sys.path.insert(0,str(ROOT/"benchmark_v1/scripts"))
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/"research/postfreeze_mali_shots_ablation_v1"))
import run_mali_full_features as base
import run_mali_batched as batching
from train_neural import ExampleCache, METHODS, fit_transform, make_model, environment_receipt
from core import atomic_json, load_context, parts_for, sha, stable
from shots_core import append_shots, fit_shot_transform, transform_shots

def main():
    context, _targets, split_receipt=load_context("E1",include_targets=False)
    parts=parts_for(context,0)
    # All transforms use only E1/fold0 inner-fit IDs.
    transform=fit_transform(context,parts,0)
    shots=[float(context["panel_by_id"][c]["shots"]) for c in parts["fit"]]
    transform["shots"]=fit_shot_transform(shots)
    node_rank=sorted(parts["fit"],key=lambda c:int(context["ledger"][c]["graph_node_count"]))
    shot_rank=sorted(parts["fit"],key=lambda c:float(context["panel_by_id"][c]["shots"]))
    strata=(node_rank[:32],node_rank[-32:],shot_rank[:32],shot_rank[-32:])
    candidates=[]
    for i in range(32):
        for seq in strata: candidates.append(seq[i])
    candidates=list(dict.fromkeys(candidates))
    for cid in node_rank:
        if len(candidates)>=128: break
        if cid not in candidates: candidates.append(cid)
    profile_ids=candidates[:128]
    if len(profile_ids)<128: raise RuntimeError("profile_fit_pool_under_128")
    node_counts=[int(context["ledger"][c]["graph_node_count"]) for c in profile_ids]
    torch=__import__("torch")
    if not torch.cuda.is_available():raise RuntimeError("CUDA_required_no_CPU_fallback")
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    from gpu_lease import acquire_gpu_lease
    lease=acquire_gpu_lease("postfreeze-qpu-workflow-expansion-batch-profile")
    try:
        gpu=base.validate_gpu_budget(torch,budget_gib=12,reserve_gib=4)
        environment=environment_receipt(torch)
        cache=ExampleCache(context,transform,"G1",4*1024**3)
        # Materialize the complete target-free workload into the bounded cache.
        for cid in profile_ids: cache.get(cid)
        synthetic={cid:float((i%17)+1) for i,cid in enumerate(profile_ids)}
        rows=[]
        for effective in (32,64,128):
            base.seed_all(42)
            model,_,projection=make_model("G1",transform,42,torch)
            optimizer=torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=0.0001)
            # Warm-up on fit-only inputs and synthetic values; never read target_seconds.
            batching.loss_step(model,optimizer,profile_ids[:effective],METHODS["G1"]["internal"],
                250000,cache,synthetic,torch)
            durations=[]; examples_per_second=[]
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(3):
                chunk=profile_ids[(repeat*effective)% (128-effective+1):(repeat*effective)% (128-effective+1)+effective]
                if len(chunk)!=effective: chunk=profile_ids[-effective:]
                begin=time.perf_counter()
                batching.loss_step(model,optimizer,chunk,METHODS["G1"]["internal"],
                    250000,cache,synthetic,torch)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin
                durations.append(elapsed);examples_per_second.append(effective/elapsed)
            rows.append({"effective_batch":effective,"repeats":3,"median_optimizer_step_seconds":statistics.median(durations),
                "median_samples_per_second":statistics.median(examples_per_second),"step_seconds":durations,
                "step_samples_per_second":examples_per_second,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
                "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "profile_effective_graph_nodes":sum(node_counts[:effective]),"max_single_graph_nodes":max(node_counts),
                "projection":projection,"uses_runtime_targets":False})
            del optimizer,model
            torch.cuda.empty_cache()
        speed32=next(r["median_samples_per_second"] for r in rows if r["effective_batch"]==32)
        chosen=32
        for r in rows:
            r["speedup_vs_32"]=(r["median_samples_per_second"]/speed32)
            r["passes_10pct_speed_gate"]=(r["effective_batch"]>32 and r["speedup_vs_32"]>=1.10)
        eligible=[r["effective_batch"] for r in rows if r["effective_batch"]>32 and r["passes_10pct_speed_gate"]]
        if eligible: chosen=max(eligible)
        receipt={"status":"PASS_TARGET_FREE_BATCH_PROFILE","study":"postfreeze_qpu_workflow_mean_expansion_v1",
          "evaluation":"E1 fold0 INNER-FIT only","outer_test_targets_loaded":False,"synthetic_targets_only":True,
          "fit_ids_sha256":base.stable_hash(parts["fit"]),"profile_ids":profile_ids,
          "profile_ids_sha256":base.stable_hash(profile_ids),"graph_nodes":{"min":min(node_counts),"median":statistics.median(node_counts),"max":max(node_counts)},
          "profile_includes_smallest_largest_graph_and_shot_regimes":True,"node_budget":250000,
          "gpu_validation":gpu,"environment":environment,"batch_results":rows,
          "chosen_common_effective_batch":chosen,"selection_rule":"largest tested batch with >=10% median samples/s gain over32 and successful budget validation; otherwise32",
          "split_receipt_sha256":sha(OUT/"splits/split_receipt.json"),"runner_sha256":sha(Path(__file__)),
          "trainer_source_sha256":sha(HERE/"train_neural.py"),"core_source_sha256":sha(HERE/"core.py"),
          "feature_preflight_sha256":sha(OUT/"cohort/feature_preflight_receipt.json")}
        atomic_json(OUT/"profiling/batch_profile.json",receipt)
        print(json.dumps({"status":receipt["status"],"chosen_batch":chosen,"batch_results":rows,
          "gpu":gpu,"environment":{"gpu":environment["gpu"],"torch":environment["torch"],"cuda":environment["cuda_runtime"]}},indent=2))
    finally:lease.release()
if __name__=="__main__":main()
