#!/usr/bin/env python3
"""Create final scientific report, provenance ledgers and checksums."""
from __future__ import annotations
import csv,hashlib,json,platform,subprocess
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata as md
ROOT=Path(__file__).resolve().parents[2];NS=ROOT/'artifacts/postfreeze_unified_analytical_completion_v1';ST=ROOT/'artifacts/postfreeze_qpu_workflow_mean_expansion_v1'
def read(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(v,d=4):return '—' if v in ('',None) else f'{float(v):.{d}f}'
def mdtable(head,rows):
 out=['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']
 out += ['| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |' for row in rows]
 return '\n'.join(out)
def main():
 panel=read(ST/'cohort/panel.csv');targets={r['canonical_observation_id']:r for r in read(ST/'cohort/targets.csv')};line={r['canonical_observation_id']:r for r in read(NS/'lineage.csv')}
 bysource=defaultdict(list)
 for r in panel:bysource[r['source_id']].append(r)
 source_rows=[]
 for s,rr in sorted(bysource.items()):
  source_rows.append({'source_id':s,'rows':len(rr),'unique_groups':len({x['group_id'] for x in rr}),'backends':len({x['backend'] for x in rr}),
   'shots_min':min(float(x['shots']) for x in rr),'shots_max':max(float(x['shots']) for x in rr),
   'unique_graph_snapshot_tasks':len({(line[x['canonical_observation_id']]['graph_file'],line[x['canonical_observation_id']]['snapshot_id']) for x in rr}),
   'structure_provenance_classes':';'.join(sorted({line[x['canonical_observation_id']]['structure_provenance'] for x in rr})),
   'timing_provenance_classes':';'.join(sorted({line[x['canonical_observation_id']]['timing_provenance'] for x in rr}))})
 write(NS/'source_inventory.csv',source_rows)
 write(NS/'unavailable_methods.csv',[{'method_id':'SCHOLTEN_ORIGINAL','status':'NOT_AVAILABLE_ON_UNIFIED_COHORT','reason':'Compatible row-aligned effective depth and CLOPS-V inputs are not established; older nominal CLOPS-V-like adaptation is not the original estimator.','evidence':'artifacts/benchmark_v3/results/two_domain_recovery_reader_v2/manifest.json; artifacts/benchmark_v3/results/two_domain_recovery_reader_v2/REPORT.md','scoring':'NOT_RUN'}])
 metrics=read(NS/'metrics.csv');paired=read(NS/'paired_comparisons.csv');corr=read(NS/'correlations.csv');sliced=read(NS/'sliced_metrics.csv');fold=read(NS/'fold_metrics.csv')
 def get(ev,case,m):return next(r for r in metrics if r['evaluation']==ev and r['case']==case and r['method_id']==m)
 labels={
  'QISKIT_SINGLE':'Qiskit Target path (single execution)',
  'QISKIT_SINGLE_SHOT_SCALED':'Qiskit Target path × shots',
  'QISKIT_SINGLE_AFFINE':'Qiskit path + train-only affine',
  'QISKIT_SINGLE_SHOT_SCALED_AFFINE':'Qiskit path × shots + train-only affine',
  'QCRE_UNITARY_SINGLE':'QCRE unitary path (single execution)',
  'QCRE_UNITARY_SINGLE_SHOT_SCALED':'QCRE unitary path × shots',
  'QCRE_UNITARY_SINGLE_AFFINE':'QCRE unitary path + train-only affine',
  'QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE':'QCRE unitary path × shots + train-only affine',
  'NOTADS_PAPER_LEVEL_SCORE_AFFINE':'NoTaDS paper score + train-only affine',
  'NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE':'NoTaDS paper score × shots + train-only affine',
  'NOTADS_CODE_FILTERED_DEPTH_AFFINE':'NoTaDS public-code score + train-only affine',
  'NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE':'NoTaDS public-code score × shots + train-only affine',
  'IBM_QUICK_USAGE_DIAGNOSTIC':'IBM quick usage formula (diagnostic only)',
  'IBM_USAGE_DURATION_DIAGNOSTIC':'IBM duration-based usage formula (diagnostic only)',
  'R0':'Qonductor (tái lập/adaptation; R0 selector)',
  'C0':'Median runtime outer-train (C0)'}
 methodorder=['QISKIT_SINGLE','QISKIT_SINGLE_SHOT_SCALED','QISKIT_SINGLE_AFFINE','QISKIT_SINGLE_SHOT_SCALED_AFFINE',
  'QCRE_UNITARY_SINGLE','QCRE_UNITARY_SINGLE_SHOT_SCALED','QCRE_UNITARY_SINGLE_AFFINE','QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE',
  'NOTADS_PAPER_LEVEL_SCORE_AFFINE','NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE','NOTADS_CODE_FILTERED_DEPTH_AFFINE','NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE',
  'IBM_QUICK_USAGE_DIAGNOSTIC','IBM_USAGE_DURATION_DIAGNOSTIC','R0','C0']
 table=[]
 for m in methodorder:
  e1=get('E1','ALL',m);e2=get('E2','ALL',m)
  unit='separate service-use diagnostic' if m.endswith('_DIAGNOSTIC') else 'seconds against unified target'
  table.append([labels[m],unit,f(e1['group_balanced_mae_seconds']),f(e2['group_balanced_mae_seconds']),f(e1['pooled_mae_seconds']),f(e2['pooled_mae_seconds']),f(e1['raw_seconds_r2']),f(e2['raw_seconds_r2'])])
 main_table=mdtable(['Method / variant','Metric scope','E1 GB-MAE','E2 GB-MAE','E1 pooled MAE','E2 pooled MAE','E1 raw R²','E2 raw R²'],table)
 contrast_ids=['QISKIT_SINGLE_SHOT_SCALED_AFFINE','QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE','NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE','NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE']
 crows=[]
 for ev in ('E1','E2'):
  for m in contrast_ids:
   r=next(x for x in paired if x['evaluation']==ev and x['case']=='ALL' and x['treatment']==m and x['control']=='R0')
   crows.append([ev,labels[m],f(r['delta_group_balanced_mae_seconds']),f(r['ci95_lower_seconds']),f(r['ci95_upper_seconds']),f(r['empirical_bootstrap_fraction_delta_below_zero'],3),'10,000 paired group draws'])
 contrast_table=mdtable(['Evaluation','Δ GB-MAE vs Qonductor (s)','95% CI low','95% CI high','Bootstrap fraction Δ<0','Resampling'],crows)
 noscore=[]
 for m in ('NOTADS_PAPER_LEVEL_SCORE','NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED','NOTADS_CODE_FILTERED_DEPTH','NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED'):
  r=next(x for x in corr if x['evaluation']=='E1' and x['case']=='ALL' and x['scope']=='overall' and x['method_id']==m)
  noscore.append([m,r['input_unit'],f(r['pearson_target'],3),f(r['spearman_target'],3),r['rows'],'Unitless association only'])
 score_table=mdtable(['NoTaDS score','Unit','Pearson r','Spearman ρ','Rows','Interpretation'],noscore)
 e3=[]
 contexts=sorted({r['case'] for r in metrics if r['evaluation']=='E3'})
 e3methods=['R0','QISKIT_SINGLE_SHOT_SCALED_AFFINE','QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE','NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE','NOTADS_PAPER_LEVEL_SCORE_SHOT_SCALED_AFFINE']
 for c in contexts:
  r0=get('E3',c,'R0');e3.append([c,r0['rows'],r0['groups']]+[f(get('E3',c,m)['group_balanced_mae_seconds']) for m in e3methods])
 e3_table=mdtable(['Held-out backend','Rows','Groups','Qonductor','Qiskit path × shots + affine','QCRE unitary × shots + affine','NoTaDS code × shots + affine','NoTaDS paper × shots + affine'],e3)
 # Per-source group-balanced MAE under the main E1 split.
 src=[]
 srcmethods=['R0','QISKIT_SINGLE_SHOT_SCALED_AFFINE','QCRE_UNITARY_SINGLE_SHOT_SCALED_AFFINE','NOTADS_CODE_FILTERED_DEPTH_SHOT_SCALED_AFFINE']
 for source in sorted({r['scope_value'] for r in sliced if r['evaluation']=='E1' and r['case']=='ALL' and r['scope']=='source'}):
  vals=[]
  for m in srcmethods:
   r=next(x for x in sliced if x['evaluation']=='E1' and x['case']=='ALL' and x['scope']=='source' and x['scope_value']==source and x['method_id']==m)
   vals.append(f(r['group_balanced_mae_seconds']))
  src.append([source,len(bysource[source]),len({x['group_id'] for x in bysource[source]})]+vals)
 source_table=mdtable(['Source','Rows','Groups','Qonductor','Qiskit scaled affine','QCRE scaled affine','NoTaDS-code scaled affine'],src)
 figs='\n\n'.join([f'![{caption}](figures/{name})' for name,caption in [('unified_accuracy_main.png','Analytical routes and matched baselines.'),('e3_backend_heatmap.png','E3 backend-context results.'),('e2_prediction_scatter.png','E2 predicted-versus-observed scatter.')]])
 report=f'''# Unified-cohort analytical estimator evaluation

Status: **COMPLETED AND INDEPENDENTLY REPLAYED** — 2026-11-01.

This post-freeze analysis adds analytical routes to the current unified QPU cohort. It preserves the saved targets, groups, E1/E2/E3 splits, and all previously frozen neural, Qonductor, and median predictions. It does not modify earlier benchmark results.

## Verified cohort and split facts

- 3,443 target observations and 203 structural groups.
- Ma–Li: 340 observations / 148 groups; Qonductor: 3,058 / 49; QPack: 45 workflow means / 6.
- QPack’s 45 rows are workflow-level means built using one predeclared representative feature input per workflow. The 3,854 underlying execution timings are not treated as 3,854 independent target rows.
- E1 is the frozen five-fold group-disjoint evaluation. E2 is the frozen five-fold, source-stratified random-row evaluation; its test groups may occur in training. E3 contains ten held-backend cases, shown separately below.
- The group-balanced MAE gives each structural group equal weight. Pooled MAE is included as a row-weighted secondary measure. All targets are read from the saved unified `targets.csv` in seconds.

Source inventory is in [`source_inventory.csv`](source_inventory.csv); full row-level results, fold slices and provenance slices are in [`metrics.csv`](metrics.csv), [`fold_metrics.csv`](fold_metrics.csv), and [`sliced_metrics.csv`](sliced_metrics.csv).

## Methods included

- **Qiskit Target path:** nominal Target-duration weighted critical path for one execution, including supported terminal measurements. The new full-cohort values are computed on saved native DAGs using the pinned Target recurrence. Exact comparison with the saved Qiskit `estimate_duration` output is available for 615 shared rows (340 Ma–Li, 230 Qonductor, 45 QPack), with zero numerical difference. The other 2,828 rows use the same frozen DAG/Target recurrence but have no row in the older Qiskit result cache.
- **QCRE unitary path:** pinned QCRE maximum-per-wire duration recurrence, adapted to the frozen Target duration map; terminal measurements are excluded. This is a nominal proxy, not historical gate calibration.
- **HyperQ paper-style weighted path:** duration-weighted dependency path including supported terminal measurements; this is a reimplementation, not the official HyperQ scheduler. On these inputs its value equals the Qiskit Target path exactly; those identical outputs must not be described as independent predictive wins.
- **NoTaDS:** both paper weighted-level score and public-code filtered-depth score. Raw and shot-scaled scores are unitless and are reported with correlations only. A score becomes a seconds predictor only after the separately named outer-train affine fit.
- **IBM usage formulas:** quick and duration-based forms appear only as service-usage boundary diagnostics. Their errors against heterogeneous source labels do not establish accuracy for actual IBM usage.
- **Scholten original estimator:** unavailable. The unified archive does not establish compatible row-aligned effective depth and CLOPS-V inputs. The older nominal CLOPS-V-like adaptation is not the original method; see [`unavailable_methods.csv`](unavailable_methods.csv) and the historical audit cited there.

Every seconds-valued raw route also has an `× shots` variant. Each route and each shot variant has an outer-train-only affine adaptation, `max(0, a + b x)`. There are 200 saved fits, each using only its applicable outer-training IDs. These are learned calibrations, not raw analytical predictions. Their coefficients and training-ID hashes are in [`calibration_receipts.csv`](calibration_receipts.csv).

## Group-balanced results on the full E1 and E2 test populations

The table shows raw-seconds R² as a secondary measure; no unlabeled R² is used. `Qonductor (R0)` is the existing Qonductor replication/adaptation: its selection procedure chooses one model family per outer case using inner-fold validation, then refits that model. It is not a per-row best-of-model merge. The outer-training median is shown as C0.

{main_table}

The raw single-execution paths have MAE above five seconds because their nominal schedule values are far below the observed unified labels. Multiplying by shots reduces raw error but does not make those quantities match the target by itself. The affine calibration learns a label-dependent scale; the shots-scaled affine variants improve further, but their E1/E2 group-balanced MAE remains above the saved Qonductor selector.

### Paired group-bootstrap comparisons against Qonductor

Each Δ is analytical method minus Qonductor GB-MAE; positive means higher error for the analytical route. The intervals use paired resampling of complete structural groups with the frozen 10,000-draw, seed-42 arrays. The bootstrap improvement fraction is descriptive, not a p-value. These comparisons are exploratory across a method family.

{contrast_table}

All four selected calibrated routes have positive intervals in E1 and E2, so the observed differences favor Qonductor on these two full-population splits. This does not establish a universal ranking across all backends, targets, or future workloads.

## NoTaDS unitless scores

{score_table}

The correlations increase after multiplying by shots, especially for the public-code filtered-depth score. This is an association in this merged cohort and can reflect source/shot composition as well as circuit-work scaling. It is not a seconds error metric or evidence that shots cause the observed runtime change.

## Backend-context holdouts (E3)

Each backend is held out and scored separately. Test population sizes differ by backend; do not average R² across these cases or treat the table as a universal backend leaderboard.

{e3_table}

The calibrated analytical routes are usually worse than Qonductor in these held-backend cases, with a few context-specific exceptions (for example, Perth). This is descriptive heterogeneity, not a confirmatory backend-generalization result. See the complete E3 rows and paired intervals in the CSV artifacts.

## Source behavior under E1

{source_table}

The group counts differ sharply by source. The 45-row QPack contribution has only six groups; its source-specific scores are therefore descriptive and relatively uncertain. The complete source, backend and structure/timing-provenance slices are in [`sliced_metrics.csv`](sliced_metrics.csv).

## Separate service-usage diagnostics

| Formula | E1 pooled MAE vs saved labels (s) | E1 GB-MAE (s) | E1 bias (s) | Status |
|---|---:|---:|---:|---|
| IBM quick: `2 + 0.00035 × shots` | {f(get('E1','ALL','IBM_QUICK_USAGE_DIAGNOSTIC')['pooled_mae_seconds'])} | {f(get('E1','ALL','IBM_QUICK_USAGE_DIAGNOSTIC')['group_balanced_mae_seconds'])} | {f(get('E1','ALL','IBM_QUICK_USAGE_DIAGNOSTIC')['bias_seconds'])} | Boundary diagnostic only |
| IBM duration: `2 + shots × (Target path + 0.00025 s)` | {f(get('E1','ALL','IBM_USAGE_DURATION_DIAGNOSTIC')['pooled_mae_seconds'])} | {f(get('E1','ALL','IBM_USAGE_DURATION_DIAGNOSTIC')['group_balanced_mae_seconds'])} | {f(get('E1','ALL','IBM_USAGE_DURATION_DIAGNOSTIC')['bias_seconds'])} | Boundary diagnostic only; see amendment |

These formulas estimate an IBM service-usage quantity under assumptions about circuits/sub-jobs. The unified target combines source-specific clocks. A low error against this label mix is not proof that the formulas recovered actual provider usage, and these rows are excluded from the matched estimator comparisons. The duration-formula addition is disclosed in [`PROTOCOL_AMENDMENT_01.md`](PROTOCOL_AMENDMENT_01.md): it was listed in the implementation plan but omitted from the initial locked protocol, so it receives diagnostic-only treatment.

## Interpretation and limits

**Verified facts:** all 3,443 cohort identities have finite raw analytical inputs, saved predictions, and coverage in E1/E2/E3; the 1,438 unique graph×snapshot tasks passed hash checks; all 200 calibration fits replay from outer-training data; the saved baseline predictions are byte-for-byte unchanged; independent metric and bootstrap replay passed.

**Interpretation:** raw schedule/path values are not interchangeable with the observed target. Shot multiplication moves the scale closer but does not by itself solve the prediction problem. Outer-train affine calibration improves the analytical inputs substantially, which means those rows are calibrated predictors rather than raw physical estimates. The tested analytical variants do not beat the Qonductor selector on E1 or E2 by group-balanced MAE. The current evidence does not identify an overhead decomposition or prove which source-specific runtime component causes the residual errors.

**Limits:** nominal fake-backend Target snapshots are proxies, not execution-day calibration snapshots. QPU target boundaries differ by source. The Qiskit old-cache parity population is 615 rows; the additional 2,828 use the saved native DAG recurrence under the same pinned Targets. QPack’s six groups make group-level uncertainty coarse. E2 allows structural-group overlap by design. E3 cases have unequal sizes. No latency/cost benchmark for the analytical extraction routes was performed in this accuracy study.

## Figures

{figs}

## Validation and reproduction

- [`validation_receipt.json`](validation_receipt.json): independent replay passed for 2,400 metric rows, 576 paired contrasts, 460 correlations, and all 200 train-only calibration fits.
- [`qiskit_parity_receipt.json`](qiskit_parity_receipt.json): exact parity on 615 overlapping graph/snapshot rows.
- [`calibration_receipt.json`](calibration_receipt.json): 200 train-only fits, 227,238 saved prediction rows after inclusion of two IBM diagnostics.
- [`evaluation_receipt.json`](evaluation_receipt.json): 10,000 group-bootstrap draws per contrast; no QPU/simulator jobs or model retraining.

Commands used:

```bash
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/extract_raw.py --check-only
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/extract_raw.py --canary
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/extract_raw.py --full
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/calibrate.py
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/evaluate.py
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/validate_qiskit_parity.py
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/validate.py
/home/server/Documents/.venv-qonductor/bin/python research/postfreeze_unified_analytical_completion_v1/make_figures.py
```

`extract_raw.py --full` intentionally refuses to overwrite an existing full extraction. Reproduction should use a clean, separately named namespace after pinning the same hashes.
'''
 (NS/'REPORT.md').write_text(report)
 githead=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
 manifest={'status':'FINAL_ANALYTICAL_RESULTS_VALIDATED','created_utc':datetime.now(timezone.utc).isoformat(),'git_head':githead,'branch':branch,
  'cohort_rows':len(panel),'group_count':len({r['group_id'] for r in panel}),'source_counts':{k:len(v) for k,v in bysource.items()},
  'protocol_sha256':sha(NS/'PROTOCOL.md'),'protocol_amendment_sha256':sha(NS/'PROTOCOL_AMENDMENT_01.md'),'method_registry_sha256':sha(NS/'method_registry.csv'),
  'cohort_panel_sha256':sha(ST/'cohort/panel.csv'),'target_sha256':sha(ST/'cohort/targets.csv'),
  'split_sha256':{e:sha(ST/f'splits/{e}_outer.csv') for e in ('E1','E2','E3')},
  'predictions_sha256':sha(NS/'predictions.csv'),'metrics_sha256':sha(NS/'metrics.csv'),'paired_comparisons_sha256':sha(NS/'paired_comparisons.csv'),
  'validation_receipt_sha256':sha(NS/'validation_receipt.json'),'calibration_receipt_sha256':sha(NS/'calibration_receipt.json'),
  'qiskit_parity_receipt_sha256':sha(NS/'qiskit_parity_receipt.json'),'python':platform.python_version(),
  'qiskit':md.version('qiskit'),'qiskit_ibm_runtime':md.version('qiskit-ibm-runtime'),'numpy':md.version('numpy'),
  'commands':['extract_raw.py --check-only','extract_raw.py --canary','extract_raw.py --full','calibrate.py','evaluate.py','validate_qiskit_parity.py','validate.py','make_figures.py'],
  'no_model_retraining':True,'no_qpu_jobs':True,'no_simulator_jobs':True,'no_commit_or_push':True}
 (NS/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 files={str(p.relative_to(NS)):sha(p) for p in sorted(NS.rglob('*')) if p.is_file() and p.name not in ('checksums.json','manifest.json')}
 (NS/'checksums.json').write_text(json.dumps(files,indent=2)+'\n')
 print(json.dumps({'status':'REPORT_AND_MANIFEST_WRITTEN','report_sha256':sha(NS/'REPORT.md'),'manifest':manifest,'checksummed_files':len(files)},indent=2))
if __name__=='__main__':main()
