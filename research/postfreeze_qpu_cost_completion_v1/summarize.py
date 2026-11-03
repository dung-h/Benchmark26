"""Validate record completeness and report scopes without promoting partial routes."""

import os as _b26_os
from pathlib import Path
import json,csv,statistics,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/completion_v2'
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 manifest=json.loads((OUT/'manifest.json').read_text())
 for p,h in manifest['pins'].items():assert sha(p)==h,('pin changed',p)
 records=[]
 for p in (OUT/'measure').rglob('*.json'):
  r=json.loads(p.read_text());assert r['manifest_fingerprint']==manifest['fingerprint']
  if r['status']=='PASS':assert len(r['durations_ns'])==3 and all(x>0 for x in r['durations_ns'])
  records.append(r)
 coverage=[]
 for kind,methods,n in [('C',('M0','M1','G0','G1','R0','C0'),len(manifest['C_requests'])),('analytical',('qiskit','qcre','notads_code','notads_paper','hyperq'),len(manifest['analytical_requests'])),('compile',('nominal',),len(manifest['C_requests']))]:
  for method in methods:
   rs=[r for r in records if (r.get('kind','C'),r['method'])==(kind,method)]
   passed=[r for r in rs if r['status']=='PASS']
   coverage.append({'route':kind,'method':method,'assigned_request_sessions':2*n,'observed_records':len(rs),'passed':len(passed),'failed':sum(r['status']=='FAILED_ROUTE' for r in rs),'unavailable':sum(r['status'].startswith('UNAVAILABLE') for r in rs),'status':'COMPLETE_MEASURED' if len(passed)==2*n else 'PARTIAL_OR_BLOCKED'})
 with (OUT/'coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
 timings=[]
 for r in records:
  if r['status']!='PASS':continue
  timings.append({'route':r.get('kind','C'),'method':r['method'],'id':r['id'],'source':r['source'],'session':r['session'],'median_ms':statistics.median(r['durations_ns'])/1e6,'min_ms':min(r['durations_ns'])/1e6,'max_ms':max(r['durations_ns'])/1e6,'boundary':r['boundary']})
 with (OUT/'timing_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(timings[0]));w.writeheader();w.writerows(timings)
 coldroot=OUT.parent/'full_cost_v1';cold=[json.loads(p.read_text()) for p in (coldroot/'cold_records').glob('*.json')];assert len(cold)==600 and all(r['status']=='PASS' for r in cold)
 coldrows=[]
 for method in ('M0','M1','G0','G1','R0','C0'):
  rs=[r for r in cold if r['method']==method]
  coldrows.append({'method':method,'launches':len(rs),'median_process_seconds':statistics.median(r['process_wall_ns'] for r in rs)/1e9,'median_state_load_seconds':statistics.median(r['state_load_ns'] for r in rs)/1e9,'boundary':'process-cold: interpreter/imports+input/model load+first raw-feature request; not disk-cold'})
 with (OUT/'cold_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(coldrows[0]));w.writeheader();w.writerows(coldrows)
 blocked=[r for r in records if r['status']!='PASS']
 (OUT/'route_failures.json').write_text(json.dumps(blocked,indent=2))
 lines=['# Remaining inference-cost measurements','','## Measured scope','','A/B are the independently validated full 3443-row warm-input benchmark. D covers 600 process-cold launches. The parallel8 study verified 2531 unique feature tasks,6886 aliases and 61974 predictions; concurrent extraction times are not isolated latency.','','This supplement measures the declared 12-observation size-stratified C diagnostic (E1 fold-correct states), a separate 12-observation analytical diagnostic from 4515, and nominal compilation reproducibility. It does not replace full-population accuracy or broaden the inference population. Input stages are explicit per row; neural logical-start and classical compiled/submitted-start durations must not be directly treated as identical end-to-end service cost.','','## Coverage','','| Route | Method | Assigned sessions | PASS | Failed | Unavailable | Status |','|---|---|---:|---:|---:|---:|---|']
 for r in coverage:lines.append('| '+ ' | '.join(str(r[k]) for k in ('route','method','assigned_request_sessions','passed','failed','unavailable','status'))+' |')
 lines+=['','## Timing artifacts','','`timing_summary.csv` retains per-observation/session medians and min/max, all measured in milliseconds; `cold_summary.csv` records process-cold seconds. `scalar_transform_timings.csv` measures only saved analytical scalar transforms; no parse/estimator cost is inferred from these numbers. Complete raw records remain in `measure/`. Compilation compares original nominal snapshot/seed, IR digest and native structural descriptors. Qonductor historical compiler policy is unavailable; a recipe reconstruction is not sufficient evidence of that policy.','','## Limits and integrity','','No new accuracy metric, fitting, hyperparameter search, QPU or simulator execution. No different runtime boundaries are collapsed into a universal leaderboard. Failed/missing routes remain in coverage and are listed in `route_failures.json`. Nominal compilation is a proxy representation pipeline, not historical execution-day compilation. Diagnostic sample quantiles were fixed before new timings; no significance claim or whole3443 latency distribution is inferred from12 rows.','','## Status','','All complete routes can be reported at their stated boundary. Partial or blocked routes require explicit qualification; this report does not turn an unavailable historical compiler into a measured value.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 status='PASS_MEASURED_WITH_EXPLICIT_UNAVAILABLE_ROUTES' if not any(r['failed'] or (r['observed_records']<r['assigned_request_sessions'] and r['route']!='compile') for r in coverage) else 'PARTIAL_WITH_EXPLICIT_BLOCKERS'
 (OUT/'validation_receipt.json').write_text(json.dumps({'status':status,'manifest_fingerprint':manifest['fingerprint'],'record_count':len(records),'coverage':coverage,'cold_records':600,'all_record_pins_checked':True,'no_scientific_results_changed':True},indent=2))
if __name__=='__main__':main()
