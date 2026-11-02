"""Portable stdlib review: checks published aggregates, not private raw timings."""
import argparse,csv,json,hashlib,math
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=parser.parse_args();root=a.root
 index_path=root/'provenance/cost_source_index.json'
 if not index_path.is_file():
  parser.error('Cost verification requires the external study bundle: provenance/cost_source_index.json and its indexed tables. See docs/DATA_SOURCES.md.')
 index=json.loads(index_path.read_text());checks=0
 for r in index['files']:
  p=root/r['destination'];assert p.is_file() and not p.is_symlink(),p
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],p;checks+=1
 folder=root/'artifacts/estimator_cost'
 def read(n):
  with (folder/n).open(newline='') as f:return list(csv.DictReader(f))
 points=read('warm_accuracy_cost.csv');lat=read('latency_reference.csv');acc=read('accuracy_reference.csv')
 assert len(points)==12 and len({(r['evaluation'],r['method']) for r in points})==12
 for r in points:
  l=next(x for x in lat if x['evaluation']==r['evaluation'] and x['method']==r['method'] and x['clock']=='B' and x['scope']=='ALL')
  m=next(x for x in acc if x['evaluation']==r['evaluation'] and x['method']==r['method'] and x['scope']=='all')
  assert l['rows']=='3443' and l['groups']=='203'
  assert math.isclose(float(l['group_balanced_mean_latency_ms']),float(r['group_balanced_mean_latency_ms']),rel_tol=1e-12)
  assert float(m['group_balanced_mae_seconds'])==float(r['saved_group_balanced_mae_seconds'])
  assert float(m['raw_seconds_r2'])==float(r['saved_raw_seconds_r2']);checks+=4
 lookup={(r['evaluation'],r['method']):float(r['group_balanced_mean_latency_ms']) for r in points}
 for r in read('paired_latency_reference.csv'):
  if r['clock']!='B' or r['evaluation'] not in ('E1','E2'):continue
  left,right=r['contrast'].split('-');delta=lookup[(r['evaluation'],left)]-lookup[(r['evaluation'],right)]
  assert math.isclose(delta,float(r['delta_group_balanced_ms']),abs_tol=1e-10)
  assert float(r['ci_low_adjusted_ms'])<=float(r['ci_high_adjusted_ms']);checks+=2
 stages=read('source_stage_cost.csv');assert len(stages)==35
 assert all(r['sample_observations']=='4' and r['sessions']=='2' for r in stages)
 for r in stages:assert 0<float(r['min_observation_ms'])<=float(r['median_ms'])<=float(r['max_observation_ms']);checks+=1
 cover=read('route_coverage.csv');assert sum(int(r['observed_records']) for r in cover)==288
 assert sum(int(r['passed']) for r in cover)==280 and sum(int(r['unavailable']) for r in cover)==8
 assert all(int(r['failed'])==0 for r in cover)
 cold=read('cold_summary.csv');assert len(cold)==6 and sum(int(r['launches']) for r in cold)==600
 reconstruction=json.loads((folder/'reconstruction_receipt.json').read_text());assert reconstruction['all_predictions']==61974 and reconstruction['all_aliases']==6886
 print(json.dumps({'status':'PASS_PORTABLE_AGGREGATE_REVIEW','checks':checks,'warm_points':12,'diagnostic_observation_session_records':288,'cold_launches':600,'raw_QPU_rows_or_checkpoints_loaded':False,'scope':'Hash integrity, saved accuracy/latency alignment, paired-delta arithmetic and aggregate coverage only; no raw timing bootstrap, source reconstruction or retraining replay.'},indent=2))
if __name__=='__main__':main()
