"""Make aggregate cost evidence; never alter accuracy/prediction artifacts."""
import csv,json,hashlib,statistics,collections,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/postfreeze_qpu_workflow_mean_cost_v1';OUT=BASE/'synthesis_v1'
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True)
 inputs=[BASE/'v2/summary/accuracy_cost_points.csv',BASE/'v2/summary/latency_summary.csv',BASE/'v2/summary/paired_comparisons.csv',BASE/'v2/summary/saved_accuracy_metrics.csv',BASE/'completion_v2/timing_summary.csv',BASE/'completion_v2/cold_summary.csv',BASE/'completion_v2/coverage.csv',BASE/'completion_v2/validation_receipt.json',BASE/'parallel8_v2/completion_receipt.json']
 points=read(inputs[0]);write('warm_accuracy_cost.csv',[r for r in points if r['clock']=='B' and r['evaluation'] in ('E1','E2')])
 by=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in read(inputs[4]):by[(r['route'],r['method'],r['source'],r['boundary'])][r['id']].append(float(r['median_ms']))
 summary=[]
 for (route,method,source,boundary),ids in sorted(by.items()):
  assert all(len(v)==2 for v in ids.values());v=[statistics.median(x) for x in ids.values()]
  summary.append({'route':route,'method':method,'source':source,'sample_observations':len(v),'sessions':2,'median_ms':statistics.median(v),'min_observation_ms':min(v),'max_observation_ms':max(v),'boundary':boundary})
 write('source_stage_cost.csv',summary)
 for src,name in [(inputs[1],'latency_reference.csv'),(inputs[2],'paired_latency_reference.csv'),(inputs[3],'accuracy_reference.csv'),(inputs[5],'cold_summary.csv'),(inputs[6],'route_coverage.csv'),(inputs[7],'route_validation_receipt.json'),(BASE/'v2/summary/validation_receipt.json','warm_validation_receipt.json'),(inputs[8],'reconstruction_receipt.json')]:shutil.copyfile(src,OUT/name)
 checks={'warm_points':len(read(OUT/'warm_accuracy_cost.csv'))==12,'source_stage_groups':len(summary)==35,'four_observations_per_source_stage':all(r['sample_observations']==4 for r in summary),'all_12_methods_routes_have_recorded_coverage':len(read(OUT/'route_coverage.csv'))==12,'cold_launches_600':sum(int(r['launches']) for r in read(OUT/'cold_summary.csv'))==600}
 assert all(checks.values()),checks
 receipt={'status':'PASS_AGGREGATE_SYNTHESIS','checks':checks,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'script_sha256':sha(__file__),'accuracy_recomputed':False,'C_and_analytical_sample_not_full_population':True}
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(checks))
if __name__=='__main__':main()
