"""Replay and time saved analytical scalar mappings on all4515 rows; no refit."""
import sys,time,json,math,csv,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/postfreeze_qpu_workflow_mean_cost_v1/completion_v2'
sys.path.insert(0,str(Path(__file__).parent));from prepare import read,sha

def main():
 folder=ROOT/'artifacts/real_qpu/qualified_logical_panel'
 refs=read(folder/'summary/oof_predictions.csv',['canonical_observation_id','outer_fold','method_id','predicted_seconds','status'])
 pred={(r['canonical_observation_id'],r['method_id']):float(r['predicted_seconds']) for r in refs if r['status']=='predicted' and (r['method_id'].startswith('qcre_') or r['method_id'].startswith('qiskit_'))}
 fits={(r['method_id'],int(r['outer_fold']),r['variant']):(float(r['slope']),float(r['intercept'])) for r in read(folder/'analytical/calibration_fits.csv') if r['variant'] in ('affine','log_affine')}
 features=read(ROOT/'artifacts/postfreeze_notads_hyperq_track_a_v1/features.csv');output=[]
 for r in features:
  cid=r['canonical_observation_id'];fold=int(r['outer_fold']);shots=float(r['shots'])
  for base in ('qcre_source_estimator','qiskit_estimate_duration_snapshot'):
   raw=pred[(cid,base+'__raw')]
   for scaled in (False,True):
    route=base if not scaled else 'qcre_source_estimator_shot_scaled' if base.startswith('qcre') else 'qiskit_shot_scaled_schedule_seconds'
    for variant in ('raw','affine','log_affine'):
     name=route+'__'+variant;coef=fits.get((route,fold,variant))
     def fn():
      v=raw*shots if scaled else raw
      if variant=='affine':v=max(0.,coef[0]*v+coef[1])
      elif variant=='log_affine':v=max(0.,math.expm1(coef[0]*math.log1p(v)+coef[1]))
      return v
     value=fn();saved=pred[(cid,name)]
     if not math.isfinite(value) or abs(value-saved)>1e-9+1e-8*abs(saved):raise RuntimeError('saved scalar replay mismatch:'+cid+':'+name)
     samples=[]
     for _ in range(30):
      t=time.perf_counter_ns();value=fn();samples.append(time.perf_counter_ns()-t)
     output.append({'id':cid,'method':name,'fold':fold,'prediction':value,'saved_prediction':saved,'median_transform_ns':statistics.median(samples),'boundary':'resident raw estimate scalar -> optional shots multiplication -> saved calibration; excludes estimator/parse/compile','status':'PASS'})
 assert len(output)==54180
 path=OUT/'scalar_transform_timings.csv'
 if path.exists():raise RuntimeError('refuse overwrite scalar timings')
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 (OUT/'scalar_receipt.json').write_text(json.dumps({'status':'PASS_ALL_54180_SAVED_ANALYTICAL_VARIANTS','rows':len(output),'cohort_rows':4515,'repetitions':30,'no_refit':True,'not_full_pipeline_latency':True,'output_sha256':sha(path)},indent=2))
if __name__=='__main__':main()
