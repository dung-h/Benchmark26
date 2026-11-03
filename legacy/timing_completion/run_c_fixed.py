"""Measure warm circuit-to-prediction latency with corrected route handling on the fixed diagnostic subset."""

import os as _b26_os
from pathlib import Path
import os,sys,json,time,signal
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
ROOT=Path(__file__).resolve().parents[2];OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/completion_v2'
sys.path[:0]=[str(ROOT/'legacy/feature_latency'),str(ROOT/'legacy/circuit_latency'),str(Path(__file__).parent)]
from common import atomic_json
from loaders import StudyInputs
from feature_proxy import Routes
from prepare import sha
import torch,numpy as np

def check_features(actual,expected,method):
 for k in (['native5'] if method=='R0' else [] if method=='C0' else ['globals']):
  if actual[k].shape!=expected[k].shape or not np.allclose(actual[k],expected[k],rtol=1e-6,atol=1e-6):raise RuntimeError('feature mismatch:'+k)
 if method.startswith('G'):
  for k,v in expected['graph_record'].items():
   if actual['graph_record'][k].shape!=v.shape or not np.allclose(actual['graph_record'][k],v,rtol=1e-6,atol=1e-6):raise RuntimeError('graph mismatch:'+k)
def main(session,canary):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 manifest=json.loads((OUT/'manifest.json').read_text())
 amendment=json.loads((OUT/'EXECUTION_AMENDMENT_01.json').read_text())
 for p,h in amendment['code_pins'].items():
  if sha(p)!=h:raise RuntimeError('execution amendment code changed:'+p)
 for p,h in manifest['pins'].items():
  if sha(p)!=h:raise RuntimeError('manifest pin changed:'+p)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800s request deadline')))
 routes=Routes();inputs=StudyInputs('E1');sample=manifest['C_requests']
 if canary:sample=[sample[0]]
 if session==2:sample=list(reversed(sample))
 for row in sample:
  cid=row['canonical_observation_id'];inputs.load_case(row['case'])
  for method in (('M0','M1','G0','G1','R0','C0') if session==1 else ('C0','R0','G1','G0','M1','M0')):
   path=OUT/('canary' if canary else 'measure')/'C'/f'session_{session}'/(str(manifest['C_requests'].index(row))+'_'+method+'.json')
   if path.exists():continue
   result={'id':cid,'method':method,'session':session,'case':row['case'],'source':row['source_id'],'manifest_fingerprint':manifest['fingerprint'],'execution_amendment':'EXECUTION_AMENDMENT_01.json','boundary':'existing-input-stage circuit bytes in RAM through parse/extraction + CPU RPC + saved transforms + 3seed median prediction; excludes compile/disk/load'}
   try:
    route=routes.get(inputs.panel_by_id[cid],inputs.ledger[cid],method);expected=inputs.raw_request(cid,method)
    def fn():
     request={} if method=='C0' else routes.extract(route,method);prepared=inputs.prepare_raw_B(method,request);prediction,seeds,_=inputs.predict_ready(method,cid,prepared);return request,prediction,seeds
    signal.alarm(1800);request,pred,seeds=fn();signal.alarm(0);check_features(request,expected,method);inputs.check(method,cid,pred,seeds)
    times=[]
    for _ in range(1 if canary else 3):
     if method.startswith(('M','G')):torch.cuda.synchronize()
     signal.alarm(1800);start=time.perf_counter_ns();request,pred,seeds=fn()
     if method.startswith(('M','G')):torch.cuda.synchronize()
     elapsed=time.perf_counter_ns()-start;signal.alarm(0);inputs.check(method,cid,pred,seeds);check_features(request,expected,method);times.append(elapsed)
    result.update(status='PASS',durations_ns=times,input_stage=route['stage'],input_sha256=route.get('sha256'),prediction=float(pred),feature_and_prediction_parity=True)
   except Exception as e:
    signal.alarm(0);result.update(status='FAILED_ROUTE',error=repr(e));atomic_json(path,result);raise
   atomic_json(path,result);print(json.dumps({k:result[k] for k in ('id','method','status')}),flush=True)
 routes.close()
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2]=='canary')
