"""Independent all-alias feature parity, then unchanged GPU model replay."""

import os as _b26_os
from pathlib import Path
import os,sys,json,time
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
ROOT=Path(__file__).resolve().parents[2];OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/parallel8_v2'
sys.path.insert(0,str(ROOT/'research/postfreeze_qpu_workflow_mean_cost_v1'))
from common import atomic_json,sha
from loaders import StudyInputs
import numpy as np
import torch

def load(task):
 path=OUT/'features'/(task['key']+'.npz');r=json.loads((OUT/'records'/(task['key']+'.json')).read_text())
 if sha(path)!=r['feature_file_sha256']:raise RuntimeError('feature output changed')
 with np.load(path,allow_pickle=False) as a:
  d={k:a[k] for k in a.files}
 g={k[7:]:v for k,v in d.items() if k.startswith('graph__')}
 result={k:v for k,v in d.items() if not k.startswith('graph__')}
 if g:result['graph_record']=g
 return result

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 manifest=json.loads((OUT/'manifest.json').read_text());ctx=StudyInputs('E1');aliases={};checked=0
 for t in manifest['tasks']:
  if not (OUT/'records'/(t['key']+'.json')).exists():raise RuntimeError('missing task:'+t['key'])
  actual=load(t)
  for cid in t['aliases']:
   expected=ctx.raw_request(cid,t['method'])
   names=['native5'] if t['method']=='R0' else ['globals']
   for name in names:
    if actual[name].shape!=expected[name].shape or not np.allclose(actual[name],expected[name],rtol=1e-6,atol=1e-6):raise RuntimeError('input feature mismatch:'+cid+':'+name)
   if t['method']=='G1':
    if set(actual['graph_record'])!=set(expected['graph_record']):raise RuntimeError('graph key mismatch:'+cid)
    for k,v in expected['graph_record'].items():
     if actual['graph_record'][k].shape!=v.shape or not np.allclose(actual['graph_record'][k],v,rtol=1e-6,atol=1e-6):raise RuntimeError('graph mismatch:'+cid+':'+k)
   aliases[(cid,t['method'])]=t;checked+=1
 if checked!=6886:raise RuntimeError('alias count mismatch')
 atomic_json(OUT/'feature_parity_receipt.json',{'status':'PASS_ALL_6886_ALIASES','observations':3443,'alias_checks':checked,'unique_tasks':len(manifest['tasks'])})
 count=0;start=time.time()
 for ev in ('E1','E2','E3'):
  inputs=StudyInputs(ev)
  for case,ids in inputs.test_by_case.items():
   inputs.load_case(case)
   for cid in ids:
    logical=load(aliases[(cid,'G1')]);native=load(aliases[(cid,'R0')])
    for method in ('M0','M1','G0','G1','R0','C0'):
     if method in ('R0','C0'):request=native if method=='R0' else {}
     else:
      request={'globals':logical['globals']}
      if method.startswith('G'):request['graph_record']=logical['graph_record']
      if method.endswith('1'):request['shots']=float(logical['shots'])
     prepared=inputs.prepare_raw_B(method,request);pred,seeds,_=inputs.predict_ready(method,cid,prepared)
     inputs.check(method,cid,pred,seeds);count+=1
   atomic_json(OUT/'gpu_replay_progress.json',{'status':'RUNNING','evaluation':ev,'case':case,'verified_predictions':count,'elapsed_seconds':time.time()-start})
 if count!=61974:raise RuntimeError('prediction count mismatch')
 atomic_json(OUT/'prediction_parity_receipt.json',{'status':'PASS_ALL_61974_PREDICTIONS','observations_per_evaluation':3443,'evaluations':3,'methods':6,'model_seeds':[42,1234,31415],'unchanged_A_B_tolerances':True,'verified_predictions':count,'no_training':True})
if __name__=='__main__':main()
