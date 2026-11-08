#!/usr/bin/env python3
"""Descriptive device-ready three-seed prediction latency; no fitting or compilation."""
import os,sys,json,csv,hashlib,time,argparse,shutil,platform,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments/qpu')]
import numpy as np,torch
from torch_geometric.data import Data
from benchmark26 import mali_training as base
from benchmark26.gpu_lease import acquire_gpu_lease
from core import load_context,parts_for
from experiments.compiled_input.train_compiled_dag import prep_fold
from experiments.compiled_input.compact_gpu import CompactCache

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',type=int,required=True);ap.add_argument('--canary',action='store_true');a=ap.parse_args()
 W=Path(os.environ['B26_RUN_DIR']);TM=W.parent/'attempt_02';out=W/'analysis_final/ready_prediction_latency';out.mkdir(exist_ok=True);tag='canary' if a.canary else f'session_{a.session}';raw=out/(tag+'.jsonl');receipt=out/(tag+'_complete.json');assert not raw.exists() and not receipt.exists(),'preserve prior timing attempts'
 assert json.loads((W/'analysis_final/full_checkpoint_replay.json').read_text())['status']=='PASS'
 mp=json.loads((W/'physical_mapping.json').read_text());draft={r['canonical_observation_id']:r for r in csv.DictReader((W/'structural_features_draft.csv').open())};glob={};sizes={}
 for k in set(mp.values()):
  with np.load(W/'physical_inputs'/f'{k}.npz',allow_pickle=False) as z:glob[k]=z['compiled_global'].astype(np.float64);sizes[k]=len(z['node_type'])
 refs={(r['evaluation'],r['arm'],r['canonical_observation_id']):float(r['predicted_seconds']) for r in csv.DictReader((W/'analysis_final/reduced_predictions_private.csv').open()) if r['arm'] in ['TM_structural','TG_structural','TGF_structural']}
 torch.set_num_threads(4);modelclass,_=base.load_pinned_model();lease=acquire_gpu_lease('transpiled-device-ready-prediction-latency');started=time.monotonic();count=0;modelpins={};total_ms=0
 try:
  with raw.open('w') as f,torch.inference_mode():
   for ev in ('E1','E2'):
    ctx=load_context(ev,include_targets=False)[0];selected=set(ctx['all_ids'])
    if a.canary:
     selected=set()
     for s in sorted({r['source_id'] for r in ctx['runs_by_id'].values()}):
      ids=sorted((c for c in ctx['all_ids'] if ctx['runs_by_id'][c]['source_id']==s),key=lambda c:(sizes[mp[c]],c));selected.update(ids[int(i)] for i in np.linspace(0,len(ids)-1,4))
    for fold in range(5):
     parts=parts_for(ctx,fold);test=sorted(set(parts['test'])&selected)
     if not test:continue
     np.random.default_rng([20261108,a.session,fold]).shuffle(test);cache=None
     for arm in ('TM_structural','TG_structural','TGF_structural'):
      ids,idx,X,mu,sd,valid,gm,gs,gvalid=prep_fold(ctx,parts,mp,glob,draft,'TG_structural' if arm=='TG_structural' else 'TGF_structural')
      if arm!='TM_structural' and cache is None:cache=CompactCache(W/'physical_inputs',mp,mu,sd,valid)
      models=[]
      for seed in (42,1234,31415):
       p=(TM if arm=='TM_structural' else W)/'training'/ev/f'fold_{fold}'/arm/f'seed_{seed}'/'best.pt';b=torch.load(p,map_location='cpu',weights_only=False);modelpins[str(p)]=sha(p);m=base.make_model(modelclass,'mali_full_features_mlp' if arm=='TM_structural' else 'mali_full_features_graph',X.shape[1]).cuda();m.load_state_dict(b['state_dict'] if arm=='TM_structural' else b['model']);m.eval()
       if arm!='TM_structural':m.mask=m.mask.cuda()
       models.append(m)
      for c in test:
       if shutil.disk_usage(W).free<8*1024**3:raise RuntimeError('low_disk_preserve_partial_no_delete')
       if arm=='TM_structural':data=Data(global_features=torch.from_numpy(X[[idx[c]]]).cuda(),x=torch.empty((0,178),device='cuda'),edge_index=torch.empty((2,0),dtype=torch.long,device='cuda'))
       else:data=cache.batch([c],X,idx)
       # Input transfer, graph expansion and normalization are outside this clock.
       def predict():return float(np.median([float(m(data).reshape(-1)[0].cpu()) for m in models]))
       reference=refs[ev,arm,c];observed=predict();assert abs(observed-reference)<=1e-4+1e-5*abs(reference)
       for _ in range(3):predict()
       samples=[]
       for _ in range(10):
        torch.cuda.synchronize();t=time.perf_counter_ns();v=predict();torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-t)/1e6;assert abs(v-reference)<=1e-4+1e-5*abs(reference);samples.append(elapsed)
       total_ms+=sum(samples);f.write(json.dumps({'evaluation':ev,'arm':arm,'session':a.session,'canonical_observation_id':c,'group_id':ctx['runs_by_id'][c]['group_id'],'source':ctx['runs_by_id'][c]['source_id'],'fold':fold,'mean_ms':float(np.mean(samples)),'samples_ms':samples})+'\n');f.flush();count+=1
       if count%100==0:print(json.dumps({'records':count,'elapsed_seconds':time.monotonic()-started,'evaluation':ev,'fold':fold,'arm':arm}),flush=True)
       (out/(tag+'_progress.json')).write_text(json.dumps({'status':'RUNNING','records':count,'elapsed_seconds':time.monotonic()-started,'evaluation':ev,'fold':fold,'arm':arm})+'\n')
      del models,m,b,data;torch.cuda.empty_cache()
     if cache is not None:del cache
     torch.cuda.empty_cache()
 finally:lease.release()
 r={'status':'PASS','scope':'canary12 inputs per evaluation,3 methods' if a.canary else 'all3443 inputs x E1/E2 x3 methods','records':count,'timed_ensemble_requests':count*10,'warmups_per_record':3,'timed_repeats_per_record':10,'elapsed_seconds':time.monotonic()-started,'timed_seconds':total_ms/1000,'raw_sha256':sha(raw),'script_sha256':sha(Path(__file__)),'boundary':'normalized inputs already resident on GPU -> three FP32 seed models -> CPU scalar median; excludes parsing, compilation, extraction, normalization, transfer, model loading and cold start','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':platform.python_version(),'hostname':socket.gethostname(),'model_pins':modelpins}
 receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='model_pins'},indent=2),flush=True)
if __name__=='__main__':main()
