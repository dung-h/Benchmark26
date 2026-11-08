#!/usr/bin/env python3
"""Train structural compiled-feature MLP, with E1/E2 and out-of-fold inference.

This is the structural-only planned arm. Calibration errors are intentionally
not claimed as present. Saved results are a separate run, not Azizov parity.
"""
from pathlib import Path
import os,sys,csv,json,time,hashlib,random
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments/qpu')]
import numpy as np,torch
from torch import nn
from benchmark26 import mali_training as base
from benchmark26.gpu_lease import acquire_gpu_lease
from core import load_context,parts_for,load_targets
O=Path(os.environ['B26_RUN_DIR']);P=O/'physical_inputs';torch.set_num_threads(4);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)
def main():
 rec=json.loads((O/'physical_preparation_receipt.json').read_text());assert rec['status']=='PASS' and rec['observations']==3443
 mp=json.loads((O/'physical_mapping.json').read_text());assert len(mp)==3443
 draft={r['canonical_observation_id']:r for r in csv.DictReader((O/'structural_features_draft.csv').open())}
 names=['count_rz','count_sx','count_x','count_ecr','count_cz','count_measure','count_reset','count_delay','count_barrier','count_other','count_total','count_two_qubit','active_qubits','depth_1q','depth_2q','unitary_depth','hyperq_full_seconds','hyperq_unitary_seconds']
 physical={}
 for key in set(mp.values()):
  with np.load(P/f'{key}.npz',allow_pickle=False) as z:physical[key]=z['compiled_global'].astype(np.float64)
 assert all(len(x)==51 and np.isfinite(x).all() for x in physical.values())
 lock={'status':'LOCKED_BEFORE_FIT','scope':'TM structural-only MLP; logical 51 + compiled 51 including SupermarQ + 18 log1p compiled descriptors + shots; excludes snapshot error features; not exact Azizov','arms':['TM_structural'],'evaluations':['E1','E2'],'folds':5,'seeds':[42,1234,31415],'batch':32,'max_epochs':500,'patience':50,'learning_rate':0.0005,'weight_decay':0.0001,'runner_sha256':sha(Path(__file__)),'preparation_receipt_sha256':sha(O/'physical_preparation_receipt.json'),'mapping_sha256':sha(O/'physical_mapping.json'),'draft_sha256':sha(O/'structural_features_draft.csv'),'run_manifest_sha256':sha(O/'run_manifest.json'),'descriptor_names':names,'fit_statistics':'training inner-fit only','no_test_labels_during_fit':True}
 lp=O/'tm_structural_lock.json'
 if lp.exists():assert json.loads(lp.read_text())==lock
 else:atomic(lp,lock)
 start=time.monotonic(); lease=acquire_gpu_lease('postfreeze-qpu-workflow-mean-expansion-v1')
 try:
  base.validate_gpu_budget(torch,budget_gib=12,reserve_gib=4)
  for ev in ('E1','E2'):
   context,_,_=load_context(ev,include_targets=False)
   for fold in range(5):
    parts=parts_for(context,fold);targets=load_targets(parts['train']);ids=sorted(context['all_ids']);idx={cid:i for i,cid in enumerate(ids)}
    def row(cid):
     logical=[float(context['ledger'][cid][f'global_{j:02d}']) for j in range(51)];extra=[np.log1p(float(draft[cid][k])) for k in names];return logical+physical[mp[cid]].tolist()+extra+[np.log1p(float(context['runs_by_id'][cid]['shots']))]
    X=np.asarray([row(cid) for cid in ids],dtype=np.float64);assert np.isfinite(X).all();fi=np.array([idx[c] for c in parts['fit']]);means=X[fi].mean(0);std=X[fi].std(0,ddof=1);mask=std>1e-6;X[:,mask]=(X[:,mask]-means[mask])/std[mask];X[:,~mask]=0;X=X.astype(np.float32)
    for seed in (42,1234,31415):
     dest=O/'training'/ev/f'fold_{fold}'/'TM_structural'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
     if (dest/'complete.json').exists():continue
     base.seed_all(seed); model_class,_=base.load_pinned_model();model=base.make_model(model_class,'mali_full_features_mlp',X.shape[1]).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0001);best=float('inf');best_epoch=-1;best_state=None;history=[]
     from torch_geometric.data import Data
     def model_input(xx):return Data(global_features=xx,x=torch.empty((0,178),device=xx.device),edge_index=torch.empty((2,0),dtype=torch.long,device=xx.device))
     for epoch in range(500):
      model.train();order=np.random.default_rng(int(base.stable_hash([fold,seed,epoch])[:16],16)).permutation(len(parts['fit']));trainloss=0.
      for j in range(0,len(order),32):
       cs=[parts['fit'][int(k)] for k in order[j:j+32]];xx=torch.from_numpy(X[[idx[c] for c in cs]]).cuda();yy=torch.tensor([targets[c] for c in cs],dtype=torch.float32,device='cuda');optimizer.zero_grad(set_to_none=True);pred=model(model_input(xx)).reshape(-1);loss=nn.functional.mse_loss(pred,yy);loss.backward();optimizer.step();trainloss+=float(loss.detach())*len(cs)
      model.eval();cs=parts['validation'];vv=0.
      with torch.no_grad():
       for j in range(0,len(cs),128):
        b=cs[j:j+128];pred=model(model_input(torch.from_numpy(X[[idx[c] for c in b]]).cuda())).reshape(-1);y=torch.tensor([targets[c] for c in b],dtype=torch.float32,device='cuda');vv+=float(((pred-y)**2).sum())
      val=vv/len(cs);assert np.isfinite(val)
      if val<best:best=val;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
      history.append({'epoch':epoch+1,'validation_mse':val,'best_epoch':best_epoch+1,'train_mse':trainloss/len(parts['fit'])})
      if (epoch+1)%10==0:print(json.dumps({'evaluation':ev,'fold':fold,'seed':seed,'epoch':epoch+1,'best_epoch':best_epoch+1,'validation_mse':val}),flush=True)
      if epoch-best_epoch>=50:break
      if time.monotonic()-start>12*3600:raise RuntimeError('12_hour_gpu_budget_exceeded')
     model.load_state_dict(best_state);model.eval();predictions=[]
     with torch.no_grad():
      cs=parts['test']
      for j in range(0,len(cs),128):
       b=cs[j:j+128];p=model(model_input(torch.from_numpy(X[[idx[c] for c in b]]).cuda())).reshape(-1).cpu().numpy()
       predictions.extend({'canonical_observation_id':c,'outer_fold':fold,'seed':seed,'predicted_seconds':float(y)} for c,y in zip(b,p))
     with (dest/'predictions.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=predictions[0].keys());w.writeheader();w.writerows(predictions)
     torch.save({'state_dict':best_state,'mean':means,'std':std,'mask':mask,'feature_names':names,'fit_ids_sha256':base.stable_hash(parts['fit'])},dest/'best.pt');atomic(dest/'history.json',history);atomic(dest/'complete.json',{'status':'COMPLETE','epochs':len(history),'best_epoch':best_epoch+1,'lock_sha256':sha(lp),'fit':len(parts['fit']),'validation':len(parts['validation']),'test':len(parts['test'])});print('CELL_COMPLETE',ev,fold,seed,flush=True)
 finally:lease.release()
 atomic(O/'tm_structural_complete.json',{'status':'COMPLETE','cells':30,'gpu_seconds':time.monotonic()-start})
if __name__=='__main__':main()
