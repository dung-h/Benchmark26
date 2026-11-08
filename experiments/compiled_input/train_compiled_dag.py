#!/usr/bin/env python3
"""Full physical-DAG graph training, fit-only scaling and node/edge-bounded batching.

Structural QPU adaptation; not an exact Azizov implementation. Calibration
error/duration node inputs are not yet implemented; preserve this limitation.
"""
import os,sys,json,csv,time,hashlib,collections,random,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments/qpu')]
import numpy as np,torch
from torch_geometric.data import Data,Batch
from benchmark26 import mali_training as base
from benchmark26.mali_features import expand_dense_x
from benchmark26.gpu_lease import acquire_gpu_lease
from core import load_context,parts_for,load_targets
from experiments.compiled_input.dynamic_batch import train_step,partition
from experiments.compiled_input.compact_gpu import CompactCache
O=Path(os.environ['B26_RUN_DIR']);P=O/'physical_inputs';M=O/'physical_moments';MODE=os.environ.get('B26_GRAPH_MODE','canary');torch.set_num_threads(4);torch.set_num_interop_threads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def prep_fold(context,parts,mp,glob,draft,arm):
 names=['count_rz','count_sx','count_x','count_ecr','count_cz','count_measure','count_reset','count_delay','count_barrier','count_other','count_total','count_two_qubit','active_qubits','depth_1q','depth_2q','unitary_depth','hyperq_full_seconds','hyperq_unitary_seconds']
 ids=sorted(context['all_ids']);idx={x:i for i,x in enumerate(ids)}
 def row(c):
  v=[float(context['ledger'][c][f'global_{j:02d}']) for j in range(51)]
  if arm=='TGF_structural':v+=glob[mp[c]].tolist()+[np.log1p(float(draft[c][k])) for k in names]
  return v+[np.log1p(float(context['runs_by_id'][c]['shots']))]
 X=np.asarray([row(c) for c in ids],dtype=np.float64);fi=[idx[c] for c in parts['fit']];mean=X[fi].mean(0);std=X[fi].std(0,ddof=1);mask=std>1e-6;X[:,mask]=(X[:,mask]-mean[mask])/std[mask];X[:,~mask]=0
 cnt=0;s=np.zeros(178);ss=np.zeros(178)
 for c in parts['fit']:
  v=json.loads((M/(mp[c]+'.json')).read_text());cnt+=v['count'];s+=v['sum'];ss+=v['sumsq']
 assert cnt>1;mu=s/cnt; var=np.maximum((ss-cnt*mu*mu)/(cnt-1),0);sd=np.sqrt(var);valid=sd>1e-6
 return ids,idx,X.astype(np.float32),mu.astype(np.float32),sd.astype(np.float32),valid,mean,std,mask
class Cache:
 def __init__(self,mp,mean,std,valid,budget=30*1024**3):self.mp=mp;self.mean=mean;self.std=std;self.valid=valid;self.budget=budget;self.cache=collections.OrderedDict();self.bytes=0;self.safe_std=np.where(valid,std,1).astype(np.float32);self.hits=0;self.misses=0
 def available(self):
  for line in Path('/proc/meminfo').read_text().splitlines():
   if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
  raise RuntimeError('memory_availability_unknown')
 def get(self,c):
  k=self.mp[c]
  if k in self.cache:self.hits+=1;self.cache.move_to_end(k);return self.cache[k][0]
  self.misses+=1
  while self.cache and self.available()<12*1024**3:_,(_,removed)=self.cache.popitem(last=False);self.bytes-=removed
  with np.load(P/(k+'.npz'),allow_pickle=False) as z:r={n:z[n] for n in z.files}
  needed=len(r['node_type'])*178*4+int(r['edge_index'].size)*8
  while self.cache and self.available()<12*1024**3+needed*2:_,(_,removed)=self.cache.popitem(last=False);self.bytes-=removed
  x=expand_dense_x(r);np.subtract(x,self.mean,out=x);np.divide(x,self.safe_std,out=x);x[:,~self.valid]=0
  data=(torch.from_numpy(x),torch.from_numpy(r['edge_index'].astype(np.int64)));size=sum(v.numel()*v.element_size() for v in data)
  if size<=self.budget and self.available()>12*1024**3:
   while self.bytes+size>self.budget or (self.cache and self.available()<12*1024**3):_,(_,removed)=self.cache.popitem(last=False);self.bytes-=removed
   self.cache[k]=(data,size);self.bytes+=size
  return data
def main():
 assert json.loads((O/'physical_preparation_receipt.json').read_text())['status']=='PASS';assert json.loads((O/'physical_moments_complete.json').read_text())['status']=='COMPLETE'
 mp=json.loads((O/'physical_mapping.json').read_text());draft={r['canonical_observation_id']:r for r in csv.DictReader((O/'structural_features_draft.csv').open())};keys=set(mp.values());glob={};sizes={}
 for k in keys:
  with np.load(P/(k+'.npz'),allow_pickle=False) as z:glob[k]=z['compiled_global'].astype(np.float64);sizes[k]=(len(z['node_type']),z['edge_index'].shape[1])
 lock={'scope':'full physical graph with Ma-Li node encoding and logical globals; TGF adds compiled globals and structural descriptors; both include shots; nominal snapshots; no gate-error or duration node features; not exact Azizov','mode':MODE,'evaluations':['E1','E2'],'arms':['TG_structural','TGF_structural'],'folds':5,'seeds':[42,1234,31415],'max_epochs':2 if MODE=='canary' else 500,'patience':50,'effective_batch':32,'node_budget':600000,'edge_budget':800000,'cache_format':'all compact graph arrays in RAM; dense expansion and normalization on GPU','compact_gpu_sha256':sha(ROOT/'experiments/compiled_input/compact_gpu.py'),'runner_sha256':sha(Path(__file__)),'batching_sha256':sha(ROOT/'experiments/compiled_input/dynamic_batch.py'),'preparation_receipt_sha256':sha(O/'physical_preparation_receipt.json'),'moments_receipt_sha256':sha(O/'physical_moments_complete.json'),'mapping_sha256':sha(O/'physical_mapping.json'),'draft_sha256':sha(O/'structural_features_draft.csv'),'train_labels':'outer train only','gpu_time_limit':'none'}
 lp=O/('physical_graph_canary_lock.json' if MODE=='canary' else 'physical_graph_lock.json')
 if lp.exists():assert json.loads(lp.read_text())==lock
 else:atomic(lp,lock)
 lease=acquire_gpu_lease('postfreeze-qpu-workflow-mean-expansion-v1');start=time.monotonic()
 try:
  base.validate_gpu_budget(torch,budget_gib=12,reserve_gib=4)
  for ev in (('E1',) if MODE=='canary' else ('E1','E2')):
   context,_,_=load_context(ev,include_targets=False)
   for fold in ((0,) if MODE=='canary' else range(5)):
    parts=parts_for(context,fold);targets=load_targets(parts['train']);obs_sizes={c:sizes[mp[c]] for c in context['all_ids']}
    for arm in (('TG_structural',) if MODE=='canary' else ('TG_structural','TGF_structural')):
     ids,idx,X,mean,std,valid,gmean,gstd,gmask=prep_fold(context,parts,mp,glob,draft,arm)
     for seed in ((42,) if MODE=='canary' else (42,1234,31415)):
      dest=O/'training'/('canary_physical' if MODE=='canary' else ev)/f'fold_{fold}'/arm/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
      if (dest/'complete.json').exists():continue
      base.seed_all(seed);model_class,_=base.load_pinned_model();model=base.make_model(model_class,'mali_full_features_graph',X.shape[1]).cuda();model.mask=model.mask.cuda();opt=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0001);cache=CompactCache(P,mp,mean,std,valid);best=float('inf');best_epoch=-1;state=None;history=[];start_epoch=0;checkpoint=dest/'checkpoint.pt'
      if not checkpoint.exists() and os.environ.get('B26_IMPORT_CHECKPOINT') and (ev,fold,arm,seed)==('E1',0,'TG_structural',42):
       source=Path(os.environ['B26_IMPORT_CHECKPOINT']);previous=torch.load(source,map_location='cpu',weights_only=False);previous_lock=source.parents[5]/'physical_graph_lock.json';assert previous['identity']['lock_sha256']==sha(previous_lock);assert (previous['identity']['evaluation'],previous['identity']['fold'],previous['identity']['arm'],previous['identity']['seed'])==(ev,fold,arm,seed)
       migration={'source_checkpoint':str(source),'source_checkpoint_sha256':sha(source),'source_lock_sha256':sha(previous_lock),'new_lock_sha256':sha(lp),'next_epoch':previous['next_epoch'],'reason':'compact RAM cache and GPU normalization; exact tensor/edge/batch parity verified; same training settings'};previous['identity']={'lock_sha256':sha(lp),'evaluation':ev,'fold':fold,'arm':arm,'seed':seed};base.save_checkpoint(checkpoint,previous);atomic(dest/'checkpoint_migration.json',migration)
      if checkpoint.exists():
       saved=base.load_saved_checkpoint(checkpoint,{'lock_sha256':sha(lp),'evaluation':ev,'fold':fold,'arm':arm,'seed':seed},torch);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);base.restore_rng_state(saved['rng_state']);best=saved['best'];best_epoch=saved['best_epoch'];state=saved['best_state'];history=saved['history'];start_epoch=saved['next_epoch']
      def batch(cs):return cache.batch(cs,X,idx)
      for epoch in range(start_epoch,lock['max_epochs']):
       tick=time.monotonic();model.train();order=np.random.default_rng(int(base.stable_hash([fold,seed,epoch])[:16],16)).permutation(len(parts['fit']));ordered=[parts['fit'][int(x)] for x in order]
       for j in range(0,len(ordered),32):
        cs=ordered[j:j+32]
        def loss(group):
         pp=model(batch(group)).reshape(-1);yy=torch.tensor([targets[c] for c in group],device='cuda:0');return torch.nn.functional.mse_loss(pp,yy,reduction='sum')
        train_step(model,opt,cs,obs_sizes,600000,800000,loss,torch)
       model.eval();total=0.
       with torch.no_grad():
        for group in partition(parts['validation'],obs_sizes,600000,800000):
         pp=model(batch(group)).reshape(-1);yy=torch.tensor([targets[c] for c in group],device='cuda:0');total+=float(((pp-yy)**2).sum())
       val=total/len(parts['validation']);assert np.isfinite(val)
       if val<best:best=val;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
       h={'epoch':epoch+1,'validation_mse':val,'best_epoch':best_epoch+1,'epoch_seconds':time.monotonic()-tick,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'cache_bytes':cache.bytes,'cache_hits':cache.hits,'cache_misses':cache.misses,'ram_available_bytes':cache.available()};history.append(h);base.save_checkpoint(checkpoint,{'identity':{'lock_sha256':sha(lp),'evaluation':ev,'fold':fold,'arm':arm,'seed':seed},'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'rng_state':base.capture_rng_state(),'best':best,'best_epoch':best_epoch,'best_state':state,'history':history,'next_epoch':epoch+1});atomic(dest/'progress.json',h);print(json.dumps({'ev':ev,'fold':fold,'arm':arm,'seed':seed,**h}),flush=True)
       if epoch-best_epoch>=50:break
      if MODE=='train':
       model.load_state_dict(state);model.eval();out=[]
       with torch.no_grad():
        for group in partition(parts['test'],obs_sizes,600000,800000):
         pp=model(batch(group)).reshape(-1).cpu().numpy();out.extend({'canonical_observation_id':c,'outer_fold':fold,'seed':seed,'predicted_seconds':float(y)} for c,y in zip(group,pp))
       with (dest/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
       torch.save({'model':state,'node_mean':mean,'node_std':std,'node_valid':valid,'global_mean':gmean,'global_std':gstd,'global_valid':gmask,'arm':arm},dest/'best.pt')
      atomic(dest/'complete.json',{'status':'COMPLETE','history':history,'lock_sha256':sha(lp),'test_labels_loaded':False});del model,opt,cache;torch.cuda.empty_cache()
 finally:lease.release()
 atomic(O/('physical_graph_canary_complete.json' if MODE=='canary' else 'physical_graph_complete.json'),{'status':'COMPLETE','seconds':time.monotonic()-start})
if __name__=='__main__':main()
