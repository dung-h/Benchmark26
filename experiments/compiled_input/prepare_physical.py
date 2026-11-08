#!/usr/bin/env python3
"""Create private compiled-circuit Ma–Li inputs from pinned saved DAGs, without labels."""
from __future__ import annotations
import os,sys,csv,json,hashlib,pickle,multiprocessing,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmark26 import mali_features as mf
import numpy as np
R=Path(os.environ['B26_ARTIFACTS']);O=Path(os.environ['B26_RUN_DIR']);P=R/'postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv'; L=R/'postfreeze_unified_analytical_completion_v1/lineage.csv';C=R/'benchmark_v3/real_qpu/compiled_unified'; REPS=C/'representation_rows.csv';STATE={}
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def init():
 extractor,pins=mf.load_pinned_global_extractor();STATE.update(extractor=extractor,pins=pins,registry=mf.load_snapshot_registry(),cal={})
def task(arg):
 key,row,rep=arg;out=O/'physical_inputs'/f'{key}.npz';meta=O/'physical_inputs'/f'{key}.json'
 if out.exists() and meta.exists():
  d=json.loads(meta.read_text());assert d['file_sha256']==sha(out) and d['key']==key;return d
 try:
  graph_path=C/'graphs'/row['graph_file'];assert sha(graph_path)==row['graph_file_sha256']
  with graph_path.open('rb') as f:graph=pickle.load(f)
  vals,gmeta=mf.extract_global_features_from_s71(graph,rep,STATE['extractor']);snap=rep['target_snapshot_id'];prop=rep['target_properties_sha256'];ck=(snap,prop)
  if ck not in STATE['cal']:STATE['cal'][ck]=mf.snapshot_t1_t2(rep,STATE['registry'])[0]
  sparse,smeta=mf.compact_graph_record_from_s71(graph,STATE['cal'][ck]);tmp=out.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez_compressed(f,**sparse,compiled_global=np.asarray(vals,dtype=np.float32))
  os.replace(tmp,out);d={'status':'PASS','key':key,'file_sha256':sha(out),'compiled_global_dim':len(vals),'nodes':smeta['node_count'],'edges':smeta['edge_count'],'graph_file':row['graph_file'],'snapshot_id':snap}
  meta.write_text(json.dumps(d)+'\n');return d
 except Exception as e:return {'status':'FAIL','key':key,'id':row['canonical_observation_id'],'error':repr(e)}
def main():
 O.mkdir(parents=True,exist_ok=True);(O/'physical_inputs').mkdir(exist_ok=True)
 dataset_rows={r['canonical_observation_id']:r for r in read(P)}; source_rows=read(L); rep={r['canonical_row_id']:r for r in read(REPS)};assert len(dataset_rows)==len(lineage)==3443
 tasks={};mapping={}
 for row in source_rows:
  cid=row['canonical_observation_id'];assert cid in dataset_rows;rp=rep[dataset_rows[cid]['feature_source_observation_id']]; assert rp['target_snapshot_id']==row['snapshot_id']
  key=hashlib.sha256(json.dumps([row['graph_file'],row['snapshot_id'],rp['target_properties_sha256'],rp['gate_counts_json'],rp['structural_depth'],rp['register_width']],separators=(',',':')).encode()).hexdigest();tasks.setdefault(key,(key,row,rp));mapping[cid]=key
 lock={'status':'PREPARATION_ONLY_NO_LABELS','code_sha256':sha(Path(__file__)),'inputs':{str(p):sha(p) for p in (P,L,REPS)},'tasks':len(tasks),'observations':len(mapping),'workers':4,'note':'Ma-Li 51-wide physical global feature analogue with SupermarQ metrics; not claimed exact Azizov feature set'}
 lp=O/'physical_preparation_lock.json'
 if lp.exists():assert json.loads(lp.read_text())==lock
 else:lp.write_text(json.dumps(lock,indent=2)+'\n')
 (O/'physical_mapping.json').write_text(json.dumps(mapping,sort_keys=True)+'\n');print('START',len(tasks),flush=True);bad=[];start=time.monotonic()
 with multiprocessing.get_context('spawn').Pool(4,initializer=init) as pool:
  for i,result in enumerate(pool.imap_unordered(task,tasks.values(),chunksize=1),1):
   if result['status']!='PASS':bad.append(result);print('FAIL',json.dumps(result),flush=True)
   if i%20==0:print('PROGRESS',i,'/',len(tasks),'seconds',round(time.monotonic()-start),flush=True)
 receipt={'status':'PASS' if not bad else 'FAIL','tasks':len(tasks),'completed':len(tasks)-len(bad),'failures':bad,'observations':len(mapping),'elapsed_seconds':time.monotonic()-start};(O/'physical_preparation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
 if bad:raise SystemExit(1)
if __name__=='__main__':main()
