#!/usr/bin/env python3
"""Prepare explicitly limited transpiled structural descriptors without target access.

Outputs are draft inputs, not the complete Azizov feature set. Snapshot-derived
calibration, SupermarQ metrics and neural graph tensors are not computed here.
"""
import collections,csv,hashlib,json,os,pickle
from pathlib import Path
R=Path(os.environ['B26_ARTIFACTS'])
O=Path(os.environ['B26_RUN_DIR']);S=R/'postfreeze_unified_analytical_completion_v1'; G=R/'benchmark_v3/real_qpu/compiled_unified/graphs'
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
source_rows=read(S/'lineage.csv');raw=read(S/'raw_features.csv');lookup={}
for r in raw:
 k=(r['graph_file'],r['snapshot_id'])
 if k in lookup:raise ValueError('duplicate graph/snapshot')
 lookup[k]=r
keys=('rz','sx','x','ecr','cz','measure','reset','delay','barrier'); cache={};out=[]
for i,row in enumerate(source_rows,1):
 f=row['graph_file'];k=(f,row['snapshot_id']);r=lookup[k]
 if f not in cache:
  with (G/f).open('rb') as h:g=pickle.load(h)
  c=collections.Counter(g['names']);n=int(g['operation_count']);assert n==sum(c.values())
  qubits=set()
  for name in ('q0','q1','q2','q3'):
   qubits.update(int(q) for q in g[name] if q>=0)
  vals={f'count_{op}':int(c[op]) for op in keys};vals.update(count_other=n-sum(c[x] for x in keys),count_total=n,count_two_qubit=sum(int(x)==2 for x in g['arity']),active_qubits=len(qubits));cache[f]=vals
 vals={'canonical_observation_id':row['canonical_observation_id'],'source_id':row['source_id'],'graph_file':f,'snapshot_id':row['snapshot_id'],**cache[f]}
 for field in ('depth_1q','depth_2q','unitary_depth','hyperq_full_seconds','hyperq_unitary_seconds','measurement_count'):vals[field]=r[field]
 out.append(vals)
 if i%500==0:print('processed',i,flush=True)
assert len(out)==3443 and len({x['canonical_observation_id'] for x in out})==3443
O.mkdir(parents=True,exist_ok=True);p=O/'structural_features_draft.csv'
if p.exists():raise FileExistsError(p)
with p.open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
meta={'status':'DRAFT_NOT_TRAINING_READY','reason':'SupermarQ and snapshot calibration features not yet implemented; graph tensors and fit-only transforms not yet implemented','rows':len(out),'unique_graphs':len(cache),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'no_targets_read':True,'fields':list(out[0])}
(O/'structural_features_draft.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
