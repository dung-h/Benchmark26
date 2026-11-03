
import os as _b26_os
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/completion_v2';SRC=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_expansion_v1'
def read(p,cols=None):
 csv.field_size_limit(100_000_000)
 with Path(p).open(newline='') as f:
  r=csv.reader(f);h=next(r);cols=cols or h;ix=[h.index(c) for c in cols];return [dict(zip(cols,(v[i] for i in ix))) for v in r]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def select(values,key,identity):
 unique={}
 for r in sorted(values,key=lambda r:r['canonical_observation_id']):unique.setdefault(identity(r),r)
 ordered=sorted(unique.values(),key=lambda r:(key(r),r['canonical_observation_id']))
 indices=sorted(set(round((len(ordered)-1)*q) for q in (0,1/3,2/3,1)))
 return [dict(ordered[i],selection_rank=i,selection_unique_population=len(ordered)) for i in indices]
def main():
 panel=read(SRC/'cohort/panel.csv');ledger={r['canonical_observation_id']:r for r in read(SRC/'cohort/feature_ledger.csv')};folds={r['canonical_observation_id']:r['outer_fold'] for r in read(SRC/'splits/E1_outer.csv')}
 c=[]
 for source in sorted({r['source_id'] for r in panel}):
  chosen=select([r for r in panel if r['source_id']==source],lambda r:int(ledger[r['canonical_observation_id']]['graph_node_count']),lambda r:ledger[r['canonical_observation_id']]['graph_artifact_sha256'])
  for r in chosen:c.append(dict(r,ledger=ledger[r['canonical_observation_id']],case=int(folds[r['canonical_observation_id']])))
 features=read(Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_notads_hyperq_track_a_v1/features.csv');rep={r['canonical_row_id']:r for r in read(Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/real_qpu/compiled_unified/representation_rows.csv')}
 a=[]
 for source in sorted({r['source_id'] for r in features}):
  a.extend(select([r for r in features if r['source_id']==source],lambda r:int(rep[r['canonical_observation_id']]['operation_count']),lambda r:r['physical_hash']))
 pins=[SRC/'cohort/panel.csv',SRC/'cohort/feature_ledger.csv',SRC/'splits/E1_outer.csv',Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_notads_hyperq_track_a_v1/features.csv',Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/real_qpu/compiled_unified/representation_rows.csv',Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/parallel8_v2/completion_receipt.json']
 pins.extend(Path(__file__).parent.glob('*.py'))
 pins.append(OUT/'PROTOCOL.md')
 pins.extend((ROOT/'research/postfreeze_qpu_workflow_mean_full_cost_v1'/n) for n in ['circuit_routes.py','feature_proxy.py','feature_server.py'])
 result={'schema':1,'selection':'unique input identity; sorted size then ID; ranks0,1/3,2/3,1 independently per source; selected before new timings; no targets or errors used','C_boundary':'existing method input stage; raw circuit bytes in RAM through parsing/features and warm prediction; CPU-feature RPC included; excludes upstream compilation, disk/network and loading','C_scope':'size-stratified diagnostic subset of expanded3443; E1 fold-correct states; no full-population cost or accuracy claim','C_requests':c,'analytical_scope':'separate frozen4515 cohort; four native-size quantiles per source; raw QCRE/Qiskit, calibrated scalar replay, NoTaDS/HyperQ native-IR routes','analytical_requests':a,'sessions':2,'warmups':1,'measured_repetitions':3,'compile_requests':'same C requests, MaLi/QPack pinned nominal policy; Qonductor historical compilation unavailable, report explicitly','pins':{str(p):sha(p) for p in pins},'no_training':True,'no_quantum_execution':True}
 result['fingerprint']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest();OUT.mkdir(exist_ok=True)
 p=OUT/'manifest.json'
 if p.exists():raise RuntimeError('refuse overwrite locked manifest')
 p.write_text(json.dumps(result,indent=2));print('C',len(c),'analytical',len(a))
if __name__=='__main__':main()
