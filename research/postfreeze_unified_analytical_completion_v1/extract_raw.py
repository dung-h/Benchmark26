#!/usr/bin/env python3
"""Extract fixed analytical scalars from identity-pinned saved native DAGs.

No targets are opened by this module. It reuses the frozen graph kernel and
Target duration lookup already reviewed in postfreeze_notads_hyperq_track_a_v1.
"""
from __future__ import annotations

import os as _b26_os
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,platform,sys,time
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NS=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_unified_analytical_completion_v1'
PANEL=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_expansion_v1'
REP=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/real_qpu/compiled_unified'
FEATURE=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_notads_hyperq_track_a_v1/features.csv'
ATTEMPTS=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/real_qpu/archived_analytical_wave_v3_20260930/attempts.csv'
CODE=ROOT/'research/postfreeze_notads_hyperq_track_a_v1/run.py'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(path,rs):
 if not rs:raise RuntimeError(f'empty_output:{path}')
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.partial')
 with tmp.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
 tmp.replace(path)
def load_module():
 spec=importlib.util.spec_from_file_location('locked_notads_hyperq_kernel',CODE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--check-only',action='store_true');g.add_argument('--canary',action='store_true');g.add_argument('--full',action='store_true');a=ap.parse_args()
 out=NS
 if a.full and (out/'raw_features.csv').exists():raise RuntimeError('refuse_overwrite_raw_features')
 source=rows(PANEL/'cohort/panel.csv');targets_not_read=True
 qmap={r['workflow_mean_id']:r for r in rows(PANEL/'cohort/qpack_feature_source_map.csv')}
 reps={r['canonical_row_id']:r for r in rows(REP/'representation_rows.csv')}
 splitfiles=[PANEL/'splits'/f'{e}_outer.csv' for e in ('E1','E2','E3')]
 ids=[r['canonical_observation_id'] for r in source]
 if len(source)!=3443 or len(set(ids))!=3443:raise RuntimeError('cohort_identity_mismatch')
 if len(set(r['group_id'] for r in source))!=203:raise RuntimeError('group_count_mismatch')
 expected={'mali_real_qpu':340,'qonductor_single_circuit_ibm':3058,'qpack_mcp':45}
 if dict(Counter(r['source_id'] for r in source))!=expected:raise RuntimeError('source_count_mismatch')
 if any(not p.is_file() for p in splitfiles):raise RuntimeError('missing_frozen_split')
 hashes={}
 pinpaths=[PANEL/'PROTOCOL.md',PANEL/'cohort/panel.csv',PANEL/'cohort/targets.csv',PANEL/'cohort/feature_ledger.csv',PANEL/'cohort/qpack_feature_source_map.csv',REP/'representation_rows.csv',REP/'graph_file_hashes.json',Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv',CODE,ROOT/'research/postfreeze_notads_hyperq_track_a_v1/path_kernel.cpp',Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_notads_hyperq_track_a_v1/path_kernel.so',*splitfiles]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in pinpaths}
 file_hashes=json.loads((REP/'graph_file_hashes.json').read_text())
 target_free=[]
 for r in source:
  cid=r['canonical_observation_id'];src=r['source_id']
  sid=qmap[cid]['feature_source_observation_id'] if src=='qpack_mcp' else cid
  if src=='qpack_mcp':
   m=qmap[cid]
   if float(m['shots'])!=float(r['shots']) or m['backend']!=r['backend']:raise RuntimeError(f'qpack_context_mismatch:{cid}')
  rep=reps.get(sid)
  if rep is None:raise RuntimeError(f'representation_missing:{cid}:{sid}')
  if rep['backend_canonical']!=r['backend'] or rep['target_snapshot_id']=='':raise RuntimeError(f'backend_snapshot_mismatch:{cid}')
  if not rep['graph_file'] or rep['graph_status']!='available':raise RuntimeError(f'graph_not_available:{cid}')
  graph=REP/'graphs'/rep['graph_file']
  if not graph.is_file() or sha(graph)!=file_hashes.get(rep['graph_file']):raise RuntimeError(f'graph_hash_mismatch:{cid}')
  if Path(rep['graph_file']).stem!=rep['graph_input_digest']:raise RuntimeError(f'graph_identity_digest_mismatch:{cid}')
  if src=='qonductor_single_circuit_ibm' and rep['lifecycle_stage']!='submitted_physical':raise RuntimeError(f'qonductor_not_exact_physical:{cid}')
  if src=='mali_real_qpu' and rep['source_qasm_sha256']!=r.get('source_qasm_sha256',rep['source_qasm_sha256']):raise RuntimeError(f'mali_source_hash_mismatch:{cid}')
  target_free.append({'canonical_observation_id':cid,'source_id':src,'group_id':r['group_id'],'backend':r['backend'],'shots':r['shots'],
   'representation_observation_id':sid,'graph_file':rep['graph_file'],'graph_file_sha256':file_hashes[rep['graph_file']],
   'graph_input_digest':rep['graph_input_digest'],'snapshot_id':rep['target_snapshot_id'],'snapshot_tier':rep['snapshot_tier'],
   'structure_provenance':('exact_submitted_physical_QASM' if src=='qonductor_single_circuit_ibm' else ('saved_nominal_compiled_DAG' if src=='mali_real_qpu' else 'workflow_representative_saved_nominal_DAG')),
   'timing_provenance':'pinned_nominal_backend_Target_proxy'})
 if len({x['canonical_observation_id'] for x in target_free})!=3443:raise RuntimeError('lineage_duplicate')
 taskkeys=sorted({(x['graph_file'],x['snapshot_id']) for x in target_free})
 # Verify the already-produced feature cache joins by exact graph and Target, not row label.
 old={r['canonical_observation_id']:r for r in rows(FEATURE)}
 shared=[]
 for x in target_free:
  o=old.get(x['representation_observation_id'])
  if o and o['graph_file']==x['graph_file'] and o['graph_sha256']==x['graph_file_sha256'] and o['snapshot_id']==x['snapshot_id']:
   shared.append(x['representation_observation_id'])
 inventory={'status':'PASS_TARGET_FREE_INVENTORY','rows':3443,'groups':203,'source_counts':expected,'unique_graph_snapshot_tasks':len(taskkeys),
  'qpack_representatives':45,'all_graph_files_hash_verified':True,'graph_snapshot_matches_existing_notads_features':len(shared),
  'targets_opened':False,'frozen_split_files':{p.name:sha(p) for p in splitfiles},'input_hashes':hashes,
  'python':platform.python_version(),'execution_environment':'must be qiskit 2.5.2 and qiskit-ibm-runtime 0.49.0'}
 (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 if a.check_only:
  print(json.dumps(inventory,indent=2));return
 if not sys.prefix.endswith('venv-qonductor') and 'qiskit-ibm-runtime' not in str(sys.executable):
  import importlib.metadata as md
  if md.version('qiskit')!='2.5.2' or md.version('qiskit-ibm-runtime')!='0.49.0':raise RuntimeError('pinned_environment_required')
 m=load_module();objects,pins=m.backends();kernel=m.kernel()
 if len(objects)!=12:raise RuntimeError(f'target_registry_count:{len(objects)}')
 lineage_by_id={x['canonical_observation_id']:x for x in target_free}
 tasks=taskkeys
 if a.canary:
  chosen=[]
  for src in expected:
   chosen.append(next(x for x in target_free if x['source_id']==src))
  taskset={(x['graph_file'],x['snapshot_id']) for x in chosen}
  tasks=sorted(taskset)
 results=[];cache={};start=time.time()
 for graphfile,snapshot in tasks:
  gp=REP/'graphs'/graphfile;g0=__import__('pickle').load(gp.open('rb'))
  if g0.get('format')!='s71-operation-dag-v1' or int(g0['operation_count'])!=len(g0['names']):raise RuntimeError(f'graph_schema:{graphfile}')
  backend=objects[snapshot]
  d=m.duration_values(g0,backend)
  feat=kernel(g0,d)
  if not all(np.isfinite(float(feat[k])) for k in ('notads_code','notads_paper','hyperq_full_seconds','hyperq_unitary_seconds')):raise RuntimeError(f'nonfinite_route:{graphfile}')
  if a.canary:
   ref=m.reference(g0,d)
   for k,v in feat.items():
    if not np.isclose(v,ref[k],rtol=1e-10,atol=1e-12):raise RuntimeError(f'kernel_reference_mismatch:{graphfile}:{k}')
  cache[(graphfile,snapshot)]=feat
  results.append({'graph_file':graphfile,'graph_file_sha256':file_hashes[graphfile],'snapshot_id':snapshot,**feat})
 if a.canary:
  canary_ids={x['canonical_observation_id'] for x in chosen}
  mapped=[]
  for x in chosen:
   f=cache[(x['graph_file'],x['snapshot_id'])]
   mapped.append({'canonical_observation_id':x['canonical_observation_id'],'source_id':x['source_id'],'group_id':x['group_id'],
    'backend':x['backend'],'shots':x['shots'],'graph_file':x['graph_file'],'snapshot_id':x['snapshot_id'],**f})
  write(out/'canary_raw_features.csv',mapped)
  receipt={'status':'PASS_CANARY','rows':len(mapped),'sources':sorted({x['source_id'] for x in mapped}),'reference_recurrence':'PASS',
   'elapsed_seconds':time.time()-start,'peak_rss_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024,
   'targets_used_only_for_nominal_durations':True,'labels_opened':False,'route_failures':0,'backend_snapshot_files_verified':len(pins)}
  (out/'canary_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return
 if a.full:
  write(out/'raw_features.csv',results)
  write(out/'lineage.csv',target_free)
  receipt={'status':'PASS_RAW_EXTRACTION','rows':len(target_free),'unique_graph_snapshot_tasks':len(results),'elapsed_seconds':time.time()-start,
   'task_failures':0,'target_labels_opened':False,'all_graphs_hash_verified':True,'snapshot_assets_verified':len(pins),
   'input_hashes':hashes,'feature_sha256':sha(out/'raw_features.csv'),'lineage_sha256':sha(out/'lineage.csv')}
  (out/'extraction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
