"""Measure one CPU extraction route and check its output against saved reference features."""

import os as _b26_os
from pathlib import Path
import os,sys,json,time,math,pickle,signal,resource,importlib.util
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','QISKIT_NUM_PROCS')})
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).parent;OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/completion_v2'
sys.path[:0]=[str(CODE),str(ROOT/'benchmark_v1/scripts'),str(ROOT/'research/postfreeze_qpu_workflow_mean_full_cost_v1'),str(ROOT/'research/postfreeze_estimator_processing_cost_v2')]
from prepare import read,sha
from circuit_routes import Routes
from qiskit import qasm2,qasm3
import numpy as np

def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def load_kernel():
 p=ROOT/'research/postfreeze_notads_hyperq_track_a_v1/run.py';spec=importlib.util.spec_from_file_location('saved_notads',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if not (m.OUT/'path_kernel.so').exists():raise RuntimeError('saved kernel missing; refuse modify old namespace')
 return m,m.kernel()
def check(v,expected):
 if not math.isfinite(v) or not math.isclose(v,float(expected),rel_tol=1e-8,abs_tol=1e-9):raise RuntimeError(f'prediction parity mismatch: {v} vs {expected}')
def main(kind,index,method,session,canary):
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 manifest=json.loads((OUT/'manifest.json').read_text())
 for p,h in manifest['pins'].items():
  if sha(p)!=h:raise RuntimeError('manifest pin changed:'+p)
 r=manifest['C_requests' if kind=='compile' else 'analytical_requests'][index];cid=r['canonical_observation_id']
 result_path=OUT/('canary' if canary else 'measure')/kind/f'session_{session}'/f'{index}_{method}.json'
 if result_path.exists():return
 result={'kind':kind,'method':method,'id':cid,'source':r['source_id'],'session':session,'sample_index':index,'manifest_fingerprint':manifest['fingerprint']}
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800s request deadline')))
 try:
  routes=Routes()
  import materialize_compiled_unified_qpu_v1 as mat
  import importlib.metadata as metadata
  env={'qiskit_version':metadata.version('qiskit'),'qiskit_ibm_runtime_version':metadata.version('qiskit-ibm-runtime')}
  registry,backends=mat.canonical_snapshot_registry(env)
  result['environment']={**env,'python':sys.version,'numpy':np.__version__}
  if kind=='compile':
   if r['source_id']=='qonductor_single_circuit_ibm':
    result.update(status='UNAVAILABLE_HISTORICAL_COMPILER_POLICY',reason='Exact submitted physical circuit is known; archived compiler policy/seed generating it is not established.');atomic(result_path,result);return
   rawroute=routes.get(r,r['ledger'],'G1');rep=routes.rep[r['feature_source_observation_id']];backend=backends[rep['backend_canonical']];seed=int(rep['transpiler_seed'])
   def fn():
    raw=rawroute['raw']
    circuit=qasm3.loads(raw.decode()) if raw.lstrip().startswith(b'OPENQASM 3') else qasm2.loads(raw.decode(),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    compiled=mat.transpile_declared(circuit,r['source_id'],backend,seed)
    return compiled
   def verify(compiled):
    digest=mat.compiled_ir_digest(compiled);f=mat.feature_values(compiled)
    for k in ('register_width','active_width','structural_depth','measurement_count','operation_count','two_qubit_count'):
     if int(f[k])!=int(rep[k]):raise RuntimeError('compiled feature mismatch:'+k)
    if digest!=rep['compiled_or_submitted_digest']:raise RuntimeError('compiled IR digest mismatch')
   result.update(boundary='logical circuit bytes in RAM -> parse -> original nominal-target transpilation; excludes model/feature extraction',snapshot=rep['target_snapshot_id'],seed=seed,structure_provenance='FROZEN_NOMINAL_RECONSTRUCTION_NOT_HISTORICAL_PHYSICAL')
  elif method in ('qiskit','qcre'):
   import run_qpu_analytical_p_retry2 as old
   import run_strict_logical_analytical as strict
   from qiskit import QuantumCircuit
   canonical=routes.canonical[cid];rep=routes.rep[cid];raw,h,_=__import__('mali_full_features').qasm_bytes_for(canonical,rep,routes.archive)
   record={'raw_bytes':raw,'backend':rep['backend_canonical'],'shots':int(r['shots']),'fold':int(r['outer_fold']),'input_stage':'EXACT_SUBMITTED_PHYSICAL_QASM' if r['source_id']=='qonductor_single_circuit_ibm' else 'FROZEN_COMPILED_QASM3'}
   ctx={'mat':mat,'backends':backends,'fits':{},'QuantumCircuit':QuantumCircuit}
   if method=='qcre':
    if sha(strict.QCRE_ROOT/'estimate.py')!=strict.QCRE_FILE_SHA256:raise RuntimeError('QCRE source changed')
    sys.path.insert(0,str(strict.QCRE_ROOT))
    from estimate import estimate_runtime
    from bqskit.ext.qiskit import qiskit_to_bqskit
    dmaps={}
    for name,b in backends.items():
     dm={}
     for opname in b.target.operation_names:
      if opname in {'measure','reset','delay'}:continue
      op=b.target.operation_from_name(opname)
      for loc,prop in b.target[opname].items():
       if loc is not None and prop is not None and prop.duration is not None:dm.setdefault(op.num_qubits,{}).setdefault(tuple(loc),{})[opname]=float(prop.duration)
     dmaps[name]=dm
    ctx.update(estimate_runtime=estimate_runtime,converter=qiskit_to_bqskit,duration_maps=dmaps)
   name='qiskit_estimate_duration_snapshot__raw' if method=='qiskit' else 'qcre_source_estimator__raw'
   refs=read(Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'real_qpu/qualified_logical_panel/summary/oof_predictions.csv',['canonical_observation_id','method_id','predicted_seconds','status'])
   expected=float(next(x['predicted_seconds'] for x in refs if x['canonical_observation_id']==cid and x['method_id']==name and x['status']=='predicted'))
   def fn():return old.compile_and_estimate(record,name,ctx)[0]
   def verify(value):check(value,expected)
   result.update(boundary='saved compiled/submitted QASM bytes in RAM -> original parser -> nominal duration estimator; historical compilation excluded',input_sha256=h,reference_prediction=expected,output_unit='seconds')
  else:
   kernel_module,kernel=load_kernel();path=mat.DEFAULT_OUT/'graphs'/r['graph_file'];raw=path.read_bytes()
   if sha(path)!=r['graph_sha256']:raise RuntimeError('native graph changed')
   backend=backends[routes.rep[cid]['backend_canonical']];field={'notads_code':'notads_code','notads_paper':'notads_paper','hyperq':'hyperq_full_seconds'}[method]
   def fn():
    graph=pickle.loads(raw)
    durations=kernel_module.duration_values(graph,backend) if method=='hyperq' else np.zeros(int(graph['operation_count']))
    return kernel(graph,durations)[field]
   expected=float(r[field])
   def verify(value):check(value,expected)
   result.update(boundary='saved native-IR bytes in RAM -> deserialize -> score; HyperQ includes backend duration assignment; QASM parsing/compilation excluded',input_sha256=r['graph_sha256'],reference_prediction=expected,output_unit='abstract weighted depth' if method.startswith('notads') else 'seconds')
  signal.alarm(1800);warm=fn();signal.alarm(0);verify(warm)
  durations=[];values=[]
  for _ in range(1 if canary else 3):
   signal.alarm(1800);start=time.perf_counter_ns();value=fn();duration=time.perf_counter_ns()-start;signal.alarm(0);verify(value)
   durations.append(duration)
   if kind!='compile':values.append(float(value))
  result.update(status='PASS',durations_ns=durations,predictions=values,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 except Exception as e:
  signal.alarm(0);result.update(status='FAILED_ROUTE',error=repr(e))
 atomic(result_path,result);print(json.dumps({k:result[k] for k in ('kind','method','id','status')}),flush=True)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),sys.argv[5]=='canary')
