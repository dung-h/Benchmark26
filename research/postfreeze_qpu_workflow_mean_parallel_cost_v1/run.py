"""Eight CPU feature workers, exact-input aliases, then separate GPU replay.

Concurrent processing times are throughput diagnostics, not isolated latency.
"""
import os,sys,json,time,csv,hashlib,signal,resource,subprocess,concurrent.futures as cf
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).parent
OUT=ROOT/'artifacts/postfreeze_qpu_workflow_mean_cost_v1/parallel8_v2'
SRC=ROOT/'artifacts/postfreeze_qpu_workflow_mean_expansion_v1'
FEATURE_ENV='/home/server/Documents/.venv-qonductor/bin/python'
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def prepare():
 panel=rows(SRC/'cohort/panel.csv');ledger={r['canonical_observation_id']:r for r in rows(SRC/'cohort/feature_ledger.csv')}
 provenance={r['canonical_observation_id']:r for r in rows(SRC/'cohort/classical_feature_provenance.csv')}
 assert len(panel)==3443 and len(ledger)==3443
 tasks={}
 for r in panel:
  cid=r['canonical_observation_id'];l=ledger[cid];p=provenance[cid]
  for method in ('G1','R0'):
   identity={'method':method,'source':r['source_id'],'shots':r['shots'],'input_sha256':l['source_qasm_sha256'] if method=='G1' else p['input_qasm_sha256']}
   if method=='G1':identity['graph_sha256']=l['graph_artifact_sha256']
   else:
    identity['adapter_signature_sha256']=p['adapter_signature_sha256']
    identity['representation_id']=p['representation_id']
   key=stable(identity)
   if key not in tasks:tasks[key]={'key':key,'identity':identity,'method':method,'panel':r,'ledger':l,'aliases':[]}
   tasks[key]['aliases'].append(cid)
 pinfiles=[SRC/'cohort/panel.csv',SRC/'cohort/feature_ledger.csv',SRC/'cohort/native_baseline_features.csv',SRC/'cohort/classical_feature_provenance.csv',ROOT/'research/postfreeze_qpu_workflow_mean_full_cost_v1/circuit_routes.py',Path(__file__)]
 manifest={'schema':1,'workers':8,'threads_per_worker':1,'cohort_rows':3443,'unique_tasks':len(tasks),'aliases':6886,'tasks':list(tasks.values()),'input_code_hashes':{str(p):sha(p) for p in pinfiles},'scope':'exact input feature reconstruction and parity; per-task concurrent durations not isolated latency; no fitted model or targets read by CPU workers','alias_rule':'same source,method,input bytes hash,shots and graph payload hash for logical path; all aliases independently feature-checked before use','feature_environment':FEATURE_ENV}
 manifest['fingerprint']=stable(manifest)
 p=OUT/'manifest.json'
 if p.exists() and json.loads(p.read_text())!=manifest:raise RuntimeError('manifest changed')
 atomic(p,manifest);print(json.dumps({'tasks':len(tasks),'logical':sum(t['method']=='G1' for t in tasks.values()),'native':sum(t['method']=='R0' for t in tasks.values())}),flush=True)
def worker(taskfile,cpu):
 os.sched_setaffinity(0,{int(cpu)})
 import numpy as np
 sys.path.insert(0,str(ROOT/'research/postfreeze_qpu_workflow_mean_full_cost_v1'))
 from circuit_routes import Routes
 task=json.loads(Path(taskfile).read_text());routes=Routes()
 def timeout(*_):raise TimeoutError('feature task exceeded1800s')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
 start=time.perf_counter_ns();r=routes.get(task['panel'],task['ledger'],task['method']);input_ns=time.perf_counter_ns()-start
 start=time.perf_counter_ns();value=routes.extract(r,task['method']);extract_ns=time.perf_counter_ns()-start;signal.alarm(0)
 arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
 for k,v in value.get('graph_record',{}).items():arrays['graph__'+k]=v
 if 'shots' in value:arrays['shots']=np.asarray(value['shots'])
 path=OUT/'features'/(task['key']+'.npz');path.parent.mkdir(exist_ok=True)
 with path.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,**arrays)
 path.with_suffix('.partial').replace(path)
 atomic(OUT/'records'/(task['key']+'.json'),{'status':'FEATURE_EXTRACTED_PENDING_INDEPENDENT_PARITY','key':task['key'],'method':task['method'],'stage':r['stage'],'input_sha256':r.get('sha256'),'input_bytes':len(r['raw']),'input_acquisition_ns_excluded_from_C':input_ns,'concurrent_extraction_ns':extract_ns,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_affinity':[int(cpu)],'feature_file_sha256':sha(path)})
def cpu_ids():
 allowed=os.sched_getaffinity(0);seen=set();chosen=[]
 for line in subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True).splitlines():
  if line.startswith('#'):continue
  cpu,core,socket=map(int,line.split(','));key=(core,socket)
  if cpu in allowed and key not in seen:seen.add(key);chosen.append(cpu)
 if len(chosen)<8:raise RuntimeError('fewer than8 distinct available physical cores')
 return chosen[:8]
def tree(pid):
 ids=[pid]
 try:
  for c in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():ids.extend(tree(int(c)))
 except FileNotFoundError:pass
 return ids
def rss(pid):
 total=0
 for c in tree(pid):
  try:total+=int(next(l.split()[1] for l in Path(f'/proc/{c}/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
  except (FileNotFoundError,StopIteration):pass
 return total
def available():
 return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
def measure():
 manifest=json.loads((OUT/'manifest.json').read_text())
 for p,h in manifest['input_code_hashes'].items():
  if sha(p)!=h:raise RuntimeError('input/code hash changed:'+p)
 cpus=cpu_ids();pending=[]
 for t in manifest['tasks']:
  record=OUT/'records'/(t['key']+'.json')
  if record.exists():continue
  path=OUT/'tasks'/(t['key']+'.json');atomic(path,t);pending.append((t,path))
 # Largest known input tasks first where archive metadata is available; stable
 # task order otherwise. Scheduling does not inspect runtime labels.
 running={};index=0;start=time.time();peak=0
 logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
 while index<len(pending) or running:
  used={x['cpu'] for x in running.values()}
  for cpu in cpus:
   if cpu in used or index>=len(pending):continue
   if available()<6*1024**3:break
   task,path=pending[index];index+=1;log=open(logdir/(task['key']+'.log'),'ab')
   proc=subprocess.Popen([FEATURE_ENV,str(Path(__file__)),'worker',str(path),str(cpu)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True);log.close()
   running[proc.pid]={'proc':proc,'task':task,'cpu':cpu,'started':time.time()}
  total=sum(rss(pid) for pid in running);peak=max(peak,total)
  if total>32*1024**3 or (running and available()<3*1024**3):
   for pid in running:
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:pass
   atomic(OUT/'resource_stop.json',{'status':'STOPPED_MEMORY_GUARD','own_rss_bytes':total,'available_bytes':available(),'completed':len(list((OUT/'records').glob('*.json')))})
   raise RuntimeError('32GiB own-RSS or3GiB available RAM guard; inspect before lower-concurrency resume')
  for pid,x in list(running.items()):
   proc=x['proc'];code=proc.poll()
   if code is None and (rss(pid)>5*1024**3 or time.time()-x['started']>1900):
    os.killpg(pid,signal.SIGTERM);code=proc.wait();atomic(OUT/'failures'/(x['task']['key']+'.json'),{'status':'TASK_RESOURCE_OR_TIMEOUT_FAILURE','exit_code':code})
   if code is not None:
    if code:
     atomic(OUT/'failures'/(x['task']['key']+'.json'),{'status':'TASK_FAILED','exit_code':code,'log':str(logdir/(x['task']['key']+'.log'))})
    del running[pid]
  atomic(OUT/'progress.json',{'status':'RUNNING_8_PROCESS_FEATURE_RECONSTRUCTION','assigned_tasks':manifest['unique_tasks'],'completed_tasks':len(list((OUT/'records').glob('*.json'))),'failed_tasks':len(list((OUT/'failures').glob('*.json'))),'active_workers':len(running),'worker_pids':list(running),'physical_cpu_ids':cpus,'own_rss_bytes':total,'peak_own_rss_bytes':peak,'available_ram_bytes':available(),'elapsed_seconds':time.time()-start})
  time.sleep(1)
 failures=list((OUT/'failures').glob('*.json'))
 atomic(OUT/'extraction_receipt.json',{'status':'COMPLETE_PENDING_PARITY' if not failures else 'PARTIAL_WITH_FAILURES','completed_tasks':len(list((OUT/'records').glob('*.json'))),'assigned_tasks':manifest['unique_tasks'],'failures':len(failures),'elapsed_seconds':time.time()-start,'peak_own_rss_bytes':peak})
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
 else:measure()
