"""Run diagnostic latency measurements sequentially with memory and time limits."""
import os,sys,json,time,subprocess,signal,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).parent;OUT=ROOT/'artifacts/postfreeze_qpu_workflow_mean_cost_v1/completion_v2'
GPU='/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python';CPU='/home/server/Documents/.venv-qonductor/bin/python'
def atomic(p,o):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2));t.replace(p)
def tree(pid):
 ids=[pid]
 try:
  for c in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():ids.extend(tree(int(c)))
 except FileNotFoundError:pass
 return ids
def rss(pid):
 n=0
 for c in tree(pid):
  try:n+=int(next(l.split()[1] for l in Path(f'/proc/{c}/status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
  except (FileNotFoundError,StopIteration):pass
 return n
def run(cmd,label):
 log=OUT/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
 with log.open('ab') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
  start=time.time();peak=0
  while p.poll() is None:
   r=rss(p.pid);peak=max(peak,r)
   available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
   if r>8*1024**3 or available<3*1024**3 or time.time()-start>8000:
    os.killpg(p.pid,signal.SIGTERM);p.wait();atomic(OUT/'resource_failure.json',{'label':label,'rss':r,'available':available,'elapsed':time.time()-start});raise RuntimeError('resource/time guard:'+label)
   atomic(OUT/'progress.json',{'status':'RUNNING','stage':label,'pid':p.pid,'elapsed_stage_seconds':time.time()-start,'rss_bytes':r,'peak_rss_bytes':peak,'log':str(log)})
   time.sleep(1)
  if p.returncode:raise RuntimeError('worker exit '+str(p.returncode)+':'+label+'; inspect '+str(log))
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());methods=('qiskit','qcre','notads_code','notads_paper','hyperq')
 # Cheap scalar mappings first; no load/fit of runtime predictors.
 if not (OUT/'scalar_receipt.json').exists():run([sys.executable,str(CODE/'scalars.py')],'scalar_all4515')
 run([GPU,str(CODE/'run_c.py'),'1','canary'],'C_canary')
 for method in methods:
  run([CPU,str(CODE/'cpu_task.py'),'analytical','0',method,'1','canary'],'analytic_canary_'+method)
  receipt=json.loads((OUT/'canary/analytical/session_1'/f'0_{method}.json').read_text())
  if receipt['status']!='PASS':atomic(OUT/('blocked_'+method+'.json'),receipt)
 run([CPU,str(CODE/'cpu_task.py'),'compile','0','nominal','1','canary'],'compile_canary')
 for session in (1,2):
  run([GPU,str(CODE/'run_c.py'),str(session),'measure'],f'C_session_{session}')
  for i in (range(len(manifest['analytical_requests'])) if session==1 else reversed(range(len(manifest['analytical_requests'])))):
   for method in (methods if session==1 else tuple(reversed(methods))):
    if (OUT/('blocked_'+method+'.json')).exists():continue
    run([CPU,str(CODE/'cpu_task.py'),'analytical',str(i),method,str(session),'measure'],f'analytical_{session}_{i}_{method}')
  for i in range(len(manifest['C_requests'])):
   run([CPU,str(CODE/'cpu_task.py'),'compile',str(i),'nominal',str(session),'measure'],f'compile_{session}_{i}')
 atomic(OUT/'measurement_receipt.json',{'status':'MEASUREMENTS_FINISHED_REQUIRES_ROUTE_VALIDATION','fingerprint':manifest['fingerprint'],'blocked_routes':[p.name for p in OUT.glob('blocked_*.json')]})
 run([sys.executable,str(CODE/'summarize.py')],'summarize_validate')
 atomic(OUT/'progress.json',{'status':'FINISHED_REVIEW_ROUTE_COVERAGE','receipt':'validation_receipt.json'})
if __name__=='__main__':
 with (OUT.parent/'.inference_timing.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except Exception as e:atomic(OUT/'progress.json',{'status':'STOPPED_WITH_ERROR','error':repr(e)});raise
