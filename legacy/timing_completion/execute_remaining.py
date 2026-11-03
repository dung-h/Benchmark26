"""Wait for the initial trial, verify its inputs, and measure remaining routes sequentially."""
import sys,os,json,time,signal,fcntl,hashlib
from pathlib import Path
from campaign import ROOT,CODE,OUT,GPU,CPU,run,rss,tree,atomic

def main():
 amendment=json.loads((OUT/'EXECUTION_AMENDMENT_01.json').read_text())
 for p,h in amendment['code_pins'].items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('execution code pin changed')
 # Existing first canary is preserved and accepted only for its matching input.
 pid=int(sys.argv[1]);start=time.time()
 while True:
  try:
   state=Path(f'/proc/{pid}/stat').read_text().split()[2]
   if state=='Z':break
  except FileNotFoundError:break
  mem=rss(pid)
  if mem>8*1024**3 or time.time()-start>4000:
   for child in reversed(tree(pid)):
    try:os.kill(child,signal.SIGTERM)
    except ProcessLookupError:pass
   raise RuntimeError('adopted canary resource/deadline guard')
  atomic(OUT/'progress.json',{'status':'ADOPTED_C_CANARY_RUNNING','pid':pid,'rss_bytes':mem,'elapsed_wait_seconds':time.time()-start})
  time.sleep(1)
 c=[json.loads(p.read_text()) for p in (OUT/'canary/C/session_1').glob('*.json')]
 if len(c)!=6 or any(r['status']!='PASS' for r in c):raise RuntimeError('first C canary did not pass all6 methods')
 # C0 correction: run a separate canary in a dedicated session so old evidence
 # is preserved. Other routes are still identical; only C0 avoids an empty RPC.
 run([GPU,str(CODE/'run_c_fixed.py'),'0','canary'],'C_fixed_canary')
 methods=('qiskit','qcre','notads_code','notads_paper','hyperq')
 for method in methods:run([CPU,str(CODE/'cpu_task.py'),'analytical','0',method,'1','canary'],'analytical_canary_'+method)
 run([CPU,str(CODE/'cpu_task.py'),'compile','0','nominal','1','canary'],'nominal_compile_canary')
 manifest=json.loads((OUT/'manifest.json').read_text())
 for session in (1,2):
  run([GPU,str(CODE/'run_c_fixed.py'),str(session),'measure'],f'C_session_{session}')
  indices=list(range(len(manifest['analytical_requests'])))
  if session==2:indices.reverse()
  for i in indices:
   for method in (methods if session==1 else tuple(reversed(methods))):
    run([CPU,str(CODE/'cpu_task.py'),'analytical',str(i),method,str(session),'measure'],f'analytical_{session}_{i}_{method}')
  for i in range(len(manifest['C_requests'])):
   run([CPU,str(CODE/'cpu_task.py'),'compile',str(i),'nominal',str(session),'measure'],f'compile_{session}_{i}')
 atomic(OUT/'measurement_receipt.json',{'status':'ALL_ASSIGNED_ROUTES_ATTEMPTED','manifest_fingerprint':manifest['fingerprint'],'execution_amendment_verified':True})
 run([sys.executable,str(CODE/'summarize.py')],'summarize_validate')
 atomic(OUT/'progress.json',{'status':'FINISHED_CHECK_VALIDATION_AND_ROUTE_FAILURES','receipt':'validation_receipt.json'})
if __name__=='__main__':
 with (OUT.parent/'.inference_timing.lock').open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except Exception as e:atomic(OUT/'progress.json',{'status':'STOPPED_WITH_ERROR','error':repr(e)});raise
