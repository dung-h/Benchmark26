"""Circuit-stage latency supplement: mandatory input/feature/output parity."""
import os,sys,json,time,signal,argparse,resource,fcntl
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/postfreeze_qpu_workflow_mean_cost_v1'))
from common import ART,SRC,METHODS,atomic_json,sha,stable
from loaders import StudyInputs
from feature_proxy import Routes
import numpy as np
import torch
OUT=ART/'full_cost_v1'/'circuit_qiskit25_rpc_v1'

def alarm(*_):raise TimeoutError('request exceeded 600 seconds')
def parity(actual,expected,method):
    fields=['native5'] if method=='R0' else ([] if method=='C0' else ['globals'])
    for k in fields:
        if not np.allclose(actual[k],expected[k],rtol=1e-6,atol=1e-6):raise RuntimeError('feature parity failed:'+k)
    if method.startswith('G'):
        for k,v in expected['graph_record'].items():
            if hasattr(v,'shape') and (k not in actual['graph_record'] or v.shape!=actual['graph_record'][k].shape or not np.allclose(v,actual['graph_record'][k],rtol=1e-6,atol=1e-6)):
                raise RuntimeError('graph feature parity failed:'+k)

def main(canary,session):
    affinity=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,affinity)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    signal.signal(signal.SIGALRM,alarm)
    routes=Routes();units=[];failures=[]
    code={str(p):sha(p) for p in (Path(__file__).parent/'run_circuit.py',Path(__file__).parent/'circuit_routes.py',Path(__file__).parent/'feature_proxy.py',Path(__file__).parent/'feature_server.py')}
    if not canary:
        receipt=json.loads((OUT/'circuit_canary_receipt.json').read_text())
        if receipt['status']!='PASS' or receipt['code']!=code:raise RuntimeError('canary not PASS or code changed')
    for ev in ('E1','E2','E3'):
        inputs=StudyInputs(ev)
        for case,ids in inputs.test_by_case.items():
            inputs.load_case(case)
            if canary:
                # Largest and median byte input per source, independently per method.
                selection={}
                for method in METHODS:
                    groups={}
                    for cid in ids:
                        try:
                            r=routes.get(inputs.panel_by_id[cid],inputs.ledger[cid],method)
                            groups.setdefault(inputs.panel_by_id[cid]['source_id'],[]).append((len(r['raw']),cid))
                        except Exception as e:failures.append({'evaluation':ev,'case':case,'id':cid,'method':method,'error':repr(e)})
                    selection[method]=sorted({c for values in groups.values() for _,c in (sorted(values)[len(values)//2],sorted(values)[-1])})
            else:selection={m:ids for m in METHODS}
            for method in METHODS:
                for cid in selection[method]:
                    path=OUT/('circuit_canary' if canary else f'circuit_session_{session}')/ev/f'case_{case}'/method/(sha_id(cid)+'.json')
                    if path.exists():continue
                    try:
                        route=routes.get(inputs.panel_by_id[cid],inputs.ledger[cid],method)
                        expected=inputs.raw_request(cid,method)
                        signal.alarm(600);request=routes.extract(route,method);signal.alarm(0)
                        parity(request,expected,method)
                        prepared=inputs.prepare_raw_B(method,request)
                        pred,seeds,_=inputs.predict_ready(method,cid,prepared);inputs.check(method,cid,pred,seeds)
                        # Reference verification above is the warmup. Timed calls
                        # rebuild circuit/features and apply transforms every time.
                        durations=[]
                        for repeat in range(1 if canary else 3):
                            if method.startswith(('M','G')):torch.cuda.synchronize()
                            signal.alarm(600);start=time.perf_counter_ns()
                            request=routes.extract(route,method)
                            prepared=inputs.prepare_raw_B(method,request)
                            pred,seeds,_=inputs.predict_ready(method,cid,prepared)
                            if method.startswith(('M','G')):torch.cuda.synchronize()
                            elapsed=time.perf_counter_ns()-start;signal.alarm(0)
                            inputs.check(method,cid,pred,seeds);durations.append(elapsed)
                            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise RuntimeError('8GiB RSS guard')
                        record={'status':'PASS','evaluation':ev,'case':case,'id':cid,'method':method,'stage':route['stage'],'sha256':route.get('sha256'),'input_bytes':len(route['raw']),'durations_ns':durations,'code':code,'feature_parity':True,'prediction_parity':True}
                        atomic_json(path,record);units.append(record)
                        print(json.dumps({k:record[k] for k in ('evaluation','case','method','id','stage')},ensure_ascii=False),flush=True)
                    except Exception as e:
                        signal.alarm(0);error={'status':'FAIL','evaluation':ev,'case':case,'id':cid,'method':method,'error':repr(e),'code':code}
                        atomic_json(path.with_suffix(f'.{time.time_ns()}.failure.json'),error);failures.append(error)
                        # Stop scientific parity failures rather than replacing inputs.
                        if 'parity' in str(e) or isinstance(e,TimeoutError) or 'worker' in str(e):raise
    atomic_json(OUT/('circuit_canary_receipt.json' if canary else f'circuit_session_{session}_receipt.json'),{'status':'PASS' if not failures else 'PARTIAL_WITH_FAILURES','new_completed':len(units),'failures':failures,'code':code})
def sha_id(cid):
    import hashlib
    return hashlib.sha256(cid.encode()).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--canary',action='store_true');p.add_argument('--session',type=int,default=1);a=p.parse_args()
    with (ART/'.inference_timing.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main(a.canary,a.session)
