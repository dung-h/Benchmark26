"""Sequential cost-measurement queue: wait for prerequisite measurements, validate a circuit-to-prediction trial, then measure the remaining sessions."""

import os as _b26_os
from pathlib import Path
import os,sys,time,json,subprocess,signal
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/full_cost_v1'/'circuit_qiskit25_rpc_v1'
CODE=Path(__file__).parent

def status(value):
    p=OUT/'queue_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def guarded(cmd):
    p=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
    while p.poll() is None:
        try:
            rss=tree_rss(p.pid)
            if rss>8*1024**3:
                os.killpg(p.pid,signal.SIGTERM);p.wait();raise RuntimeError('8GiB RSS guard terminated own worker')
        except FileNotFoundError:pass
        time.sleep(.5)
    if p.returncode:raise RuntimeError(f'worker exit={p.returncode}: {cmd}')
def tree_pids(pid):
    values=[pid]
    try:
        for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():values.extend(tree_pids(int(child)))
    except FileNotFoundError:pass
    return values
def tree_rss(pid):
    total=0
    for child in tree_pids(pid):
        try:total+=int(next(x.split()[1] for x in Path(f'/proc/{child}/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
        except (FileNotFoundError,StopIteration):pass
    return total
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    adopt=int(sys.argv[1]) if len(sys.argv)>1 else None
    status({'status':'C_CANARY_RUNNING','adopted_pid':adopt})
    if adopt:
        start=time.time()
        while not (OUT/'circuit_canary_receipt.json').exists():
            try:
                state=Path(f'/proc/{adopt}/stat').read_text().split()[2]
                if state=='Z':raise RuntimeError('adopted canary exited without receipt')
            except FileNotFoundError:raise RuntimeError('adopted canary exited without receipt')
            if tree_rss(adopt)>8*1024**3:
                for pid in reversed(tree_pids(adopt)):
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                raise RuntimeError('8GiB process-tree RSS guard')
            if time.time()-start>7200:raise RuntimeError('canary wait exceeded2h')
            time.sleep(.5)
    else:guarded([sys.executable,str(CODE/'run_circuit.py'),'--canary'])
    gate=json.loads((OUT/'circuit_canary_receipt.json').read_text())
    if gate['status']!='PASS':raise RuntimeError('C canary incomplete: inspect failures')
    # Conservative workload projection from the median/largest input canaries.
    import csv
    with (Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_expansion_v1/cohort/panel.csv').open() as f:
        panel={r['canonical_observation_id']:r for r in csv.DictReader(f)}
    sys.path.insert(0,str(ROOT/'research/postfreeze_qpu_workflow_mean_cost_v1'))
    from common import load_split
    by={}
    for path in (OUT/'circuit_canary').rglob('*.json'):
        if 'failure' in path.name:continue
        rec=json.loads(path.read_text());key=(rec['evaluation'],rec['case'],rec['method'],panel[rec['id']]['source_id'])
        by.setdefault(key,[]).extend(rec['durations_ns'])
    projection=0
    for ev in ('E1','E2','E3'):
        for case,ids in load_split(ev).items():
            for method in ('M0','M1','G0','G1','R0','C0'):
                for cid in ids:
                    values=by.get((ev,case,method,panel[cid]['source_id']))
                    if not values:raise RuntimeError('projection missing source/case/method')
                    projection+=max(values)*8/1e9 # two sessions x (warmup + three timed)
    (OUT/'canary_workload_projection.json').write_text(json.dumps({'conservative_seconds':projection,'budget_seconds':48*3600},indent=2))
    if projection>48*3600:raise RuntimeError('projected C workload exceeds48h; inspect projection before budget decision')
    for session in (1,2):
        status({'status':'C_MEASURING','session':session})
        guarded([sys.executable,str(CODE/'run_circuit.py'),'--session',str(session)])
    status({'status':'C_D_MEASUREMENTS_COMPLETE_REQUIRES_AGGREGATION_AND_ANALYTICAL_ROUTES'})
if __name__=='__main__':
    try:main()
    except Exception as e:status({'status':'STOPPED_WITH_ERROR','error':repr(e)});raise
