"""Lock a cold-service manifest, validate first launch, then sequential 600 launches."""
import os,sys,json,pickle,time,subprocess,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/postfreeze_qpu_workflow_mean_cost_v1'))
from common import ART,SRC,atomic_json,sha,stable,load_split,METHODS,SEEDS
OUT=ART/'full_cost_v1';CODE=Path(__file__).parent

def prepare():
    from loaders import StudyInputs
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True)
    payloads=[]
    for ev in ('E1','E2','E3'):
        inputs=StudyInputs(ev)
        for case,ids in inputs.test_by_case.items():
            cid=sorted(ids)[0]
            transform=json.loads((SRC/'training'/ev/f'fold_{case}'/'transform.json').read_text())
            for method in METHODS:
                p={'method':method,'cid':cid,'request':inputs.raw_request(cid,method),'transform':transform}
                if method.startswith(('M','G')):
                    p['checkpoints']=[str(SRC/'training'/ev/f'fold_{case}'/method/f'seed_{s}'/'checkpoint.pt') for s in SEEDS]
                    expected={}
                    from common import rows
                    for seed in SEEDS:
                        pred=next(r for r in rows(SRC/'training'/ev/f'fold_{case}'/method/f'seed_{seed}'/'predictions.csv') if r['canonical_observation_id']==cid)
                        expected[str(seed)]=float(pred['predicted_seconds'])
                    import numpy as np
                    reference={'prediction':float(np.median(list(expected.values()))),'seeds':expected}
                else:
                    from common import rows
                    folder=SRC/'classical'/ev/f'fold_{case}'
                    name='regression' if method=='R0' else 'constant'
                    pred=next(r for r in rows(folder/f'{name}_predictions.csv') if r['canonical_observation_id']==cid)
                    reference={'prediction':float(pred['predicted_seconds'])}
                    if method=='R0':p['state']=str(ART/'recovered_classical'/ev/f'fold_{case}'/'estimator.joblib')
                    else:p['constant_manifest']=str(folder/'constant_manifest.json')
                path=OUT/'payloads'/f'{ev}_{case}_{method}.pkl'
                data=pickle.dumps(p,protocol=5)
                if path.exists() and path.read_bytes()!=data:raise RuntimeError('payload changed')
                path.write_bytes(data)
                payloads.append({'evaluation':ev,'case':case,'method':method,'cid':cid,'path':str(path),'sha256':sha(path),'reference':reference})
    manifest={'clock':'D','definition':'fresh Python process including imports, CUDA init where needed, full original selected checkpoint load, raw-feature payload load, first Clock-B-equivalent request; OS page cache not reset; excludes QASM extraction','repetitions':5,'population':'one prespecified lexicographically first heldout ID per case/method; 20 cases x 6 methods; not full population latency','payloads':payloads,'code':{str(p):sha(p) for p in CODE.glob('*.py')},'a_b_manifest_sha256':sha(ART/'v2/timing_manifest.json'),'no_training':True}
    manifest['fingerprint']=stable(manifest)
    path=OUT/'cold_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('manifest changed')
    atomic_json(path,manifest)

def execute(canary):
    from timing import check_prediction
    manifest=json.loads((OUT/'cold_manifest.json').read_text())
    for path,digest in manifest['code'].items():
        if sha(Path(path))!=digest:raise RuntimeError('code changed')
    records=OUT/('cold_canary' if canary else 'cold_records');records.mkdir(exist_ok=True)
    payloads=manifest['payloads']
    if canary:
        payloads=[next(p for p in payloads if p['method']==m) for m in METHODS]
    for p in payloads:
        if sha(Path(p['path']))!=p['sha256']:raise RuntimeError('payload changed')
        for repeat in range(1 if canary else 5):
            path=records/f"{p['evaluation']}_{p['case']}_{p['method']}_{repeat}.json"
            if path.exists():continue
            start=time.perf_counter_ns()
            r=subprocess.run([sys.executable,str(CODE/'cold_worker.py'),p['path']],capture_output=True,text=True,timeout=600,cwd=ROOT,check=True)
            elapsed=time.perf_counter_ns()-start
            value=json.loads(r.stdout.strip().splitlines()[-1]);check_prediction(value['prediction'],p['reference'],p['method'],value['seeds'])
            atomic_json(path,{'status':'PASS','fingerprint':manifest['fingerprint'],**{k:p[k] for k in ('evaluation','case','method','cid')},'repeat':repeat,'process_wall_ns':elapsed,**value})
            print(json.dumps({'phase':'D_canary' if canary else 'D','evaluation':p['evaluation'],'case':p['case'],'method':p['method'],'repeat':repeat,'wall_seconds':elapsed/1e9}),flush=True)
    expected=len(payloads)*(1 if canary else 5)
    assert len(list(records.glob('*.json')))==expected
    atomic_json(OUT/('cold_canary_receipt.json' if canary else 'cold_receipt.json'),{'status':'PASS','fingerprint':manifest['fingerprint'],'launches':expected,'prediction_parity':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','canary','measure']);args=a.parse_args()
    if args.phase=='prepare':prepare()
    else:execute(args.phase=='canary')
