"""Bounded scalar analytical applications; no transpilation or circuit execution."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, ctypes, hashlib, importlib.metadata, json, math, pickle
import platform, resource, subprocess, sys, sqlite3
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/postfreeze_notads_hyperq_track_a_v1'
PANEL=ROOT/'artifacts/real_qpu/qualified_logical_panel/panel.csv'
SPLITS=PANEL.parent/'outer_splits.csv'
REP=ROOT/'artifacts/benchmark_v3/real_qpu/compiled_unified'
REG=ROOT/'artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv'
TIMING=ROOT/'artifacts/postfreeze_track_a_transpiled_representation_v1/features/timing_features.csv'
CANON=ROOT/'artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv'
NATIVE=ROOT/'artifacts/real_qpu/qonductor_native_features/feature_rows.csv'
DB=ROOT.parent/'Qonductor-SC25/data/database/quantum_scheduler.db'
FIELDS=['depth_1q','depth_2q','unitary_depth','paper_kappa1','paper_kappa2',
        'notads_code','notads_paper','hyperq_full_seconds','hyperq_unitary_seconds','measurement_count']

def read(path):
    csv.field_size_limit(100_000_000)
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def idsha(ids):return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def write(name,rows):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def guard():
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss>8*1024**3:raise RuntimeError('RSS_above_8GiB')
    return rss

def tables():
    panel=read(PANEL);ids=[r['canonical_observation_id'] for r in panel]
    assert len(panel)==len(set(ids))==4515
    assert len({r['group_id'] for r in panel})==165
    assert Counter(r['source_id'] for r in panel)=={'mali_real_qpu':340,'qonductor_single_circuit_ibm':230,'qpack_mcp':3945}
    splits=read(SPLITS);split={r['canonical_observation_id']:r for r in splits}
    assert len(split)==len(splits)==4515 and set(split)==set(ids)
    groups=defaultdict(set)
    for r in panel:
        s=split[r['canonical_observation_id']]
        assert s['group_id']==r['group_id']
        groups[r['group_id']].add(int(s['outer_fold']))
    assert all(len(v)==1 for v in groups.values()) and {next(iter(v)) for v in groups.values()}==set(range(5))
    reps={r['canonical_row_id']:r for r in read(REP/'representation_rows.csv') if r['canonical_row_id'] in set(ids)}
    assert set(reps)==set(ids)
    canon={r['canonical_row_id']:r for r in read(CANON) if r['canonical_row_id'] in set(ids)}
    native={r['canonical_row_id']:r for r in read(NATIVE) if r['canonical_row_id'] in set(ids)}
    mismatches=[];db=sqlite3.connect(f'file:{DB}?mode=ro&immutable=1',uri=True)
    for r in panel:
        rid=r['canonical_observation_id'];rep=reps[rid];c=canon[rid];n=native[rid]
        native_hash=rep['source_qasm_sha256'] if rep['lifecycle_stage']=='submitted_physical' else rep['compiled_qasm3_sha256']
        assert r['compiled_qasm_sha256']==native_hash
        assert r['backend'].replace('ibmq_','ibm_')==rep['backend_canonical']
        shots=int(float(r['shots']))
        assert shots==int(float(c['shots']))==int(float(n['shots'])) and float(n['circuit_count'])==1
        dbshots='not_applicable'
        if r['source_id']=='qonductor_single_circuit_ibm':
            job=json.loads(c['raw_row_json'])['job_id']
            dbshots=db.execute('SELECT shots FROM Job WHERE id=?',(job,)).fetchone()[0]
            assert shots==dbshots
        if shots!=int(float(rep['shots'])):
            mismatches.append({'canonical_observation_id':rid,'panel_shots':shots,'canonical_shots':c['shots'],
                'native_feature_shots':n['shots'],'database_shots':dbshots,'representation_table_shots':rep['shots'],
                'applied_shots_source':'frozen panel; independently matches canonical/native/SQL','graph_byte_identity':'PASS'})
    db.close()
    assert len(mismatches)==82 and all(x['database_shots']!='not_applicable' for x in mismatches)
    write('representation_shots_discrepancies.csv',mismatches)
    return panel,split,reps

def kernel():
    so=OUT/'path_kernel.so'
    if not so.exists():subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(CODE/'path_kernel.cpp'),'-o',str(so)],check=True)
    dll=ctypes.CDLL(str(so));fn=dll.paths
    ptr=np.ctypeslib.ndpointer
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ptr(np.uint8,flags='C_CONTIGUOUS'),ptr(np.int32,flags='C_CONTIGUOUS'),ptr(np.int32,flags='C_CONTIGUOUS'),ptr(np.uint8,flags='C_CONTIGUOUS'),ptr(np.float64,flags='C_CONTIGUOUS'),ptr(np.float64,flags='C_CONTIGUOUS')]
    fn.restype=ctypes.c_int
    def call(g,durations):
        n=int(g['operation_count']);arity=np.frombuffer(g['arity'],dtype=np.uint8)
        q0=np.frombuffer(g['q0'],dtype=np.int32);q1=np.frombuffer(g['q1'],dtype=np.int32)
        measure=np.array([x=='measure' for x in g['names']],dtype=np.uint8)
        output=np.empty(10,dtype=np.float64)
        code=fn(n,int(g['num_qubits']),arity,q0,q1,measure,np.ascontiguousarray(durations,dtype=np.float64),output)
        if code:raise RuntimeError(f'path_kernel_failure_{code}')
        return dict(zip(FIELDS,output.tolist()))
    return call

def reference(g,durations):
    # Independent explicit graph recurrence, not a last-use resource simulation.
    n=g['operation_count'];arity=np.frombuffer(g['arity'],dtype=np.uint8)
    measure=np.array([x=='measure' for x in g['names']])
    src=np.frombuffer(g['edge_src'],dtype=np.int32);dst=np.frombuffer(g['edge_dst'],dtype=np.int32)
    assert len(src)==len(dst) and np.all(src<dst)
    predecessors=[[] for _ in range(n)]
    for s,t in zip(src,dst):predecessors[int(t)].append(int(s))
    vals=np.zeros((n,5));has2=set()
    for i in range(n):
        start=np.max(vals[predecessors[i]],axis=0) if predecessors[i] else np.zeros(5)
        inc=np.array([not measure[i] and arity[i]==1,not measure[i] and arity[i]==2,not measure[i],durations[i],0 if measure[i] else durations[i]])
        vals[i]=start+inc
        if not measure[i] and arity[i]==2:has2.add(int(vals[i,2]))
    d1,d2,depth,full,unit=np.max(vals,axis=0)
    return dict(zip(FIELDS,[d1,d2,depth,depth-len(has2),len(has2),d1+10*d2,depth+9*len(has2),full,unit,int(measure.sum())]))

def backends():
    from qiskit_ibm_runtime import fake_provider
    assert importlib.metadata.version('qiskit')=='2.5.2'
    assert importlib.metadata.version('qiskit-ibm-runtime')=='0.49.0'
    rows=read(REG);objects={};pins={}
    for r in rows:
        for pkey,hkey in [('configuration_path','configuration_sha256'),('properties_path','properties_sha256'),('class_module_path','class_module_sha256')]:
            p=Path(r[pkey]);assert sha(p)==r[hkey],p;pins[str(p)]=r[hkey]
        objects[r['snapshot_id']]=getattr(fake_provider,r['fake_backend_class'])()
    return objects,pins

def duration_values(g,backend):
    n=g['operation_count'];q0=np.frombuffer(g['q0'],dtype=np.int32);q1=np.frombuffer(g['q1'],dtype=np.int32)
    names=np.asarray(g['names']);values=np.full(n,np.nan)
    for name in np.unique(names):
        assert name in {'rz','sx','x','cx','ecr','measure'},name
        mask=names==name;indices=np.flatnonzero(mask)
        op=backend.target.operation_from_name(str(name));arity=op.num_qubits
        shape=(g['num_qubits'],) if arity==1 else (g['num_qubits'],g['num_qubits'])
        lut=np.full(shape,np.nan)
        for qs in backend.target[str(name)]:
            v=float(backend.target.durations().get(str(name),qs,unit='s'))
            assert math.isfinite(v) and v>=0
            lut[qs[0] if arity==1 else qs]=v
        values[indices]=lut[q0[indices]] if arity==1 else lut[q0[indices],q1[indices]]
    assert np.all(np.isfinite(values)) and np.all(values>=0)
    return values

def check_qiskit_depths(g,features):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(g['num_qubits'])
    for name,a,x,y in zip(g['names'],g['arity'],g['q0'],g['q1']):
        if name=='measure':continue
        if a==1:qc.h(x)
        else:qc.cx(x,y)
    assert qc.depth()==features['unitary_depth']
    assert qc.depth(lambda i:i.operation.num_qubits==1)==features['depth_1q']
    assert qc.depth(lambda i:i.operation.num_qubits==2)==features['depth_2q']

def extract():
    panel,split,reps=tables()
    # Target values deliberately removed before structural calculation.
    safe=[{k:v for k,v in r.items() if k!='target_seconds'} for r in panel]
    del panel
    objects,snapshot_pins=backends();call=kernel()
    hashes=json.loads((REP/'graph_file_hashes.json').read_text());cache={};files={}
    pairs=sorted({(r['graph_file'],r['target_snapshot_id']) for r in reps.values()})
    canaries={}
    for name,sid in pairs:
        size=(REP/'graphs'/name).stat().st_size
        if sid not in canaries or size<canaries[sid][0]:canaries[sid]=(size,name)
    largest=max(pairs,key=lambda k:(REP/'graphs'/k[0]).stat().st_size)
    validated=[]
    for index,(name,sid) in enumerate(pairs,1):
        path=REP/'graphs'/name;expected=hashes[name]
        assert sha(path)==expected,name
        files[str(path.relative_to(ROOT))]=expected
        g=pickle.load(path.open('rb'));assert g['format']=='s71-operation-dag-v1'
        assert len(g['names'])==len(g['arity'])==g['operation_count']
        ds=duration_values(g,objects[sid]);features=call(g,ds)
        check=(canaries[sid][1]==name or (name,sid)==largest)
        if check:
            ref=reference(g,ds)
            for k,v in features.items():assert math.isclose(v,ref[k],rel_tol=1e-10,abs_tol=1e-12),(name,k,v,ref[k])
            if canaries[sid][1]==name:check_qiskit_depths(g,features)
            validated.append({'graph_file':name,'snapshot_id':sid,'nodes':g['operation_count'],'explicit_DAG_replay':'PASS','qiskit_depth':'PASS' if canaries[sid][1]==name else 'not_requested_for_largest'})
        cache[name,sid]=features
        if index%20==0 or index==len(pairs):
            dump('progress.json',{'stage':'feature_extraction','completed':index,'assigned':len(pairs),'peak_rss_bytes':guard()})
            print(f'graphs {index}/{len(pairs)}',flush=True)
    timing={r['canonical_observation_id']:float(r['nominal_single_shot_duration_seconds']) for r in read(TIMING)}
    rows=[];maxdiff=0
    for r in safe:
        rid=r['canonical_observation_id'];rep=reps[rid];feature=cache[rep['graph_file'],rep['target_snapshot_id']]
        expected=timing[rid];actual=feature['hyperq_full_seconds']
        maxdiff=max(maxdiff,abs(actual-expected))
        assert math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-9),(rid,actual,expected)
        rows.append({'canonical_observation_id':rid,'source_id':r['source_id'],'group_id':r['group_id'],
            'outer_fold':int(split[rid]['outer_fold']),'backend':r['backend'],'shots':int(float(r['shots'])),
            'physical_hash':rep['compiled_or_submitted_digest'],'graph_file':rep['graph_file'],
            'graph_sha256':hashes[rep['graph_file']],'snapshot_id':rep['target_snapshot_id'],
            'structure_provenance':'EXACT_SUBMITTED_PHYSICAL' if rep['compiler_route']=='preserve_exact_submitted_physical_qasm_no_retranspile' else 'FROZEN_RECONSTRUCTION',
            'timing_provenance':rep['snapshot_tier'],**feature,'qiskit_saved_duration_seconds':expected})
    write('features.csv',rows)
    inputpaths=[PANEL,SPLITS,TIMING,REG,CANON,NATIVE,REP/'representation_rows.csv',REP/'graph_file_hashes.json',OUT/'PROTOCOL.md',OUT/'PROTOCOL_AMENDMENT_01.md',Path(__file__),CODE/'path_kernel.cpp']
    dump('feature_receipt.json',{'status':'PASS','rows':len(rows),'groups':165,'graph_snapshot_pairs':len(pairs),
        'explicit_DAG_canaries':validated,'qiskit_full_measurement_max_abs_difference_seconds':maxdiff,
        'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputpaths},
        'graph_sha256':files,'snapshot_pins':snapshot_pins,'feature_sha256':sha(OUT/'features.csv'),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'python':platform.python_version(),'numpy':np.__version__,'qiskit':importlib.metadata.version('qiskit'),
        'qiskit_ibm_runtime':importlib.metadata.version('qiskit-ibm-runtime'),
        'target_values_used_in_feature_extraction':False,'peak_rss_bytes':guard(),
        'representation_shots_discrepancies':82,'panel_canonical_native_shots_equal':'4515/4515 PASS',
        'qonductor_SQL_shots_equal':'230/230 PASS','shots_policy':'Use frozen panel; never cached representation shots'})

def metric(rows,method,scope):
    y=np.array([float(r['target_seconds']) for r in rows]);pred=np.array([float(r['prediction_seconds']) for r in rows])
    errors=pred-y;ae=np.abs(errors);groups=defaultdict(list);sources=defaultdict(list)
    for r,e in zip(rows,ae):groups[r['group_id']].append(e);sources[r['source_id']].append(e)
    sst=float(np.sum((y-y.mean())**2))
    return {'method':method,'scope':scope,'rows':len(rows),'groups':len(groups),
        'mae_seconds':float(ae.mean()),'group_balanced_mae_seconds':float(np.mean([np.mean(v) for v in groups.values()])),
        'source_macro_mae_seconds':float(np.mean([np.mean(v) for v in sources.values()])),
        'rmse_seconds':float(np.sqrt(np.mean(errors**2))),'raw_r2':1-float(np.sum(errors**2))/sst if sst>0 else '',
        'bias_seconds':float(errors.mean()),'median_absolute_error_seconds':float(np.median(ae)),
        'p90_absolute_error_seconds':float(np.quantile(ae,.9)),'p99_absolute_error_seconds':float(np.quantile(ae,.99)),
        'max_absolute_error_seconds':float(ae.max())}

def score():
    panel,split,reps=tables();receipt=json.loads((OUT/'feature_receipt.json').read_text())
    assert receipt['status']=='PASS' and sha(OUT/'features.csv')==receipt['feature_sha256']
    for p,h in receipt['inputs_sha256'].items():assert sha(ROOT/p)==h,p
    feature={r['canonical_observation_id']:r for r in read(OUT/'features.csv')}
    assert set(feature)=={r['canonical_observation_id'] for r in panel}
    ids=[r['canonical_observation_id'] for r in panel];folds=np.array([int(split[i]['outer_fold']) for i in ids])
    y=np.array([float(r['target_seconds']) for r in panel]);shots=np.array([int(float(r['shots'])) for r in panel])
    assert np.all(np.isfinite(y)) and np.all(y>0) and np.all(shots>0)
    predictions={};fits=[];leakage=[]
    for route,field in [('notads_paper','notads_paper'),('notads_code','notads_code'),('hyperq_paper','hyperq_full_seconds')]:
        raw=np.array([float(feature[i][field]) for i in ids])
        for scale,x in [('raw',raw),('shots',raw*shots)]:
            method=f'{route}_{scale}_affine';pred=np.full(len(ids),np.nan)
            if route=='hyperq_paper':predictions[f'{route}_{scale}']=x
            for fold in range(5):
                train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
                tg={panel[j]['group_id'] for j in train};vg={panel[j]['group_id'] for j in test}
                assert not tg&vg
                matrix=np.column_stack([np.ones(len(train)),x[train]])
                coef,_,rank,_=np.linalg.lstsq(matrix,y[train],rcond=None)
                pred[test]=np.maximum(0,coef[0]+coef[1]*x[test])
                fits.append({'method':method,'fold':fold,'intercept':float(coef[0]),'slope':float(coef[1]),'rank':int(rank),
                    'train_rows':len(train),'test_rows':len(test),'train_id_sha256':idsha(ids[j] for j in train),'test_id_sha256':idsha(ids[j] for j in test)})
                leakage.append({'method':method,'fold':fold,'train_groups':len(tg),'test_groups':len(vg),'overlapping_groups':0})
            assert np.all(np.isfinite(pred));predictions[method]=pred
    predictions['qiskit_saved_shots_reference']=np.array([float(feature[i]['qiskit_saved_duration_seconds']) for i in ids])*shots
    allrows=[];metricrows=[];groups=sorted({r['group_id'] for r in panel});draws=np.random.default_rng(42).integers(0,len(groups),size=(10000,len(groups)))
    group_errors={}
    for method,pred in predictions.items():
        rows=[{'canonical_observation_id':ids[j],'source_id':r['source_id'],'group_id':r['group_id'],'outer_fold':int(folds[j]),
               'method':method,'target_seconds':float(y[j]),'prediction_seconds':float(pred[j]),'status':'predicted'} for j,r in enumerate(panel)]
        allrows.extend(rows);metricrows.append(metric(rows,method,'all'))
        for source in sorted({r['source_id'] for r in rows}):metricrows.append(metric([r for r in rows if r['source_id']==source],method,source))
        for fold in range(5):metricrows.append(metric([r for r in rows if r['outer_fold']==fold],method,f'fold_{fold}'))
        group_errors[method]=np.array([np.mean([abs(pred[j]-y[j]) for j,r in enumerate(panel) if r['group_id']==g]) for g in groups])
    paired=[]
    for a,b in [('notads_paper_shots_affine','notads_code_shots_affine'),('hyperq_paper_shots_affine','notads_code_shots_affine'),('hyperq_paper_shots','qiskit_saved_shots_reference')]:
        delta=group_errors[a]-group_errors[b];boot=delta[draws].mean(axis=1)
        paired.append({'method_a':a,'method_b':b,'delta_group_balanced_mae_seconds':float(delta.mean()),
            'ci95_low_seconds':float(np.quantile(boot,.025)),'ci95_high_seconds':float(np.quantile(boot,.975)),
            'replicates':10000,'seed':42,'resampling_unit':'structural group','inference':'descriptive; no confirmatory discovery claim'})
    write('predictions.csv',allrows);write('metrics.csv',metricrows);write('calibration_fits.csv',fits);write('leakage_receipts.csv',leakage);write('paired_comparisons.csv',paired)
    np.savez_compressed(OUT/'bootstrap_draws.npz',draws=draws,groups=np.array(groups))
    dump('completion_receipt.json',{'status':'PASS','observations':4515,'groups':165,'folds':5,'prediction_methods':len(predictions),
         'prediction_rows':len(allrows),'calibration_fits':len(fits),'feature_sha256':sha(OUT/'features.csv'),
         'no_neural_tree_training':True,'scalar_affine_fitting':'outer-train only; two coefficients; no selection',
         'no_QPU_or_simulator_jobs':True,'no_retranspilation':True,
         'output_sha256':{p:sha(OUT/p) for p in ['predictions.csv','metrics.csv','calibration_fits.csv','leakage_receipts.csv','paired_comparisons.csv','bootstrap_draws.npz']}})
    print(json.dumps([m for m in metricrows if m['scope']=='all'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['extract','score'],required=True);a=p.parse_args()
    if a.stage=='extract':extract()
    else:score()
