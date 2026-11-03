"""Fresh-process, first raw-feature request; no labels, training or search."""
import time
START = time.perf_counter_ns()
import os, sys, json, pickle, resource
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
payload = pickle.loads(Path(sys.argv[1]).read_bytes())
method = payload['method']
if method.startswith(('M','G')):
    sys.path.insert(0, str(ROOT/'legacy/feature_latency'))
    from loaders import StudyInputs, base
    from common import SEEDS
    import torch
    from shots_core import expand_global_projection
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    t = time.perf_counter_ns(); torch.cuda.init(); torch.cuda.synchronize()
    cuda_init_ns = time.perf_counter_ns()-t
    obj = object.__new__(StudyInputs)
    obj.transforms={'fold':payload['transform']}; obj.models={method:{}}
    cls, _ = base.load_pinned_model()
    load_start = time.perf_counter_ns()
    for seed,path in zip(SEEDS,payload['checkpoints']):
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        base.seed_all(seed)
        graph=method.startswith('G')
        variant='mali_full_features_graph' if graph else 'mali_full_features_mlp'
        model=base.make_model(cls,variant,len(payload['transform']['global']['retained_indices']),checkpoint_layers=graph)
        if graph:model.mask=model.mask.to('cuda:0')
        if method.endswith('1'):
            rng=base.capture_rng_state();expand_global_projection(model,torch);base.restore_rng_state(rng)
        model.load_state_dict(ckpt['best_model_state'],strict=True)
        obj.models[method][seed]=model.to('cuda:0').eval()
        del ckpt
    torch.cuda.synchronize(); load_ns=time.perf_counter_ns()-load_start
    prepared=obj.prepare_raw_B(method,payload['request'])
    with torch.inference_mode():pred,seeds,_=obj.predict_ready(method,payload['cid'],prepared)
    torch.cuda.synchronize()
else:
    import numpy as np
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    cuda_init_ns=0; seeds={};load_start=time.perf_counter_ns()
    if method=='R0':
        import joblib
        estimator=joblib.load(payload['state']);load_ns=time.perf_counter_ns()-load_start
        pred=float(estimator.predict(np.asarray(payload['request']['native5']).reshape(1,-1))[0])
    else:
        constant=json.loads(Path(payload['constant_manifest']).read_text())
        load_ns=time.perf_counter_ns()-load_start;pred=float(constant['outer_train_median_seconds'])
end=time.perf_counter_ns()
print(json.dumps({'prediction':float(pred),'seeds':{str(k):float(v) for k,v in seeds.items()},'worker_ns':end-START,'cuda_init_ns':cuda_init_ns,'state_load_ns':load_ns,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
