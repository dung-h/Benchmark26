#!/usr/bin/env python3
"""Compute sparse per-graph moments for later train-only physical-node scaling."""
import os,sys,json,multiprocessing,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
O=Path(os.environ['B26_RUN_DIR']);source=O/'physical_inputs';dest=O/'physical_moments';dest.mkdir(exist_ok=True)
def work(path):
 out=dest/(path.stem+'.json')
 if out.exists():return 'cached'
 with np.load(path,allow_pickle=False) as z:
  n=len(z['node_type']);s=np.zeros(178,np.float64);ss=np.zeros(178,np.float64)
  types=z['node_type'].astype(np.int64);s[:46]=np.bincount(types,minlength=46)[:46];ss[:46]=s[:46]
  w0=z['wire0'].astype(np.int64);w1=z['wire1'].astype(np.int64);w=np.concatenate([w0[w0!=255],w1[(w1!=255)&(w1!=w0)]]);s[46:173]=np.bincount(w,minlength=127)[:127];ss[46:173]=s[46:173]
  for j,k in enumerate(('t1_0','t2_0','t1_1','t2_1','node_index'),173):
   a=z[k].astype(np.float64);s[j]=a.sum();ss[j]=np.dot(a,a)
 d={'count':n,'sum':s.tolist(),'sumsq':ss.tolist()};tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d)+'\n');os.replace(tmp,out);return 'done'
if __name__=='__main__':
 files=sorted(source.glob('*.npz'));assert len(files)==1438;t=time.monotonic()
 with multiprocessing.get_context('spawn').Pool(4) as pool:
  for i,_ in enumerate(pool.imap_unordered(work,files,chunksize=1),1):
   if i%100==0:print('moments',i,'/',len(files),'s',round(time.monotonic()-t),flush=True)
 (O/'physical_moments_complete.json').write_text(json.dumps({'status':'COMPLETE','graphs':len(files),'elapsed_seconds':time.monotonic()-t})+'\n')
