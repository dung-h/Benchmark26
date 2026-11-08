"""Keep sparse graph encodings in RAM; construct normalized model inputs on CUDA."""
import numpy as np
import torch
from torch_geometric.data import Data
FIELDS=('node_type','wire0','wire1','t1_0','t2_0','t1_1','t2_1','node_index')
class CompactCache:
 def __init__(self,root,mapping,mean,std,valid):
  self.mapping=mapping;self.records={};self.bytes=0;self.hits=0;self.misses=0
  self.mean=torch.as_tensor(mean,device='cuda');self.div=torch.as_tensor(np.where(valid,std,1).astype(np.float32),device='cuda');self.invalid=torch.as_tensor(~valid,device='cuda')
  for key in sorted(set(mapping.values())):
   with np.load(root/(key+'.npz'),allow_pickle=False) as z:r={k:z[k].copy() for k in FIELDS+('edge_index',)}
   self.records[key]=r;self.bytes+=sum(a.nbytes for a in r.values())
 def available(self):
  from pathlib import Path
  return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 def batch(self,ids,X,index):
  rs=[self.records[self.mapping[c]] for c in ids];self.hits+=len(ids);counts=[len(r['node_type']) for r in rs];n=sum(counts);x=torch.zeros((n,178),dtype=torch.float32,device='cuda');rows=torch.arange(n,device='cuda');types=torch.from_numpy(np.concatenate([r['node_type'] for r in rs]).astype(np.int64)).cuda();x[rows,types]=1
  for key in ('wire0','wire1'):
   w=torch.from_numpy(np.concatenate([r[key] for r in rs]).astype(np.int64)).cuda();v=w!=255;x[rows[v],46+w[v]]=1
  for col,key in enumerate(('t1_0','t2_0','t1_1','t2_1','node_index'),173):x[:,col]=torch.from_numpy(np.concatenate([r[key] for r in rs]).astype(np.float32)).cuda()
  x.sub_(self.mean).div_(self.div);x[:,self.invalid]=0
  edges=[];offset=0
  for r,cnt in zip(rs,counts):edges.append(r['edge_index'].astype(np.int64)+offset);offset+=cnt
  edge=torch.from_numpy(np.concatenate(edges,axis=1)).cuda();batch=torch.repeat_interleave(torch.arange(len(ids),device='cuda'),torch.tensor(counts,device='cuda'));gf=torch.from_numpy(X[[index[c] for c in ids]]).cuda()
  return Data(x=x,edge_index=edge,batch=batch,global_features=gf)
