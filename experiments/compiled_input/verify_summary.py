#!/usr/bin/env python3
"""Reconstruct exported group/pooled MAE and paired intervals from anonymous group errors."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[2]/'results/compiled_input');ap.add_argument('--receipt',type=Path);a=ap.parse_args();root=a.results
 rows=lambda p:list(csv.DictReader((root/p).open()));errors=rows('group_errors.csv');metrics=rows('metrics.csv');contrasts=rows('contrasts.csv');n=0
 for m in metrics:
  rs=[r for r in errors if r['evaluation']==m['evaluation'] and r['arm']==m['arm'] and (m['source']=='all' or r['source']==m['source'])];assert len(rs)==int(m['groups']);values=np.array([float(r['group_mae_s']) for r in rs]);weights=np.array([int(r['observations']) for r in rs]);assert weights.sum()==int(m['observations']);assert abs(values.mean()-float(m['group_balanced_mae_s']))<1e-12;assert abs(np.average(values,weights=weights)-float(m['pooled_mae_s']))<1e-12;n+=2
 for ev in ('E1','E2'):
  rs=[r for r in errors if r['evaluation']==ev];groups=sorted({int(r['group_index']) for r in rs});assert len(groups)==203
  source={int(r['group_index']):r['source'] for r in rs};lookup={(r['arm'],int(r['group_index'])):float(r['group_mae_s']) for r in rs};rng=np.random.default_rng(42);indices=[]
  for s in sorted(set(source.values())):
   gs=[g for g in groups if source[g]==s];indices.append(np.array(gs)[rng.integers(0,len(gs),size=(10000,len(gs)))])
  picks=np.concatenate(indices,axis=1)
  for row in [r for r in contrasts if r['evaluation']==ev]:
   delta=np.array([lookup[row['first'],g]-lookup[row['second'],g] for g in groups]);boot=delta[picks].mean(axis=1);level=float(row['ci_level']);ci=np.quantile(boot,[(1-level)/2,1-(1-level)/2]);assert abs(delta.mean()-float(row['difference_gb_mae_s']))<1e-12;assert np.max(abs(ci-np.array([float(row['ci_low_s']),float(row['ci_high_s'])])))<1e-12;n+=3
 receipt={'status':'PASS','checks':n,'scope':'Group/pooled MAE and fourteen paired whole-group confidence intervals from anonymous aggregates. Does not reproduce checkpoints, raw correlation metrics, training or latency.','inputs_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ('group_errors.csv','metrics.csv','contrasts.csv')}}
 if a.receipt:a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
