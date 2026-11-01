#!/usr/bin/env python3
"""Independent CSV-level arithmetic replay, without importing analysis_core."""
from __future__ import annotations
import collections
import datetime
import json
import math
from pathlib import Path
import numpy as np
from core import OUT, ROOT, rows, sha, atomic_json

DEST=OUT/'analysis_v1'

def close(actual,expected):
    if expected is None:
        assert actual in ('',None)
    else:
        assert math.isclose(float(actual),float(expected),rel_tol=1e-11,abs_tol=1e-11),(actual,expected)

def independent(records):
    y=np.array([float(r['target_seconds']) for r in records]); p=np.array([float(r['predicted_seconds']) for r in records])
    ae=abs(p-y); groups=collections.defaultdict(list); sources=collections.defaultdict(list)
    for r,e in zip(records,ae): groups[r['group_id']].append(e); sources[r['source_id']].append(e)
    denominator=sum((y-y.mean())**2)
    l2=None
    if (p>0).all():
        ly=np.log1p(y); lp=np.log1p(p); den=sum((ly-ly.mean())**2)
        if den>0 and len(y)>1: l2=1-sum((ly-lp)**2)/den
    return {'rows':len(y),'groups':len(groups),'sources':len(sources),
            'pooled_mae_seconds':np.mean(ae),'group_balanced_mae_seconds':np.mean([np.mean(v) for v in groups.values()]),
            'source_macro_mae_seconds':np.mean([np.mean(v) for v in sources.values()]),
            'medae_seconds':np.median(ae),'rmse_seconds':np.sqrt(np.mean((p-y)**2)),
            'raw_seconds_r2':1-sum((p-y)**2)/denominator if denominator>0 and len(y)>1 else None,
            'log1p_seconds_r2':l2,'bias_seconds':np.mean(p-y),
            'p90_absolute_error_seconds':np.quantile(ae,.9),'p99_absolute_error_seconds':np.quantile(ae,.99),
            'max_absolute_error_seconds':np.max(ae),'negative_predictions':sum(p<0),'zero_predictions':sum(p==0),
            'overprediction_fraction':np.mean(p>y),'underprediction_fraction':np.mean(p<y),
            'target_mean_seconds':np.mean(y),'target_std_seconds':np.std(y),'prediction_mean_seconds':np.mean(p)}

def main():
    receipt=DEST/'independent_statistics_receipt.json'
    if receipt.exists(): raise RuntimeError('Refuse receipt overwrite')
    checks=0
    saved=rows(DEST/'reduced_predictions.csv'); seedrows=rows(DEST/'per_seed_predictions.csv')
    assert len(saved)==61974 and len(seedrows)==123948
    panel={r['canonical_observation_id']:r for r in rows(OUT/'cohort/panel.csv')}
    targets={r['canonical_observation_id']:float(r['target_seconds']) for r in rows(OUT/'cohort/targets.csv')}
    groups=collections.defaultdict(list)
    lookup={}
    for r in saved:
        key=(r['evaluation'],r['treatment'],r['canonical_observation_id'])
        assert key not in lookup; lookup[key]=r
        cid=r['canonical_observation_id']
        assert float(r['target_seconds'])==targets[cid]
        for k in ('group_id','source_id','backend','shots'): assert r[k]==panel[cid][k]
        for field,expected in [('signed_error_seconds',float(r['predicted_seconds'])-targets[cid]),
                               ('absolute_error_seconds',abs(float(r['predicted_seconds'])-targets[cid]))]: close(r[field],expected)
        groups[r['evaluation'],r['treatment']].append(r)
        checks+=7
    seedgroups=collections.defaultdict(dict)
    for r in seedrows:
        key=(r['evaluation'],r['treatment'],r['canonical_observation_id'])
        assert int(r['seed']) not in seedgroups[key]
        seedgroups[key][int(r['seed'])]=float(r['predicted_seconds'])
    for key,vals in seedgroups.items():
        assert set(vals)=={42,1234,31415}
        assert float(lookup[key]['predicted_seconds'])==sorted(vals.values())[1]
        checks+=2
    for row in rows(DEST/'input_prediction_hashes.csv'):
        assert sha(ROOT/row['path'])==row['sha256']; checks+=1
    for row in rows(DEST/'pooled_metrics.csv')+rows(DEST/'sliced_metrics.csv'):
        recs=groups[row['evaluation'],row['treatment']]
        if 'scope' in row:
            field={'source':'source_id','backend':'backend','outer_fold':'outer_fold'}[row['scope']]
            recs=[r for r in recs if r[field]==row['scope_value']]
        for k,v in independent(recs).items(): close(row[k],v); checks+=1
    contrasts=rows(DEST/'paired_comparisons.csv')
    for row in contrasts:
        ev,scope=row['evaluation'],row['scope']
        a,b=row['treatment'],row['control']
        first=groups[ev,a]; second=groups[ev,b]
        if ev=='E3':
            first=[r for r in first if r['backend']==scope]; second=[r for r in second if r['backend']==scope]
        ga,gb=collections.defaultdict(list),collections.defaultdict(list); gs={}
        for r in first: ga[r['group_id']].append(float(r['absolute_error_seconds'])); gs[r['group_id']]=r['source_id']
        for r in second: gb[r['group_id']].append(float(r['absolute_error_seconds']))
        go=sorted(ga); assert set(go)==set(gb)
        diff=np.array([np.mean(ga[g])-np.mean(gb[g]) for g in go])
        npz=DEST/'bootstrap'/f'{ev}_{scope}.npz'
        assert sha(npz)==row['bootstrap_draws_sha256']
        with np.load(npz,allow_pickle=False) as artifact:
            assert list(artifact['groups'])==go
            labels=np.array([gs[g] for g in go]); rng=np.random.default_rng(42)
            strata=[np.where(labels==s)[0] for s in sorted(set(labels))] if ev!='E3' else [np.arange(len(go))]
            replay=np.concatenate([rng.choice(ix,(10000,len(ix)),replace=True) for ix in strata],axis=1)
            assert np.array_equal(artifact['draws'],replay)
        dist=np.mean(diff[replay],axis=1)
        close(row['delta_group_balanced_mae_seconds'],np.mean(diff))
        close(row['ci95_lower_seconds'],np.percentile(dist,2.5)); close(row['ci95_upper_seconds'],np.percentile(dist,97.5))
        pct=.3125 if ev!='E3' else .0625
        close(row['adjusted_ci_lower_seconds'],np.percentile(dist,pct)); close(row['adjusted_ci_upper_seconds'],np.percentile(dist,100-pct))
        close(row['empirical_bootstrap_fraction_delta_below_zero'],np.mean(dist<0))
        assert len(dist)==int(row['valid_replicates'])==10000
        checks+=9
    # Independently replay all individual-seed metric summaries.
    for row in rows(DEST/'seed_metrics.csv'):
        recs=[r for r in seedrows if r['evaluation']==row['evaluation'] and r['treatment']==row['treatment'] and r['seed']==row['seed']]
        if row.get('outer_fold','')!='': recs=[r for r in recs if r['outer_fold']==row['outer_fold']]
        for k,v in independent(recs).items(): close(row[k],v); checks+=1
    from train_neural import protocol_lock
    _,locksha=protocol_lock()
    output={'status':'PASS','checks':checks,'prediction_rows':len(saved),'per_seed_rows':len(seedrows),
            'paired_contrasts':len(contrasts),'bootstrap_replicates_per_contrast':10000,'bootstrap_seed':42,
            'all_saved_prediction_hashes_unchanged':True,'execution_lock_sha256':locksha,
            'metrics_and_bootstrap_independently_replayed':True,'original_targets_and_group_metadata_aligned':True,
            'no_fit_training_simulation_or_clipping':True,'script_sha256':sha(Path(__file__)),
            'aggregation_receipt_sha256':sha(DEST/'aggregation_receipt.json'),
            'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(receipt,output); print(json.dumps(output,indent=2))

if __name__=='__main__': main()
