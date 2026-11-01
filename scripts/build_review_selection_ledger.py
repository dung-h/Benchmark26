#!/usr/bin/env python3
"""Explain inclusion/exclusion using retained evidence; do not rebuild data."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or ROOT / 'artifacts' in args.output.resolve().parents:
        parser.error('Use a new output outside frozen artifacts')
    study = ROOT / 'artifacts/postfreeze_qpu_workflow_mean_expansion_v1'
    panel = {r['canonical_observation_id']: r for r in read(study / 'cohort/panel.csv')}
    legacy = read(ROOT / 'artifacts/real_qpu/qualified_logical_panel/panel.csv')
    qualified = {r['canonical_observation_id']: r for r in read(ROOT / 'artifacts/postfreeze_qonductor_generator_version_semantics_v1/semantics_v2_legacy_parser/observation_semantic_results.csv') if r['version'] == '1.0.3'}
    screen = {r['canonical_observation_id']: r for r in read(ROOT / 'upstream/qonductor_recipe_screen.csv')}
    corpus = ROOT / 'artifacts/benchmark_v1/canonical_corpus_8767_20260926'
    result = []
    for r in legacy:
        if r['source_id'] == 'mali_real_qpu':
            cid = r['canonical_observation_id']
            result.append(dict(source='mali',source_record_id=cid,final_observation_id=cid,
                decision='INCLUDED_OBSERVATION',reason='RETAINED_PUBLISHED_LABEL_WITH_LOGICAL_INPUT',
                reason_evidence='qualified_logical_panel/panel.csv',group_id=panel[cid]['group_id']))
    for r in read(corpus / 'qonductor_single_circuit_ibm.csv'):
        cid = r['canonical_row_id']; evidence = qualified.get(cid)
        if evidence and evidence['semantic_status'] == 'PASS_PREPARED_READOUT_STATE':
            if cid not in panel:
                raise ValueError('Qualified row absent from final cohort: ' + cid)
            decision,reason='INCLUDED_OBSERVATION','PASS_PREPARED_READOUT_STATE_MQT_1_0_3'
        else:
            if cid in panel:
                raise ValueError('Final row lacks retained functional PASS: ' + cid)
            decision='EXCLUDED'
            reason=evidence['semantic_status'] if evidence else screen[cid]['recipe_test_status']
        result.append(dict(source='qonductor',source_record_id=cid,final_observation_id=cid if cid in panel else '',
            decision=decision,reason=reason,reason_evidence='observation_semantic_results.csv' if evidence else 'upstream/qonductor_recipe_screen.csv',group_id=panel.get(cid,{}).get('group_id','')))
    for r in read(corpus / 'qpack_mcp.csv'):
        workflow = json.loads(r['raw_row_json'])['workflow_id']
        cid = 'qpack_workflow_mean|' + hashlib.sha256(workflow.encode()).hexdigest()
        included = cid in panel
        result.append(dict(source='qpack',source_record_id=r['canonical_row_id'],final_observation_id=cid if included else '',
            decision='AGGREGATED_CONSTITUENT' if included else 'EXCLUDED',
            reason='UPSTREAM_TIMING_CONFIG_PARITY_WORKFLOW_MEAN' if included else 'WORKFLOW_NOT_IN_RECONCILED_45_UPSTREAM_SET',
            reason_evidence='workflow_final_parameters.csv',group_id=panel.get(cid,{}).get('group_id','')))
    represented = {r['final_observation_id'] for r in result if r['final_observation_id']}
    if len(result) != 8767 or represented != set(panel):
        raise ValueError('Selection ledger coverage differs from locked cohort')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    print(json.dumps({'status':'PASS_RETAINED_SELECTION_LEDGER','input_source_records':len(result),
        'final_observations':len(represented),'source_decisions':dict(collections.Counter(r['source']+'/'+r['decision'] for r in result)),
        'excluded_qonductor_reason_codes':dict(collections.Counter(r['reason'] for r in result if r['source']=='qonductor' and r['decision']=='EXCLUDED')),
        'scope':'Explains membership from retained tables and original screening statuses. Combined reason codes retain their original granularity; this does not re-execute candidate screening or prove logical equivalence.'},indent=2))


if __name__=='__main__':
    main()
