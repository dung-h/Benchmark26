#!/usr/bin/env python3
"""Render verified saved-result tables and figures; no new estimates or training."""
import collections
import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
from core import OUT, ROOT, rows, sha, atomic_json

NAMES={'M0':'MLP không shots','M1':'MLP thêm shots','G0':'Graph+MLP không shots','G1':'Graph+MLP thêm shots',
       'R0':'Một pipeline hồi quy được chọn bằng validation','C0':'Trung vị nhãn tập train'}
EVALS={'E1':'Mạch chưa gặp: chia theo nhóm','E2':'Chia ngẫu nhiên, phân tầng theo nguồn','E3':'Giữ lại từng backend'}

def table(headers, body):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in body])

def f(x): return 'không xác định' if x in ('',None) else f'{float(x):.6f}'

def main():
    dest=OUT/'analysis_v1'
    final=OUT/'FINAL_REPORT.md'
    if final.exists() or (OUT/'completion_receipt.json').exists(): raise RuntimeError('Refuse final report overwrite')
    vr=OUT/'validation/final_v2/receipt.json'; sr=dest/'independent_statistics_receipt.json'
    assert json.loads(vr.read_text())['status']=='PASS' and json.loads(sr.read_text())['status']=='PASS'
    assert (dest/'diagnostics/receipt.json').exists()
    metrics=rows(dest/'pooled_metrics.csv'); contrasts=rows(dest/'paired_comparisons.csv')
    slices=rows(dest/'sliced_metrics.csv'); cases=rows(dest/'diagnostics/case_input_support.csv')
    panel=rows(OUT/'cohort/panel.csv'); source_counts=collections.Counter(r['source_id'] for r in panel)
    source_groups={s:len({r['group_id'] for r in panel if r['source_id']==s}) for s in source_counts}
    histories=rows(dest/'diagnostics/training_curve_summary.csv')
    selected=np.array([int(r['selected_epoch']) for r in histories])
    checkpoint_replay=json.loads(vr.read_text())
    # Scientific figures: standalone vector PDFs plus PNG previews.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    figs=dest/'figures'; figs.mkdir(exist_ok=True)
    short={'M0':'MLP','M1':'MLP + shots','G0':'Graph+MLP','G1':'Graph+MLP + shots','R0':'Classical selected','C0':'Train median'}
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,ev,title in zip(axes,('E1','E2','E3'),('Unseen circuit groups','Random rows, source-stratified','Held-backend OOF (descriptive)')):
        mm=[next(r for r in metrics if r['evaluation']==ev and r['treatment']==m) for m in short]
        values=[float(r['group_balanced_mae_seconds']) for r in mm]
        bars=ax.bar(range(6),values,color=['#537aa2','#8bb1d2','#368479','#80b7a6','#9d7caf','#bcbcbc'])
        ax.set_xticks(range(6),[short[m] for m in short],rotation=35,ha='right'); ax.set_title(title)
        ax.set_ylabel('Group-balanced MAE (seconds)'); ax.grid(axis='y',alpha=.2)
        for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    fig.savefig(figs/'accuracy_by_evaluation.pdf'); fig.savefig(figs/'accuracy_by_evaluation.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    cc=[r for r in contrasts if r['evaluation'] in ('E1','E2')]
    for i,r in enumerate(cc):
        point=float(r['delta_group_balanced_mae_seconds'])
        ax.plot([float(r['adjusted_ci_lower_seconds']),float(r['adjusted_ci_upper_seconds'])],[i,i],color='#9db9ce',lw=6)
        ax.plot([float(r['ci95_lower_seconds']),float(r['ci95_upper_seconds'])],[i,i],color='#234f70',lw=2)
        ax.scatter([point],[i],color='#b44b46',zorder=3)
    ax.set_yticks(range(len(cc)),[f"{r['evaluation']}: {short[r['treatment']]} − {short[r['control']]}" for r in cc])
    ax.axvline(0,color='black',lw=.8); ax.set_xlabel('Δ group-balanced MAE (s); negative favors first model')
    ax.set_title('Paired structural-group bootstrap: 95% (dark), 99.375% (light)')
    fig.savefig(figs/'paired_neural_contrasts.pdf'); fig.savefig(figs/'paired_neural_contrasts.png',dpi=180); plt.close(fig)
    text=['# Kết quả benchmark QPU mở rộng: 3.443 observations / 203 nhóm',
          '', '## Câu hỏi và phạm vi', '',
          'Thí nghiệm hỏi: dự đoán runtime thay đổi thế nào khi gặp mạch chưa thấy, khi có các lần đo tương tự trong train, và khi gặp backend chưa có nhãn train? Đồng thời kiểm tra việc bổ sung số shots cho MLP và Graph+MLP. Đây là một bộ dữ liệu và thí nghiệm bổ sung riêng; benchmark đã đóng băng 4.515 observations / 165 nhóm giữ nguyên.',
          '', '## Dữ liệu đã kiểm tra', '',
          table(['Nguồn','Observations','Nhóm mạch'],[(s,source_counts[s],source_groups[s]) for s in sorted(source_counts)]),
          '', 'Qonductor gồm 230 bản ghi cũ và 2.828 bản ghi bổ sung có bằng chứng ghép/qualification. QPack gồm 45 nhãn trung bình workflow, được tính từ 3.854 giá trị thời gian thành phần; mỗi nhãn trung bình được dự đoán như một observation tổng hợp. Đại diện circuit của QPack không phục hồi góc lịch sử của từng constituent. Logical candidate của Qonductor được qualification theo nguồn đã lưu, không được mô tả là khôi phục chính xác mọi logical program lịch sử.',
          '', 'Nguồn: [protocol](PROTOCOL.md), [amendment đọc nhãn](PROTOCOL_AMENDMENT_01.md), [panel](cohort/panel.csv), [targets](cohort/targets.csv), [feature preflight](cohort/feature_preflight_receipt.json), [locked inputs/code](execution_lock_v2.json).',
          '', '## Các phương pháp và cách đánh giá', '',
          '- MLP và Graph+MLP dùng các features của pipeline Ma–Li đã khóa. Graph là logical circuit DAG; graph-side calibration/context giữ nguyên giữa treatment có/không shots. Đây không phải một thí nghiệm về transpiled DAG hoặc QPU coupling-GNN.',
          '- Với MLP thêm shots và Graph+MLP thêm shots, chỉ bổ sung một input shots raw, chuẩn hóa bằng trung bình/độ lệch chuẩn inner-fit. Cột trọng số mới khởi tạo bằng 0; không rescale target theo shots.',
          '- Mỗi neural treatment dùng ba seeds 42, 1234, 31415; dự đoán chính cho một observation là median của ba predictions held-out. Đây là kết quả của phép gộp ba seeds, không phải một checkpoint đơn đã được chọn để deploy.',
          '- Classical comparator chọn đúng một pipeline cho mỗi outer case từ sáu họ ứng viên đã định trước bằng mean inner raw-seconds R². Không chọn model bằng test error hoặc ghép best prediction theo từng test row. Năm inputs: `swap, depth, num_qubits, shots, circuit_count`; representation có nguồn gốc riêng, khác neural inputs.',
          '- Train-median dùng trung vị nhãn outer-train. Neural preprocessing chỉ fit inner-fit; validation chọn earliest minimum MSE trong 500 epochs. Mọi test label chỉ được đọc để chấm sau khi training hoàn tất.',
          '', 'Sai số chính là MAE tính đều theo nhóm mạch: tính sai số trung bình trong mỗi nhóm, rồi lấy trung bình các nhóm. Pooled MAE tính đều theo observations. Source-macro MAE tính đều ba nguồn. Các thước đo này trả lời các câu hỏi có trọng số khác nhau.',
          '', '## Kết quả chính từ predictions đã lưu', '',
          table(['Cách chia','Phương pháp','MAE đều nhóm (s)','MAE mọi observation (s)','MAE đều nguồn (s)','R² giây gốc','R² log1p'],
                [(r['evaluation'],NAMES[r['treatment']],f(r['group_balanced_mae_seconds']),f(r['pooled_mae_seconds']),f(r['source_macro_mae_seconds']),f(r['raw_seconds_r2']),f(r['log1p_seconds_r2'])) for r in metrics]),
          '', '![Sai số theo cách đánh giá](analysis_v1/figures/accuracy_by_evaluation.png)',
          '', 'E1 giữ toàn bộ một nhóm mạch trong cùng fold; E2 chia observations ngẫu nhiên và phân tầng theo nguồn, nên có thể gặp cùng nhóm mạch trong train/test; E3 giữ riêng từng backend và cho phép cùng mạch ở backend khác trong train. Số E3 gộp chỉ mô tả toàn bộ held-backend predictions; kết luận từng backend cần đọc bảng riêng.',
          '', '## So sánh ghép cặp và độ không chắc chắn', '',
          'Δ = phương pháp thứ nhất − phương pháp thứ hai; Δ âm nghĩa là phương pháp thứ nhất có sai số thấp hơn. Resample toàn bộ structural group, giữ ghép cặp predictions. E1/E2 giữ số nhóm của từng nguồn; E3 resample nhóm trong từng backend. Mỗi phép dùng 10.000 draws, seed 42. Khoảng 99,375% cho tám contrasts E1/E2 và 99,875% cho 40 contrasts E3 là quy tắc Bonferroni đã định trước. Khoảng chứa 0 nghĩa là chưa xác lập khác biệt rõ ràng, không phải chứng minh hai phương pháp tương đương.',
          '', table(['Cách chia','Contrast','Δ MAE nhóm (s)','CI 95% (s)','CI 99,375% (s)'],
                    [(r['evaluation'],r['contrast'],f(r['delta_group_balanced_mae_seconds']),f"[{f(r['ci95_lower_seconds'])}, {f(r['ci95_upper_seconds'])}]",f"[{f(r['adjusted_ci_lower_seconds'])}, {f(r['adjusted_ci_upper_seconds'])}]") for r in cc]),
          '', '![So sánh ghép cặp](analysis_v1/figures/paired_neural_contrasts.png)',
          '', 'Bảng đủ 48 contrasts, bao gồm 40 contrasts theo backend: [paired_comparisons.csv](analysis_v1/paired_comparisons.csv). Tỷ lệ bootstrap Δ<0 trong CSV chỉ mô tả phân bố bootstrap; không phải p-value hoặc xác suất hậu nghiệm Bayes.',
          '', '## Theo từng nguồn và backend', '',
          table(['Cách chia','Nguồn','Phương pháp','Rows','Nhóm','MAE đều nhóm (s)','MAE mọi observation (s)'],
                [(r['evaluation'],r['scope_value'],NAMES[r['treatment']],r['rows'],r['groups'],f(r['group_balanced_mae_seconds']),f(r['pooled_mae_seconds'])) for r in slices if r['scope']=='source']),
          '', 'Các case/backend và nguồn nhỏ cần được đọc cùng số nhóm độc lập. Xem [toàn bộ slices và tails](analysis_v1/sliced_metrics.csv), [fold deltas](analysis_v1/fold_differences.csv), [25 lỗi lớn nhất mỗi phương pháp/cách chia](analysis_v1/tail_errors.csv). Không loại bất kỳ lỗi lớn hợp lệ nào khỏi score.',
          '', '## Mức độ trùng lặp đầu vào và kiểm tra training', '',
          table(['Cách chia','Test rows có cùng nhóm ở neural fit','Tổng test rows'],
                [(ev,sum(int(r['test_rows_fit_same_group']) for r in cases if r['evaluation']==ev),sum(int(r['test_rows']) for r in cases if r['evaluation']==ev)) for ev in EVALS]),
          '', f'240/240 neural cells và 20/20 classical cases hoàn tất. Selected epoch: min {selected.min()}, median {np.median(selected):.1f}, max {selected.max()}; {int((selected>=450).sum())} cells chọn epoch ≥450. Các số này mô tả optimization đã chạy; không dùng test performance để đổi epoch budget.',
          '', 'Xem [curve summary](analysis_v1/diagnostics/training_curve_summary.csv), [input-support audit](analysis_v1/diagnostics/case_input_support.csv), [error theo vùng có/chưa có input support](analysis_v1/diagnostics/support_error_slices.csv) và [Pearson/Spearman](analysis_v1/diagnostics/correlations.csv). Correlation là thống kê mô tả trên observations lặp; không chứng minh feature nào gây ra gain.',
          '', '## Kiểm định và các vấn đề đã công bố', '',
          f"- Replayed toàn bộ {checkpoint_replay['prediction_replay_rows']:,} neural predictions từ selected checkpoints; sai khác tuyệt đối lớn nhất {checkpoint_replay['max_abs_prediction_replay_difference']:.12g} s. Checkpoint states finite; đủ 500 epochs/cell, chọn đúng earliest min validation, transforms/splits/output hashes đã kiểm.",
          '- Lần replay đầu dùng tolerance tuyệt đối 1e-6 s và dừng tại sai khác 1,9073486328125e-6 s. Hồ sơ giữ nguyên tại `validation/final/`. Replay v2 ghi tolerance `abs=1e-5 s + rel=1e-6 × |saved prediction|`; có kiểm tra lặp cùng input để đo sai khác reduction float32 trên CUDA. Không sửa, clip hoặc thay predictions gốc bằng giá trị replay. Tolerance chỉ dùng kiểm tra inference, không đổi metric.',
          '- Counter vận hành cuối ghi 414 vì cộng lại 174 cells đã có khi resume. Manifest duy nhất xác nhận đúng 240 cells. Producer/counter lịch sử giữ nguyên.',
          '- Classical candidate-selection và stored predictions được kiểm; producer không lưu fitted estimator states. Vì vậy chưa có classical checkpoint-to-prediction replay; không refit để che khoảng thiếu này.',
          '- Independent arithmetic replay kiểm tra median-seed reduction, targets, metrics, source/group identity, hashes và deterministic bootstrap draws. Xem [checkpoint receipt](validation/final_v2/receipt.json) và [statistics receipt](analysis_v1/independent_statistics_receipt.json).',
          '', '## Diễn giải được phép và giới hạn', '',
          'Ba cách chia đánh giá ba loại tình huống. Lợi thế trên E2 có thể gắn với việc đã gặp các đầu vào tương tự; E1 kiểm tra mạch chưa gặp; E3 kiểm tra backend context chưa có nhãn train. Không kết luận một cách chia “công bằng hơn” cho mọi deployment. Việc random-source stratification giữ tỷ lệ nguồn, không làm ba nguồn có số observations hoặc fitting weights bằng nhau.',
          '', 'Graph+MLP so với MLP là so sánh toàn bộ hai pipelines; nó không cô lập đóng góp của connectivity. Shots là thông tin workload có cơ sở, nhưng có thêm input chưa tự bảo đảm học được scaling ngoài miền shots đã train. Snapshot/calibration features không được mô tả là calibration lịch sử chính xác nếu nguồn chỉ là reconstruction/proxy.',
          '', 'Bootstrap đo biến động theo các nhóm trong cohort đã có, có điều kiện trên splits và median ba seeds đã train. Nó không bao trùm mọi uncertainty của training, missing provenance hay deployment khác. QPack chỉ có sáu nhóm; backend/source/workload có thể bị confounded. Runtime labels và observation units vẫn giữ nguồn riêng; workflow mean không tương đương một execution riêng lẻ.',
          '', '## Kết luận tiếng Anh dùng cho thesis', '',
          'The supplementary benchmark evaluates 3,443 QPU-runtime observations from 203 circuit groups using group-disjoint evaluation, source-stratified random-observation evaluation, and ten held-backend contexts. All neural comparisons use the median of three held-out seed predictions, with group-balanced MAE as the primary metric and prespecified paired structural-group bootstrap intervals. The evaluation protocols address different forms of novelty and are reported separately. The QPack observations are workflow-mean targets, and the expanded Qonductor inputs retain their reconstruction qualifications. Graph-versus-MLP comparisons concern the full prediction pipelines; connectivity-specific or physical causal effects are not identified by these comparisons.',
          '', '## Artifact index và tái lập', '',
          '- [Protocol](PROTOCOL.md), [lock](execution_lock_v2.json), [cohort](cohort/panel.csv), [splits](splits/split_receipt.json).',
          '- [Reduced predictions](analysis_v1/reduced_predictions.csv), [per-seed predictions](analysis_v1/per_seed_predictions.csv), [primary/secondary metrics](analysis_v1/pooled_metrics.csv), [group metrics](analysis_v1/group_metrics.csv).',
          '- [Source/backend/fold metrics](analysis_v1/sliced_metrics.csv), [seed metrics](analysis_v1/seed_metrics.csv), [paired comparisons](analysis_v1/paired_comparisons.csv), draws under `analysis_v1/bootstrap/`.',
          '- [Validation cells](validation/final_v2/neural_cells.csv), [preprocessing/splits](validation/final_v2/split_and_preprocessing_cases.csv), [classical selection](validation/final_v2/classical_selection.csv).',
          '', 'Các lệnh dưới đây ghi ra namespace mới; aggregation mặc định từ chối overwrite. Dùng cùng environment đã lưu trong cell identities. Không cần train lại để kiểm tra predictions/metrics.',
          '', '```bash',
          'python -m unittest discover -s research/postfreeze_qpu_workflow_mean_expansion_v1 -p test_analysis.py -v',
          'python research/postfreeze_qpu_workflow_mean_expansion_v1/validate_completed.py --replay-all --output artifacts/postfreeze_qpu_workflow_mean_expansion_v1/validation/NEW_ATTEMPT',
          'python research/postfreeze_qpu_workflow_mean_expansion_v1/aggregate_results.py --validation artifacts/postfreeze_qpu_workflow_mean_expansion_v1/validation/NEW_ATTEMPT/receipt.json --output artifacts/postfreeze_qpu_workflow_mean_expansion_v1/NEW_ANALYSIS',
          '```',
          '', 'Lệnh thực tế cho bản tham chiếu: validator → aggregate_results.py → replay_statistics.py → diagnose_results.py → write_final_report.py. `replay_statistics.py`/diagnostics/report mặc định trỏ `analysis_v1`; dùng bản bàn giao portable hoặc copy vào workspace review riêng để tránh overwrite. Bản nguồn hiện tại không được tự commit/push. Repo team review là bước bàn giao riêng, không nằm trong scientific completion receipt này.',
          '']
    # Explicit claim table determined only after full validation and aggregation.
    claims=[]
    for r in cc:
        lo=float(r['adjusted_ci_lower_seconds']); hi=float(r['adjusted_ci_upper_seconds'])
        verdict='LOWER_ERROR_SUPPORTED' if hi<0 else ('HIGHER_ERROR_SUPPORTED' if lo>0 else 'DIRECTION_NOT_ESTABLISHED')
        claims.append({**{k:r[k] for k in ('evaluation','contrast','delta_group_balanced_mae_seconds','adjusted_ci_lower_seconds','adjusted_ci_upper_seconds')},
                       'status':verdict,'scope':'This supplementary cohort/protocol only',
                       'forbidden_claim':'equivalence; physical causality; universal estimator superiority'})
    from core import atomic_csv
    atomic_csv(dest/'claim_matrix.csv',claims)
    final.write_text('\n'.join(text),encoding='utf-8')
    artifacts=[p for p in dest.rglob('*') if p.is_file()]+[final,vr]
    receipt={'status':'PASS_SCIENTIFIC_AGGREGATION_COMPLETE','cohort_rows':3443,'groups':203,'neural_cells':240,'classical_cases':20,
             'execution_lock_sha256':sha(OUT/'execution_lock_v2.json'),'report_sha256':sha(final),
             'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in sorted(artifacts)},
             'no_new_training_or_refit':True,'frozen4515_benchmark_unchanged':True,
             'classical_state_replay':'NOT_AVAILABLE_PRODUCER_DID_NOT_SAVE_STATES',
             'team_repository_release':'SEPARATE_PENDING_DELIVERY',
             'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(OUT/'completion_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts'},indent=2))

if __name__=='__main__': main()
