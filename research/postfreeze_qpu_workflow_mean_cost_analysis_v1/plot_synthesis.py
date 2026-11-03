
import os as _b26_os
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(_b26_os.environ.get("B26_ARTIFACTS", str(ROOT / "artifacts"))) / 'postfreeze_qpu_workflow_mean_cost_v1/synthesis_v1'
names={'M0':'MLP','M1':'MLP + shots','G0':'Graph+MLP','G1':'Graph+MLP + shots','R0':'Selected classical','C0':'Training median'}
def read(n):
 with (OUT/n).open() as f:return list(csv.DictReader(f))
def main():
 colors=dict(zip(names,plt.get_cmap('tab10').colors));fig,axes=plt.subplots(1,2,figsize=(10,4.8));fig.subplots_adjust(bottom=.26,top=.77,wspace=.28)
 for ax,ev,title in zip(axes,['E1','E2'],['E1: held-out structural groups','E2: source-stratified random rows']):
  for r in read('warm_accuracy_cost.csv'):
   if r['evaluation']!=ev or r['method']=='C0':continue
   m=r['method'];x=float(r['group_balanced_mean_latency_ms']);y=float(r['saved_group_balanced_mae_seconds'])
   ax.scatter(x,y,color=colors[m],label=names[m],s=45);ax.annotate(m,(x,y),xytext=((-20 if m.startswith('G') else 5),7),textcoords='offset points',fontsize=8)
  ax.set(xscale='log',xlabel='Group-balanced request latency (ms)',ylabel='Group-balanced MAE (s)',title=title);ax.grid(alpha=.2)
 axes[0].set_xlim(.45,15);axes[1].set_xlim(.45,15)
 handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=3,frameon=False);fig.suptitle('Learned pipelines: 3,443 observations / 203 structural groups\nConstant lookup baseline reported separately at the harness floor.',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(OUT/f'accuracy_cost.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
 sources=['mali_real_qpu','qonductor_single_circuit_ibm','qpack_mcp']
 for ax,source,title in zip(axes,sources,['Ma–Li','Qonductor','QPack workflow means']):
  rows=[r for r in read('source_stage_cost.csv') if r['route']=='C' and r['source']==source and r['method']!='C0']
  rows.sort(key=lambda r:list(names).index(r['method']))
  for i,r in enumerate(rows):
   x=float(r['median_ms']);lo=float(r['min_observation_ms']);hi=float(r['max_observation_ms'])
   ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=colors[r['method']],capsize=3)
  ax.set_yticks(range(len(rows)),[names[r['method']] for r in rows],fontsize=8)
  ax.set(xscale='log',xlabel='Request latency (ms; log scale)',title=title);ax.grid(axis='x',alpha=.2)
 fig.suptitle('Diagnostic only: 4 observations/source; method-specific input stages\nNeural: logical QASM; classical: submitted/nominal compiled QASM. Bars = sample min–max.',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(OUT/f'input_processing_diagnostic.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
