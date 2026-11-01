"""Wait for complete CPU extraction, then run independent feature/GPU replay."""
import pathlib,json,time,subprocess,sys,os
ROOT=pathlib.Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/postfreeze_qpu_workflow_mean_cost_v1/parallel8_v2'
start=time.time()
while not (OUT/'extraction_receipt.json').exists():
 if (OUT/'resource_stop.json').exists():raise RuntimeError('resource guard stop; inspect resource_stop.json')
 if time.time()-start>48*3600:raise RuntimeError('extraction wait exceeded48h')
 owner=json.loads((OUT/'owner.json').read_text())
 try:
  state=pathlib.Path(f"/proc/{owner['pid']}/stat").read_text().split()[2]
  if state=='Z':raise RuntimeError('CPU coordinator exited without receipt')
 except FileNotFoundError:raise RuntimeError('CPU coordinator exited without receipt')
 time.sleep(10)
receipt=json.loads((OUT/'extraction_receipt.json').read_text())
if receipt['status']!='COMPLETE_PENDING_PARITY':raise RuntimeError('CPU tasks have failures; preserve and inspect')
subprocess.run(['/home/server/Documents/Quantum-Execution-Time-Prediction/.venv-mali-gpu/bin/python',str(pathlib.Path(__file__).with_name('validate.py'))],cwd=ROOT,check=True)
(OUT/'completion_receipt.json').write_text(json.dumps({'status':'PASS_FEATURE_AND_PREDICTION_RECONSTRUCTION_8_CPU_WORKERS','all_aliases':6886,'all_predictions':61974,'canonical_isolated_latency_results_replaced':False,'remaining':'Analytical routes/common-logical compilation and isolated-C sampling protocol not completed by this receipt'},indent=2))
