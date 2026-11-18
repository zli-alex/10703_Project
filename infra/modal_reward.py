"""Bounded step-D reward audit; one T4, no function retries."""
from pathlib import Path
import sys,json,subprocess
from datetime import datetime,timezone
import modal
if Path('/root/project').exists(): sys.path.insert(0,'/root/project')
from infra.modal_app import image as base_image,volume,RETRIES,PROJECT,_gate_is_fresh
app=modal.App('bailandopp-batch01-reward')
image=base_image
@app.function(image=image,gpu='T4',cpu=2.0,memory=8192,timeout=900,retries=RETRIES,volumes={'/vol':volume})
def audit(run_id:str):
 out=Path('/vol/runs')/run_id; out.mkdir(parents=True,exist_ok=False)
 with (out/'legacy_cuda.log').open('w') as log:
  result=subprocess.run(['/opt/legacy/bin/python','/root/project/scripts/reward_audit_gpu.py',str(out)],stdout=log,stderr=subprocess.STDOUT)
 volume.commit()
 return {'run_id':run_id,'exit_code':result.returncode}
@app.local_entrypoint()
def main():
 if not _gate_is_fresh(): raise SystemExit('Run account gate first')
 with volume.batch_upload(force=True) as batch:
  for name in ['inputs.npz','manifest.json']: batch.put_file(PROJECT/'data/batch01/reward_audit'/name,'reward_audit/'+name)
 run_id='rewardD-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 result=audit.remote(run_id); dest=PROJECT/'runs/batch01'/run_id; dest.mkdir(parents=True,exist_ok=True)
 for name in ['results.npz','summary.json','legacy_cuda.log']:
  try: (dest/name).write_bytes(b''.join(volume.read_file('runs/'+run_id+'/'+name)))
  except Exception as exc: print('Download failed:',name,type(exc).__name__)
 (dest/'modal_calls.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
 if result['exit_code']: raise SystemExit(result['exit_code'])
 if not (dest/'results.npz').exists(): raise SystemExit('GPU completed, results download missing; retrieve from volume without rerun')
