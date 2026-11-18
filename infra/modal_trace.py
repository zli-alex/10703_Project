"""Forward-only step-E trace replay, one bounded T4 and no function retries."""
from pathlib import Path
import sys,json,subprocess
from datetime import datetime,timezone
import modal
if Path('/root/project').exists(): sys.path.insert(0,'/root/project')
from infra.modal_app import image,volume,RETRIES,PROJECT,_gate_is_fresh
app=modal.App('bailandopp-batch01-trace')
@app.function(image=image,gpu='T4',cpu=2.0,memory=8192,timeout=900,retries=RETRIES,volumes={'/vol':volume})
def trace(run_id:str):
    out=Path('/vol/runs')/run_id; out.mkdir(parents=True,exist_ok=False)
    with (out/'legacy_cuda.log').open('w') as log:
        result=subprocess.run(['/opt/legacy/bin/python','/root/project/scripts/rollout_trace_gpu.py',str(out)],stdout=log,stderr=subprocess.STDOUT)
    volume.commit()
    return {'run_id':run_id,'exit_code':result.returncode}
@app.local_entrypoint()
def main():
    if not _gate_is_fresh(): raise SystemExit('Run account gate first')
    with volume.batch_upload(force=True) as batch:
        batch.put_file(PROJECT/'data/batch01/rollout_trace/inputs.npz','rollout_trace/inputs.npz')
    run_id='traceE-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result=trace.remote(run_id); dest=PROJECT/'runs/batch01'/run_id; dest.mkdir(parents=True,exist_ok=True)
    for name in ['window_000.npz','window_020.npz','summary.json','critic_initial.pt','legacy_cuda.log']:
        try: (dest/name).write_bytes(b''.join(volume.read_file('runs/'+run_id+'/'+name)))
        except Exception as exc: print('Download failed:',name,type(exc).__name__)
    (dest/'modal_calls.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    if result['exit_code']: raise SystemExit(result['exit_code'])
    if not all((dest/n).exists() for n in ['window_000.npz','window_020.npz','summary.json','critic_initial.pt']):
        raise SystemExit('Results download incomplete; retrieve from volume without rerunning')
