"""Three sequential step-C rollouts, one bounded T4 container."""
from datetime import datetime, timezone
import json, subprocess
from pathlib import Path
import modal
import sys
if Path('/root/project').exists():
    sys.path.insert(0, '/root/project')
from infra.modal_app import image, volume, RETRIES, PROJECT, _gate_is_fresh, _download
app=modal.App('bailandopp-batch01-music')

@app.function(image=image,gpu='T4',cpu=2.0,memory=8192,timeout=900,retries=RETRIES,volumes={'/vol':volume})
def music_cases(run_id: str):
    results=[]
    for case in ['A-repeat','B-swap','A-shift']:
        output=f'/vol/runs/{run_id}/{case}'
        Path(output).mkdir(parents=True,exist_ok=False)
        with open(output+'/legacy_cuda.log','w') as log:
            result=subprocess.run(['/opt/legacy/bin/python','/root/project/scripts/music_intervention_infer.py','--case',case,'--output',output],stdout=log,stderr=subprocess.STDOUT)
        results.append({'case':case,'exit_code':result.returncode,'output':output})
        volume.commit()
        if result.returncode: break
    return results

@app.local_entrypoint()
def main():
    if not _gate_is_fresh(): raise SystemExit('Refresh account gate first')
    # Only add the second song; existing baseline assets are retained.
    with volume.batch_upload(force=True) as upload:
        upload.put_file(PROJECT/'assets/audio/mJB5.wav','assets/audio/mJB5.wav')
        upload.put_file(PROJECT/'data/batch01/music_feat_60fps/mJB5.json','assets/features/mJB5.json')
    run_id='musicC-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    results=music_cases.remote(run_id)
    root=PROJECT/'runs/batch01'/run_id
    for result in results:
        destination=root/result['case']; remote=result['output'].removeprefix('/vol/')
        _download(remote,destination)
        for name in ['intervention.json']:
            try: (destination/name).write_bytes(b''.join(volume.read_file(remote+'/'+name)))
            except Exception: pass
    (root/'modal_calls.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'run_id':run_id,'results':results}))
    if len(results)!=3 or any(r['exit_code'] for r in results): raise SystemExit(1)
