"""M4 update verification: one actor step and one critic step, one bounded T4, no retries."""
from pathlib import Path
import sys, json, subprocess
from datetime import datetime, timezone
import modal

if Path('/root/project').exists():
    sys.path.insert(0, '/root/project')

from infra.modal_app import image, volume, RETRIES, PROJECT, _gate_is_fresh

app = modal.App('bailandopp-batch01-update')


@app.function(
    image=image,
    gpu='T4',
    cpu=2.0,
    memory=8192,
    timeout=900,
    retries=RETRIES,
    volumes={'/vol': volume},
)
def update_verify(run_id: str, trace_run_id: str) -> dict:
    out = Path('/vol/runs') / run_id
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = Path('/vol/runs') / trace_run_id

    # Upload latest update_reference.json from local reports so GPU can compare
    with (out / 'legacy_cuda.log').open('w') as log:
        result = subprocess.run(
            [
                '/opt/legacy/bin/python',
                '/root/project/scripts/verify_update_gpu.py',
                str(out),
                str(trace_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    volume.commit()
    return {'run_id': run_id, 'exit_code': result.returncode}


@app.local_entrypoint()
def main() -> None:
    if not _gate_is_fresh():
        raise SystemExit('Account gate is missing or stale. Run scripts/check_modal_account.py first.')

    # Upload local update_reference.json so GPU can compare
    ref_path = PROJECT / 'reports' / 'batch01' / 'update_reference.json'
    if not ref_path.exists():
        raise SystemExit('update_reference.json not found; run scripts/independent_update_reference.py first.')

    trace_run_id = 'traceE-20261106T221045Z'
    run_id = 'updateM4-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    print(f'Run ID: {run_id}', flush=True)
    print(f'Trace:  {trace_run_id}', flush=True)

    # Upload the reference JSON so GPU can read it from the volume
    with volume.batch_upload(force=True) as batch:
        # GPU script checks /vol/reports/batch01/update_reference.json
        batch.put_file(ref_path, 'reports/batch01/update_reference.json')

    result = update_verify.remote(run_id, trace_run_id)
    print(json.dumps(result))

    dest = PROJECT / 'runs' / 'batch01' / run_id
    dest.mkdir(parents=True, exist_ok=True)

    for name in ['update_results.json', 'legacy_cuda.log']:
        try:
            (dest / name).write_bytes(b''.join(volume.read_file(f'runs/{run_id}/{name}')))
            print(f'Downloaded {name}')
        except Exception as exc:
            print(f'Download failed: {name}  {type(exc).__name__}: {exc}')

    (dest / 'modal_calls.json').write_text(json.dumps(result, indent=2))

    if result['exit_code'] != 0:
        raise SystemExit(result['exit_code'])
    if not (dest / 'update_results.json').exists():
        raise SystemExit('update_results.json not downloaded; retrieve from volume.')
