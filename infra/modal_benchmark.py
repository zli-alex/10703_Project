"""Launch one bounded Bailando++ training-cycle benchmark on a T4."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import modal

if Path('/root/project').exists():
    sys.path.insert(0, '/root/project')

from infra.modal_app import PROJECT, RETRIES, _gate_is_fresh, image, volume
from infra.benchmark_inputs import SPLIT_CONFIGS, rollout_pairs

app = modal.App('bailandopp-cycle-benchmark')


@app.function(image=image, gpu='T4', cpu=2.0, memory=8192, timeout=900,
              retries=RETRIES, volumes={'/vol': volume})
def benchmark(run_id: str, batch_cap: int, split: str) -> dict:
    output = Path('/vol/runs') / run_id
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'legacy_cuda.log').open('w') as log:
        completed = subprocess.run(
            ['/opt/legacy/bin/python', '-u', '-m', 'scripts.benchmark_cycle_gpu',
             str(output), str(output / 'inputs'), '--batch-cap', str(batch_cap), '--split', split],
            cwd='/root/project', stdout=log, stderr=subprocess.STDOUT,
        )
    volume.commit()
    return {'run_id': run_id, 'exit_code': completed.returncode}


@app.local_entrypoint()
def main(batch_cap: int = 64, split: str = 'development') -> None:
    if not _gate_is_fresh():
        raise SystemExit('Refresh the account check with scripts/check_modal_account.py.')
    if not 1 <= batch_cap <= 64:
        raise SystemExit('Batch cap must be between 1 and 64.')
    if split not in SPLIT_CONFIGS:
        raise SystemExit('Split must be development or pilot.')
    pairs = rollout_pairs(split)
    run_id = 'cycle-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    local = PROJECT / 'runs' / 'batch02' / run_id
    local.mkdir(parents=True)
    print(f'Run ID: {run_id}', flush=True)
    remote = f'runs/{run_id}'
    with volume.batch_upload() as upload:
        for pair in pairs:
            upload.put_file(PROJECT / pair['motion_record'], f'{remote}/inputs/motion/{pair["record_id"]}.json')
            upload.put_file(PROJECT / pair['external_features'], f'{remote}/inputs/features/{pair["song_id"]}.json')

    result = benchmark.remote(run_id, batch_cap, split)
    (local / 'modal_calls.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    names = ['legacy_cuda.log']
    if result['exit_code'] == 0:
        names.append('benchmark_results.json')
    for name in names:
        (local / name).write_bytes(b''.join(volume.read_file(f'{remote}/{name}')))
    if result['exit_code'] != 0:
        print((local / 'legacy_cuda.log').read_text()[-5000:], file=sys.stderr)
        raise SystemExit(result['exit_code'])
    print(f'Measurements saved to {local}. Checkpoint available on the volume at {remote}/benchmark_checkpoint.pt.')
