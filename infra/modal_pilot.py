"""Run and retrieve the bounded Bailando++ pilot and matched evaluation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import modal

if Path('/root/project').exists():
    sys.path.insert(0, '/root/project')

from infra.modal_app import PROJECT, RETRIES, _gate_is_fresh, image, volume
from infra.benchmark_inputs import rollout_pairs

app = modal.App('bailandopp-five-epoch-pilot')


@app.function(image=image, gpu='T4', cpu=2.0, memory=8192, timeout=900,
              retries=RETRIES, volumes={'/vol': volume})
def worker(run_id: str, stage: str, resume: bool) -> dict:
    root = Path('/vol/runs') / run_id
    output = root if stage == 'train' else root / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    module = 'scripts.run_pilot_gpu' if stage == 'train' else 'scripts.evaluate_pilot_gpu'
    command = ['/opt/legacy/bin/python', '-u', '-m', module, str(output)]
    if resume:
        command.append('--resume')
    with (output / 'legacy_cuda.log').open('a' if resume else 'w') as log:
        process = subprocess.Popen(command, cwd='/root/project', stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(('Epoch ', 'Evaluation ', '{"saved_epoch"')):
                    print(line.rstrip(), flush=True)
                if line.startswith('{"saved_epoch"'):
                    volume.commit()
        code = process.wait()
    volume.commit()
    return {'run_id': run_id, 'stage': stage, 'exit_code': code}


@app.local_entrypoint()
def main(stage: str = 'train', run_id: str = '', resume: bool = False) -> None:
    if not _gate_is_fresh():
        raise SystemExit('Refresh the account check with scripts/check_modal_account.py.')
    if stage not in {'train', 'evaluate'} or (resume and stage != 'train'):
        raise SystemExit('Use train or evaluate. Resume applies to training.')
    if run_id and not re.fullmatch(r'pilot-\d{8}T\d{12}Z', run_id):
        raise SystemExit('Invalid pilot run ID.')
    if (resume or stage == 'evaluate') and not run_id:
        raise SystemExit('Supply the existing pilot run ID.')
    if stage == 'train' and not resume:
        if run_id:
            raise SystemExit('Omit the run ID to start a new pilot.')
        run_id = 'pilot-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    remote = f'runs/{run_id}'
    print(f'Run ID: {run_id}', flush=True)
    if stage == 'train' and not resume:
        with volume.batch_upload() as upload:
            for pair in rollout_pairs('pilot'):
                upload.put_file(PROJECT / pair['motion_record'], f'{remote}/inputs/motion/{pair["record_id"]}.json')
                upload.put_file(PROJECT / pair['external_features'], f'{remote}/inputs/features/{pair["song_id"]}.json')
    elif stage == 'evaluate':
        pairs = json.loads((PROJECT / 'configs/pilot_split.json').read_text())['pairs']
        with volume.batch_upload(force=True) as upload:
            for pair in pairs:
                upload.put_file(PROJECT / pair['external_features'], f'{remote}/evaluation/features/{pair["song_id"]}.json')
        remote += '/evaluation'

    local = PROJECT / 'runs' / 'batch03' / run_id
    if stage == 'evaluate':
        local /= 'evaluation'
    local.mkdir(parents=True, exist_ok=True)
    result = worker.remote(run_id, stage, resume)
    (local / 'modal_calls.json').write_text(json.dumps(result, indent=2) + '\n')
    names = ['legacy_cuda.log']
    if not result['exit_code']:
        names.extend(['pilot_results.json'] if stage == 'train' else ['evaluation.json', 'motions.npz'])
    for name in names:
        with (local / name).open('wb') as output:
            for chunk in volume.read_file(f'{remote}/{name}'):
                output.write(chunk)
    if result['exit_code']:
        print((local / 'legacy_cuda.log').read_text()[-5000:], file=sys.stderr)
        raise SystemExit(result['exit_code'])
    print(f'{stage.title()} completed. Results saved to {local}.')
