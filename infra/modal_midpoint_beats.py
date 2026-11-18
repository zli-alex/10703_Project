"""Serial, finite Modal stages for the midpoint comparison."""
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import modal

if Path('/root/project').exists():
    sys.path.insert(0, '/root/project')

from infra.modal_app import PROJECT, RETRIES, _gate_is_fresh, image, volume

app = modal.App('bailandopp-midpoint-beats')
PROTOCOL = PROJECT / 'configs/midpoint_beat_protocol.json'
SPLIT = PROJECT / 'configs/midpoint_beat_split.json'
FROZEN = PROJECT / 'configs/midpoint_beat_frozen.json'
ACCOUNT = PROJECT / 'reports/batch01/modal_account.json'
BENCH_TIMEOUT = 900
COMPARE_TIMEOUT = 3600


def execute(run_id, stage):
    root = Path('/vol/runs') / run_id
    dest = root / stage
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'summary.json').exists() or (dest / 'legacy_cuda.log').exists():
        raise ValueError('Stage already attempted. Recover its recorded output by run ID.')
    command = ['/opt/legacy/bin/python', '-u', '-m', 'scripts.run_midpoint_beats_gpu', str(root), stage]
    with (dest / 'legacy_cuda.log').open('x') as log:
        volume.commit()
        process = subprocess.Popen(command, cwd='/root/project', stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line.rstrip(), flush=True)
            if line.startswith('Round '):
                volume.commit()
        code = process.wait()
    volume.commit()
    return code


@app.function(image=image, gpu='T4', cpu=2, memory=8192, timeout=BENCH_TIMEOUT,
              retries=RETRIES, max_containers=1, volumes={'/vol': volume})
def benchmark_worker(run_id: str) -> int:
    return execute(run_id, 'benchmark')


@app.function(image=image, gpu='T4', cpu=2, memory=8192, timeout=COMPARE_TIMEOUT,
              retries=RETRIES, max_containers=1, volumes={'/vol': volume})
def comparison_worker(run_id: str, stage: str) -> int:
    return execute(run_id, stage)


def rate():
    rates = json.loads(ACCOUNT.read_text())['rates']
    return sum((float(rates['gpu_hour_cost_t4']), 2*float(rates['cpu_hour_cost']),
                8*float(rates['mem_gib_hour_cost'])))


def download(run_id, stage, directory):
    remote = f'runs/{run_id}/{stage}'
    directory.mkdir(parents=True, exist_ok=True)
    for entry in volume.listdir(remote):
        name = Path(entry.path).name
        keep = (name.endswith('.json') or name.startswith(('test_', 'validation_'))
                or name in {'legacy_cuda.log', 'decoded_continuation.npy', 'decoded_before_reload.npz',
                            'reward_screen.npz'})
        if keep and '.' in name:
            with (directory / name).open('wb') as handle:
                for chunk in volume.read_file(f'{remote}/{name}'):
                    handle.write(chunk)


def stage_inputs(run_id, config_path, split, test=False):
    with volume.batch_upload() as upload:
        remote = 'runs/' + run_id
        if not test:
            upload.put_file(config_path, remote + '/protocol.json')
            upload.put_file(SPLIT, remote + '/split.json')
        for pair in split['pairs']:
            if (pair['split'] == 'test') == test:
                upload.put_file(PROJECT / pair['external_features'],
                                f"{remote}/features/{pair['song_id']}.json")


@app.local_entrypoint()
def main(stage: str = 'benchmark', run_id: str = '', approved_cap_usd: float = 0.0):
    if not _gate_is_fresh():
        raise SystemExit('Refresh scripts/check_modal_account.py before launch.')
    if stage not in {'benchmark', 'comparison', 'retrieve'}:
        raise SystemExit('Use benchmark, comparison, or retrieve.')
    if run_id and not re.fullmatch(r'midbeats-\d{8}T\d{12}Z', run_id):
        raise SystemExit('Invalid run ID.')
    if stage == 'retrieve':
        if not run_id:
            raise SystemExit('Supply the existing run ID.')
        base = f'runs/{run_id}'
        for entry in volume.listdir(base):
            name = Path(entry.path).name
            if re.fullmatch(r'benchmark|validation|test|(?:bas|bin)-\d+', name):
                download(run_id, name, PROJECT / 'runs/midpoint_beats' / run_id / name)
        return
    if run_id:
        raise SystemExit('Omit the run ID for a new job.')
    config_path = PROTOCOL if stage == 'benchmark' else FROZEN
    config = json.loads(config_path.read_text())
    split = json.loads(SPLIT.read_text())
    run_id = 'midbeats-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = PROJECT / 'runs/midpoint_beats' / run_id
    print(f'Run ID: {run_id}', flush=True)
    hourly = rate()
    if stage == 'benchmark':
        cap = config['budget']['benchmark_compute_cap_usd']
        if hourly * BENCH_TIMEOUT / 3600 > cap:
            raise SystemExit('Current rates exceed the benchmark compute cap.')
        stage_inputs(run_id, config_path, split)
        code = benchmark_worker.remote(run_id)
        download(run_id, 'benchmark', root / 'benchmark')
        if code:
            raise SystemExit(code)
        return
    from scripts.rescore_midpoint_pilot import digest

    if (config['status'] != 'frozen' or split['status'] != 'frozen'
            or config['budget']['split_sha256'] != digest(SPLIT)):
        raise SystemExit('A measured protocol and verified fresh split must be frozen first.')
    cap = config['budget']['comparison_compute_cap_usd']
    if not math.isfinite(approved_cap_usd) or approved_cap_usd < cap:
        raise SystemExit(f'Comparison requires explicit authorization for the ${cap:.2f} cap.')
    stages = ['validation'] + [f'{o}-{s}' for o, s in product(config['objectives'], config['training_seeds'])] + ['test']
    timeouts = config['budget']['stage_timeouts_seconds']
    if (sum(timeouts.values()) * hourly / 3600 > cap or set(stages) != set(timeouts)
            or any(not 0 < seconds <= COMPARE_TIMEOUT for seconds in timeouts.values())):
        raise SystemExit('Current rates or stage schedule exceed the frozen cap.')
    stage_inputs(run_id, config_path, split)
    for item in stages:
        if item == 'test':
            stage_inputs(run_id, config_path, split, test=True)
        timeout = timeouts[item]
        code = comparison_worker.with_options(timeout=timeout).remote(run_id, item)
        download(run_id, item, root / item)
        if code:
            raise SystemExit(code)
