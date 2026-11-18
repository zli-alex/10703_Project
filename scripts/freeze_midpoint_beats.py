"""Freeze the common comparison only after a representative completed T4 benchmark."""
import argparse
import json
import math
from pathlib import Path

from infra.modal_midpoint_beats import COMPARE_TIMEOUT, FROZEN, PROTOCOL, SPLIT, rate
from scripts.rescore_midpoint_pilot import digest


def main(bench):
    cfg = json.loads(PROTOCOL.read_text())
    split = json.loads(SPLIT.read_text())
    summary = json.loads((bench / 'summary.json').read_text())
    if split['status'] != 'frozen' or not summary['reload_verified'] or not summary['frozen_verified']:
        raise ValueError('Require the verified split and successful benchmark')
    if summary['manifest']['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('Benchmark used a different common configuration')
    screen = json.loads((bench / 'reward_screen.json').read_text())
    bench_split_ids = sorted({r['song'] for r in screen['cases']})
    if bench_split_ids != sorted(p['song_id'] for p in split['pairs'] if p['split'] == 'adaptation'):
        raise ValueError('Benchmark and comparison adaptation differ')
    generation = summary['generation_seconds'] + summary['decode_score_seconds']
    update = summary['update']['update_seconds']
    setup = max(20, summary['worker_seconds'] - generation - update)
    val = sum(p['split'] == 'validation' for p in split['pairs']) * len(split['starts']) * cfg['validation_draws']
    test = sum(p['split'] == 'test' for p in split['pairs']) * len(split['starts']) * cfg['test_draws']
    val_batches, test_batches = (math.ceil(n / cfg['batch']) for n in (val, test))
    proposal = cfg['rounds']
    # Retain the longest common schedule whose benchmark-based deadline fits
    # the worker's limit. Keep both objectives and both independent seeds.
    for endpoint in reversed(cfg['validation_rounds'][1:]):
        val_rounds = [round_id for round_id in cfg['validation_rounds'] if round_id <= endpoint]
        train = endpoint * (generation + update)
        valid = (len(val_rounds) - 1) * (generation * val_batches + 10)
        if (setup + train + valid) * 1.5 + 30 <= COMPARE_TIMEOUT:
            break
    else:
        raise ValueError('Benchmark cannot support even the first common endpoint')
    cfg['rounds'], cfg['validation_rounds'] = endpoint, val_rounds
    predicted = {'validation': setup + generation * val_batches, 'test': setup + 5 * (generation * test_batches + 10)}
    for objective in cfg['objectives']:
        for seed in cfg['training_seeds']:
            predicted[f'{objective}-{seed}'] = setup + train + valid
    timeouts = {stage: math.ceil((seconds * 1.5 + 30) / 30) * 30 for stage, seconds in predicted.items()}
    if max(timeouts.values()) > COMPARE_TIMEOUT:
        raise ValueError('Measured schedule exceeds the single-stage deadline. Freeze a common shorter schedule first.')
    hourly = rate()
    cap = math.ceil(sum(timeouts.values()) * hourly / 36) / 100
    cfg['status'] = 'frozen'
    cfg['proposed_rounds'] = proposal
    cfg['budget'].update(comparison_compute_cap_usd=cap, stage_timeouts_seconds=timeouts,
                         rate_usd_per_worker_hour=hourly,
                         predicted_worker_seconds=predicted,
                         predicted_worker_compute_usd=sum(predicted.values())*hourly/3600,
                         predicted_train_compute_usd=4*train*hourly/3600,
                         predicted_eval_compute_usd=(4*valid+generation*val_batches+
                                                    5*(generation*test_batches+10))*hourly/3600,
                         predicted_setup_compute_usd=6*setup*hourly/3600,
                         split_sha256=digest(SPLIT), benchmark_summary_sha256=digest(bench / 'summary.json'),
                         benchmark_run_id=bench.parent.name)
    cfg['expected_counts'] = dict(training=4*cfg['rounds']*cfg['batch'],
                                  validation=val*(1+4*(len(cfg['validation_rounds'])-1)),
                                  test=test*5)
    if FROZEN.exists():
        if json.loads(FROZEN.read_text()) != cfg:
            raise ValueError('Frozen comparison would change')
    else:
        FROZEN.write_text(json.dumps(cfg, indent=2) + '\n')
    print(json.dumps({'expected_counts': cfg['expected_counts'], 'budget': cfg['budget']}, indent=2))
    print(f'modal run -m infra.modal_midpoint_beats --stage comparison --approved-cap-usd {cap:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_dir', type=Path)
    main(parser.parse_args().benchmark_dir)
