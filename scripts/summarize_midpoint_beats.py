"""Rebuild song-weighted midpoint tables, paired deltas, and figures from saved runs."""
import argparse
from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.rescore_midpoint_pilot import METRICS, OUTPUT, ROOT, digest, write_csv

FIELDS = (*METRICS, 'precision_50ms', 'recall_50ms', 'f1_50ms',
          'precision_150ms', 'recall_150ms', 'f1_150ms', 'root_travel')
HEADLINE = ('bas', 'coverage_100ms', 'precision_100ms', 'recall_100ms',
            'f1_100ms', 'hit_fraction', 'motion_events_per_s',
            'mean_joint_speed', 'rms_jerk', 'stationary_fraction', 'within_song_diversity')
LABELS = ('BAS', 'Cov. 100', 'P 100', 'R 100', 'F1 100', 'Bin hit',
          'Events/s', 'Speed', 'RMS jerk', 'Stationary', 'Diversity')


def read(path):
    return json.loads(path.read_text())


def mean(values):
    available = [x for x in values if x is not None]
    return float(np.mean(available)) if available else None


def song_means(rows, within_start):
    all_songs = defaultdict(list)
    for row in rows:
        all_songs[row['song']].append(row)
    result = {}
    for song, cases in sorted(all_songs.items()):
        groups = defaultdict(list)
        for case in cases:
            groups[case['start'] if within_start else 0].append(case['speed_signature'])
        distances = [np.linalg.norm(np.asarray(a)-b) for group in groups.values()
                     for a, b in combinations(group, 2)]
        result[song] = {k: mean([r[k] for r in cases]) for k in FIELDS}
        result[song]['within_song_diversity'] = mean(distances)
    return result


def aggregate(songs):
    return {k: mean([r[k] for r in songs.values()]) for k in (*FIELDS, 'within_song_diversity')}


def table(rows):
    head = '| Policy | ' + ' | '.join(LABELS) + ' |'
    rule = '|---|' + '|'.join(['---:'] * len(LABELS)) + '|'
    body = []
    for name, values in rows:
        content = ['—' if values[k] is None else f'{values[k]:.3f}' for k in HEADLINE]
        body.append('| ' + name + ' | ' + ' | '.join(content) + ' |')
    return '\n'.join([head, rule, *body])


def pilot(output):
    payload = read(output / 'pilot_rescore.json')
    notes = ['# Cached pilot rescoring', '',
             f"Reproduced {payload['complete_cases']} of {payload['expected_cases']} cached dances. "
             f"Missing: {len(payload['missing'])}. New generations: 0.", '',
             'Rows weight music IDs equally after averaging starts. Historical diversity compares the two starts within a song. '
             'Legacy BAS is shown only as a raw-coordinate diagnostic. The position panels exclude the first eight conditioning frames.', '']
    for panel in ('legacy_full', 'position_full', 'position_20s'):
        timeline = [r for r in payload['cases'] if r['panel'] == panel]
        notes.extend([f'## {panel}', '', table([
            (f'{group} / {policy}', aggregate(song_means(
                [r for r in timeline if r['split'] == group and r['policy'] == policy], False)))
            for group, policy in product(('adaptation', 'held_out_from_rl'), ('supervised', 'epoch5'))
        ]), ''])
    (output / 'pilot_rescore.md').write_text('\n'.join(notes) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), sharex=True, sharey=True)
    for ax, panel in zip(axes, ('position_full', 'position_20s')):
        for group in ('adaptation', 'held_out_from_rl'):
            rows = [r for r in payload['pairs'] if r['panel'] == panel and r['split'] == group]
            ax.scatter([r['bas'] for r in rows], [r['f1_100ms'] for r in rows], label=group.replace('_', ' '))
        ax.axhline(0, color='0.7', linewidth=1)
        ax.axvline(0, color='0.7', linewidth=1)
        ax.set(title=panel.replace('_', ' '), xlabel='BAS change')
    axes[0].set_ylabel('Matched F1 change (100 ms)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'pilot_metric_changes.png', dpi=150)
    plt.close(fig)


def delta(post, pre):
    return {k: post[k] - pre[k] if post[k] is not None and pre[k] is not None else None for k in post}


def panel_cases(path, split, group, draws):
    rows = read(path)
    expected = set(product([p['song_id'] for p in split['pairs'] if p['split'] == group],
                           range(len(split['starts'])), range(draws)))
    actual = {(r['song'], r['start'], r['draw']) for r in rows}
    if expected != actual or len(rows) != len(actual):
        raise ValueError(f'{path}: missing {sorted(expected - actual)}, unexpected {sorted(actual - expected)}')
    return rows


def comparison(run, output, usage, app_id):
    config_path = ROOT / 'configs/midpoint_beat_frozen.json'
    split_path = ROOT / 'configs/midpoint_beat_split.json'
    cfg, split = read(config_path), read(split_path)
    arms = [f'{obj}-{seed}' for obj, seed in product(cfg['objectives'], cfg['training_seeds'])]
    policies = ['baseline', *arms]
    val, tr, test_cases, endpoints, costs = [], [], [], {}, {}
    phase_seconds = defaultdict(float)
    lineage = None
    for name in ['validation', *arms, 'test']:
        summary = read(run / name / 'summary.json')
        manifest = summary['manifest']
        if (not summary['frozen_verified'] or manifest['protocol_sha256'] != digest(config_path)
                or manifest['split_sha256'] != digest(split_path)):
            raise ValueError(f'Unverified comparison provenance for {name}')
        fields = ('baseline_sha256', 'initial_actor', 'initial_critic', 'frozen', 'trainable')
        comparability = {field: manifest[field] for field in fields}
        if lineage is None:
            lineage = comparability
        elif lineage != comparability:
            raise ValueError(f'Initial model lineage changed in {name}')
        costs[name] = summary['worker_seconds']
        phase_seconds['training'] += summary.get('train_seconds', 0)
        evaluation = (summary['worker_seconds'] - summary['setup_seconds']
                      if name in ('validation', 'test') else summary['evaluation_seconds'])
        phase_seconds['evaluation'] += evaluation
        phase_seconds['setup_checkpoint_and_checks'] += (
            summary['worker_seconds'] - summary.get('train_seconds', 0) - evaluation)
    baseline_val = panel_cases(run / 'validation/validation_000.json', split, 'validation', cfg['validation_draws'])
    for policy in policies:
        rows = panel_cases(run / 'test' / f'test_{policy}.json', split, 'test', cfg['test_draws'])
        endpoints[policy] = song_means(rows, True)
        test_cases.extend(dict(policy=policy, **row) for row in rows)
    for arm in arms:
        val.extend(dict(policy=arm, round=0, **r) for r in baseline_val)
        for step in cfg['validation_rounds'][1:]:
            rows = panel_cases(run / arm / f'validation_{step:03d}.json', split, 'validation', cfg['validation_draws'])
            val.extend(dict(policy=arm, round=step, **r) for r in rows)
        rounds = read(run / arm / 'training.json')
        if [r['round'] for r in rounds] != list(range(1, cfg['rounds']+1)):
            raise ValueError(f'Incomplete training for {arm}')
        if any(len(r['cases']) != cfg['batch'] for r in rounds):
            raise ValueError(f'Incomplete collection batch for {arm}')
        tr.extend(dict(policy=arm, **r) for r in rounds)
    if sum(len(r['cases']) for r in tr) != cfg['expected_counts']['training']:
        raise ValueError('Training collection count differs from the frozen protocol')
    if len(test_cases) != cfg['expected_counts']['test'] or (
            len(val) - (len(arms)-1)*len(baseline_val) != cfg['expected_counts']['validation']):
        raise ValueError('Evaluation collection count differs from the frozen protocol')
    for objective in cfg['objectives']:
        endpoints[objective] = {
            song: {key: mean([endpoints[f'{objective}-{seed}'][song][key]
                              for seed in cfg['training_seeds']]) for key in values}
            for song, values in endpoints['baseline'].items()
        }
    contrasts = [('bas', 'baseline', 'BAS seed mean vs baseline'),
                 ('bas', 'bin', 'BAS vs bin, seed mean')]
    contrasts.extend((arm, 'baseline', arm + ' vs baseline') for arm in arms)
    contrasts.extend((f'bas-{s}', f'bin-{s}', f'BAS vs bin, seed {s}') for s in cfg['training_seeds'])
    pairs = [dict(contrast=label, song=song, **delta(endpoints[post][song], endpoints[pre][song]))
             for post, pre, label in contrasts for song in endpoints[pre]]
    uncertainty = []
    for _, _, contrast in contrasts:
        rows = [r for r in pairs if r['contrast'] == contrast]
        rng = np.random.default_rng(cfg['eval_seed'])
        for metric in ('bas', 'f1_100ms'):
            values = np.asarray([r[metric] for r in rows])
            draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
            uncertainty.append(dict(contrast=contrast, metric=metric, n_songs=len(values),
                                    mean=float(values.mean()), song_bootstrap_95=np.quantile(draws, [.025, .975]).tolist()))
    ledger = {'worker_seconds': costs, 'phase_seconds': dict(phase_seconds)}
    rate = cfg['budget']['rate_usd_per_worker_hour']
    ledger['phase_compute_estimates_usd'] = {key: seconds*rate/3600 for key, seconds in phase_seconds.items()}
    if usage:
        from decimal import Decimal

        resources = defaultdict(Decimal)
        for row in read(usage):
            if row['object_id'] == app_id:
                resources[row['resource']] += Decimal(row['cost'])
        if not {'T4', 'Memory', 'CPU'} <= resources.keys():
            raise ValueError('No complete billing rows found for this application')
        ledger['app_id'] = app_id
        ledger['metered_resources_usd'] = {resource: str(value) for resource, value in resources.items()}
        ledger['metered_total_usd'] = str(sum(resources.values()))
    (output / 'comparison_costs.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (output / 'comparison.json').write_text(json.dumps(dict(endpoints=endpoints, test_cases=test_cases,
        validation_cases=val, rounds=tr, per_song_deltas=pairs, song_uncertainty=uncertainty, worker_seconds=costs),
        indent=2, allow_nan=False) + '\n')
    write_csv(output / 'test_cases.csv', test_cases)
    write_csv(output / 'test_song_deltas.csv', pairs)
    write_csv(output / 'training_rounds.csv', tr)
    scores = {p: aggregate(endpoints[p]) for p in policies}
    report = ['# Controlled comparison', '',
              f"Completed {cfg['expected_counts']['training']:,} training, "
              f"{cfg['expected_counts']['validation']:,} validation, and "
              f"{cfg['expected_counts']['test']:,} final test continuations.",
              '', table([
        ('Frozen', scores['baseline']),
        *[(obj + ' RL, seed mean', {k: mean([scores[f'{obj}-{s}'][k] for s in cfg['training_seeds']])
                                   for k in HEADLINE}) for obj in cfg['objectives']],
        *[(p, scores[p]) for p in arms]]), '',
        'Diversity compares stochastic draws within the same song and start. '
        'Song bootstrap intervals in comparison.json are conditional on each seed.', '',
        '## Paired primary outcomes', '']
    report.extend(f"- {r['contrast']}, {r['metric']} {r['mean']:+.4f} "
                  f"(song interval {r['song_bootstrap_95'][0]:+.4f} to {r['song_bootstrap_95'][1]:+.4f})"
                  for r in uncertainty)
    (output / 'comparison.md').write_text('\n'.join(report) + '\n')
    plot_comparison(cfg, arms, tr, val, pairs, scores, output)


def plot_comparison(cfg, arms, tr, val, pairs, scores, output):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for arm in arms:
        rows = [r for r in tr if r['policy'] == arm]
        axes[0, 0].plot([r['round'] for r in rows], [r['reward_mean'] for r in rows], label=arm)
        axes[0, 1].plot([r['round'] for r in rows], [r['reference_kl'] for r in rows], label=arm)
        for ax, key in zip(axes[1], ('bas', 'f1_100ms')):
            series = [aggregate(song_means([r for r in val if r['policy'] == arm and r['round'] == step], True))[key]
                      for step in cfg['validation_rounds']]
            ax.plot(cfg['validation_rounds'], series, marker='o', label=arm)
    for ax, title in zip(axes.flat, ('Training objective return', 'Training KL to frozen policy',
                                    'Validation BAS', 'Validation F1 (100 ms)')):
        ax.set(title=title, xlabel='Round')
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in sorted({r['contrast'] for r in pairs}):
        rows = [r for r in pairs if r['contrast'] == label]
        axes[0].scatter([r['bas'] for r in rows], [r['f1_100ms'] for r in rows], label=label)
    axes[0].axhline(0, color='0.7')
    axes[0].axvline(0, color='0.7')
    axes[0].set(xlabel='Paired BAS change', ylabel='Paired F1 change (100 ms)')
    axes[0].legend(fontsize=7)
    for name, values in scores.items():
        axes[1].plot([50, 100, 150], [values[f'coverage_{m}ms'] for m in (50, 100, 150)], marker='o', label=name)
    axes[1].set(xlabel='Tolerance (ms)', ylabel='Music event coverage', xticks=[50, 100, 150])
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'test_timing.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--usage', type=Path)
    parser.add_argument('--app-id')
    args = parser.parse_args()
    pilot(args.output)
    if args.run:
        comparison(args.run, args.output, args.usage, args.app_id)
