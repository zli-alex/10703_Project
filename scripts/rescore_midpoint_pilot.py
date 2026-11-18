"""Reproduce the historical raw path and rescore all cached case pairs."""
import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from scripts.beat_metrics import FPS, joints, score_motion

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports/midpoint_beats'
PILOT = ROOT / 'runs/batch03/pilot-20261117T062722443178Z/evaluation'
METRICS = ('bas', 'mean_reward', 'hit_fraction', 'coverage_50ms', 'coverage_100ms',
           'coverage_150ms', 'precision_100ms', 'recall_100ms', 'f1_100ms',
           'music_to_motion_mean_ms', 'motion_to_music_mean_ms', 'motion_events_per_s',
           'mean_joint_speed', 'rms_jerk', 'stationary_fraction')


def digest(path):
    with path.open('rb') as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    scalars = [{key: val for key, val in row.items() if not isinstance(val, (dict, list))}
               for row in rows]
    keys = list(dict.fromkeys(key for row in scalars for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(scalars)


def reproduce(case, raw, codes, beats, cache, key):
    score = score_motion(raw, beats, 0, len(raw), legacy=True)
    np.testing.assert_array_equal(score['bin_rewards'], cache[key + '_reward'])
    np.testing.assert_array_equal(score['motion_events'], cache[key + '_motion_beats'])
    aliases = {'music_coverage_100ms': 'coverage_100ms', 'motion_to_music_ms': 'motion_to_music_mean_ms'}
    keys = ('mean_reward', 'hit_fraction', 'hit_bins', 'music_beat_bins', 'music_beats',
            'motion_beats', 'music_coverage_100ms', 'motion_to_music_ms', 'mean_joint_speed',
            'rms_jerk', 'root_travel')
    for field in keys:
        np.testing.assert_allclose(score[aliases.get(field, field)], case[field], rtol=1e-6, atol=1e-7)
    positive = np.asarray(score['bin_rewards']) > 0
    np.testing.assert_allclose(positive.mean(), case['positive_bin_fraction'])
    music_bins = beats.reshape(-1, 8).max(axis=1).astype(bool)
    np.testing.assert_allclose(np.asarray(score['bin_rewards'])[music_bins].mean(),
                               case['mean_reward_music_bins'], rtol=1e-6)
    relative = joints(raw).copy()
    relative[:, 0] = 0
    acceleration = np.diff(relative, n=2, axis=0) * FPS**2
    np.testing.assert_allclose(np.sqrt(np.mean(acceleration**2)), case['rms_acceleration'], rtol=1e-6)
    if codes[0].tolist() != case['initial_codes']:
        raise ValueError(f'Cached initial codes differ for {key}')
    scored_codes = codes[:len(raw)//8]
    for channel, name in enumerate(('unique_upper', 'unique_lower')):
        if np.unique(scored_codes[:, channel]).size != case[name]:
            raise ValueError(f'Cached code inventory differs for {key}')
    changes = np.flatnonzero(np.any(scored_codes[1:] != scored_codes[:-1], axis=1)) + 1
    if np.diff(np.r_[0, changes, len(scored_codes)]).max() != case['max_repeated_pair_run']:
        raise ValueError(f'Cached code repetition differs for {key}')
    before_key = key.replace('_epoch5', '_supervised') + '_codes'
    if case['policy'] == 'epoch5' and before_key in cache:
        before = cache[before_key]
        np.testing.assert_equal((before[1:] != codes[1:]).mean(),
                                case['code_disagreement_fraction_vs_supervised'])
    return score


def main(pilot, output):
    manifest = ROOT / 'configs/pilot_split.json'
    split = json.loads(manifest.read_text())
    evaluation = json.loads((pilot / 'evaluation.json').read_text())
    if evaluation['split_manifest_sha256'] != digest(manifest):
        raise ValueError('Cached evaluation split hash mismatch')
    cases = {case['key']: case for case in evaluation['cases']}
    rows, missing, pairs = [], [], []
    expected = {f"{p['song_id']}_I{i}_{policy}"
                for p, i, policy in product(split['pairs'], (1, 2), ('supervised', 'epoch5'))}
    with np.load(pilot / 'motions.npz', allow_pickle=False) as cache:
        for key in sorted(expected):
            fields = [key + '_' + name for name in ('raw', 'codes', 'reward', 'motion_beats', 'music_beats')]
            absent = [name for name in fields if name not in cache]
            if key not in cases or absent:
                missing.append({'key': key, 'missing_record': key not in cases, 'missing_arrays': absent})
                continue
            case = cases[key]
            raw, codes, _, _, beats = (cache[name] for name in fields)
            if len(raw) != case['frames'] or len(codes) * 8 != case['generated_frames']:
                raise ValueError(f'Cached length mismatch for {key}')
            identity = {field: case[field] for field in ('key', 'song', 'split', 'start', 'policy')}
            historical = {field: value for field, value in case.items() if field not in identity}
            rows.append({**identity, 'panel': 'legacy_full', **historical,
                         **reproduce(case, raw, codes, beats, cache, key)})
            for panel, stop in [('position_full', len(raw)), ('position_20s', 1208)]:
                row = {**identity, 'panel': panel, **score_motion(raw, beats, 8, stop)}
                rows.append(row)
    lookup = {(r['panel'], r['song'], r['start'], r['policy']): r for r in rows}
    for (panel, song, start, policy), pre in lookup.items():
        after = lookup.get((panel, song, start, 'epoch5'))
        if policy != 'supervised' or after is None:
            continue
        pairs.append({'panel': panel, 'song': song, 'start': start, 'split': pre['split'],
                      **{key: after[key] - pre[key] if after[key] is not None and pre[key] is not None
                         else None for key in METRICS}})
    output.mkdir(parents=True, exist_ok=True)
    result = {'expected_cases': len(expected), 'complete_cases': len(rows) // 3,
              'missing': missing, 'unexpected_records': sorted(set(cases) - expected),
              'baseline_sha256': evaluation['baseline_sha256'],
              'pilot_checkpoint_sha256': evaluation['checkpoint_sha256'],
              'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                               (manifest, pilot / 'evaluation.json', pilot / 'motions.npz')},
              'cases': rows, 'pairs': pairs}
    (output / 'pilot_rescore.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    write_csv(output / 'pilot_cases.csv', rows)
    write_csv(output / 'pilot_pairs.csv', pairs)
    print(f"Historical cases: {result['complete_cases']}/{len(expected)}. Missing: {len(missing)}.")
    if missing or result['unexpected_records']:
        raise SystemExit('Incomplete or unexpected pilot cases; see pilot_rescore.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=PILOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.pilot, args.output)
