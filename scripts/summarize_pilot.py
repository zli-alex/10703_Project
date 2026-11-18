"""Build a paired comparison, training curves, and review clips for a completed pilot."""
import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from scripts.render_skeleton import _probe, render_skeleton_mp4
from scripts.smoke_infer import global_positions, require_hash

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ('mean_reward', 'hit_fraction', 'motion_to_music_ms', 'music_coverage_100ms',
           'mean_joint_speed', 'rms_acceleration', 'rms_jerk', 'root_travel')
GROUPS = {'adaptation': 'Adaptation', 'held_out_from_rl': 'Held out from RL'}


def mean(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def delta(before, after):
    return after - before if before is not None and after is not None else None


def positions(raw):
    integrated = raw.copy()
    integrated[0, :3] = 0
    integrated[1:, :3] = np.cumsum(raw[:-1, :3], axis=0)
    return global_positions(integrated)


def speed_description(raw):
    xyz = raw.reshape(-1, 24, 3).copy()
    xyz[:, 0] = 0
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=2) * 60
    return np.concatenate((speed.mean(axis=0), speed.std(axis=0)))


def aggregate(rows, motions):
    starts = defaultdict(list)
    for case in rows:
        starts[case['song'], case['start']].append(case)
    if len(starts) != 12 or any([x['policy'] for x in values] != ['supervised', 'epoch5'] for values in starts.values()):
        raise ValueError('Expected two matched conditions for twelve song/start combinations')
    paired = []
    for (song, start), (before, after) in starts.items():
        payload = {key: before[key] for key in ('split', 'frames', 'initial_codes')}
        paired.append({'song': song, 'start': start, **payload,
                       'keys': [before['key'], after['key']],
                       'before': {metric: before[metric] for metric in METRICS},
                       'after': {metric: after[metric] for metric in METRICS},
                       'delta': {metric: delta(before[metric], after[metric]) for metric in METRICS},
                       'before_hits': before['hit_bins'], 'after_hits': after['hit_bins'],
                       'music_beat_bins': before['music_beat_bins'],
                       'changed_code_fraction': after['code_disagreement_fraction_vs_supervised']})
    by_song = defaultdict(list)
    for pair in paired:
        by_song[pair['song']].append(pair)
    common_frames = min(case['frames'] for case in rows)
    descriptors = {case['key']: speed_description(motions[case['key'] + '_raw'][:common_frames]) for case in rows}
    songs = []
    for song, matches in by_song.items():
        if len(matches) != 2:
            raise ValueError(f'Expected two matched starts for {song}')
        songs.append({'song': song, 'split': matches[0]['split'],
                      **{condition: {metric: mean(row[condition][metric] for row in matches) for metric in METRICS}
                         for condition in ('before', 'after', 'delta')},
                      'before_hits': sum(row['before_hits'] for row in matches),
                      'after_hits': sum(row['after_hits'] for row in matches),
                      'music_beat_bins': sum(row['music_beat_bins'] for row in matches),
                      'start_distance': [float(np.linalg.norm(descriptors[matches[0]['keys'][i]]
                                                               - descriptors[matches[1]['keys'][i]])) for i in range(2)]})
    groups = {}
    for split in GROUPS:
        selected = [song for song in songs if song['split'] == split]
        group_pairs = [row for row in paired if row['split'] == split]
        group = {condition: {metric: mean(song[condition][metric] for song in selected) for metric in METRICS}
                 for condition in ('before', 'after', 'delta')}
        for j, name in enumerate(('before', 'after')):
            group[name]['diversity'] = mean(np.linalg.norm(descriptors[a['keys'][j]] - descriptors[b['keys'][j]])
                                           for a, b in combinations(group_pairs, 2))
            group[name]['within_song_start_distance'] = mean(song['start_distance'][j] for song in selected)
        groups[split] = group
    return paired, songs, groups, common_frames


def plot(output, training, songs):
    epochs = [epoch['epoch'] for epoch in training['epochs']]
    rewards = [np.average([row['masked_reward_mean'] for row in epoch['updates']],
                          weights=[row['actor_mask_count'] for row in epoch['updates']]) for epoch in training['epochs']]
    errors = [np.average([row['td_mse'] for row in epoch['updates']],
                         weights=[row['size'] for row in epoch['updates']]) for epoch in training['epochs']]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3), layout='constrained')
    for ax, values, label in zip(axes, (rewards, errors), ('Mean sampled reward', 'Mean squared TD error')):
        ax.plot(epochs, values, marker='o', color='#126e82')
        ax.set(xlabel='Epoch', ylabel=label, xticks=epochs)
        ax.grid(axis='y', alpha=.25)
    fig.savefig(output / 'training_curves.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout='constrained', sharey=True)
    for ax, metric, label, scale in zip(axes, ('mean_reward', 'hit_fraction', 'rms_jerk'),
                                        ('Mean reward', 'Beat-bin hits (pp)', 'RMS jerk'), (1, 100, 1)):
        labels = [song['song'] for song in songs]
        ax.barh(labels, [scale * song['delta'][metric] for song in songs],
                color=['#126e82' if song['split'] == 'adaptation' else '#ba6b35' for song in songs])
        ax.axvline(0, color='#333333', linewidth=.8)
        ax.set_xlabel(label)
    axes[0].invert_yaxis()
    fig.suptitle('Epoch 5 minus supervised, averaged across the two starts per song')
    fig.savefig(output / 'paired_deltas.png', dpi=150)
    plt.close(fig)
    return [dict(epoch=i, sampled_reward=float(r), td_mse=float(e)) for i, r, e in zip(epochs, rewards, errors)]


def render_pair(destination, pair, motions, source):
    count = min(pair['frames'], 16 * 60)
    views = [positions(motions[key + '_raw'][:count]) for key in pair['keys']]
    joint_offsets = np.concatenate([view - view[:, :1] for view in views])
    span = float(np.percentile(np.abs(joint_offsets[:, :, :2]), 99))
    with TemporaryDirectory() as temp:
        paths = [Path(temp) / f'{i}.mp4' for i in range(2)]
        for path, view, label in zip(paths, views, ('Supervised', 'RL epoch 5')):
            render_skeleton_mp4(view, path, None, size=(384, 320), projection_span=span, label=label)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(paths[0]), '-i', str(paths[1]),
                        '-i', str(source), '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]', '-map', '[v]',
                        '-map', '2:a:0', '-t', f'{count/60:.6f}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', str(destination)], check=True)
    probe = _probe(destination)
    streams = {stream['codec_type']: stream for stream in json.loads(probe['json'])['streams']}
    if (int(streams['video']['nb_frames']) != count or streams['video']['r_frame_rate'] != '60/1'
            or abs(float(streams['audio']['duration']) - count/60) > .05):
        raise ValueError(f'Unexpected comparison clip alignment in {destination}')


def table(output, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with output.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads((args.run / 'pilot_results.json').read_text())
    evaluation = json.loads((args.run / 'evaluation/evaluation.json').read_text())
    if evaluation['checkpoint_sha256'] != training['checkpoint']['sha256']:
        raise ValueError('Evaluation checkpoint does not match the completed pilot')
    split = json.loads((ROOT / 'configs/pilot_split.json').read_text())
    entries = {row['song_id']: row for row in split['pairs']}
    with np.load(args.run / 'evaluation/motions.npz') as motions:
        pairs, songs, groups, common_frames = aggregate(evaluation['cases'], motions)
        ordered = sorted(pairs, key=lambda pair: (pair['delta']['mean_reward'], pair['song'], pair['start']))
        selected = [('largest', ordered[-1]), ('middle', ordered[len(ordered)//2]), ('smallest', ordered[0])]
        disagreements = [pair for pair in pairs if pair['delta']['mean_reward'] > 0 and pair['delta']['rms_jerk'] > 0]
        if disagreements:
            candidate = max(disagreements, key=lambda pair: pair['delta']['rms_jerk'])
            if all(pair != candidate for _, pair in selected):
                selected.append(('reward_and_jerk_up', candidate))
        clips = []
        for label, pair in selected:
            title = f'{label}_{pair["song"]}_I{pair["start"]}.mp4'
            entry = entries[pair['song']]
            source = ROOT / entry['audio']
            require_hash(source, entry['audio_sha256'])
            render_pair(output / title, pair, motions, source)
            clips.append({'selection': label, 'file': title, 'song': pair['song'], 'start': pair['start'],
                          'reward_delta': pair['delta']['mean_reward']})
    curves = plot(output, training, songs)
    table(output / 'evaluation_cases.csv', evaluation['cases'])
    table(output / 'paired_song_deltas.csv', [{'song': s['song'], 'split': s['split'], **s['delta']} for s in songs])
    for name, data in (('pilot_results.json', training),
                       ('paired_results.json', {'groups': groups, 'songs': songs, 'pairs': pairs, 'clips': clips,
                                                'common_diversity_frames': common_frames, 'training_curves': curves,
                                                'evaluation_worker_seconds': evaluation['worker_seconds'],
                                                'checkpoint_sha256': evaluation['checkpoint_sha256']})):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'groups': groups, 'clips': clips, 'curves': curves}, indent=2))


if __name__ == '__main__':
    main()
