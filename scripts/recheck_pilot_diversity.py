"""Recompute pilot diversity from cached motion; no model execution or GPU use."""
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/batch03/pilot-20261117T062722443178Z/evaluation'


def descriptor(raw, frames):
    xyz = raw[:frames].reshape(frames, 24, 3).copy()
    # Decoder root channels contain displacements, not root-relative positions.
    xyz[:, 0] = 0
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=2) * 60
    return np.concatenate([speed.mean(axis=0), speed.std(axis=0)])


def main():
    saved = json.loads((ROOT / 'reports/batch03/paired_results.json').read_text())
    cases = json.loads((RUN / 'evaluation.json').read_text())['cases']
    frames = min(row['frames'] for row in cases)
    output = {'source_run': str(RUN.relative_to(ROOT)), 'frames': frames,
              'seconds': frames / 60, 'training_seeds': 1,
              'starts_per_song': 2, 'decoding': 'greedy', 'groups': {}, 'songs': []}
    with np.load(RUN / 'motions.npz') as motion:
        features = {row['key']: descriptor(motion[row['key'] + '_raw'], frames) for row in cases}
        for song in saved['songs']:
            row = {'song': song['song'], 'split': song['split']}
            for i, policy in enumerate(('supervised', 'epoch5')):
                selected = [c for c in cases if c['song'] == song['song'] and c['policy'] == policy]
                assert len(selected) == 2
                distance = float(np.linalg.norm(features[selected[0]['key']] - features[selected[1]['key']]))
                np.testing.assert_allclose(distance, song['start_distance'][i], rtol=1e-6)
                row[policy] = distance
            row['delta'] = row['epoch5'] - row['supervised']
            output['songs'].append(row)
        for group, prior in saved['groups'].items():
            group_result = {}
            for old, policy in (('before', 'supervised'), ('after', 'epoch5')):
                selected = [c for c in cases if c['split'] == group and c['policy'] == policy]
                distances = [(a['song'] == b['song'], float(np.linalg.norm(features[a['key']] - features[b['key']])))
                             for a, b in combinations(selected, 2)]
                within = [d for same, d in distances if same]
                across = [d for same, d in distances if not same]
                assert (len(within), len(across)) == (3, 12)
                result = {'pooled': float(np.mean([d for _, d in distances])),
                          'within_song_across_starts': float(np.mean(within)),
                          'cross_song_only': float(np.mean(across)),
                          'within_song_pairs': len(within), 'cross_song_pairs': len(across)}
                np.testing.assert_allclose(result['pooled'], prior[old]['diversity'], rtol=1e-6)
                np.testing.assert_allclose(result['within_song_across_starts'],
                                           prior[old]['within_song_start_distance'], rtol=1e-6)
                group_result[policy] = result
            group_result['percent_change'] = {
                key: 100 * (group_result['epoch5'][key] / group_result['supervised'][key] - 1)
                for key in ('pooled', 'within_song_across_starts', 'cross_song_only')}
            output['groups'][group] = group_result
    output['source_sha256'] = {
        name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in ('motions.npz', 'evaluation.json')}
    output['validation'] = 'All recomputed pooled and within-song values match batch03 within rtol=1e-6.'
    output['limitations'] = [
        'Two starts give one within-song pair; this is not fixed-start stochastic diversity.',
        'The 48-dimensional speed summary loses ordering, pose semantics, and musical phrasing.',
        'Distances are in source units/s; larger distance alone does not establish better dance.',
        'Each group has three songs; genre and RL assignment coincide.',
        'One training seed and an already-inspected evaluation set support exploratory conclusions only.',
    ]
    path = ROOT / 'reports/midpoint/diversity_recheck.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
