"""Analytic and released-source differential checks runnable without Torch."""
import ast
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

from scripts.beat_metrics import exported_positions, motion_events, score_events, score_motion

ROOT = Path(__file__).resolve().parents[1]


def main():
    near, far = (score_events([8], [d], 0, 24) for d in (7, 15))
    assert near['hit_fraction'] == 0 and far['hit_fraction'] == 1
    assert score_events([8], [7], 1, 25)['hit_fraction'] == 1
    np.testing.assert_allclose([near['bas'], far['bas']], np.exp(-np.array([1, 49]) / 18))
    for offset in (3, 6, 9):
        row = score_events([20], [20 + offset], 0, 40)
        assert row[f'f1_{offset * 50 // 3}ms'] == 1
        np.testing.assert_allclose(row['bas'], np.exp(-offset**2 / 18))
    duplicates = score_events([8, 8, 17], [8, 8, 12, 17], 0, 32)
    assert duplicates['precision_100ms'] == .75 and duplicates['recall_100ms'] == 1
    assert duplicates['matches_100ms'] == [[8, 8], [8, 8], [17, 17]]
    assert score_events([8, 14], [2, 9], 0, 24)['matches_100ms'] == [[8, 2], [14, 9]]
    assert score_events([8, 14], [8], 0, 24)['recall_100ms'] == .5
    empty = score_events([8], [], 0, 24)
    assert empty['bas'] == empty['f1_100ms'] == 0 and empty['precision_100ms'] is None
    assert empty['music_to_motion_mean_ms'] is None
    assert score_events([], [], 0, 24)['bas'] is None
    assert score_events([], [8], 0, 24)['precision_100ms'] is None
    stationary = score_motion(np.zeros((48, 72)), np.eye(1, 48, 10)[0], 8, 48)
    assert stationary['status'] == 'no_motion' and stationary['stationary_fraction'] == 1
    assert score_events([7, 8, 23, 24], [7, 8, 23, 24], 8, 24)['music_events'] == [8, 23]

    upstream = ROOT / 'third_party/Bailando/tpami_bailandopp/utils/beat_align_score.py'
    functions = [node for node in ast.parse(upstream.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in {'calc_db', 'BA'}]
    env = {'np': np, 'G': gaussian_filter, 'argrelextrema': argrelextrema}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(upstream), 'exec'), env)
    fixture_count = 0
    for directory in sorted({p.parent for p in (ROOT / 'runs/batch01').glob('*/**/motion_global_positions.npy')}):
        raw = np.load(directory / 'motion_raw_reward.npy')
        saved = np.load(directory / 'motion_global_positions.npy')
        reconstructed = exported_positions(raw)
        np.testing.assert_array_equal(reconstructed, saved)
        for positions in (reconstructed, reconstructed[:1200]):
            (events,), length = env['calc_db'](positions)
            music = np.arange(15, length, 25)
            np.testing.assert_array_equal(events, motion_events(positions))
            np.testing.assert_allclose(score_events(music, events, 0, length)['bas'],
                                       env['BA'](music, (events,)), atol=1e-14, rtol=0)
            fixture_count += 1
    assert fixture_count
    result = {'analytic_and_empty_fixtures': 'passed', 'upstream_bas_fixtures': fixture_count,
              'cached_export_exact': fixture_count // 2}
    (ROOT / 'reports/midpoint_beats/evaluator_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
