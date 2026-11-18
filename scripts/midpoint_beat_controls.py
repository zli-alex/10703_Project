"""Check cached motion controls and nonwrapping timing offsets on common support."""
import json
from pathlib import Path

import numpy as np

from scripts.beat_metrics import exported_positions, motion_events, score_events, score_motion
from scripts.rescore_midpoint_pilot import OUTPUT, ROOT, digest, write_csv


def main():
    path = ROOT / 'runs/batch01/rewardD-20261106T215539Z/results.npz'
    with np.load(path, allow_pickle=False) as cache:
        rows = []
        for i, name in enumerate(cache['names']):
            raw, beats = cache['raw'][i], cache['beats']
            legacy = score_motion(raw, beats, 0, 240, legacy=True)
            np.testing.assert_array_equal(legacy['bin_rewards'], cache['beat'][i])
            position = score_motion(raw, beats, 0, 240)
            rows.append(dict(control=str(name), legacy_reward=legacy['mean_reward'], **position))
        shifts = []
        shifts_frames = (-12, -6, -3, 0, 3, 6, 12)
        start, stop = 24, 216
        for i, name in enumerate(cache['names']):
            if name.lower() not in ('generated', 'real', 'reconstructed'):
                continue
            motion = motion_events(exported_positions(cache['raw'][i]))
            for shift in shifts_frames:
                score = score_events(np.flatnonzero(cache['beats']), motion+shift, start, stop)
                shifts.append(dict(control=str(name), shift_frames=shift, **score))
    if len(shifts) != 3 * len(shifts_frames):
        raise ValueError('Incomplete shifted control panel')
    write_csv(OUTPUT / 'motion_controls.csv', rows)
    write_csv(OUTPUT / 'offset_controls.csv', shifts)
    (OUTPUT / 'controls.json').write_text(json.dumps(
        dict(input_sha256=digest(path), motion_controls=rows, nonwrapping_shifts=shifts),
        indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
