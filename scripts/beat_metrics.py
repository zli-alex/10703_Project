"""CPU timing metrics for Bailando++ positional output at the native 60 Hz."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

FPS = 60
BIN = 8
TOLERANCES = {50: 3, 100: 6, 150: 9}


def joints(array):
    array = np.asarray(array)
    if array.ndim not in (2, 3) or array.shape[1:] not in ((72,), (24, 3)):
        raise ValueError(f'Expected [T,72] or [T,24,3], got {array.shape}')
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError('Expected finite floating-point coordinates')
    return array.reshape(-1, 24, 3)


def exported_positions(raw):
    """Integrate the previous frame's root delta, then translate other joints."""
    positions = joints(raw).copy()
    root = np.zeros_like(positions[:, 0])
    root[1:] = np.cumsum(positions[:-1, 0], axis=0)
    positions[:, 1:] += root[:, None]
    positions[:, 0] = root
    return positions


def motion_events(positions):
    """Left index of each strict interior minimum, with reflect smoothing."""
    speed = np.sqrt(np.sum(np.diff(joints(positions), axis=0) ** 2, axis=2)).mean(axis=1)
    return argrelextrema(gaussian_filter(speed, 5, mode='reflect'), np.less)[0]


def nearest(query, reference):
    if len(query) and len(reference):
        return np.abs(query[:, None] - reference[None, :]).min(axis=1)
    return None


def match_events(music, motion, tolerance):
    """Ordered absolute-error matching. Uncrossing preserves feasibility and cost."""
    count = np.zeros((len(music) + 1, len(motion) + 1), dtype=np.int32)
    error = np.zeros_like(count)
    choice = np.zeros_like(count, dtype=np.uint8)
    for i, m in enumerate(music, 1):
        for j, d in enumerate(motion, 1):
            candidates = [(count[i-1, j], -error[i-1, j], 1),
                          (count[i, j-1], -error[i, j-1], 2)]
            delta = abs(int(m) - int(d))
            if delta <= tolerance:
                candidates.append((count[i-1, j-1] + 1, -error[i-1, j-1] - delta, 3))
            hits, neg_error, move = max(candidates)
            count[i, j], error[i, j], choice[i, j] = hits, -neg_error, move
    result = []
    i, j = len(music), len(motion)
    while i and j:
        move = choice[i, j]
        if move == 3:
            result.append([int(music[i-1]), int(motion[j-1])])
        i -= move in (1, 3)
        j -= move in (2, 3)
    return result[::-1]


def score_events(music, motion, start, stop, full=True):
    """Absolute frame indices, scored on [start, stop), bins anchored at start.

    Only complete bins enter the bin reward. All frames in the interval enter
    event metrics. Duplicates remain distinct occurrences for matching.
    """
    if not 0 <= start < stop:
        raise ValueError('Invalid scoring interval')
    selected = []
    for events in (music, motion):
        events = np.asarray(events)
        if events.ndim != 1 or (events.size and not np.issubdtype(events.dtype, np.integer)):
            raise ValueError('Expected a vector of integer frame indices')
        selected.append(np.sort(events[(events >= start) & (events < stop)].astype(np.int64)))
    music, motion = selected
    bins = (stop - start) // BIN
    music_bins = np.unique((music[music < start + BIN * bins] - start) // BIN)
    motion_bins = np.unique((motion[motion < start + BIN * bins] - start) // BIN)
    hits = np.intersect1d(music_bins, motion_bins)
    rewards = np.full(bins, 5.0)
    rewards[np.setdiff1d(music_bins, hits)] = -5
    dist = nearest(music, motion)
    bas = float(np.exp(-dist**2 / 18).mean()) if dist is not None else (0.0 if len(music) else None)
    result = {
        'bas': bas,
        'mean_reward': float(rewards.mean()) if bins and len(music) else None,
        'normalized_bin': float((rewards > 0).mean()) if bins and len(music) else None,
        'hit_fraction': float(len(hits) / len(music_bins)) if len(music_bins) else None,
        'music_beat_bins': len(music_bins), 'hit_bins': len(hits),
        'music_beats': len(music), 'motion_beats': len(motion),
        'motion_events_per_s': len(motion) * FPS / (stop - start),
    }
    if not full:
        return result
    result.update(start_frame=start, stop_frame=stop, bin_origin=start,
                  music_events=music.tolist(), motion_events=motion.tolist(),
                  bin_rewards=rewards.tolist(),
                  status='no_music' if not len(music) else 'no_motion' if not len(motion) else 'ok')
    for direction, distances in [('music_to_motion', dist), ('motion_to_music', nearest(motion, music))]:
        for name, reduce in [('mean', np.mean), ('median', np.median)]:
            result[f'{direction}_{name}_ms'] = (float(reduce(distances) * 1000 / FPS)
                                                if distances is not None else None)
    for ms, frames in TOLERANCES.items():
        pairs = match_events(music, motion, frames)
        k = len(pairs)
        result.update({
            f'coverage_{ms}ms': float((dist <= frames).mean()) if dist is not None else (0.0 if len(music) else None),
            f'precision_{ms}ms': k / len(motion) if len(music) and len(motion) else None,
            f'recall_{ms}ms': k / len(music) if len(music) else None,
            f'f1_{ms}ms': 2 * k / (len(music) + len(motion)) if len(music) else None,
            f'matches_{ms}ms': pairs,
        })
    return result


def quality(raw):
    """Root-relative motion statistics, in decoder units, sampled at 60 Hz."""
    relative = joints(raw).copy()
    relative[:, 0] = 0
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1) * FPS
    return {
        'mean_joint_speed': float(speed.mean()),
        'stationary_fraction': float((speed.mean(axis=1) < 0.05).mean()),
        'rms_jerk': float(np.sqrt(np.mean((np.diff(relative, n=3, axis=0) * FPS**3) ** 2))),
        'root_travel': float(np.linalg.norm(joints(raw)[:-1, 0], axis=-1).sum()),
        'speed_signature': np.r_[speed.mean(axis=0), speed.std(axis=0)].tolist(),
    }


def score_motion(raw, beats, start, stop, *, legacy=False, full=True):
    if stop > len(raw) or stop > len(beats) or not np.isin(beats, [0, 1]).all():
        raise ValueError('Invalid motion/music support or nonbinary beat stream')
    events = motion_events(raw if legacy else exported_positions(raw))
    result = score_events(np.flatnonzero(beats), events, start, stop, full=full)
    if full:
        result.update(quality(raw[start:stop]))
    return result
