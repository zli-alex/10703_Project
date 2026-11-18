"""Generate and score matched full-track rollouts from supervised and epoch 5."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

from scripts import benchmark_cycle_gpu as cycle
from scripts import smoke_infer as smoke


def load_features(directory, pairs, padding):
    if {path.name for path in directory.iterdir()} != {p['song_id'] + '.json' for p in pairs}:
        raise ValueError('Evaluation feature set differs from the declared split')
    songs = {}
    for pair in pairs:
        song = pair['song_id']
        path = directory / (song + '.json')
        smoke.require_hash(path, pair['external_features_sha256'])
        source = json.loads(path.read_text())
        native = np.asarray(source['music_array'], dtype=np.float32)
        if source['id'] != song or native.shape != (pair['music_frames'], 55) or not np.isfinite(native).all():
            raise ValueError(f'Invalid evaluation features for {song}')
        # Released loader at external_wav_rate=1 always appends one last frame.
        music = np.pad(np.concatenate((native, native[-1:])), ((padding, padding), (0, 0)))
        songs[song] = (torch.from_numpy(music[None]).cuda(), native[:, 53])
    return songs


def nearest(query, reference):
    if not len(query) or not len(reference):
        return None
    return np.abs(query[:, None] - reference[None, :]).min(axis=1)


def motion_metrics(raw, beat, upper, lower, rewards):
    frames = len(raw)
    relative = raw.reshape(frames, 24, 3).copy()
    relative[:, 0] = 0
    speed_per_joint = np.linalg.norm(np.diff(relative, axis=0), axis=2) * 60
    mixed_speed = np.sqrt(np.sum(np.diff(raw.reshape(frames, 24, 3), axis=0) ** 2, axis=2)).mean(axis=1)
    motion_beats = argrelextrema(gaussian_filter(mixed_speed, 5), np.less)[0]
    music_beats = np.flatnonzero(beat)
    hit_bins = np.intersect1d(motion_beats // 8, music_beats // 8).size
    music_bin = beat.reshape(-1, 8).max(axis=1).astype(bool)
    expected = np.full(len(rewards), 5.0)
    expected[music_bin] = -5
    expected[np.intersect1d(motion_beats // 8, music_beats // 8)] = 5
    if not np.array_equal(expected, rewards):
        raise ValueError('Independent beat-bin rewards differ from released reward')
    motion_distance = nearest(motion_beats, music_beats)
    music_distance = nearest(music_beats, motion_beats)
    repeating = (upper[1:] == upper[:-1]) & (lower[1:] == lower[:-1])
    boundaries = np.r_[0, np.flatnonzero(~repeating) + 1, len(upper)]
    return {
        'frames': frames, 'mean_reward': float(rewards.mean()),
        'mean_reward_music_bins': float(rewards[music_bin].mean()),
        'positive_bin_fraction': float((rewards > 0).mean()), 'music_beat_bins': int(music_bin.sum()),
        'hit_bins': int(hit_bins), 'hit_fraction': float(hit_bins / music_bin.sum()),
        'music_beats': len(music_beats), 'motion_beats': len(motion_beats),
        'motion_to_music_ms': float(motion_distance.mean() * 1000 / 60) if motion_distance is not None else None,
        'music_coverage_100ms': float((music_distance <= 6).mean()) if music_distance is not None else 0.0,
        'mean_joint_speed': float(speed_per_joint.mean()),
        'rms_acceleration': float(np.sqrt(np.mean((np.diff(relative, n=2, axis=0) * 60**2) ** 2))),
        'rms_jerk': float(np.sqrt(np.mean((np.diff(relative, n=3, axis=0) * 60**3) ** 2))),
        'root_travel': float(np.linalg.norm(raw[:-1, :3], axis=1).sum()),
        'unique_upper': len(np.unique(upper)), 'unique_lower': len(np.unique(lower)),
        'max_repeated_pair_run': int(np.diff(boundaries).max()),
    }, motion_beats


def evaluate(policy, decoder, reward, songs, pairs, starts, stride, condition):
    records, arrays = [], {}
    policy.eval()
    with torch.no_grad():
        for pair in pairs:
            song = pair['song_id']
            music, beat = songs[song]
            for index, codes in enumerate(starts, 1):
                print(f'Evaluation {condition} {song} start {index}', flush=True)
                initial = tuple(torch.tensor([[code]], device='cuda', dtype=torch.long) for code in codes)
                sampled = policy.sample(initial, cond=music[:, stride:])
                upper, lower = (part[0][0].cpu().numpy() for part in sampled)
                raw = decoder.decode(sampled)
                if tuple(upper[:1]) + tuple(lower[:1]) != tuple(codes) or raw.size(1) != len(upper) * stride:
                    raise ValueError(f'Rollout length or start mismatch for {song}')
                frames = min(len(beat), raw.size(1)) // stride * stride
                if len(beat) - frames >= stride:
                    raise ValueError(f'Rollout dropped a complete music bin for {song}')
                window = raw[:, :frames]
                native_beats = beat[:frames]
                score = reward(window, torch.from_numpy(native_beats[None]).cuda(), stride)[0].cpu().numpy()
                decoded = window[0].cpu().numpy()
                if not np.isfinite(score).all() or not np.isfinite(decoded).all():
                    raise ValueError(f'Nonfinite rollout or reward for {song}')
                metrics, motion_beats = motion_metrics(decoded, native_beats, upper[:frames//8], lower[:frames//8], score)
                key = f'{song}_I{index}_{condition}'
                records.append({'key': key, 'song': song, 'split': pair['split'], 'start': index,
                                'initial_codes': codes, 'policy': condition, 'generated_frames': raw.size(1), **metrics})
                arrays.update({f'{key}_{name}': array for name, array in
                               {'raw': decoded, 'codes': np.stack((upper, lower), axis=1),
                                'reward': score, 'motion_beats': motion_beats, 'music_beats': native_beats}.items()})
    return records, arrays


def main(output):
    started = cycle.mark()
    manifest_path = cycle.ROOT / 'configs/pilot_split.json'
    manifest = json.loads(manifest_path.read_text())
    config = yaml.safe_load((cycle.SOURCE / 'configs/actor_critic_music_trans_400.yaml').read_text())
    stride = config['ds_rate'] // config['music_relative_rate']
    pilot = json.loads((output.parent / 'pilot_results.json').read_text())
    provenance = pilot['provenance']
    if (len(pilot['epochs']) != 5 or pilot['epochs'][-1]['epoch'] != 5 or not pilot['checkpoint']['reload_pass']
            or provenance['split_manifest_sha256'] != smoke.sha256_file(manifest_path)):
        raise ValueError('Expected a verified epoch-5 checkpoint with the selected evaluation split')
    checkpoint_path = output.parent / 'pilot_checkpoint.pt'
    smoke.require_hash(checkpoint_path, pilot['checkpoint']['sha256'])
    songs = load_features(output / 'features', manifest['pairs'], config['wav_padding'] * stride)
    policy, decoder, _, hashes = cycle.load_models(output, Path('/vol/assets'), config)
    if hashes != provenance['baseline_sha256']:
        raise ValueError('Evaluation baseline differs from the training baseline')
    reward_type = cycle.released_module('released_reward', cycle.SOURCE / 'models/up_down_half_reward.py').UpDownReward
    reward = reward_type(EasyDict(config['reward'])).cuda()
    starts = manifest['evaluation_initial_upper_lower_codes']
    baseline, arrays = evaluate(policy, decoder, reward, songs, manifest['pairs'], starts, stride, 'supervised')
    checkpoint = smoke.torch_load(checkpoint_path)
    if checkpoint['epoch'] != 5 or checkpoint['provenance'] != provenance:
        raise ValueError('Checkpoint lineage mismatch')
    policy.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    trained, rl_arrays = evaluate(policy, decoder, reward, songs, manifest['pairs'], starts, stride, 'epoch5')
    arrays.update(rl_arrays)
    for pre, post in zip(baseline, trained):
        paired_fields = ('song', 'split', 'start', 'initial_codes', 'frames', 'generated_frames', 'music_beats')
        if any(pre[field] != post[field] for field in paired_fields):
            raise ValueError(f'Unmatched endpoint for {pre["key"]}')
        changed = arrays[pre['key'] + '_codes'] != arrays[post['key'] + '_codes']
        post['code_disagreement_fraction_vs_supervised'] = float(changed[1:].mean())
    np.savez_compressed(output / 'motions.npz', **arrays)
    (output / 'evaluation.json').write_text(json.dumps({
        'checkpoint_sha256': pilot['checkpoint']['sha256'], 'baseline_sha256': hashes,
        'split_manifest_sha256': provenance['split_manifest_sha256'], 'cases': baseline + trained,
        'worker_seconds': cycle.mark() - started, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
