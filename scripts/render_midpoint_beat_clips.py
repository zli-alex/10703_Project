"""Render synchronized coded panels for prespecified song/start/draw comparisons."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import requests
import soundfile as sf

from scripts.beat_metrics import exported_positions
from scripts.render_skeleton import _probe, render_skeleton_mp4
from scripts.rescore_midpoint_pilot import ROOT, digest

AUDIO_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/audio/wav/'


def waveform(song, frames, fetch):
    path = ROOT / 'assets/audio' / (song + '.wav')
    if fetch and not path.exists():
        with requests.get(AUDIO_URL + path.name, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            payload = bytearray()
            for block in response.iter_content(1024**2):
                payload.extend(block)
                if len(payload) > 24 * 1024**2:
                    raise ValueError(f'Audio exceeded the bounded download for {song}')
        partial = path.with_suffix('.midpoint.partial')
        partial.write_bytes(payload)
        path = partial
    info = sf.info(path)
    if info.samplerate != 48000 or abs(info.duration - frames / 60) > 1/60 + 1e-9:
        raise ValueError(f'Audio/feature clock mismatch for {song}')
    if path.suffix == '.partial':
        path = path.rename(path.with_name(song + '.wav'))
    return path


def render_panel(paths, labels, audio, start, destination):
    output = [exported_positions(raw)[start:] for raw in paths]
    span = float(np.percentile(np.abs(np.concatenate([p - p[:, :1] for p in output])[:, :, :2]), 99))
    with TemporaryDirectory() as tmp:
        files = [Path(tmp) / f'{label}.mp4' for label in labels]
        for path, label, pose in zip(files, labels, output):
            render_skeleton_mp4(pose, path, None, size=(384, 320), projection_span=span, label=label)
        command = ['ffmpeg', '-y', '-loglevel', 'error']
        for file in files:
            command.extend(['-i', str(file)])
        command.extend(['-ss', f'{start/60:.9f}', '-i', str(audio)])
        if len(files) > 1:
            layout = ''.join(f'[{i}:v]' for i in range(len(files)))
            command.extend(['-filter_complex', layout + f'hstack=inputs={len(files)}[v]', '-map', '[v]'])
        else:
            command.extend(['-map', '0:v:0'])
        command.extend(['-map', f'{len(files)}:a:0', '-t', f'{len(output[0])/60:.9f}',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(destination)])
        subprocess.run(command, check=True)
    streams = {s['codec_type']: s for s in json.loads(_probe(destination)['json'])['streams']}
    if (int(streams['video']['nb_frames']) != len(output[0])
            or streams['video']['r_frame_rate'] != '60/1'
            or abs(float(streams['audio']['duration']) - len(output[0])/60) > .05):
        raise ValueError(f'Unexpected panel duration: {destination}')


def main(run, benchmark, fetch, output):
    split = json.loads((ROOT / 'configs/midpoint_beat_split.json').read_text())
    start, _ = split['scored_interval']
    test = [p for p in split['pairs'] if p['split'] == 'test'][:2]
    if fetch:
        for pair in test:
            audio = waveform(pair['song_id'], pair['music_frames'], True)
            print(pair['song_id'], digest(audio))
    output.mkdir(parents=True, exist_ok=True)
    if benchmark:
        case = json.loads((benchmark / 'reward_screen.json').read_text())['cases'][0]
        pair = next(p for p in split['pairs'] if p['song_id'] == case['song'])
        with np.load(benchmark / 'decoded_before_reload.npz', allow_pickle=False) as archive:
            render_panel([archive['raw'][0]], ['Smoke rollout'], ROOT / pair['audio'], start,
                         output / 'smoke_continuation.mp4')
    if run:
        cfg = json.loads((ROOT / 'configs/midpoint_beat_frozen.json').read_text())
        seed = cfg['training_seeds'][0]
        policies = ['baseline', f'bin-{seed}', f'bas-{seed}']
        rng = np.random.default_rng(cfg['eval_seed'])
        key = []
        for pair in test:
            song = pair['song_id']
            order = rng.permutation(policies)
            raw = []
            for policy in order:
                prefix = run / 'test' / f'test_{policy}'
                rows = json.loads(prefix.with_suffix('.json').read_text())
                i, = [i for i, r in enumerate(rows) if (r['song'], r['start'], r['draw']) == (song, 0, 0)]
                batch, within = divmod(i, cfg['batch'])
                with np.load(prefix.with_name(f'{prefix.name}_{batch*cfg["batch"]:03d}.npz'),
                             allow_pickle=False) as archive:
                    raw.append(archive['raw'][within])
            labels = ['A', 'B', 'C']
            audio = waveform(song, pair['music_frames'], False)
            render_panel(raw, labels, audio, start, output / f'{song}_review.mp4')
            key.append(dict(song=song, seed=seed, start=0, draw=0, audio_sha256=digest(audio),
                            identities=dict(zip(labels, order.tolist()))))
        (run / 'clip_key.json').write_text(json.dumps(key, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--fetch-audio', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/midpoint_beats/review')
    args = parser.parse_args()
    main(args.run, args.benchmark, args.fetch_audio, args.output)
