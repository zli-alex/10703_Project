"""Verify the six pilot records and export their native policy features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.smoke_infer import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RECORDS = {
    'adaptation': ('gHO_sBM_cAll_d20_mHO5_ch01', 'gJB_sBM_cAll_d08_mJB5_ch01',
                   'gPO_sBM_cAll_d10_mPO1_ch01'),
    'held_out_from_rl': ('gBR_sBM_cAll_d04_mBR0_ch01', 'gJS_sBM_cAll_d01_mJS3_ch01',
                         'gKR_sBM_cAll_d28_mKR2_ch01'),
}
AUDIO_SOURCE = 'https://aistdancedb.ongaaccel.jp/v1.0.0/audio/wav/'


def prepare_pair(record_id, split, downloaded, train_songs, excluded, features):
    name = record_id + '.json'
    genre, song = record_id[:3], record_id.split('_')[-2]
    record_path = ROOT / 'assets/preprocessed_data/data/aistpp_test_full_wav' / name
    relative = str(record_path.relative_to(ROOT))
    digest = sha256_file(record_path)
    if name in excluded or digest != downloaded.get(relative):
        raise ValueError(f'Excluded record or archive hash mismatch: {record_id}')
    record = json.loads(record_path.read_text())
    music = np.asarray(record['music_array'], dtype=np.float64)
    motion = np.asarray(record['dance_array'], dtype=np.float64)
    if (record['id'] != record_id or music.ndim != 2 or music.shape[1] != 438
            or motion.ndim != 2 or motion.shape[1] != 72
            or len(music) < 1200 or len(motion) < 8
            or not (np.isfinite(music).all() and np.isfinite(motion).all())):
        raise ValueError(f'Unexpected ID, shape or values: {record_id}')
    beats = music[:, 53]
    if not np.isin(beats, [0.0, 1.0]).all() or beats.sum() == 0:
        raise ValueError(f'Invalid beat stream: {record_id}')

    audio = ROOT / 'assets/audio' / f'{song}.wav'
    audio_input = audio if audio.exists() else audio.with_suffix('.wav.partial')
    info = sf.info(audio_input)
    audio_seconds = info.frames / info.samplerate
    error = len(music) / 60 - audio_seconds
    if info.format != 'WAV' or info.samplerate != 48000 or info.channels != 2 or abs(error) > 1 / 60 + 1e-9:
        raise ValueError(f'Invalid waveform or feature clock disagreement: {song} ({error:.6f}s)')
    if audio_input != audio:
        audio_input.rename(audio)

    # Policy channels only, kept at the source timebase and precision.
    feature_path = features / f'{song}.json'
    feature_path.write_text(json.dumps({'id': song, 'music_array': [frame[:55] for frame in record['music_array']]}) + '\n')
    exported = np.asarray(json.loads(feature_path.read_text())['music_array'])
    if not np.array_equal(exported, music[:, :55]):
        raise ValueError(f'Feature export validation failed: {song}')
    return {'song_id': song, 'record_id': record_id, 'genre_code': genre, 'split': split,
            'music_frames': len(music), 'motion_frames': len(motion), 'music_beats': int(beats.sum()),
            'audio_seconds': round(audio_seconds, 6), 'clock_difference_ms': round(error * 1000, 6),
            'audio': str(audio.relative_to(ROOT)), 'audio_source': AUDIO_SOURCE + audio.name,
            'audio_sha256': sha256_file(audio), 'motion_record': relative, 'motion_record_sha256': digest,
            'external_features': str(feature_path.relative_to(ROOT)),
            'external_features_sha256': sha256_file(feature_path),
            'song_in_released_train_wav_index': song in train_songs}


def main():
    reports = ROOT / 'reports/batch01'
    archive = json.loads((reports / 'preprocessed_data_index.json').read_text())
    train_songs = {Path(member['name']).stem.split('_')[-2] for member in archive['members']
                   if member['name'].startswith('data/aistpp_train_wav/') and member['name'].endswith('.json')}
    downloads = json.loads((reports / 'downloaded_assets.json').read_text())
    downloaded = {row['path']: row['sha256'] for row in downloads if row['zip_crc_verified']}
    excluded = set((ROOT / 'third_party/Bailando/ignore_list.txt').read_text().splitlines())
    feature_dir = ROOT / 'data/pilot/music_feat_60fps'
    feature_dir.mkdir(parents=True, exist_ok=True)
    pairs = [prepare_pair(record, split, downloaded, train_songs, excluded, feature_dir)
             for split, records in RECORDS.items() for record in records]
    if len({pair['song_id'] for pair in pairs}) != 6:
        raise ValueError('Expected six unique music IDs')
    manifest = {'frame_rate_hz': 60, 'policy_feature_slice': [0, 55], 'music_beat_channel': 53,
                'evaluation_initial_upper_lower_codes': [[436, 289], [179, 387]],
                'selection': 'One full native track per genre in the released full-wave split. The available tracks imply different genres across groups.',
                'supervised_overlap_scope': 'Song IDs checked against the released train_wav archive index. Exact training membership of the supervised checkpoint is unverified.',
                'pairs': pairs}
    destination = ROOT / 'configs/pilot_split.json'
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    print(destination)
    print([(pair['song_id'], pair['split'], pair['music_frames'], pair['clock_difference_ms']) for pair in pairs])


if __name__ == '__main__':
    main()
