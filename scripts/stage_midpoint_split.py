"""Stage verified native 60 Hz features for four new music IDs from the indexed ZIP."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.public_archive import RangeReader
from scripts.rescore_midpoint_pilot import ROOT, digest

PILOT = ROOT / 'configs/pilot_split.json'
INDEX = ROOT / 'reports/batch01/preprocessed_data_index.json'
CONFIG = ROOT / 'configs/midpoint_beat_split.json'
TEST_IDS = ('mLH4', 'mLO2', 'mMH3', 'mWA0')
STAGED = ROOT / 'data/midpoint_beats'
START_FRAME = 8
STOP_FRAME = 1208


def main(fetch):
    pilot = json.loads(PILOT.read_text())
    index = json.loads(INDEX.read_text())
    valid = [m for m in index['members'] if m['name'].endswith('.json')]
    train = {Path(m['name']).stem.split('_')[-2] for m in valid
             if m['name'].startswith('data/aistpp_train_wav/')}
    exclude = set((ROOT / 'third_party/Bailando/ignore_list.txt').read_text().splitlines())
    source = {}
    for music in TEST_IDS:
        source[music] = min((m for m in valid if m['name'].startswith('data/aistpp_test_full_wav/')
                             and Path(m['name']).stem.split('_')[-2] == music
                             and Path(m['name']).name not in exclude), key=lambda m: m['name'])
    pairs = [{**pair, 'split': 'validation' if pair['split'] == 'held_out_from_rl' else 'adaptation',
              'previously_explored': True} for pair in pilot['pairs']]
    if fetch:
        STAGED.mkdir(parents=True, exist_ok=True)
        asset = index['source']
        url = 'https://drive.usercontent.google.com/download?id=' + asset['drive_id'] + '&export=download&confirm=t'
        with RangeReader(url, asset['reported_bytes']) as remote, zipfile.ZipFile(remote) as archive:
            for music, member in source.items():
                path = STAGED / (Path(member['name']).stem + '.json')
                info = archive.getinfo(member['name'])
                if info.file_size != member['bytes'] or f'{info.CRC:08x}' != member['crc32']:
                    raise ValueError(f'Archive index mismatch for {music}')
                if not path.exists():
                    path.write_bytes(archive.read(info))
                print(f'Staged {music}', flush=True)
    missing = []
    for music, member in source.items():
        record = Path(member['name']).stem
        path = STAGED / (record + '.json')
        if not path.exists():
            missing.append({'song_id': music, 'archive_member': member['name']})
            continue
        import zlib
        raw = path.read_bytes()
        if len(raw) != member['bytes'] or f'{zlib.crc32(raw):08x}' != member['crc32']:
            raise ValueError(f'Cached ZIP data mismatch for {music}')
        data = json.loads(raw)
        array = np.asarray(data['music_array'])
        if (data['id'] != record or array.ndim != 2 or array.shape[1] != 438
                or not np.isfinite(array).all()):
            raise ValueError(f'Invalid native features for {music}')
        features = STAGED / (music + '.json')
        features.write_text(json.dumps({'id': music, 'music_array': array[:, :55].tolist()}) + '\n')
        pairs.append({'song_id': music, 'genre_code': record[:3], 'record_id': record,
                      'split': 'test', 'previously_explored': False, 'archive_member': member['name'],
                      'source_zip_crc32': member['crc32'], 'source_sha256': digest(path),
                      'music_frames': len(array), 'external_features': str(features.relative_to(ROOT)),
                      'external_features_sha256': digest(features),
                      'song_in_released_train_wav_index': music in train})
    for pair in pairs:
        feature = ROOT / pair['external_features']
        native = np.asarray(json.loads(feature.read_text())['music_array'], dtype=np.float32)
        if (digest(feature) != pair['external_features_sha256']
                or native.shape != (pair['music_frames'], 55) or len(native) < STOP_FRAME
                or not np.isfinite(native).all() or not np.isin(native[:, 53], [0, 1]).all()
                or not native[START_FRAME:STOP_FRAME, 53].any()):
            raise ValueError(f"Ineligible timing support for {pair['song_id']}")
        pair['scored_music_beats'] = int(native[START_FRAME:STOP_FRAME, 53].sum())
        if pair['song_in_released_train_wav_index'] != (pair['song_id'] in train):
            raise ValueError(f"Train archive membership mismatch for {pair['song_id']}")
    if len({p['song_id'] for p in pairs}) != len(pairs):
        raise ValueError('Music identities overlap')
    manifest = {'status': 'test_assets_pending' if missing else 'frozen',
                'archive_index_sha256': digest(INDEX), 'historical_split_sha256': digest(PILOT),
                'released_full_track_music_ids': sorted({Path(m['name']).stem.split('_')[-2]
                    for m in valid if m['name'].startswith('data/aistpp_test_full_wav/')}),
                'music_fps': 60, 'scored_interval': [START_FRAME, STOP_FRAME],
                'starts': pilot['evaluation_initial_upper_lower_codes'],
                'pretraining_membership': 'Exact checkpoint membership is unknown.',
                'pairs': sorted(pairs, key=lambda p: (p['split'], p['song_id'])),
                'missing_test_assets': missing}
    if CONFIG.exists() and json.loads(CONFIG.read_text())['status'] == 'frozen':
        if json.loads(CONFIG.read_text()) != manifest:
            raise ValueError('Frozen split would change')
    else:
        CONFIG.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"{manifest['status']}: {len(pairs)} verified songs, {len(missing)} missing test tracks")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    main(parser.parse_args().fetch)
