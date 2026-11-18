"""Select the eligible rollout records for a benchmark."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CONFIGS = {'development': 'batch01_data.json', 'pilot': 'pilot_split.json'}


def rollout_pairs(split):
    manifest = json.loads((ROOT / 'configs' / SPLIT_CONFIGS[split]).read_text())
    pairs = manifest['pairs']
    for key in ('song_id', 'record_id'):
        if len({pair[key] for pair in pairs}) != len(pairs):
            raise ValueError(f'Duplicate {key} in {split} manifest')
    if split == 'development':
        if len(pairs) != 2 or any(pair['split'] != 'development_not_held_out' for pair in pairs):
            raise ValueError('Expected two development songs')
        return pairs

    adaptation = [pair for pair in pairs if pair['split'] == 'adaptation']
    held_out = [pair for pair in pairs if pair['split'] == 'held_out_from_rl']
    if len(pairs) != 6 or len(adaptation) != 3 or len(held_out) != 3:
        raise ValueError('Expected three adaptation songs and three songs held out from RL')
    return adaptation
