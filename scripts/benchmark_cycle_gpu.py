"""Measure one disposable Bailando++ epoch on an explicit adaptation split."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from easydict import EasyDict

from infra.benchmark_inputs import SPLIT_CONFIGS, rollout_pairs
from infra.compat.released_test_loader import load_test_data_aist
from scripts import smoke_infer as smoke

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party' / 'Bailando'
SOURCE = UPSTREAM / 'tpami_bailandopp'


def released_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adaptation_inputs(directory, config, padding, split):
    pairs = rollout_pairs(split)
    expected = {pair['record_id'] + '.json': pair for pair in pairs}
    records = directory / 'motion'
    features = directory / 'features'
    if {path.name for path in records.iterdir()} != set(expected):
        raise ValueError('Rollout directory does not match the selected adaptation records')
    if {path.name for path in features.iterdir()} != {pair['song_id'] + '.json' for pair in pairs}:
        raise ValueError('Feature directory does not match the selected adaptation songs')
    for name, pair in expected.items():
        smoke.require_hash(records / name, pair['motion_record_sha256'])
        smoke.require_hash(features / (pair['song_id'] + '.json'), pair['external_features_sha256'])

    music, poses, names = load_test_data_aist(
        str(records), rotmat=config['rotmat'], move=config['move'],
        external_wav=str(features), external_wav_rate=config['external_wav_rate'],
        music_normalize=config['music_normalize'], wav_padding=padding,
    )
    loader_type = released_module('released_md_seq', SOURCE / 'dataset/md_seq.py').MoDaSeq
    loader = torch.utils.data.DataLoader(loader_type(music, poses), batch_size=1, shuffle=False)
    result = []
    for name in names:
        pair = expected[name]
        song = name.split('_')[-2]
        if song != pair['song_id']:
            raise ValueError(f'Unexpected song ID in {name}')
        beats = np.asarray(json.loads((records / name).read_text())['music_array'])[:, 53]
        if len(beats) != pair['music_frames'] or not np.isin(beats, [0, 1]).all():
            raise ValueError(f'Invalid native beat stream for {song}')
        result.append((song, beats))
    return loader, result


def load_models(output, assets, config):
    baseline = json.loads((ROOT / 'configs/baseline_assets.json').read_text())
    policy_path = assets / 'policy/epoch_400.pt'
    decoder_path = assets / 'decoder/epoch_500.pt'
    smoke.require_hash(policy_path, baseline['policy']['sha256'])
    smoke.require_hash(decoder_path, baseline['position_decoder']['sha256'])
    pc, dc = smoke.torch_load(policy_path), smoke.torch_load(decoder_path)
    models = smoke.import_models(output / 'work', UPSTREAM)
    decoder = models.SepVQVAER(dc['config'].structure)
    decoder_state, _ = smoke.strip_module_prefix(dc['model'])
    report = smoke.load_strict(decoder, decoder_state)
    if not report['loaded']:
        raise RuntimeError(report)
    policy_config = copy.deepcopy(pc['config'].structure_generate)
    policy_config.name = 'CrossCondGPT2MWAC'
    policy_config.critic_net = EasyDict(config['structure_generate']['critic_net'])
    policy = models.CrossCondGPT2MWAC(policy_config)
    actor_state, _ = smoke.strip_module_prefix(pc['model'])
    report = smoke.load_actor_critic(policy, actor_state)
    if not report['loaded']:
        raise RuntimeError(report)
    return policy.cuda(), decoder.cuda().eval(), policy_config, {
        name: baseline[name]['sha256'] for name in ('policy', 'position_decoder')
    }


def mark():
    torch.cuda.synchronize()
    return perf_counter()


def fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def generate(policy, decoder, loader, songs, config, stride):
    music_tracks, upper, lower, beats, records = [], [], [], [], []
    policy.eval()
    with torch.no_grad():
        for (padded_music, pose), (song, beat) in zip(loader, songs):
            started = mark()
            pose = pose.float().cuda()
            pose[:, :, :3] = 0
            music = padded_music[:, :, :config['structure_generate']['n_music']].float().cuda().contiguous()
            initial = tuple(part[0][:, :1] for part in decoder.encode(pose))
            sampled = policy.sample(initial, cond=music[:, stride:])
            upper.append(sampled[0][0][0].cpu().numpy())
            lower.append(sampled[1][0][0].cpu().numpy())
            expected_codes = (music.size(1) - 2 * config['wav_padding'] * stride) // stride
            if len(upper[-1]) != expected_codes or upper[-1].shape != lower[-1].shape:
                raise RuntimeError(f'Unexpected sampled length for {song}')
            music_tracks.append(music[0].cpu().numpy())
            beats.append(beat)
            records.append({'song_id': song, 'native_frames': len(beat), 'codes': len(upper[-1]),
                            'initial_codes': [int(upper[-1][0]), int(lower[-1][0])],
                            'rollout_seconds': mark() - started})
    return music_tracks, (upper, lower), beats, records


def gradients(module):
    present = [p.grad for p in module.parameters() if p.grad is not None]
    return {'present': len(present), 'nonzero': sum(bool(g.ne(0).any().item()) for g in present)}


def train_cycle(policy, decoder, reward, optimizer, loader, config, stride):
    records = []
    cycle = config['actor_iter'] + config['critic_iter']
    heads = {'actor': policy.gpt_head, 'critic': policy.critic_net}
    # Fingerprints and gradient inspection sit outside the phase timers.
    for index, batch in enumerate(loader):
        branch = 'actor' if index % cycle < config['actor_iter'] else 'critic'
        switch = not records or records[-1]['branch'] != branch
        before = {name: fingerprint(module) for name, module in heads.items()} if switch else {}
        started = mark()
        music, upper, lower, beats, mask = (tensor.cuda() for tensor in batch)
        codes = (upper.long(), lower.long())
        critic_input = tuple(code[:, :-1].contiguous() for code in codes)
        actor_input = tuple(code[:, :-2].contiguous() for code in codes)
        targets = tuple(code[:, 1:-1].contiguous() for code in codes)
        critic_music, actor_music = music[:, stride:], music[:, stride:-stride]
        with torch.no_grad():
            raw = decoder.decode(tuple([part] for part in targets))
            rewards = reward(raw, beats[:, stride:-stride], stride)
        scoring_finished = mark()

        policy.train()
        policy.freeze_drop()
        # Use the historical runtime's default zero_grad behavior with shared Adam state.
        optimizer.zero_grad()
        values = policy.critic(critic_input, critic_music)[:, :, 0]
        td = (rewards + config['gamma'] * values[:, 1:]).detach() - values[:, :-1]
        with torch.no_grad():
            policy.eval()
            _, _, entropy = policy.actor(actor_input, actor_music, targets, reduction=False)
        policy.train()
        policy.freeze_drop()
        alpha = config['entropy_alpha']
        td = td.reshape(-1) + alpha * entropy.detach()
        mask = mask.reshape(-1)
        if branch == 'actor':
            _, nll, entropy = policy.actor(actor_input, actor_music, targets, reduction=False)
            loss = ((nll * td.detach() - alpha * entropy) * mask).sum() / mask.sum()
            loss = loss * config['actor_loss_decay']
        else:
            loss = td.pow(2).mean()
        if not torch.isfinite(loss).item() or not torch.isfinite(rewards).all().item():
            raise ValueError(f'Nonfinite loss or reward at minibatch {index}')
        loss.backward()
        optimizer.step()
        finished = mark()
        entry = {'minibatch': index, 'branch': branch, 'size': len(upper),
                 'loss': loss.item(), 'td_mse': td.detach().pow(2).mean().item(),
                 'masked_reward_mean': ((rewards.reshape(-1) * mask).sum() / mask.sum()).item(),
                 'actor_mask_count': int(mask.sum().item()),
                 'scoring_seconds': scoring_finished - started, 'update_seconds': finished - scoring_finished}
        if switch:
            entry['heads'] = {name: {'changed': fingerprint(module) != before[name],
                                      'gradients': gradients(module)} for name, module in heads.items()}
            if not entry['heads'][branch]['changed'] or not entry['heads'][branch]['gradients']['nonzero']:
                raise RuntimeError(f'Inactive intended update at minibatch {index}')
        records.append(entry)
        print(json.dumps(entry), flush=True)
    return records


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def checkpoint(path, policy, optimizer, scheduler, config, policy_config, provenance, updates,
               *, epoch=1, history=None, verify=True):
    # The next update's legacy zero_grad acts on existing buffers. Save their
    # presence so a resumed epoch decays Adam momentum on the same parameters.
    state = {'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
             'python_rng': random.getstate(), 'config': config, 'policy_config': policy_config,
             'provenance': provenance, 'epoch': epoch, 'updates': updates, 'history': history,
             'gradient_buffers': [name for name, param in policy.named_parameters() if param.grad is not None]}
    started = mark()
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    saved = mark()
    details = {'bytes': path.stat().st_size, 'save_seconds': saved - started}
    if not verify:
        return details
    restored = smoke.torch_load(path)
    # Restore on fresh instances so the original Adam state cannot satisfy the check.
    fresh = type(policy)(policy_config).cuda()
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), **config['optimizer']['kwargs'])
    fresh_scheduler = torch.optim.lr_scheduler.MultiStepLR(fresh_optimizer, **config['optimizer']['schedular_kwargs'])
    restore_training_state(restored, fresh, fresh_optimizer, fresh_scheduler)
    current_numpy = np.random.get_state()
    passed = (
        equal_state(fresh.state_dict(), state['model'])
        and equal_state(fresh_optimizer.state_dict(), state['optimizer'])
        and equal_state(fresh_scheduler.state_dict(), state['scheduler'])
        and torch.equal(torch.get_rng_state(), state['torch_rng'])
        and equal_state(torch.cuda.get_rng_state_all(), state['cuda_rng'])
        and random.getstate() == state['python_rng']
        and current_numpy[0] == state['numpy_rng'][0]
        and np.array_equal(current_numpy[1], state['numpy_rng'][1])
        and current_numpy[2:] == state['numpy_rng'][2:]
        and [name for name, p in fresh.named_parameters() if p.grad is not None] == state['gradient_buffers']
        and restored['epoch'] == epoch and restored['updates'] == updates
        and restored['provenance'] == provenance and restored['history'] == history
    )
    if not passed:
        raise RuntimeError('Checkpoint reload did not reproduce model, optimizer, scheduler, RNG and progress')
    return {**details, 'reload_and_compare_seconds': mark() - saved, 'reload_pass': passed}


def restore_training_state(state, policy, optimizer, scheduler):
    policy.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    existing = set(state['gradient_buffers'])
    for name, parameter in policy.named_parameters():
        parameter.grad = torch.zeros_like(parameter) if name in existing else None
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])


def run_epoch(policy, decoder, reward, optimizer, rollout_loader, songs, config, stride, batch_cap):
    music, dances, beats, rollouts = generate(policy, decoder, rollout_loader, songs, config, stride)
    dataset_cls = released_module('released_ac_dataset', SOURCE / 'dataset/md_seq_ac.py').MoDaSeqAC
    interval = config['structure_generate']['block_size'] + 1
    dataset_start = mark()
    dataset = dataset_cls(music, dances, beats, interval=interval, music_dance_ratio=stride,
                          wav_padding=config['wav_padding'] * stride)
    windows = [record['codes'] - interval + 1 for record in rollouts]
    if len(dataset) != sum(windows) or len(dataset) < 10:
        raise RuntimeError('Expected at least ten usable released training windows')
    batch_size = min(batch_cap, config['batch_size'], (len(dataset) - 1) // 9)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                         num_workers=0, pin_memory=True)
    dataset_elapsed = mark() - dataset_start
    print(f'{len(dataset)} windows, batch size {batch_size}, {len(loader)} minibatches', flush=True)
    train_started = mark()
    updates = train_cycle(policy, decoder, reward, optimizer, loader, config, stride)
    train_elapsed = mark() - train_started
    branches = ['actor' if i % 10 < 5 else 'critic' for i in range(math.ceil(len(dataset) / batch_size))]
    if [row['branch'] for row in updates] != branches:
        raise RuntimeError('Incomplete epoch or branch schedule')
    return {'rollouts': rollouts, 'batch_size': batch_size, 'updates': updates,
            'windows_by_song': dict(zip((song for song, _ in songs), windows)),
            'dataset_seconds': dataset_elapsed, 'train_wall_seconds': train_elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('--assets', type=Path, default=Path('/vol/assets'))
    parser.add_argument('--split', choices=SPLIT_CONFIGS, default='development')
    parser.add_argument('--batch-cap', type=int, default=64, choices=range(1, 65), metavar='1–64')
    args = parser.parse_args()
    config = yaml.safe_load((SOURCE / 'configs/actor_critic_music_trans_400.yaml').read_text())
    if (config['actor_iter'], config['critic_iter'], config['pretrain_critic_epoch']) != (5, 5, 0):
        raise ValueError('Expected the released five-actor/five-critic schedule in epoch one')
    stride = config['ds_rate'] // config['music_relative_rate']
    padding = config['wav_padding'] * stride
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    started = mark()
    rollout_loader, songs = adaptation_inputs(args.inputs, config, padding, args.split)
    policy, decoder, policy_config, hashes = load_models(args.output, args.assets, config)
    frozen = {'music_encoder': policy.music_trans, 'gpt_base': policy.gpt_base, 'decoder': decoder}
    before = {name: fingerprint(module) for name, module in frozen.items()}
    reward_cls = released_module('released_reward', SOURCE / 'models/up_down_half_reward.py').UpDownReward
    reward = reward_cls(EasyDict(config['reward'])).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), **config['optimizer']['kwargs'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['optimizer']['schedular_kwargs'])
    record = run_epoch(policy, decoder, reward, optimizer, rollout_loader, songs, config, stride, args.batch_cap)
    rollouts, batch_size = record.pop('rollouts'), record.pop('batch_size')
    frozen_unchanged = {name: fingerprint(module) == before[name] for name, module in frozen.items()}
    if not all(frozen_unchanged.values()):
        raise RuntimeError(f'Unexpected changes to frozen modules: {frozen_unchanged}')
    peak_training = torch.cuda.max_memory_allocated()
    scheduler.step()
    provenance = {'baseline_sha256': hashes, 'rollouts': rollouts, 'batch_size': batch_size,
                  'split': args.split, 'split_manifest_sha256': smoke.sha256_file(ROOT / 'configs' / SPLIT_CONFIGS[args.split]),
                  'upstream_commit': json.loads((ROOT / 'configs/smoke_batch01.json').read_text())['upstream_commit']}
    ckpt = checkpoint(args.output / 'benchmark_checkpoint.pt', policy, optimizer, scheduler, config,
                      policy_config, provenance, len(record['updates']))
    result = {'provenance': provenance, **record,
              'checkpoint': ckpt, 'frozen_unchanged': frozen_unchanged,
              'worker_seconds': mark() - started, 'peak_training_bytes': peak_training,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'data_loader_workers': 0}
    (args.output / 'benchmark_results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
