"""Run the declared five-epoch Bailando++ pilot on the adaptation split."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict

from scripts import benchmark_cycle_gpu as cycle
from scripts import smoke_infer as smoke

EPOCHS = 5
BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output
    path = output / 'pilot_checkpoint.pt'
    if path.exists() != args.resume:
        raise ValueError('An existing pilot checkpoint requires --resume')
    config = yaml.safe_load((cycle.SOURCE / 'configs/actor_critic_music_trans_400.yaml').read_text())
    if (config['seed'], config['batch_size'], config['actor_iter'], config['critic_iter'],
            config['pretrain_critic_epoch']) != (42, BATCH_SIZE, 5, 5, 0):
        raise ValueError('Released configuration disagrees with the declared pilot')
    stride = config['ds_rate'] // config['music_relative_rate']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = True
    started = cycle.mark()
    loader, songs = cycle.adaptation_inputs(output / 'inputs', config, config['wav_padding'] * stride, 'pilot')
    policy, decoder, policy_config, hashes = cycle.load_models(output, Path('/vol/assets'), config)
    frozen = {'music_encoder': policy.music_trans, 'gpt_base': policy.gpt_base, 'decoder': decoder}
    before = {name: cycle.fingerprint(module) for name, module in frozen.items()}
    reward_type = cycle.released_module('released_reward', cycle.SOURCE / 'models/up_down_half_reward.py').UpDownReward
    reward = reward_type(EasyDict(config['reward'])).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), **config['optimizer']['kwargs'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config['optimizer']['schedular_kwargs'])
    provenance = {
        'baseline_sha256': hashes,
        'upstream_commit': json.loads((cycle.ROOT / 'configs/smoke_batch01.json').read_text())['upstream_commit'],
        'split_manifest_sha256': smoke.sha256_file(cycle.ROOT / 'configs/pilot_split.json'),
        'songs': [song for song, _ in songs], 'seed': config['seed'], 'batch_size': BATCH_SIZE,
        'planned_epochs': EPOCHS, 'data_loader_workers': 0,
        'initial_critic_sha256': cycle.fingerprint(policy.critic_net),
    }
    history = []
    if args.resume:
        restored = smoke.torch_load(path)
        history = restored['history']
        if (restored['provenance'] != provenance or restored['config'] != config
                or restored['epoch'] != len(history) or not 1 <= len(history) <= EPOCHS):
            raise ValueError('Checkpoint does not match the declared pilot')
        cycle.restore_training_state(restored, policy, optimizer, scheduler)
        del restored
    checkpoint_details = None
    for epoch in range(len(history) + 1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}', flush=True)
        result = cycle.run_epoch(policy, decoder, reward, optimizer, loader, songs, config, stride, BATCH_SIZE)
        if result['batch_size'] != BATCH_SIZE or len(result['updates']) != 12:
            raise RuntimeError('Unexpected batch size or update count in pilot epoch')
        history.append({'epoch': epoch, **result})
        scheduler.step()
        checkpoint_details = cycle.checkpoint(
            path, policy, optimizer, scheduler, config, policy_config, provenance, epoch * 12,
            epoch=epoch, history=history, verify=epoch == EPOCHS,
        )
        print(json.dumps({'saved_epoch': epoch, 'checkpoint': checkpoint_details}), flush=True)

    if checkpoint_details is None:
        # A completed checkpoint can survive an interruption during final verification or reporting.
        checkpoint_details = cycle.checkpoint(path, policy, optimizer, scheduler, config, policy_config,
                                             provenance, EPOCHS * 12, epoch=EPOCHS, history=history)
    unchanged = {name: cycle.fingerprint(module) == before[name] for name, module in frozen.items()}
    if not all(unchanged.values()) or not checkpoint_details['reload_pass']:
        raise RuntimeError(f'Pilot verification failed: {unchanged}')
    checkpoint_details['sha256'] = smoke.sha256_file(path)
    results = {'provenance': provenance, 'epochs': history, 'checkpoint': checkpoint_details,
               'frozen_unchanged': unchanged, 'worker_seconds': cycle.mark() - started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'gpu': torch.cuda.get_device_name(),
               'torch': torch.__version__}
    temporary = output / 'pilot_results.tmp'
    temporary.write_text(json.dumps(results, indent=2) + '\n')
    temporary.replace(output / 'pilot_results.json')


if __name__ == '__main__':
    main()
