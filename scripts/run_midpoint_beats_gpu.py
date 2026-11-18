"""Execute one bounded midpoint stage in the established GPU runtime."""
import copy
import hashlib
import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts import benchmark_cycle_gpu as runtime
from scripts.beat_metrics import exported_positions, score_motion
from scripts.midpoint_actor_critic import collect, update
from scripts.rescore_midpoint_pilot import digest


def save_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def check_optimizers(actor_opt, critic_opt, policy):
    ids = [{id(p) for group in opt.param_groups for p in group['params']}
           for opt in (actor_opt, critic_opt)]
    intended = [{id(p) for p in module.parameters()} for module in (policy.gpt_head, policy.critic_net)]
    if ids != intended or ids[0] & ids[1] or ids[0] | ids[1] != {
            id(p) for p in policy.parameters() if p.requires_grad}:
        raise ValueError('Unexpected optimizer membership')


def state_to_cpu(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {key: state_to_cpu(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(state_to_cpu(value) for value in state)
    return state


class Run:
    def __init__(self, root, stage):
        self.root = root
        self.output = root / stage
        self.output.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load((runtime.SOURCE / 'configs/actor_critic_music_trans_400.yaml').read_text())
        self.protocol = json.loads((root / 'protocol.json').read_text())
        split = json.loads((root / 'split.json').read_text())
        self.starts = split['starts']
        self.first, self.stop = split['scored_interval']
        self.protocol_hash = digest(root / 'protocol.json')
        self.split_hash = digest(root / 'split.json')
        torch.manual_seed(self.protocol['critic_initialization_seed'])
        self.policy, self.decoder, _, self.hashes = runtime.load_models(self.output, Path('/vol/assets'), config)
        self.policy.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)
        self.actor, self.critic = self.policy.gpt_head, self.policy.critic_net
        self.reference = copy.deepcopy(self.actor)
        self.actor.requires_grad_(True)
        self.critic.requires_grad_(True)
        if self.first != self.policy.music_sample_size or self.stop != self.first + self.protocol['scored_frames']:
            raise ValueError('Protocol and model stride disagree')
        self.steps = (self.stop - self.first) // self.first
        self.temperature = self.protocol['temperature']
        self.frozen = (self.policy.music_trans, self.policy.gpt_base, self.decoder)
        self.initial_frozen = [runtime.fingerprint(m) for m in self.frozen]
        self.manifest = dict(baseline_sha256=self.hashes, split_sha256=self.split_hash,
                             protocol_sha256=self.protocol_hash, frozen=self.initial_frozen,
                             initial_actor=runtime.fingerprint(self.actor),
                             initial_critic=runtime.fingerprint(self.critic),
                             gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                             trainable=[n for n, p in self.policy.named_parameters() if p.requires_grad])
        betas = tuple(self.protocol['adam_betas'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.protocol['actor_lr'], betas=betas)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.protocol['critic_lr'], betas=betas)
        check_optimizers(self.actor_opt, self.critic_opt, self.policy)
        self.songs, self.groups = {}, {}
        for pair in split['pairs']:
            group, song = pair['split'], pair['song_id']
            self.groups.setdefault(group, []).append(song)
            if group == 'test' and stage != 'test':
                continue
            path = root / 'features' / (song + '.json')
            if digest(path) != pair['external_features_sha256']:
                raise ValueError(f'Feature hash mismatch for {song}')
            source = json.loads(path.read_text())
            native = np.asarray(source['music_array'], dtype=np.float32)
            if source['id'] != song or native.shape != (pair['music_frames'], 55):
                raise ValueError(f'Unexpected feature support for {song}')
            pad = self.policy.padding_size * self.first
            padded = np.pad(np.concatenate((native, native[-1:])), ((pad, pad), (0, 0)))
            self.songs[song] = (torch.from_numpy(padded[:self.stop + 2*pad]).cuda(), native[:, 53])

    def generate(self, panel, seed, training=False, audit=False, archive=None):
        music = torch.stack([self.songs[song][0] for song, _, _ in panel])
        initial = torch.tensor([self.starts[i] for _, i, _ in panel], device=music.device)
        started = runtime.mark()
        samples = collect(self.policy, music, initial, self.steps, self.temperature, seed, training, audit)
        generation_seconds = runtime.mark() - started
        with torch.no_grad():
            raw = self.decoder.decode(([samples.codes[:, :, 0].contiguous()],
                                       [samples.codes[:, :, 1].contiguous()])).cpu().numpy()
        if raw.shape != (len(panel), self.stop, 72) or samples.log_prob.shape != (len(panel), self.steps, 2):
            raise ValueError('Padding or decoded support differs from protocol')
        if not torch.isfinite(samples.log_prob).all() or not torch.equal(samples.codes[:, 0], initial):
            raise ValueError('Nonfinite likelihood or changed conditioning prefix')
        codes, logp = samples.codes.cpu().numpy(), samples.log_prob.cpu().numpy()
        rows = []
        for i, (song, start, draw) in enumerate(panel):
            row = score_motion(raw[i], self.songs[song][1], self.first, self.stop, full=not training)
            if row['normalized_bin'] is None or row['bas'] is None:
                raise ValueError(f'No eligible music beats for {song}')
            rows.append(dict(song=song, start=start, draw=draw,
                             generated_actions=self.steps, generated_frames=self.stop-self.first,
                             code_sha256=hashlib.sha256(codes[i, 1:].tobytes()).hexdigest(), **row))
        if archive is not None:
            np.savez_compressed(archive, codes=codes, log_prob=logp, raw=raw)
        timings = dict(generation_seconds=generation_seconds, decode_score_seconds=runtime.mark()-started-generation_seconds)
        return samples, rows, timings

    def evaluate(self, group, filename):
        panel = list(product(self.groups[group], range(len(self.starts)),
                             range(self.protocol[group + '_draws'])))
        rows = []
        batch = self.protocol['batch']
        for i in range(0, len(panel), batch):
            _, cases, _ = self.generate(panel[i:i+batch], self.protocol['eval_seed'] + i,
                                        archive=self.output / f'{filename}_{i:03d}.npz')
            rows.extend(cases)
        save_json(self.output / (filename + '.json'), rows)
        return rows

    def checkpoint(self, round_id, identity):
        path = self.output / 'last.pt'
        tmp = path.with_suffix('.pt.tmp')
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                        manifest=self.manifest, round=round_id, identity=identity,
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), tmp)
        tmp.replace(path)
        return path

    def restore(self, path, identity, round_id):
        state = torch.load(str(path), map_location='cpu')
        for key in ('baseline_sha256', 'split_sha256', 'protocol_sha256', 'frozen',
                    'initial_actor', 'initial_critic', 'trainable'):
            if state['manifest'][key] != self.manifest[key]:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        if state['identity'] != identity or state['round'] != round_id:
            raise ValueError('Checkpoint endpoint mismatch')
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        check_optimizers(self.actor_opt, self.critic_opt, self.policy)
        for name in ('actor', 'critic', 'actor_opt', 'critic_opt'):
            if not runtime.equal_state(state_to_cpu(getattr(self, name).state_dict()), state[name]):
                raise ValueError(f'Checkpoint round trip mismatch: {name}')

    def train_step(self, samples, rows, objective, audit):
        key = {'bin': 'normalized_bin', 'bas': 'bas'}[objective]
        rewards = [row[key] for row in rows]
        started = runtime.mark()
        numbers = update(self.policy, self.reference, samples,
                         torch.tensor(rewards, device=samples.codes.device, dtype=torch.float32),
                         self.actor_opt, self.critic_opt, self.temperature, audit=audit)
        return dict(update_seconds=runtime.mark()-started, reward_mean=float(np.mean(rewards)),
                    reward_std=float(np.std(rewards)), **numbers)

    def benchmark(self):
        # Four fixed song/start conditions; each yields four independent categorical trajectories.
        conditions = [(song, 0) for song in self.groups['adaptation']] + [(self.groups['adaptation'][0], 1)]
        panel = [(song, start, draw) for song, start in conditions for draw in range(4)]
        samples, rows, timings = self.generate(panel, self.protocol['reward_screen_seed'], True, True,
                                               self.output / 'reward_screen.npz')
        first = self.train_step(samples, rows, 'bas', True)
        del samples
        paired = []
        tol = self.protocol['reward_screen_tie_tolerance']
        for idx, (song, start) in enumerate(conditions):
            group = rows[idx*4:(idx+1)*4]
            deltas = np.asarray([(a['normalized_bin']-b['normalized_bin'], a['bas']-b['bas'])
                                 for a, b in combinations(group, 2)])
            paired.append(dict(song=song, start=start,
                               exact_ties=(deltas == 0).sum(axis=0).tolist(),
                               near_ties=(np.abs(deltas) <= tol).sum(axis=0).tolist(),
                               opposite_rank=int((deltas.prod(axis=1) < 0).sum()),
                               duplicate_pairs=sum(a['code_sha256'] == b['code_sha256']
                                                   for a, b in combinations(group, 2))))
        save_json(self.output / 'reward_screen.json', dict(cases=rows, paired=paired,
                  near_tie_tolerance=tol, reward_order=['bin', 'bas']))
        # Two one-trajectory collections check reload under the updated actor.
        args = (panel[:1], self.protocol['eval_seed'])
        expected, expected_metrics, _ = self.generate(*args, archive=self.output / 'decoded_before_reload.npz')
        archive = np.load(self.output / 'decoded_before_reload.npz')
        np.save(self.output / 'decoded_continuation.npy', exported_positions(archive['raw'][0])[self.first:])
        saved = self.checkpoint(1, 'benchmark')
        self.actor.load_state_dict(self.reference.state_dict())
        self.restore(saved, 'benchmark', 1)
        actual, actual_metrics, _ = self.generate(*args)
        if (not torch.equal(expected.codes, actual.codes)
                or not torch.equal(expected.log_prob, actual.log_prob) or expected_metrics != actual_metrics):
            raise ValueError('Reload changed fixed-seed evaluation')
        return dict(update=first, **timings, reload_verified=True,
                    checkpoint_sha256=digest(saved), counts={'reward_screen': 16, 'reload_evaluations': 2,
                                                             'train_updates': 1, 'sampled_actions': 18*self.steps,
                                                             'scored_frames': 18*(self.stop-self.first)})

    def train(self, objective, seed):
        records = []
        train_seconds = evaluation_seconds = checkpoint_seconds = 0.0
        songs = self.groups['adaptation']
        rounds = self.protocol['rounds']
        batch = self.protocol['batch']
        for idx in range(rounds):
            training_started = runtime.mark()
            slots = range(idx*batch, (idx+1)*batch)
            panel = [(songs[(i+seed) % len(songs)], (i//len(songs)) % len(self.starts), i) for i in slots]
            samples, cases, times = self.generate(panel, seed*100000+idx, True, idx == 0,
                                                   self.output / f'collection_{idx+1:03d}.npz')
            info = self.train_step(samples, cases, objective, idx in (0, rounds-1))
            del samples
            records.append(dict(round=idx+1, cases=cases, **times, **info))
            save_json(self.output / 'training.json', records)
            train_seconds += runtime.mark() - training_started
            if idx+1 in self.protocol['validation_rounds']:
                checkpoint_started = runtime.mark()
                self.checkpoint(idx+1, f'{objective}-{seed}')
                checkpoint_seconds += runtime.mark() - checkpoint_started
                evaluation_started = runtime.mark()
                self.evaluate('validation', f'validation_{idx+1:03d}')
                evaluation_seconds += runtime.mark() - evaluation_started
            print(f'Round {idx+1}/{rounds}', flush=True)
        final = self.output / 'last.pt'
        checkpoint_started = runtime.mark()
        self.restore(final, f'{objective}-{seed}', rounds)
        checkpoint_seconds += runtime.mark() - checkpoint_started
        return dict(rounds=rounds, train_cases=batch*rounds, checkpoint_sha256=digest(final),
                    train_seconds=train_seconds, evaluation_seconds=evaluation_seconds,
                    checkpoint_seconds=checkpoint_seconds)

    def test(self):
        # Check all endpoint records before reading any test generations.
        arms = []
        for objective, seed in product(self.protocol['objectives'], self.protocol['training_seeds']):
            name = f'{objective}-{seed}'
            path = self.root / name / 'last.pt'
            record = json.loads((path.parent / 'summary.json').read_text())
            if record['rounds'] != self.protocol['rounds'] or digest(path) != record['checkpoint_sha256']:
                raise ValueError(f'Unverified final checkpoint: {name}')
            arms.append((name, path))
        self.evaluate('test', 'test_baseline')
        for name, path in arms:
            self.restore(path, name, self.protocol['rounds'])
            self.evaluate('test', f'test_{name}')
        return dict(test_policies=1+len(arms))


def main(root, stage):
    started = runtime.mark()
    run = Run(root, stage)
    setup_seconds = runtime.mark()-started
    if stage == 'benchmark':
        summary = run.benchmark()
    elif stage == 'validation':
        summary = dict(validation_cases=len(run.evaluate('validation', 'validation_000')))
    elif stage == 'test':
        summary = run.test()
    else:
        objective, seed = stage.split('-')
        summary = run.train(objective, int(seed))
    if [runtime.fingerprint(m) for m in run.frozen] != run.initial_frozen:
        raise ValueError('A frozen component changed')
    save_json(run.output / 'summary.json',
              dict(**summary, manifest=run.manifest, setup_seconds=setup_seconds,
                   worker_seconds=runtime.mark()-started,
                   peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3, frozen_verified=True))


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2])
