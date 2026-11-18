"""Stochastic midpoint rollout and Monte Carlo update on the released modules."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class Collection:
    codes: torch.Tensor
    log_prob: torch.Tensor
    states: list


def last_logits(head, state):
    upper, lower, _, _, _, _ = head(state)
    return torch.stack((upper[:, -1], lower[:, -1]), dim=1)


def conditions(starts, stop_frame, stride, padding):
    """Reproduce released sampler music windows after the one-code offset."""
    for k in range(stop_frame // stride - 1):
        end = (k + 1 + 2 * padding) * stride
        left = max(0, k + 1 - starts)
        yield left, slice(left * stride, end)


@torch.no_grad()
def collect(policy, music, initial, steps, temperature, seed, keep_states=False, audit=False):
    if policy.training or any(m.training for m in policy.modules()):
        raise ValueError('Policy and all child modules must be in eval mode')
    rng = torch.Generator(device=music.device)
    rng.manual_seed(seed)
    history = [initial]
    log_prob, states = [], []
    stride = policy.music_sample_size
    cond = music[:, stride:]
    windows = conditions(policy.get_block_size()-1, (steps+1)*stride, stride, policy.padding_size)
    for k, (left, span) in enumerate(windows):
        tokens = torch.stack(history[left:], dim=1)
        music_window = cond[:, span]
        if music_window.shape[1] != (tokens.shape[1] + 2*policy.padding_size)*stride:
            raise ValueError('Music and autoregressive history differ in support')
        state = policy.gpt_base(tokens[:, :, 0], tokens[:, :, 1], policy.music_trans(music_window))
        logits = last_logits(policy.gpt_head, state)
        if audit and k in {0, policy.get_block_size()-2, policy.get_block_size()-1, steps-1}:
            # Independently replay the released sample loop's default shift using forward().
            released_left = 0 if k < policy.get_block_size()-1 else k+2-policy.get_block_size()
            end = (k+1+2*policy.padding_size)*stride
            exact_history = torch.stack(history[released_left:], dim=1)
            upstream, _ = policy((exact_history[:, :, 0], exact_history[:, :, 1]),
                                 cond[:, released_left*stride:end])
            expected = torch.stack([part[:, -1] for part in upstream], dim=1)
            if (expected - logits).abs().max().item() > 3e-5:
                raise ValueError(f'Generation differs from released forward logits at action {k}')
        logp = F.log_softmax(logits / temperature, dim=-1)
        action = torch.multinomial(logp.exp().reshape(-1, logp.shape[-1]), 1, generator=rng)
        action = action.reshape(initial.shape)
        log_prob.append(logp.gather(-1, action[..., None]).squeeze(-1))
        history.append(action)
        if keep_states:
            states.append(state)
    return Collection(torch.stack(history, dim=1), torch.stack(log_prob, dim=1), states)


def grad_norm(module):
    value = sum(p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None)
    norm = value.sqrt().item()
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f'Unexpected gradient norm: {norm}')
    return norm


def update(policy, reference, collection, returns, actor_opt, critic_opt, temperature, audit=False):
    """Accumulate one update across exact generation states, without replay reuse."""
    actor, critic = policy.gpt_head, policy.critic_net
    if not collection.states or not torch.isfinite(returns).all():
        raise ValueError('Missing behavior states or invalid returns')
    actor_opt.zero_grad()
    critic_opt.zero_grad()
    advantages, rows = [], []
    normalizer = len(collection.states) * len(returns)
    for k, state in enumerate(collection.states):
        action = collection.codes[:, k+1]
        logits = last_logits(actor, state) / temperature
        logp = F.log_softmax(logits, dim=-1)
        actual = logp.gather(-1, action[..., None]).squeeze(-1)
        error = (actual.detach() - collection.log_prob[:, k]).abs().max().item()
        if error > 2e-5:
            raise ValueError(f'Behavior/update log likelihood differs at step {k}: {error}')
        value = last_logits(critic, state).sum(dim=1).squeeze(-1)
        advantage = returns - value.detach()
        actor_loss = -(actual.sum(dim=1) * advantage).sum() / normalizer
        critic_loss = (returns - value).square().sum() / normalizer
        actor_loss.backward()
        critic_loss.backward()
        with torch.no_grad():
            ref = F.log_softmax(last_logits(reference, state) / temperature, dim=-1)
            entropy = -(logp.exp() * logp).sum(dim=(1, 2)).mean().item()
            kl = (logp.exp() * (logp - ref)).sum(dim=(1, 2)).mean().item()
        rows.append((actor_loss.item(), critic_loss.item(), entropy, kl, error))
        advantages.append(advantage)
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite update diagnostics')
    actor_norm, critic_norm = grad_norm(actor), grad_norm(critic)
    if audit:
        from scripts.benchmark_cycle_gpu import fingerprint

        critic_before = fingerprint(critic)
        actor_before = fingerprint(actor)
    actor_opt.step()
    if audit:
        if fingerprint(critic) != critic_before or fingerprint(actor) == actor_before:
            raise ValueError('Actor step changed the wrong parameters')
        actor_after = fingerprint(actor)
    critic_opt.step()
    if audit and (fingerprint(actor) != actor_after or fingerprint(critic) == critic_before):
        raise ValueError('Critic step changed the wrong parameters')
    advantages = torch.cat(advantages)
    values = np.asarray(rows)
    return dict(actor_loss=float(values[:, 0].sum()), critic_loss=float(values[:, 1].sum()),
                entropy=float(values[:, 2].mean()), reference_kl=float(values[:, 3].mean()),
                likelihood_max_error=float(values[:, 4].max()), actor_grad_norm=actor_norm,
                critic_grad_norm=critic_norm, advantage_mean=advantages.mean().item(),
                advantage_std=advantages.std(unbiased=False).item())
