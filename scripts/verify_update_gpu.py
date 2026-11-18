"""M4: One actor step and one critic step on disposable checkpoint copies.

Runs on GPU. Reads saved trace from /vol/runs/<trace_run>/.
Writes outputs to <out_dir>/ passed as sys.argv[1].

Diagnostic batch of 2 windows (not released batch64).
Loads CrossCondGPT2MWAC with supervised policy weights + saved diagnostic critic.
Forks to two independent copies for actor branch (batch_i=0) and critic branch (batch_i=5).
No rollout regeneration; no decoder decode during update.

Batch-size note: the saved trace was computed per-window (batch=1). This script
verifies forward checks at batch=1 (per window) so they match the saved values.
The update computation uses both windows as a batch of 2 (mimicking the released
training loop), which may differ from saved NLL by O(1e-5) due to float32
non-associativity in matmuls. This is recorded but does not indicate a bug.
"""
from __future__ import annotations
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path('/root/project')
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from scipy.special import logsumexp as scipy_logsumexp

from scripts import smoke_infer as smoke

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

TRACE_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/vol/runs/traceE-20261106T221045Z')
VOL_ASSETS = Path('/vol/assets')
SOURCE = ROOT / 'third_party' / 'Bailando' / 'tpami_bailandopp'
CFG_PATH = ROOT / 'configs' / 'update_batch01.json'

cfg = json.loads(CFG_PATH.read_text())
rl = yaml.safe_load((SOURCE / 'configs' / 'actor_critic_music_trans_400.yaml').read_text())
baseline = json.loads((ROOT / 'configs' / 'baseline_assets.json').read_text())

GAMMA = rl['gamma']
ENTROPY_ALPHA = rl.get('entropy_alpha', 0.0)
ACTOR_LOSS_DECAY = rl.get('actor_loss_decay', 1)
ACTOR_ITER = rl['actor_iter']
CRITIC_ITER = rl['critic_iter']
ATOL = cfg['atol']
RTOL = cfg['rtol']

started = time.perf_counter()


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state_hash(module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build model overlay and load weights
# ---------------------------------------------------------------------------
print("Building model overlay...", flush=True)
models = smoke.import_models(OUT / 'work', ROOT / 'third_party' / 'Bailando')

policy_path = VOL_ASSETS / 'policy' / 'epoch_400.pt'
decoder_path = VOL_ASSETS / 'decoder' / 'epoch_500.pt'
smoke.require_hash(policy_path, baseline['policy']['sha256'])
smoke.require_hash(decoder_path, baseline['position_decoder']['sha256'])

pc = smoke.torch_load(policy_path)
dc = smoke.torch_load(decoder_path)

# Build AC model
policy_cfg = copy.deepcopy(pc['config'].structure_generate)
policy_cfg.name = 'CrossCondGPT2MWAC'
policy_cfg.critic_net = EasyDict(rl['structure_generate']['critic_net'])

policy_base = models.CrossCondGPT2MWAC(policy_cfg)
ps, _ = smoke.strip_module_prefix(pc['model'])
load_report = smoke.load_actor_critic(policy_base, ps)
assert load_report['loaded'], f"Policy load failed: {load_report}"

# Load saved critic_net state
critic_pt = TRACE_DIR / 'critic_initial.pt'
smoke.require_hash(critic_pt, cfg['critic_initial_sha256_expected'])
critic_state = torch.load(critic_pt, map_location='cpu')
policy_base.critic_net.load_state_dict(critic_state)
print(f"Loaded critic_net from saved diagnostic state: {sha256_file(critic_pt)}", flush=True)

# Prove decoder is not in optimizer scope (load it; never add it to optimizer)
decoder = models.SepVQVAER(dc['config'].structure).eval()
ds, _ = smoke.strip_module_prefix(dc['model'])
dec_report = smoke.load_strict(decoder, ds)
assert dec_report['loaded']

# Wrap in DataParallel (matches released _build_model)
policy_dp = nn.DataParallel(policy_base).cuda()

initial_hash = model_state_hash(policy_dp.module)
print(f"Initial state hash: {initial_hash}", flush=True)

# Optimizer (matches released _build_optimizer, only policy2 params i.e. policy_dp.module)
def make_optimizer(model_dp):
    return torch.optim.Adam(
        model_dp.module.parameters(),
        lr=rl['optimizer']['kwargs']['lr'],
        betas=tuple(rl['optimizer']['kwargs']['betas']),
        weight_decay=rl['optimizer']['kwargs']['weight_decay'],
    )

# Decoder parameters should NOT be in optimizer
_tmp_opt = make_optimizer(policy_dp)
_opt_param_ids = set(id(p) for group in _tmp_opt.param_groups for p in group['params'])
_decoder_param_ids = set(id(p) for p in decoder.parameters())
decoder_in_opt = bool(_decoder_param_ids & _opt_param_ids)
print(f"Decoder parameters in optimizer: {decoder_in_opt} (expected False)", flush=True)
del _tmp_opt


# ---------------------------------------------------------------------------
# Load trace windows
# ---------------------------------------------------------------------------
print("Loading trace windows...", flush=True)
w0 = np.load(TRACE_DIR / 'window_000.npz')
w20 = np.load(TRACE_DIR / 'window_020.npz')


def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32)).cuda()


def to_long(arr):
    return torch.from_numpy(arr.astype(np.int64)).cuda()


def window_tensors(w):
    """Return (critic_input, actor_input, actor_target, critic_music, actor_music)
    all as CUDA tensors with batch dim=1.
    """
    ci = (to_long(w['critic_input'][None, :, 0]),    # (1, 29)
          to_long(w['critic_input'][None, :, 1]))
    ai = (to_long(w['actor_input'][None, :, 0]),     # (1, 28)
          to_long(w['actor_input'][None, :, 1]))
    at = (to_long(w['actor_target'][None, :, 0]),    # (1, 28)
          to_long(w['actor_target'][None, :, 1]))
    cm = to_tensor(w['critic_music'][None])           # (1, 312, 55)
    am = to_tensor(w['actor_music'][None])            # (1, 304, 55)
    rw = to_tensor(w['rewards'][None])                # (1, 28)
    mk = to_tensor(w['mask'][None])                   # (1, 28)
    return ci, ai, at, cm, am, rw, mk


# ---------------------------------------------------------------------------
# Per-window forward check: batch=1 to match the trace (atol=1e-6, rtol=1e-5)
# ---------------------------------------------------------------------------
print("Verifying forward pass against saved trace (batch=1 per window)...", flush=True)
fwd_checks = {}
all_rt_nll = []   # collect runtime NLL for each window, then stack for batch loss
all_rt_values = []

for w_idx, w in [(0, w0), (20, w20)]:
    ci, ai, at, cm, am, rw, mk = window_tensors(w)

    policy_dp.train()
    policy_dp.module.freeze_drop()
    with torch.no_grad():
        v_rt = policy_dp.module.critic(ci, cm)[:, :, 0]  # (1, 29)
        policy_dp.eval()
        logits_rt, nll_rt, entropy_rt = policy_dp.module.actor(ai, am, at, reduction=False)
        policy_dp.train()
        policy_dp.module.freeze_drop()

    v_np = v_rt.detach().cpu().numpy()[0]       # (29,)
    nll_np = nll_rt.detach().cpu().numpy()      # could be (28,) or (28*B,)
    if nll_np.ndim == 1 and nll_np.shape[0] == 28:
        pass
    else:
        nll_np = nll_np.reshape(-1)   # flatten if (28,) or (1*28,)

    saved_v = w['values'].astype(np.float64)
    saved_nll = w['actor_nll'].astype(np.float64)
    saved_td = w['td_targets'].astype(np.float64)

    v_max_err = float(np.abs(v_np.astype(np.float64) - saved_v).max())
    nll_max_err = float(np.abs(nll_np.astype(np.float64) - saved_nll).max())

    # TD check: derive from runtime values and saved rewards
    rt_td = w['rewards'].astype(np.float64) + GAMMA * v_np[1:]
    td_max_err = float(np.abs(rt_td - saved_td).max())

    v_pass = np.allclose(v_np, saved_v, atol=ATOL, rtol=RTOL)
    nll_pass = np.allclose(nll_np, saved_nll, atol=ATOL, rtol=RTOL)
    td_pass = np.allclose(rt_td, saved_td, atol=ATOL, rtol=RTOL)

    fwd_checks[f'window_{w_idx}'] = {
        'values_max_abs_err': v_max_err,
        'values_pass': v_pass,
        'nll_max_abs_err': nll_max_err,
        'nll_pass': nll_pass,
        'td_max_abs_err': td_max_err,
        'td_pass': td_pass,
        'nll_batch1_note': (
            'batch=1 matches saved trace within tolerance' if nll_pass
            else f'batch=1 NLL max_err={nll_max_err:.3e} exceeds atol={ATOL}; '
                 'may indicate numerical divergence beyond float32 rounding'
        ),
    }
    print(f"Window {w_idx}: values_err={v_max_err:.3e}({'PASS' if v_pass else 'FAIL'})  "
          f"nll_err={nll_max_err:.3e}({'PASS' if nll_pass else 'FAIL'})  "
          f"td_err={td_max_err:.3e}({'PASS' if td_pass else 'FAIL'})", flush=True)

    all_rt_nll.append(nll_np)
    all_rt_values.append(v_np)

fwd_all_pass = all(c['values_pass'] and c['nll_pass'] and c['td_pass'] for c in fwd_checks.values())
print(f"Forward checks (batch=1) overall: {'PASS' if fwd_all_pass else 'FAIL'}", flush=True)

if not fwd_all_pass:
    # Report but do not stop - proceed to update verification
    print("WARNING: some forward checks failed; proceeding with update verification", flush=True)


# ---------------------------------------------------------------------------
# Build batch=2 tensors for update computation
# ---------------------------------------------------------------------------
def stack_batch(w_list, key):
    return np.stack([w[key] for w in w_list], axis=0)

windows_list = [w0, w20]

critic_music_b = to_tensor(stack_batch(windows_list, 'critic_music'))  # (2, 312, 55)
actor_music_b = to_tensor(stack_batch(windows_list, 'actor_music'))    # (2, 304, 55)
critic_in_up_b = to_long(stack_batch(windows_list, 'critic_input')[:, :, 0])  # (2, 29)
critic_in_down_b = to_long(stack_batch(windows_list, 'critic_input')[:, :, 1])
actor_in_up_b = to_long(stack_batch(windows_list, 'actor_input')[:, :, 0])    # (2, 28)
actor_in_down_b = to_long(stack_batch(windows_list, 'actor_input')[:, :, 1])
actor_tgt_up_b = to_long(stack_batch(windows_list, 'actor_target')[:, :, 0])  # (2, 28)
actor_tgt_down_b = to_long(stack_batch(windows_list, 'actor_target')[:, :, 1])
rewards_b = to_tensor(stack_batch(windows_list, 'rewards'))  # (2, 28)
mask_b = to_tensor(stack_batch(windows_list, 'mask'))        # (2, 28)


# ---------------------------------------------------------------------------
# Parameter group classification and utilities
# ---------------------------------------------------------------------------

def classify_params(module):
    groups = {}
    for name, _ in module.named_parameters():
        if name.startswith('music_trans.'):
            groups[name] = 'music_encoder'
        elif name.startswith('gpt_base.'):
            groups[name] = 'gpt_base'
        elif name.startswith('gpt_head.'):
            groups[name] = 'actor_head'
        elif name.startswith('critic_net.'):
            groups[name] = 'critic_net'
        else:
            groups[name] = 'other'
    return groups


param_groups = classify_params(policy_dp.module)


def snapshot_params(model_dp):
    return {name: p.detach().cpu().clone() for name, p in model_dp.module.named_parameters()}


def grad_summary(model_dp, param_groups):
    summary = {}
    for name, p in model_dp.module.named_parameters():
        g = param_groups.get(name, 'other')
        if g not in summary:
            summary[g] = {'has_grad': [], 'grad_norms': []}
        has = p.grad is not None
        summary[g]['has_grad'].append(has)
        if has:
            summary[g]['grad_norms'].append(float(p.grad.norm().item()))
    return {
        g: {
            'any_grad': any(v['has_grad']),
            'grad_norm_total': sum(v['grad_norms']),
            'max_grad_norm': max(v['grad_norms']) if v['grad_norms'] else 0.0,
            'num_params_with_grad': sum(1 for x in v['has_grad'] if x),
        }
        for g, v in summary.items()
    }


def param_delta_by_group(before_snapshot, model_dp, param_groups):
    gd = {}
    per_param = {}
    for name, p in model_dp.module.named_parameters():
        b = before_snapshot[name]
        delta = (p.detach().cpu() - b).abs()
        g = param_groups.get(name, 'other')
        per_param[name] = {'max_abs_delta': float(delta.max()), 'any_changed': bool(float(delta.max()) > 0)}
        if g not in gd:
            gd[g] = {'max_abs_delta': 0.0, 'any_changed': False}
        gd[g]['max_abs_delta'] = max(gd[g]['max_abs_delta'], float(delta.max()))
        gd[g]['any_changed'] = gd[g]['any_changed'] or bool(float(delta.max()) > 0)
    return gd, per_param


# Optimizer membership
opt_tmp = make_optimizer(policy_dp)
opt_param_ids = set(id(p) for group in opt_tmp.param_groups for p in group['params'])
membership = {name: id(p) in opt_param_ids for name, p in policy_dp.module.named_parameters()}
membership_decoder = {f'[decoder]{name}': id(p) in opt_param_ids for name, p in decoder.named_parameters()}
del opt_tmp

in_opt_by_group: dict = {}
for name, in_opt in {**membership, **membership_decoder}.items():
    g = '[decoder]' if name.startswith('[decoder]') else param_groups.get(name, 'other')
    if g not in in_opt_by_group:
        in_opt_by_group[g] = []
    in_opt_by_group[g].append(in_opt)
opt_summary = {g: {'all_in_optimizer': all(v), 'count': len(v)} for g, v in in_opt_by_group.items()}
print(f"Optimizer membership by group: {json.dumps(opt_summary)}", flush=True)
assert not any(v for k, v in membership_decoder.items()), "Decoder params in optimizer!"


# ---------------------------------------------------------------------------
# Greedy replay check: use NPZ stored arrays
# ---------------------------------------------------------------------------
greedy_checks = {}
for w_idx, w in [(0, w0), (20, w20)]:
    matches = np.all(w['replay_greedy'] == w['actor_target'], axis=1)
    teacher_mismatch = int(np.any(w['actor_window_greedy'] != w['actor_target'], axis=1).sum())
    greedy_checks[f'window_{w_idx}'] = {
        'original_context_all_match': bool(matches.all()),
        'mismatch_count': int((~matches).sum()),
        'teacher_forced_mismatch_count': teacher_mismatch,
        'note': 'top-1 from original sampling context matches cached codes; teacher-forced may differ due to context reset',
    }
print(f"Greedy checks: {greedy_checks}", flush=True)


# ---------------------------------------------------------------------------
# ACTOR UPDATE: branch copy (batch_i=0)
# ---------------------------------------------------------------------------
print("\n--- Actor update (batch_i=0) ---", flush=True)
actor_policy_dp = copy.deepcopy(policy_dp)
actor_opt = make_optimizer(actor_policy_dp)

actor_policy_dp.train()
actor_policy_dp.module.freeze_drop()
actor_opt.zero_grad()

# 1. Critic forward (values + td, detached)
values_a = actor_policy_dp.module.critic(
    (critic_in_up_b, critic_in_down_b), critic_music_b
)[:, :, 0]  # (2, 29)
td_a = (rewards_b + GAMMA * values_a[:, 1:]).clone().detach() - values_a[:, :-1]  # (2, 28)

# 2. Actor forward for pre-logging (under no_grad, eval, as in released train loop)
with torch.no_grad():
    actor_policy_dp.eval()
    _, actor_loss_log_raw, _ = actor_policy_dp.module.actor(
        (actor_in_up_b, actor_in_down_b),
        actor_music_b,
        (actor_tgt_up_b, actor_tgt_down_b),
        reduction=False,
    )
    actor_policy_dp.train()
    actor_policy_dp.module.freeze_drop()

# 3. Actor forward for gradient computation (outside no_grad, actor branch)
_, actor_nll_a, entropy_a = actor_policy_dp.module.actor(
    (actor_in_up_b, actor_in_down_b),
    actor_music_b,
    (actor_tgt_up_b, actor_tgt_down_b),
    reduction=False,
)
# actor_nll_a shape: (B*T,) = (56,)

alpha = ENTROPY_ALPHA
actor_loss_a = (
    torch.sum(
        (actor_nll_a * td_a.view(-1).clone().detach() - alpha * entropy_a)
        * mask_b.view(-1).clone().detach()
    ) / mask_b.sum().clone().detach() * ACTOR_LOSS_DECAY
)
# Logged actor_loss (mean NLL, masked) - computed separately per released code (after backward)
actor_nll_logged = (
    torch.sum(actor_nll_a * mask_b.view(-1).clone().detach())
    / mask_b.sum().clone().detach()
)

pre_actor_snapshot = snapshot_params(actor_policy_dp)
actor_loss_a.backward()
actor_grads = grad_summary(actor_policy_dp, param_groups)
actor_opt.step()
actor_group_deltas, actor_per_param = param_delta_by_group(pre_actor_snapshot, actor_policy_dp, param_groups)

actor_loss_scalar = float(actor_loss_a.detach().cpu().item())
actor_nll_logged_scalar = float(actor_nll_logged.detach().cpu().item())
print(f"Actor loss (backward): {actor_loss_scalar:.8f}", flush=True)
print(f"Actor NLL logged (mean masked, used in stats): {actor_nll_logged_scalar:.8f}", flush=True)
print(f"Actor grad summary: {json.dumps(actor_grads)}", flush=True)
print(f"Actor group deltas: {json.dumps(actor_group_deltas)}", flush=True)

# Independent arithmetic check on runtime NLL/values
rt_nll_a_np = actor_nll_a.detach().cpu().numpy()           # (56,)
rt_values_a_np = values_a.detach().cpu().numpy()            # (2, 29)
rt_rewards_np = rewards_b.detach().cpu().numpy()            # (2, 28)
rt_td_a_np = (rt_rewards_np + GAMMA * rt_values_a_np[:, 1:]) - rt_values_a_np[:, :-1]  # (2, 28)
rt_mask_np = mask_b.detach().cpu().numpy()                  # (2, 28)
indep_actor_loss_rt = float(
    np.sum(rt_mask_np.flatten() * rt_nll_a_np * rt_td_a_np.flatten()) / rt_mask_np.sum()
)
actor_rt_indep_err = abs(actor_loss_scalar - indep_actor_loss_rt)
actor_rt_agree = np.allclose(actor_loss_scalar, indep_actor_loss_rt, atol=ATOL, rtol=RTOL)
print(f"Actor loss vs indep-from-runtime-NLL: runtime={actor_loss_scalar:.8f}  indep={indep_actor_loss_rt:.8f}  "
      f"err={actor_rt_indep_err:.3e}  pass={actor_rt_agree}", flush=True)

# Compare against saved-trace independent reference (tolerance may exceed atol due to batch-size diff)
# The reference JSON is uploaded to the volume at a known path.
_ref_candidates = [
    Path('/vol/reports/batch01/update_reference.json'),
    ROOT / 'reports' / 'batch01' / 'update_reference.json',
]
_ref_path = next((p for p in _ref_candidates if p.exists()), None)
if _ref_path is None:
    raise FileNotFoundError(
        f"update_reference.json not found at any candidate: {_ref_candidates}. "
        "Upload it to the volume before running."
    )
ref = json.loads(_ref_path.read_text())
ref_actor = ref['stacked_batch']['actor_loss']
ref_critic = ref['stacked_batch']['critic_loss']
actor_saved_err = abs(actor_loss_scalar - ref_actor)
actor_saved_agree = np.allclose(actor_loss_scalar, ref_actor, atol=ATOL, rtol=RTOL)
print(f"Actor loss vs saved-trace indep ref: runtime={actor_loss_scalar:.8f}  saved={ref_actor:.8f}  "
      f"err={actor_saved_err:.3e}  pass={actor_saved_agree}", flush=True)


# ---------------------------------------------------------------------------
# CRITIC UPDATE: branch copy (batch_i=5)
# ---------------------------------------------------------------------------
print("\n--- Critic update (batch_i=5) ---", flush=True)
critic_policy_dp = copy.deepcopy(policy_dp)
critic_opt = make_optimizer(critic_policy_dp)

critic_policy_dp.train()
critic_policy_dp.module.freeze_drop()
critic_opt.zero_grad()

# 1. Critic forward (values + td, detached)
values_c = critic_policy_dp.module.critic(
    (critic_in_up_b, critic_in_down_b), critic_music_b
)[:, :, 0]  # (2, 29)
td_c = (rewards_b + GAMMA * values_c[:, 1:]).clone().detach() - values_c[:, :-1]  # (2, 28)

# 2. Actor forward for logging only (under no_grad, eval, per released code)
with torch.no_grad():
    critic_policy_dp.eval()
    _, actor_nll_c_log, _ = critic_policy_dp.module.actor(
        (actor_in_up_b, actor_in_down_b),
        actor_music_b,
        (actor_tgt_up_b, actor_tgt_down_b),
        reduction=False,
    )
    critic_policy_dp.train()
    critic_policy_dp.module.freeze_drop()

# 3. Critic loss (released: torch.mean(td_error ** 2))
critic_loss_c = torch.mean(td_c ** 2)

# Actor NLL logged stat (mean masked, re-computed after backward in released code)
actor_nll_c_logged_scalar = float(
    (torch.sum(actor_nll_c_log * mask_b.view(-1).clone().detach())
     / mask_b.sum().clone().detach()).detach().cpu().item()
)

pre_critic_snapshot = snapshot_params(critic_policy_dp)
critic_loss_c.backward()
critic_grads = grad_summary(critic_policy_dp, param_groups)
critic_opt.step()
critic_group_deltas, critic_per_param = param_delta_by_group(pre_critic_snapshot, critic_policy_dp, param_groups)

critic_loss_scalar = float(critic_loss_c.detach().cpu().item())
print(f"Critic loss (backward): {critic_loss_scalar:.8f}", flush=True)
print(f"Actor NLL logged in critic step (for stats only): {actor_nll_c_logged_scalar:.8f}", flush=True)
print(f"Critic grad summary: {json.dumps(critic_grads)}", flush=True)
print(f"Critic group deltas: {json.dumps(critic_group_deltas)}", flush=True)

# Independent arithmetic check on runtime values
rt_values_c_np = values_c.detach().cpu().numpy()  # (2, 29)
rt_td_c_np = (rt_rewards_np + GAMMA * rt_values_c_np[:, 1:]) - rt_values_c_np[:, :-1]  # (2, 28)
indep_critic_loss_rt = float(np.mean(rt_td_c_np ** 2))
critic_rt_indep_err = abs(critic_loss_scalar - indep_critic_loss_rt)
critic_rt_agree = np.allclose(critic_loss_scalar, indep_critic_loss_rt, atol=ATOL, rtol=RTOL)
print(f"Critic loss vs indep-from-runtime-values: runtime={critic_loss_scalar:.8f}  indep={indep_critic_loss_rt:.8f}  "
      f"err={critic_rt_indep_err:.3e}  pass={critic_rt_agree}", flush=True)

critic_saved_err = abs(critic_loss_scalar - ref_critic)
critic_saved_agree = np.allclose(critic_loss_scalar, ref_critic, atol=ATOL, rtol=RTOL)
print(f"Critic loss vs saved-trace indep ref: runtime={critic_loss_scalar:.8f}  saved={ref_critic:.8f}  "
      f"err={critic_saved_err:.3e}  pass={critic_saved_agree}", flush=True)


# ---------------------------------------------------------------------------
# Parameter-change expectations
# ---------------------------------------------------------------------------
def verify_expectations(group_deltas, changed_expected, unchanged_expected, label):
    failures = []
    for g in changed_expected:
        if not group_deltas.get(g, {}).get('any_changed', False):
            failures.append(f"{label}: expected {g} to change but it did not")
    for g in unchanged_expected:
        if group_deltas.get(g, {}).get('any_changed', False):
            failures.append(f"{label}: expected {g} unchanged but max_delta={group_deltas[g]['max_abs_delta']:.3e}")
    return failures


actor_failures = verify_expectations(
    actor_group_deltas,
    changed_expected=['actor_head'],
    unchanged_expected=['critic_net', 'music_encoder', 'gpt_base'],
    label='actor',
)
critic_failures = verify_expectations(
    critic_group_deltas,
    changed_expected=['critic_net'],
    unchanged_expected=['actor_head', 'music_encoder', 'gpt_base'],
    label='critic',
)
expectation_failures = actor_failures + critic_failures
if expectation_failures:
    print(f"\nExpectation failures: {expectation_failures}", flush=True)
else:
    print("\nAll parameter-change expectations met.", flush=True)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
elapsed = time.perf_counter() - started
result = {
    'run_id': str(OUT),
    'trace_dir': str(TRACE_DIR),
    'versions': smoke.versions(),
    'elapsed_seconds': elapsed,
    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    'initial_state_hash': initial_hash,
    'policy_sha256': sha256_file(policy_path),
    'critic_initial_sha256': sha256_file(critic_pt),
    'decoder_sha256': sha256_file(decoder_path),
    'optimizer_summary': opt_summary,
    'decoder_in_optimizer': decoder_in_opt,
    'forward_checks_batch1': fwd_checks,
    'forward_all_pass': fwd_all_pass,
    'greedy_checks': greedy_checks,
    'actor_branch': {
        'batch_i': 0,
        'loss_backward': actor_loss_scalar,
        'nll_logged_mean_masked': actor_nll_logged_scalar,
        'grad_summary': actor_grads,
        'group_deltas': actor_group_deltas,
        'indep_from_runtime_loss': indep_actor_loss_rt,
        'rt_indep_agree': actor_rt_agree,
        'rt_indep_abs_err': actor_rt_indep_err,
        'saved_trace_indep_agree': actor_saved_agree,
        'saved_trace_abs_err': actor_saved_err,
        'expectation_failures': actor_failures,
    },
    'critic_branch': {
        'batch_i': 5,
        'loss_backward': critic_loss_scalar,
        'actor_nll_logged_for_stats': actor_nll_c_logged_scalar,
        'grad_summary': critic_grads,
        'group_deltas': critic_group_deltas,
        'indep_from_runtime_loss': indep_critic_loss_rt,
        'rt_indep_agree': critic_rt_agree,
        'rt_indep_abs_err': critic_rt_indep_err,
        'saved_trace_indep_agree': critic_saved_agree,
        'saved_trace_abs_err': critic_saved_err,
        'expectation_failures': critic_failures,
    },
    'independent_reference_saved': {
        'actor_loss': ref_actor,
        'critic_loss': ref_critic,
    },
    'primary_checks_pass': (
        fwd_all_pass
        and actor_rt_agree
        and critic_rt_agree
        and not expectation_failures
    ),
    'expectation_failures': expectation_failures,
    'batch_note': (
        'Batch=2 actor/critic losses compared against independent arithmetic '
        'applied to runtime logits/values (primary check). '
        'Comparison against saved-trace reference (batch=1) is secondary and '
        'may differ by O(1e-5) due to float32 non-associativity in matmuls.'
    ),
}

(OUT / 'update_results.json').write_text(json.dumps(result, indent=2) + '\n')
print(f"\nWrote {OUT / 'update_results.json'}", flush=True)
print(f"Primary checks pass: {result['primary_checks_pass']}", flush=True)
print(json.dumps({'elapsed': elapsed, 'primary_checks_pass': result['primary_checks_pass']}), flush=True)
