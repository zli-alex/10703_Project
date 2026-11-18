"""Export two released training windows and forward diagnostics; no optimizer step."""
from pathlib import Path
import sys, json, time, importlib.util, copy, math
sys.path.insert(0, '/root/project')
import numpy as np
import torch
import yaml
from easydict import EasyDict
from scripts import smoke_infer as smoke

ROOT = Path('/root/project')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
config = json.loads((ROOT/'configs/trace_batch01.json').read_text())
baseline = json.loads((ROOT/'configs/baseline_assets.json').read_text())
source = ROOT/'third_party/Bailando/tpami_bailandopp'
rl = yaml.safe_load((source/'configs/actor_critic_music_trans_400.yaml').read_text())
inputs_path = Path('/vol/rollout_trace/inputs.npz')
smoke.require_hash(inputs_path, config['input_sha256'])
data = np.load(inputs_path)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

# Execute the actual upstream dataset, including its overlapping-window masks.
dataset_module = module('released_ac_dataset', source/'dataset/md_seq_ac.py')
dataset = dataset_module.MoDaSeqAC([data['music']], ([data['upper']], [data['lower']]),
                                 [data['beats']], interval=30, music_dance_ratio=8, wav_padding=40)
assert len(dataset) == 121
mask_counts = np.zeros(150, dtype=np.int64)
for index in range(len(dataset)):
    mask_counts[index+1:index+29] += dataset[index][-1].astype(np.int64)
assert np.array_equal(mask_counts[1:149], np.ones(148, dtype=np.int64))
assert mask_counts[0] == mask_counts[149] == 0

models = smoke.import_models(OUT/'work', ROOT/'third_party/Bailando')
torch.manual_seed(config['critic_initialization_seed'])
policy_path = Path('/vol/assets/policy/epoch_400.pt')
decoder_path = Path('/vol/assets/decoder/epoch_500.pt')
smoke.require_hash(policy_path, baseline['policy']['sha256'])
smoke.require_hash(decoder_path, baseline['position_decoder']['sha256'])
pc, dc = smoke.torch_load(policy_path), smoke.torch_load(decoder_path)
decoder = models.SepVQVAER(dc['config'].structure).cuda().eval()
ds, _ = smoke.strip_module_prefix(dc['model'])
decoder_load = smoke.load_strict(decoder, ds); assert decoder_load['loaded']
policy_cfg = copy.deepcopy(pc['config'].structure_generate)
policy_cfg.name = 'CrossCondGPT2MWAC'
policy_cfg.critic_net = EasyDict(rl['structure_generate']['critic_net'])
policy = models.CrossCondGPT2MWAC(policy_cfg).cuda().eval()
ps, _ = smoke.strip_module_prefix(pc['model'])
policy_load = smoke.load_actor_critic(policy, ps); assert policy_load['loaded']
# Preserve the exact random critic for later independent-update work.
torch.save(policy.critic_net.state_dict(), OUT/'critic_initial.pt')
reward_module = module('released_reward', source/'models/up_down_half_reward.py')
reward = reward_module.UpDownReward(EasyDict(rl['reward'])).cuda()
all_upper = torch.from_numpy(data['upper'][None]).long().cuda()
all_lower = torch.from_numpy(data['lower'][None]).long().cuda()
rollout_cond = torch.from_numpy(data['music'][None,8:]).float().cuda()
results = []

def arr(t):
    return t.detach().cpu().numpy()

with torch.no_grad():
    for index in config['window_starts_codes']:
        music, upper, lower, beats, mask = dataset[index]
        assert music.shape == (320,55) and len(upper) == 30 and len(beats) == 240
        # Independent native timestamp map, including explicit zero padding.
        native = np.arange(8*index-40, 8*(index+30)+40)
        expected_music = np.zeros_like(music)
        valid = (native>=0) & (native<1201)
        expected_music[valid] = data['music'][40+native[valid]]
        assert np.array_equal(expected_music, music)
        u = torch.from_numpy(upper[None]).long().cuda()
        l = torch.from_numpy(lower[None]).long().cuda()
        critic_cond = torch.from_numpy(music[None,8:]).float().cuda()
        actor_cond = critic_cond[:,:-8]
        critic_input = (u[:,:-1],l[:,:-1])
        actor_input = (u[:,:-2],l[:,:-2])
        actor_target = (u[:,1:-1],l[:,1:-1])
        decoded = decoder.decode(tuple([t] for t in actor_target))
        reward_beats = torch.from_numpy(beats[None,8:-8]).float().cuda()
        rewards = reward(decoded, reward_beats.clone(),8)
        cached_raw = data['full_raw'][8*(index+1):8*(index+29)]
        cached_rewards = reward(torch.from_numpy(cached_raw[None]).cuda(),reward_beats.clone(),8)
        policy.train(); policy.freeze_drop()
        values = policy.critic(critic_input,critic_cond)[:,:,0]
        policy.eval()
        logits, nll, entropy = policy.actor(actor_input,actor_cond,actor_target,reduction=False)
        actor_features = policy.music_trans(actor_cond)
        critic_features = policy.music_trans(critic_cond)
        assert tuple(actor_features.shape)==(1,28,768)
        assert tuple(critic_features.shape)==(1,29,768)
        targets = rewards + rl['gamma']*values[:,1:]
        td = targets-values[:,:-1]
        replay_pairs, replay_ranges, replay_feature_diffs = [], [], []
        # Replay original sampling contexts, without extending/regenerating the trajectory.
        for j in range(28):
            k = index+j
            length = min(k+1,28)
            history_start = k+1-length
            cond_start = 0 if k<28 else history_start*8
            cond_end = (k+1+10)*8
            cond = rollout_cond[:,cond_start:cond_end]
            policy.eval()
            replay_logits, _ = policy((all_upper[:,history_start:k+1],all_lower[:,history_start:k+1]),cond)
            chosen = [int(torch.topk(torch.softmax(x[:,-1],dim=-1),1,dim=-1)[1].item()) for x in replay_logits]
            replay_pairs.append(chosen)
            replay_ranges.append([history_start,k+1,cond_start-32,cond_end-32])
            sampled_feature = policy.music_trans(cond)[:,-1]
            replay_feature_diffs.append(float((sampled_feature-actor_features[:,j]).abs().max()))
        replay_pairs = np.asarray(replay_pairs)
        cached_pairs = np.stack([upper[1:-1],lower[1:-1]],axis=1)
        if not np.array_equal(replay_pairs,cached_pairs):
            raise RuntimeError('Original-context greedy replay differs from cached supervised rollout')
        actor_greedy = np.stack([arr(x.argmax(-1))[0] for x in logits],axis=1)
        payload = dict(dataset_music=music, native_music_indices=native, critic_music=arr(critic_cond)[0],
                       actor_music=arr(actor_cond)[0], codes=np.stack([upper,lower],axis=1),
                       critic_input=np.stack([upper[:-1],lower[:-1]],axis=1),
                       actor_input=np.stack([upper[:-2],lower[:-2]],axis=1), actor_target=cached_pairs,
                       beats_full=beats, reward_beats=beats[8:-8], mask=mask, decoded_raw=arr(decoded)[0],
                       cached_full_decode_slice=cached_raw, rewards=arr(rewards)[0], cached_slice_rewards=arr(cached_rewards)[0],
                       values=arr(values)[0], td_targets=arr(targets)[0], td_errors=arr(td)[0],
                       actor_logits_up=arr(logits[0])[0], actor_logits_down=arr(logits[1])[0],
                       actor_nll=arr(nll), entropy=arr(entropy), actor_music_features=arr(actor_features)[0],
                       critic_music_features=arr(critic_features)[0], replay_greedy=replay_pairs,
                       actor_window_greedy=actor_greedy, replay_ranges=np.asarray(replay_ranges),
                       replay_vs_actor_feature_max_abs=np.asarray(replay_feature_diffs))
        for key,value in payload.items():
            assert np.isfinite(value).all(),key
        np.savez_compressed(OUT/f'window_{index:03d}.npz',**payload)
        result = {'window_start_code':index,'codes_inclusive':[index,index+29],
                  'decoded_native_frames':[8*(index+1),8*(index+29)],'actor_mask_nonzero':np.flatnonzero(mask).tolist(),
                  'mean_reward_all_bins':float(rewards.mean()),
                  'mean_reward_actor_masked':float((arr(rewards)[0]*mask).sum()/mask.sum()),
                  'standalone_vs_cached_max_abs':float(np.abs(arr(decoded)[0]-cached_raw).max()),
                  'standalone_vs_cached_reward_changed_bins':int((arr(rewards)!=arr(cached_rewards)).sum()),
                  'sampling_context_greedy_replay_matches':True,
                  'actor_window_greedy_mismatch_pairs':int(np.any(actor_greedy!=cached_pairs,axis=1).sum()),
                  'actor_window_greedy_mismatch_local_indices':np.flatnonzero(np.any(actor_greedy!=cached_pairs,axis=1)).tolist(),
                  'max_music_feature_difference_from_sampling':max(replay_feature_diffs)}
        results.append(result)

# Every computation above is forward-only; explicitly ensure no accumulated gradients.
assert all(p.grad is None for p in policy.parameters())
assert all(p.grad is None for p in decoder.parameters())
torch.cuda.synchronize()
summary = {'config':config,'versions':smoke.versions(),'decoder_load':decoder_load,'policy_load':policy_load,
           'dataset_windows':len(dataset),'actor_mask_coverage_codes':[1,149],'batch_size':rl['batch_size'],
           'minibatches_one_20s_rollout':math.ceil(len(dataset)/rl['batch_size']),
           'sampling':'RandomSampler(replacement=True), num_samples=dataset length; masks do not guarantee per-epoch unique coverage',
           'critic_initial_sha256':smoke.sha256_file(OUT/'critic_initial.pt'),
           'critic_status':'new random critic, seed42; numerical values are diagnostic, not learned estimates',
           'optimizer_steps':0,'gradient_verification':'No backward call; all .grad are None. Update gradients are not yet verified.',
           'total_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
           'source_hashes':{str(path.relative_to(source)):smoke.sha256_file(path) for path in [source/'dataset/md_seq_ac.py',source/'models/cross_cond_gpt2_music_window_ac.py',source/'models/up_down_half_reward.py',source/'actor_critic.py']},
           'windows':results}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'windows':results,'total_seconds':summary['total_seconds']}))
