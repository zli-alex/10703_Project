"""Package the cached supervised rollout and exact loader music for step E."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/batch01/smokeB-20261106T204747Z'
out=ROOT/'data/batch01/rollout_trace'; out.mkdir(parents=True,exist_ok=True)
codes=np.load(run/'codes.npz')
features=np.asarray(json.loads((ROOT/'data/batch01/music_feat_60fps/mHO5.json').read_text())['music_array'],dtype=np.float32)
music=np.concatenate([np.zeros((40,55),np.float32),features[:1200,:55],features[1199:1200,:55],np.zeros((40,55),np.float32)])
assert np.array_equal(music[8:],np.load(run/'music_cond.npy'))
np.savez(out/'inputs.npz',music=music,upper=codes['upper'],lower=codes['lower'],beats=features[:,53],full_raw=np.load(run/'motion_raw_reward.npy'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
config={'source_run':run.name,'song':'mHO5','window_starts_codes':[0,20],'window_length_codes':30,'music_frames_per_code':8,'music_context_frames':40,'fps':60,'critic_initialization_seed':42,'critic_status':'newly initialized diagnostic critic, not trained','optimizer_steps':0,'input_sha256':sha(out/'inputs.npz'),'cached_codes_sha256':sha(run/'codes.npz'),'cached_raw_sha256':sha(run/'motion_raw_reward.npy'),'scope':'Replay cached supervised trajectory through released training dataset and forward/reward paths; not a regenerated RL rollout. Supervised initialization retains root; training initialization zeros root.'}
(ROOT/'configs/trace_batch01.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(config,indent=2))
