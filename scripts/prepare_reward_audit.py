"""Prepare step-D controls in released decoder/reward coordinates, locally."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/batch01/reward_audit'; OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'assets/preprocessed_data/data/aistpp_test_full_wav/gHO_sBM_cAll_d20_mHO5_ch01.json'
record=json.loads(source.read_text()); points=np.asarray(record['dance_array'],dtype=np.float32).reshape(-1,24,3)
relative=points-points[:,:1]
real=relative.copy(); real[:-1,0]=np.diff(points[:,0],axis=0); real[-1,0]=0
run=ROOT/'runs/batch01/smokeB-20261106T204747Z'
generated=np.load(run/'motion_raw_reward.npy').reshape(-1,24,3)
# All controls are constructed with native time origin, then cropped identically.
start,end=120,360
controls={'Real':real[start:end], 'Generated':generated[start:end], 'Shifted':generated[start+8:end+8]}
segment=generated[120:240].copy()
controls['Repetitive']=np.tile(segment,(2,1,1))
pose=relative[120].copy(); pose[0]=0
up=np.linalg.norm(np.cross(pose[14],pose[13])); down=np.linalg.norm(np.cross(pose[4],pose[5]))
assert min(up,down)>1e-8, 'Degenerate reference normals'
stationary=np.repeat(pose[None],240,axis=0)
controls['Stationary']=stationary
angle=np.deg2rad(.5)*np.sin(2*np.pi*.5*np.arange(240)/60)
yaw=stationary.copy()
yaw[:,:,0]=stationary[:,:,0]*np.cos(angle[:,None])+stationary[:,:,2]*np.sin(angle[:,None])
yaw[:,:,2]=-stationary[:,:,0]*np.sin(angle[:,None])+stationary[:,:,2]*np.cos(angle[:,None])
controls['Nearly stationary']=yaw
names=list(controls); raw=np.stack([controls[n].reshape(240,72) for n in names]).astype(np.float32)
beats=np.asarray(record['music_array'],dtype=np.float32)[start:end,53]
codes=np.load(run/'codes.npz')
assert np.isfinite(raw).all() and raw.shape==(6,240,72)
assert np.array_equal(raw[names.index('Shifted')],generated[128:368].reshape(240,72))
np.savez(OUT/'inputs.npz',names=np.asarray(names),raw=raw,beats=beats,real_upper=codes['real_upper'],real_lower=codes['real_lower'])
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'frames':[120,360],'seconds':[2,6],'fps':60,'bin_frames':8,'song':'mHO5','source_run':run.name,'input_sha256':digest(OUT/'inputs.npz'),'source_record_sha256':digest(source),'generated_raw_sha256':digest(run/'motion_raw_reward.npy'),'real_coordinates':'Root-relative non-root joints, forward root displacement per frame in channels 0:3 (not velocity per second).','reconstruction':'Decode full 54-code real sequence cached during supervised smoke encoding of 432 padded frames; score only valid frames 120:360.','repetitive':'Repeat source generated frames 120:240 twice; seam at native frame240 (4 s), unsmoothed.','nearly_stationary':'Real pose at frame120, fixed root, rigid yaw amplitude0.5degrees at0.5Hz; phase zero at audit origin.','scoring_context':'Released reward called on cropped 240-frame controls. Gaussian smoothing sees crop boundaries; no full-rollout context. This is a reward-function audit, not exact replay of RL minibatch decoding (typically224 frames).','norm_checks':{'upper':float(up),'lower':float(down)}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
