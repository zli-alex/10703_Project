"""One decoder pass and unchanged released CUDA reward; save every input/output."""
from pathlib import Path
import sys,json,time,importlib.util
sys.path.insert(0,'/root/project')
import numpy as np
import torch
from easydict import EasyDict
from scripts import smoke_infer as smoke
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter(); torch.manual_seed(42)
inputs=Path('/vol/reward_audit/inputs.npz')
manifest=json.loads(Path('/vol/reward_audit/manifest.json').read_text())
smoke.require_hash(inputs,manifest['input_sha256'])
data=np.load(inputs); names=data['names'].tolist()
root=Path('/root/project'); cfg=json.loads((root/'configs/smoke_batch01.json').read_text())
checkpoint=Path('/vol/assets/decoder/epoch_500.pt'); smoke.require_hash(checkpoint,cfg['position_decoder']['sha256'])
models=smoke.import_models(out/'work',root/'third_party/Bailando')
ckpt=smoke.torch_load(checkpoint)
decoder=models.SepVQVAER(ckpt['config'].structure).cuda().eval()
state,_=smoke.strip_module_prefix(ckpt['model']); loading=smoke.load_strict(decoder,state)
assert loading['loaded']
with torch.no_grad():
 codes=tuple([torch.from_numpy(data[k][None]).long().cuda()] for k in ['real_upper','real_lower'])
 reconstruction=decoder.decode(codes)[0,120:360].cpu().numpy()
names.insert(1,'Reconstructed')
raw=np.insert(data['raw'],1,reconstruction,axis=0).astype(np.float32)
source=root/'third_party/Bailando/tpami_bailandopp/models/up_down_half_reward.py'
spec=importlib.util.spec_from_file_location('released_reward',source); reward=importlib.util.module_from_spec(spec); spec.loader.exec_module(reward)
poses=torch.from_numpy(raw).cuda(); beats=torch.from_numpy(np.repeat(data['beats'][None],len(names),axis=0)).cuda()
torch.cuda.synchronize(); scoring_started=time.perf_counter()
with torch.no_grad():
 total=reward.UpDownReward(EasyDict(rate=0)).cuda()(poses.clone(),beats.clone(),8)
 beat=reward.ba_reward(poses.clone(),beats.clone())
 with_body=reward.UpDownReward(EasyDict(rate=1)).cuda()(poses.clone(),beats.clone(),8)
 body=with_body-beat
 dance_bins=reward.calc_db(raw)
torch.cuda.synchronize(); scoring_seconds=time.perf_counter()-scoring_started
# Primary total uses rate zero. Rate-one call only exposes the unweighted body term.
np.savez(out/'results.npz',names=np.asarray(names),raw=raw,beats=data['beats'],total=total.cpu().numpy(),beat=beat.cpu().numpy(),body=body.cpu().numpy(),dance_bins=dance_bins.cpu().numpy())
# Supplementary numerical edge case: finite all-zero geometry, explicitly not a valid pose.
with torch.no_grad():
 degenerate=reward.UpDownReward(EasyDict(rate=0)).cuda()(torch.zeros_like(poses[:1]),beats[:1].clone(),8).cpu().numpy()
summary={'versions':smoke.versions(),'decoder_sha256':cfg['position_decoder']['sha256'],'decoder_missing':loading['missing'],'decoder_unexpected':loading['unexpected'],'reward_source_sha256':smoke.sha256_file(source),'input_manifest':manifest,'body_rate_primary':0,'scoring_seconds':scoring_seconds,'total_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'finite_total_per_control':dict(zip(names,np.isfinite(total.cpu().numpy()).all(axis=1).tolist())),'degenerate_zero_geometry_probe':{'valid_pose':False,'input_finite':True,'nonfinite_reward_bins':int((~np.isfinite(degenerate)).sum())}}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
