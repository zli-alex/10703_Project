"""Independently check reward bins and report cached step-D evidence locally."""
from pathlib import Path
import os,sys,json,csv,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib')); os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.render_skeleton import render_skeleton_mp4
run=ROOT/'runs/batch01'/sys.argv[1]; report=ROOT/'reports/batch01'
d=np.load(run/'results.npz'); names=d['names'].tolist(); music=d['beats'].reshape(-1,8).max(axis=1).astype(bool)
assert d['raw'].shape==(7,240,72) and int(music.sum())==7
rows=[]; speeds=[]; minima_list=[]
for i,name in enumerate(names):
 raw=d['raw'][i].reshape(240,24,3)
 speed=gaussian_filter1d(np.linalg.norm(np.diff(raw,axis=0),axis=2).mean(axis=1),5)
 minima=argrelextrema(speed,np.less)[0] # Preserve released index convention, no +1.
 bins=np.zeros(240); bins[minima]=1; bins=bins.reshape(30,8).max(axis=1).astype(bool)
 expected=np.where(~music|bins,5.,-5.)
 assert np.array_equal(bins,d['dance_bins'][i].astype(bool)),name
 assert np.array_equal(expected,d['beat'][i]),name
 if np.isfinite(d['total'][i]).all(): assert np.array_equal(expected,d['total'][i]),name
 # Physical metrics exclude the root-velocity pseudo-joint used by the reward.
 rel=raw.copy(); rel[:,0]=0
 physical_speed=np.linalg.norm(np.diff(rel,axis=0),axis=2).mean(axis=1)*60
 row={'control':name,'mean_total_reward':float(d['total'][i].mean()),'mean_beat_reward':float(d['beat'][i].mean()),'mean_unweighted_body':float(d['body'][i].mean()),'mean_reward_music_bins':float(d['total'][i,music].mean()),'mean_reward_no_music_bins':float(d['total'][i,~music].mean()),'music_bin_fraction':float(music.mean()),'music_bins_hit':int((music&bins).sum()),'music_bins_total':int(music.sum()),'motion_bins_without_music':int((bins&~music).sum()),'motion_beat_count':len(minima),'mean_joint_speed':float(physical_speed.mean()),'rms_jerk':float(np.sqrt(np.mean((np.diff(rel,n=3,axis=0)*60**3)**2))),'nonfinite_total_bins':int((~np.isfinite(d['total'][i])).sum()),'positive_reward_from_no_music_bins':float(5*(~music).sum()),'total_reward_sum':float(d['total'][i].sum())}
 rows.append(row); speeds.append(speed*60); minima_list.append(minima)
(report/'reward_audit.json').write_text(json.dumps({'run_id':run.name,'independent_truth_table_check_passed':True,'rows':rows},indent=2)+'\n')
with (report/'reward_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True,gridspec_kw={'height_ratios':[3,1]})
axes[0].imshow(d['total'],aspect='auto',cmap='RdYlGn',vmin=-5,vmax=5,extent=[2,6,6.5,-.5]); axes[0].set_yticks(range(7),names); axes[0].set_title('Released total reward: green +5, red −5 (body weight 0)')
axes[1].imshow(music[None],aspect='auto',cmap='Greys',vmin=0,vmax=1,extent=[2,6,.5,-.5]); axes[1].set_yticks([0],['Music beat present']); axes[1].set_xlabel('Native seconds; each column is an eight-frame bin')
fig.savefig(report/'reward_heatmap.png',dpi=150); plt.close(fig)
fig,axes=plt.subplots(7,1,figsize=(12,13),constrained_layout=True)
for i,ax in enumerate(axes):
 t=2+np.arange(239)/60
 ax.plot(t,speeds[i]); ax.scatter(2+minima_list[i]/60,speeds[i][minima_list[i]],s=15,c='black')
 for beat in np.flatnonzero(d['beats']): ax.axvline(2+beat/60,c='orange',alpha=.6)
 if names[i]=='Repetitive': ax.axvline(4,c='red',linestyle='--')
 ax.set_xlim(2,6); ax.set_ylabel(names[i]); ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
axes[0].set_title('Reward detector speed: orange music beats, black motion minima; red repetition seam')
axes[-1].set_xlabel('Native seconds; speed uses released mixed coordinates and source units/s')
fig.savefig(report/'reward_timeline.png',dpi=140); plt.close(fig)
audio=run/'audit_audio.wav'
subprocess.run(['ffmpeg','-v','error','-y','-i',str(ROOT/'assets/audio/mHO5.wav'),'-af','atrim=start_sample=96000:end_sample=288000,asetpts=PTS-STARTPTS','-c:a','pcm_s24le',str(audio)],check=True)
for i,name in enumerate(names):
 raw=d['raw'][i].reshape(240,24,3); integrated=np.vstack([np.zeros((1,3),dtype=np.float32),np.cumsum(raw[:-1,0],axis=0)])
 positions=raw.copy(); positions[:,0]=0; positions+=integrated[:,None]
 slug=name.lower().replace(' ','_')
 render_skeleton_mp4(positions,report/f'reward_{slug}.mp4',audio)
shutil.copyfile(run/'summary.json',report/'reward_runtime.json')
print(json.dumps(rows,indent=2))
