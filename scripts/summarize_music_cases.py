"""Local diagnostics on cached step C results; not the released reward."""
import os
os.environ.setdefault('MPLCONFIGDIR','.cache/matplotlib')
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
run=root/'runs/batch01'/sys.argv[1]
original=root/'runs/batch01/smokeB-20261106T204747Z'
paths=[('A-original',original)]+[(x,run/x) for x in ['A-repeat','B-swap','A-shift']]
base=np.load(original/'motion_global_positions.npy'); base=base-base[:,:1]
bc=np.load(original/'codes.npz'); basecodes=np.stack([bc['upper'],bc['lower']],axis=1)
rows=[]; fig,axes=plt.subplots(4,1,figsize=(12,10),constrained_layout=True)
for ax,(case,path) in zip(axes,paths):
    m=json.loads((path/'manifest.json').read_text())
    # Original manifest retains the historically over-strict round-trip failure.
    if case!='A-original': assert not m['validation_failures']
    c=np.load(path/'codes.npz'); codes=np.stack([c['upper'],c['lower']],axis=1)
    assert codes[0].tolist()==[436,289]
    xyz=np.load(path/'motion_global_positions.npy'); rel=xyz-xyz[:,:1]
    speed=np.linalg.norm(np.diff(rel,axis=0),axis=2).mean(axis=1)*60
    smooth=gaussian_filter1d(speed,5)
    minima=find_peaks(-smooth)[0]+1
    minima=minima[(minima>=300)&(minima<900)]
    song='mJB5' if case=='B-swap' else 'mHO5'; shift=8 if case=='A-shift' else 0
    f=np.asarray(json.loads((root/f'data/batch01/music_feat_60fps/{song}.json').read_text())['music_array'])
    beats=np.flatnonzero(f[shift:shift+1200,53]); beats=beats[(beats>=300)&(beats<900)]
    distances=np.abs(beats[:,None]-minima[None,:]).min(axis=1)/60 if len(minima) else np.full(len(beats),np.nan)
    # Code bins whose full eight frames lie inside [300,900).
    code_slice=slice(38,112)
    row={'case':case,'code_disagreement_fraction':float((codes[code_slice]!=basecodes[code_slice]).mean()),'root_relative_position_rmse':float(np.sqrt(np.mean((rel[300:900]-base[300:900])**2))),'mean_joint_speed':float(speed[300:899].mean()),'rms_jerk':float(np.sqrt(np.mean((np.diff(rel[300:900],n=3,axis=0)*60**3)**2))),'mean_music_beat_to_speed_minimum_seconds':float(np.mean(distances)),'music_beats_within_100ms_fraction':float(np.mean(distances<=.1)),'all_codes_equal_original':bool(np.array_equal(codes,basecodes))}
    rows.append(row)
    original_features=np.asarray(json.loads((root/'data/batch01/music_feat_60fps/mHO5.json').read_text())['music_array'])
    original_beats=np.flatnonzero(original_features[:1200,53])
    original_beats=original_beats[(original_beats>=300)&(original_beats<900)]
    row['mean_original_A_beat_to_minimum_seconds']=float(np.abs(original_beats[:,None]-minima[None,:]).min(axis=1).mean()/60) if len(minima) else None
    row['unique_upper_codes']=int(len(np.unique(codes[code_slice,0])))
    row['unique_lower_codes']=int(len(np.unique(codes[code_slice,1])))
    ax.plot((np.arange(len(smooth))+.5)/60,smooth,label=case)
    for b in beats: ax.axvline(b/60,color='orange',alpha=.5)
    ax.scatter(minima/60,smooth[minima-1],s=12,color='black')
    ax.set_xlim(5,15); ax.set_ylabel('Joint speed'); ax.legend(loc='upper right')
axes[-1].set_xlabel('Seconds; orange: music beats, black: speed minima')
fig.savefig(root/'reports/batch01/music_sensitivity.png',dpi=150)
(root/'reports/batch01/music_sensitivity.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Fixed-initialization music sensitivity','',f'Run: `{run.name}`. Reuses the step B clip as A-original. All four start from codes 436/289.','', '| Case | Code disagreement | Position RMSE | Mean speed | RMS jerk | Beat-to-minimum distance (s) | Beats within 100 ms |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows: lines.append('| '+r['case']+' | '+' | '.join(f'{r[k]:.4f}' for k in ['code_disagreement_fraction','root_relative_position_rmse','mean_joint_speed','rms_jerk','mean_music_beat_to_speed_minimum_seconds','music_beats_within_100ms_fraction'])+' |')
lines += ['', 'Metrics use [5,15) seconds; code comparison uses only fully contained eight-frame bins. Positions are root-relative in source units. Speed averages all 24 joints, including the zero-speed root; jerk is RMS over coordinate components. Music timing diagnostic matches each supplied beat to its nearest local minimum of Gaussian-smoothed mean joint speed (sigma five frames). This is an exploratory diagnostic, not the released reward, and may favor frequent minima. No statistical or perceptual dance-quality claim follows from four runs.', '', '![Music and speed](music_sensitivity.png)', '', 'A-shift advances native features by eight frames and audio by 6400 samples at 48 kHz, without wrapping. All runs preserve the original endpoint zero-padding convention. Source timestamps must be advanced by eight frames when interpreting the A-shift manifest.', '', 'Human audiovisual review and comparison against the original song’s beats remain pending. Reward controls belong to step D.']
(root/'reports/batch01/music_sensitivity.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
