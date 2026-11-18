"""Local independent alignment/arithmetic checks and figures for step E."""
from pathlib import Path
import os,sys,json,csv
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=ROOT/'runs/batch01'/sys.argv[1]; REPORT=ROOT/'reports/batch01'
summary=json.loads((RUN/'summary.json').read_text())
inputs=np.load(ROOT/'data/batch01/rollout_trace/inputs.npz')
all_rows=[]; checks=[]; windows=[]
for i in [0,20]:
    d=np.load(RUN/f'window_{i:03d}.npz'); windows.append(d)
    assert np.array_equal(d['codes'][:,0],inputs['upper'][i:i+30])
    assert np.array_equal(d['codes'][:,1],inputs['lower'][i:i+30])
    assert np.array_equal(d['reward_beats'],inputs['beats'][8*(i+1):8*(i+29)])
    assert np.array_equal(d['actor_music'],d['dataset_music'][8:-8])
    assert np.array_equal(d['critic_music'],d['dataset_music'][8:])
    assert np.array_equal(d['replay_greedy'],d['actor_target'])
    assert np.isfinite(d['decoded_raw']).all()
    raw=d['decoded_raw'].reshape(224,24,3)
    speed=gaussian_filter1d(np.linalg.norm(np.diff(raw,axis=0),axis=2).mean(axis=1),5)
    framebeats=np.zeros(224); framebeats[argrelextrema(speed,np.less)[0]]=1
    musicbins=d['reward_beats'].reshape(28,8).max(1).astype(bool)
    motionbins=framebeats.reshape(28,8).max(1).astype(bool)
    expected=np.where(~musicbins|motionbins,5.,-5.)
    assert np.array_equal(expected,d['rewards'])
    target=d['rewards'].astype(np.float64)+d['values'][1:].astype(np.float64)
    assert np.allclose(target,d['td_targets'],atol=1e-6,rtol=1e-6)
    independent_nll=np.zeros(28)
    for branch,key in enumerate(['actor_logits_up','actor_logits_down']):
        logits=d[key].astype(np.float64)
        independent_nll+=logsumexp(logits,axis=-1)-logits[np.arange(28),d['actor_target'][:,branch]]
    assert np.allclose(independent_nll,d['actor_nll'],atol=1e-4,rtol=1e-5)
    delta=target-d['values'][:-1]
    checks.append({'window':i,'alignment_passed':True,'reward_truth_table_passed':True,
                   'td_target_max_abs_error':float(np.max(np.abs(target-d['td_targets']))),
                   'summed_nll_max_abs_error':float(np.max(np.abs(independent_nll-d['actor_nll']))),
                   'illustrative_actor_loss':float(np.sum(independent_nll*delta*d['mask'])/d['mask'].sum()),
                   'illustrative_critic_loss':float(np.mean(delta**2)),
                   'note':'Forward arithmetic only; freshly initialized critic; no gradient or optimizer verification.'})
    for j in range(28):
        row={'window':i,'local_action':j,'history_last_code':i+j,'target_code_index':i+j+1,
             'target_upper':int(d['actor_target'][j,0]),'target_lower':int(d['actor_target'][j,1]),
             'native_frame_start':8*(i+j+1),'native_frame_end_exclusive':8*(i+j+2),
             'music_beat_present':bool(musicbins[j]),'motion_beat_present':bool(motionbins[j]),
             'reward':float(d['rewards'][j]),'cached_full_slice_reward':float(d['cached_slice_rewards'][j]),
             'actor_mask':float(d['mask'][j]),'value':float(d['values'][j]),
             'bootstrap_value':float(d['values'][j+1]),'td_target':float(d['td_targets'][j]),
             'nll':float(d['actor_nll'][j]),'sampling_history_start':int(d['replay_ranges'][j,0]),
             'sampling_music_start_native':int(d['replay_ranges'][j,2]),
             'sampling_music_end_native_exclusive':int(d['replay_ranges'][j,3]),
             'actor_window_greedy_matches':bool(np.array_equal(d['actor_window_greedy'][j],d['actor_target'][j]))}
        all_rows.append(row)
with (REPORT/'rollout_trace_actions.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
(REPORT/'rollout_trace_validation.json').write_text(json.dumps({'run_id':RUN.name,'checks':checks,'actions':all_rows},indent=2)+'\n')
(REPORT/'rollout_trace_runtime.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True,gridspec_kw={'height_ratios':[2,1,1]})
for col,(i,d) in enumerate(zip([0,20],windows)):
    ax=axes[0,col]
    spans=[('Dataset music',8*i-40,8*(i+30)+40),('Critic music',8*i-32,8*(i+30)+40),('Actor music',8*i-32,8*(i+29)+40),('30 code bins',8*i,8*(i+30)),('28 rewarded code bins',8*(i+1),8*(i+29))]
    colors=['#b7d8d2','#62aaa2','#1d7b73','#a58ac0','#735199']
    for y,((label,start,end),color) in enumerate(zip(spans,colors)):
        ax.barh(y,(end-start)/60,left=start/60,height=.55,color=color)
    if i==0: ax.axvspan(-40/60,0,color='gray',alpha=.15)
    ax.set_yticks(range(5),[x[0] for x in spans]); ax.invert_yaxis()
    ax.set_title(f'Window starting at code {i}'); ax.set_xlabel('Native seconds (negative time is zero padding)')
    start,end=8*(i+1)/60,8*(i+29)/60
    for row,key,title in [(1,'rewards','Reward: green +5 / red −5'),(2,'mask','Actor loss mask: black included')]:
        ax=axes[row,col]
        if key=='rewards': ax.imshow(d[key][None],aspect='auto',cmap='RdYlGn',vmin=-5,vmax=5,extent=[start,end,1,0])
        else: ax.imshow(d[key][None],aspect='auto',cmap='Greys',vmin=0,vmax=1,extent=[start,end,1,0])
        ax.set_yticks([]); ax.set_title(title,fontsize=10); ax.set_xlabel('Native seconds; each bin = 8 motion frames')
fig.suptitle('Step E: exact dataset windows, reward interval, and actor mask',fontsize=15)
fig.savefig(REPORT/'rollout_trace_timeline.png',dpi=150);plt.close(fig)
# Show why matching the decoder context matters, and how the two forward contexts differ.
fig,axes=plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
for col,(i,d) in enumerate(zip([0,20],windows)):
    frame_error=np.max(np.abs(d['decoded_raw']-d['cached_full_decode_slice']),axis=1)
    axes[0,col].plot((np.arange(224)+8*(i+1))/60,frame_error)
    axes[0,col].set_title(f'Window {i}: standalone decode vs full decode slice')
    axes[0,col].set_ylabel('Maximum coordinate difference');axes[0,col].set_xlabel('Native seconds')
    axes[1,col].plot(np.arange(28),d['replay_vs_actor_feature_max_abs'],marker='.',markersize=3)
    axes[1,col].set_title(f'Window {i}: music features differ with context')
    axes[1,col].set_xlabel('Local action index');axes[1,col].set_ylabel('Maximum feature difference')
fig.savefig(REPORT/'rollout_trace_context.png',dpi=150);plt.close(fig)
print(json.dumps({'checks':checks,'windows':summary['windows'],'selected_actions':[r for r in all_rows if (r['window'],r['local_action']) in [(0,0),(0,27),(20,27)]]},indent=2))
