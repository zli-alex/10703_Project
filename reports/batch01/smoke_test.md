# Supervised Bailando++ smoke test

Run `smokeB-20261106T204747Z`, November 6, 2026. One Tesla T4, legacy PyTorch
1.6.0+cu101, Python 3.8.10. The rollout is valid.

Command:

```sh
cd /Users/zhengfeili/2026_Fall/10703/10703_Project
/Users/zhengfeili/opt/anaconda3/bin/conda run --no-capture-output -n 10703 python scripts/launch_batch01_smoke.py
```

App: [bailandopp-batch01-smoke](https://modal.com/apps/zhengfel/main/ap-q3SlvRYzti1dmeK3P7qeJs).
Clip: [smoke_mHO5_skeleton.mp4](smoke_mHO5_skeleton.mp4).
Arrays: `runs/batch01/smokeB-20261106T204747Z/`.

## Loading

Generation used `CrossCondGPT2MW.sample` with top-1 selection. The actor-critic
class was loaded only to account for its new critic. Critic weights were not used.

| Check | Result |
|---|---|
| Policy file / stored epoch | `epoch_400.pt` / 440 |
| Policy SHA256 | `ed84ccd47d861c9b8a8a61a60a67aa3b8cbb59c928487daec1fdb65190b02da2` |
| Policy missing / unexpected keys | 0 / 0, strict load, 269 tensors |
| Decoder SHA256 | `35670f42a3b3092438f73f0af3ace7b52e318a8b5c00b2b05c92078176b21716` |
| Decoder stored epoch | 500 |
| Decoder missing / unexpected keys | 0 / 0, strict load, 228 tensors |
| New `critic_net` tensors | 55, all absent from the supervised file |
| Critic change while loading the actor | 0 |
| Runtime | Python 3.8.10, torch 1.6.0+cu101, NumPy 1.19.5, Tesla T4 |

## Rollout

| Item | Value |
|---|---|
| Song | mHO5, record `gHO_sBM_cAll_d20_mHO5_ch01` |
| Initial upper / lower code | 436 / 289 |
| Codes, including initialization | 150 |
| Decoded frames | 1,200 (20.0 s at 60 Hz) |
| Code range | 1–511 |
| Unique upper / lower codes | 84 / 75 |
| Raw motion and global positions | finite |

The released loader padded 1,200 native music frames to 1,281: 40 left zeros,
the 1,200 real frames, one repeated last frame, and 40 right zeros. Sampling
then dropped the first 8 frames. Generated code `k+1` is aligned with native
music frames `[8+8k, 16+8k)`. The music Transformer also attends across its
±40-frame band.

Raw decoder coordinates are `motion_raw_reward.npy`. Root integration follows
the released rule `position[0]=0`, `position[t]=position[t-1]+velocity[t-1]`,
saved in `motion_root_integrated.npy`. Global joints are
`motion_global_positions.npy`. Rebuilding the integrated pose from those joints
differs by at most `5.96e-8`, one float32 rounding step. The job originally
exited 3 because the checker used a tighter tolerance. That checker now allows
`1e-5`. The saved arrays were not regenerated.

The first code comes from encoding all 432 padded real-motion frames, not an
8-frame crop. Holding the pose from frame 64 changed the upper code from 436 to
440, so that code depends on later frames. Holds at 8, 16, 32, 128, 256, and
the full 432 frames kept 436/289.

## Timing, memory, and cost

| Item | Value |
|---|---|
| Checkpoint read | 0.49 s |
| Build and load | 10.37 s |
| Encode and context probe | 0.98 s |
| Sample | 2.13 s |
| Decode | 0.15 s |
| Render | 3.48 s |
| GPU function compute | 17.93 s |
| Peak allocated GPU memory | 710.2 MiB |
| Peak reserved GPU memory | 760.0 MiB |
| This app's T4 / CPU / memory | $0.00492 / $0.00231 / $0.00153 |
| Workspace metered / billed after the run | $0.03039 / $0.00 |
| Credits applied this cycle | $0.03000 |

Reward scoring was not part of this smoke test. No second GPU was used.

## Audio

The file has 1,200 video frames at 60 fps and a 20.000-second container.
The AAC stream reports 19.968 seconds, one encoder frame short of the video.
The clip starts at the native audio origin. Perceptual beat alignment is still
a viewing question; the clock and mux were checked, not whether the dance looks
locked to the beats.

## Follow-up closure review

The updated validator was executed locally on the saved arrays and passed with
no failures. See `smoke_revalidation.json`: maximum position round-trip error
is 5.960464477539063e-8, with rtol/atol both 1e-5. The original manifest and
historical exit status are preserved. No GPU regeneration was needed.

The five-frame contact sheet (`smoke_contact_sheet.png`) shows visible skeletons
and changing poses without obvious clipping in those samples. This does not
establish smoothness or audiovisual synchronization; human playback review remains
pending. The renderer recenters the root, so it cannot show global drift reliably.

Complete container cold-start time was not recorded; reported build/load time is
inside the function. Reward computation and its timing are deferred to step D.
The 20-second music crop uses zero right context rather than the genuine future
song frames. Step C retains that convention and scores only seconds 5–15.
