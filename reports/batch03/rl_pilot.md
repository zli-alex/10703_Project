# Bailando++ RL pilot

Run `pilot-20261117T062722443178Z` completed five epochs from the released
supervised policy with seed 42, batch size 64, and one shared Adam optimizer. Each
epoch regenerated the three adaptation rollouts and followed the released
five-actor/five-critic schedule. The 12 minibatches per epoch yielded **35 actor
and 25 critic loss steps**. The music encoder, GPT base, and position decoder
matched their initial state fingerprints. The data loader used zero worker
subprocesses, as in the benchmark. [Modal training run](https://modal.com/apps/zhengfel/main/ap-uAPRi3ezD2fDnFDwLQnnM1)

The epoch-5 checkpoint reloaded on a fresh model and optimizer. The verification
covered model, Adam, scheduler, RNG, progress, and the presence of legacy
gradient buffers. Those buffers determine which Adam moments decay when the
next epoch starts. This run trained continuously through epoch 5. The rolling
checkpoint is on the `bailandopp-batch01` volume at
`runs/pilot-20261117T062722443178Z/pilot_checkpoint.pt`. Its SHA-256 is
`69498660b8d110b29205a571e490c233a48251efc3437ffb545952ce5bb84079`.
[Training records](pilot_results.json)

Training took 205.89 worker seconds and peaked at 11.75 GiB of allocated GPU
memory. Sampled masked reward across the five epochs was 2.84, 3.11, 2.96, 3.12,
and 3.08. The rollout policy and sampled windows change between epochs, so use
the paired evaluation below to assess the endpoint. [Training curves](training_curves.png)

## Matched evaluation

Each checkpoint generated from two fixed upper/lower code pairs for each of six
songs, producing 12 before/after pairs. Both conditions used greedy inference,
native features, the same seed codes and position decoder, and the same complete
eight-frame scoring bins. The final five native music frames on `mJB5` do not
form a complete bin. The three songs held out from RL were absent from the
training rollout loader. [Split and scope](../batch02/pilot_readiness.md)

The reward gives +5 in every bin without a music beat. For paired clips, the
music and bin counts are fixed, so any reward delta comes from changes in
music-beat-bin hits. Timing metrics use the released motion-beat detector and
report distance from each detected motion beat to its nearest music beat and
the percentage of music beats within 100 ms of a motion beat. Jerks and speeds
use root-relative joints, and root travel uses the decoded root displacements.
Group means weight songs equally after averaging the two starts. The motion
diversity statistic is mean pairwise Euclidean distance between per-clip vectors
of per-joint speed means and standard deviations, using a shared duration of
1,768 frames. The dataset split and a single training seed support descriptive
paired comparisons.

| Group | Mean reward | Music-beat bins hit | Music-beat coverage within 100 ms | Motion-to-music distance | RMS jerk |
|---|---:|---:|---:|---:|---:|
| Adaptation | 3.315 → 3.277 | 25.51% → 24.19% | 35.11% → 38.62% | 226 → 228 ms | 392.6 → 388.8 |
| Held out from RL | 3.368 → 3.367 | 25.14% → 25.19% | 38.51% → 39.95% | 260 → 270 ms | 351.5 → 369.1 |

The endpoint showed no average reward improvement in either group. The 100 ms
coverage rose in both groups while the average motion-to-music distance rose.
The two measures use opposite directions of the nearest-beat match. Ten of 12
pairs changed at least one generated token. Reward rose in five pairs, fell in
four, and tied in three. On `mJB5`, start 1 added four music-beat-bin hits and
start 2 lost nine, on the same music and beat grid. [Paired song changes](paired_deltas.png) · [Per-case metrics](evaluation_cases.csv) · [Paired data](paired_results.json)

## Review clips

Each clip shows the first 16 seconds with common audio and projection scale.
The skeleton camera recenters the root, so the numeric root-travel measure is
the source for global displacement. Reward and jerk below use each full scoring
interval.

- `mJB5`, start 1, largest reward gain (+0.181). Hits increased from 12 to 16. [Side-by-side clip](largest_mJB5_I1.mp4)
- `mJB5`, start 2, largest reward drop (−0.407). Hits decreased from 16 to 7. [Side-by-side clip](smallest_mJB5_I2.mp4)
- `mKR2`, start 1, middle-ranked reward change (0.000). The greedy code sequences were identical. [Side-by-side clip](middle_mKR2_I1.mp4)
- `mJS3`, start 2, reward improved (+0.037) and RMS jerk rose by 76.1. Check whether the two versions feel different in continuity or accent placement. [Side-by-side clip](reward_and_jerk_up_mJS3_I2.mp4)

Beat-bin metrics and jerk quantify local timing and motion changes. Musical
phrasing, such as whether a combo lands on a coherent count when it returns,
still needs audiovisual review. [Modal evaluation run](https://modal.com/apps/zhengfel/main/ap-lOkAk0rGsTdHXA1lzgYKmI)

Modal itemized $0.04575 for training and $0.02662 for evaluation, **$0.07236
combined**, excluding storage. To reproduce, use the `10703` Conda environment
and the existing Modal volume.

```sh
python scripts/check_modal_account.py
modal run -m infra.modal_pilot
modal run -m infra.modal_pilot --stage evaluate --run-id <pilot-run-id>
python -m scripts.summarize_pilot runs/batch03/<pilot-run-id> reports/batch03
```
