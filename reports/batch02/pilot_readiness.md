# Bailando++ pilot split and readiness

The split is keyed by music ID. All other recordings of the same track inherit
its assignment. One source record per song is prepared, with its full 60 Hz
feature sequence and matching 48 kHz WAV. The policy input preserves source
channels 0–54, including the beat stream in channel 53.

| RL assignment | Song | Genre code | Native feature frames |
|---|---|---|---:|
| Adaptation | `mHO5` | `gHO` | 1,920 |
| Adaptation | `mJB5` | `gJB` | 1,773 |
| Adaptation | `mPO1` | `gPO` | 2,720 |
| Held out from RL | `mBR0` | `gBR` | 3,240 |
| Held out from RL | `mJS3` | `gJS` | 2,160 |
| Held out from RL | `mKR2` | `gKR` | 2,880 |

The authors' full-wave release has one music ID per genre. This selection keeps
the two existing development songs and uses six genres so all songs can use
the same full-track preprocessing. Genre and group therefore coincide. Compare
the RL checkpoint with the supervised policy on the same song and starting
codes, and treat between-group differences as descriptive. The six song IDs are
absent from the released `train_wav` archive index. The exact supervised
checkpoint's training membership remains unverified.

The [split manifest](../../configs/pilot_split.json) records provenance, hashes,
and the two fixed evaluation starts `(436, 289)` and `(179, 387)`. The adaptation
loader checks its exact set of motion files and feature IDs and verifies hashes
before rollout. Local preflight decoded all six WAVs, built 714 adaptation
windows with the released dataset, and confirmed that adding a held-out record
fails the loader check. The largest waveform-to-feature duration difference
was 11.54 ms. Perceptual review remains pending.

## Adaptation benchmark

Run `cycle-20261117T044620385396Z` completed on one Tesla T4 in the legacy
PyTorch 1.6.0 runtime. **Batch size 64** is the released ceiling and passed on all
three adaptation tracks. The 714 eligible windows yielded eleven full batches
and one batch of ten with the released replacement sampler. The schedule ran
five actor, five critic, and two actor updates per epoch. The worker peaked at
**11.75 GiB** of allocated GPU memory. [Modal run](https://modal.com/apps/zhengfel/main/ap-cv2sz6gjWSzNLkWrx7LqXu)

The bench started from the released supervised policy and used one Adam
optimizer. At both branch switches the active head had nonzero gradients and
the inactive head moved with zero-valued gradient tensors from legacy Adam
momentum. The encoder, GPT base, and position decoder stayed unchanged. The
checkpoint restored model, optimizer, scheduler, RNG, and progress. Reload and
state comparison passed at the epoch boundary. [Machine-readable result](pilot_benchmark_results.json)

| Phase | Time |
|---|---:|
| Three full-track rollouts | 10.88 s |
| Decode and reward scoring | 3.14 s |
| Forward, backward, and optimizer steps | 16.92 s |
| Training loop wall time including instrumentation | 24.57 s |
| Checkpoint save / reload and comparison | 2.71 s / 3.42 s |
| Worker including setup and verification | 61.45 s |

Modal itemized **$0.01580** for the benchmark ($0.01246 T4, $0.00200 CPU,
$0.00135 memory). Five comparable app runs project to about $0.079. Budget
**$0.10 in worker compute** as a rounded planning estimate before storage,
cold-start variation, and matched evaluation. This fits within the $16 training portion of
the proposed $20 pilot limit. The declared pilot is five complete epochs at batch
64, seed 42, from the immutable supervised starting checkpoint, with epoch 5 as
the evaluation endpoint. This benchmark checkpoint is disposable. The later
[five-epoch pilot report](../batch03/rl_pilot.md) records training and evaluation.

To regenerate the split manifest and rerun the benchmark after staging the six
source records and WAVs, use Conda environment `10703`:

```sh
python -m scripts.prepare_pilot_split
python scripts/check_modal_account.py
modal run -m infra.modal_benchmark --split pilot
```
