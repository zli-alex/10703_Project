# Bailando++ midpoint beat-objective comparison

All four controlled training runs and their prespecified evaluations completed.
Gaussian beat proximity did not improve held-out BAS or 100 ms F1 over the bin
objective in this screen. Both matched seeds favored bin on both metrics.
Four test songs and two training seeds limit the strength of that finding.
[Comparison run](https://modal.com/apps/zhengfel/main/ap-AWQ5zKf2hW5Okeb659QqrN)

## Research question and method

The experiment replaces the released eight-frame bin rule with a Gaussian
proximity reward while holding the stochastic actor–critic algorithm, motion
detector, pretrained weights, and generation budget constant. The state is the
music window and sampled motion-token history. The action samples the upper and
lower codes independently conditional on this state, then appends both. A frozen
position decoder maps the entire code sequence to motion. The critic predicts
one state value by summing its two output branches.

Each rollout receives a sequence-terminal return with discount 1. The actor
uses the sum of the two sampled log probabilities at each state, weighted by
the terminal return minus the detached pre-update value. The critic minimizes
mean squared return error. Both losses average over generated actions. There is
one actor and one critic optimizer step per fresh collection batch. The music
encoder, GPT base, and decoder remain fixed. Dropout is disabled in both
collection and the update. Reference-policy KL is logged with coefficient zero.
Matched-event F1 is evaluation-only.

The bin arm receives the fraction of complete bins without a missed music
event, equivalent to the released mean bin reward shifted and scaled to
`[0,1]`. The BAS arm averages Gaussian proximity to the nearest motion event
for each music event, with sigma three frames at 60 Hz. Both arms score the same
exported-position events and generated frames. For music at frame 8, motion at
frame 7 misses the bin but scores 0.946 BAS. Motion at frame 15 hits the bin but
scores 0.066 BAS. This tests a difference in reward ranking on shared events.

## Evaluator and historical evidence

The position export integrates frame `t−1` root displacement into the frame
`t` root position, then adds that root position to the other joints. The CPU
adapter matched seven cached exported arrays exactly. Fourteen differential
fixtures agreed with the released detector and BAS functions. Event detection
uses mean interframe joint displacement, Gaussian smoothing with sigma five
frames and reflect boundaries, and strict local minima indexed at the left
frame of the difference. Detection uses the available decoded context before
filtering events to the scored interval. In the new protocol that interval is
`[8, 1208)` on the native 60 Hz timeline. The first code provides context and
receives no credit. Bins start at frame 8. The historical full-position panel
uses the available common support after frame 8. The legacy raw panel preserves
its original complete-bin interval from frame 0.

One-to-one matching uses maximum cardinality and then minimum total absolute
timing error at inclusive 50, 100, and 150 ms. Coverage uses many-to-one nearest
neighbors in the music-to-motion direction. With music and no detected motion,
BAS, coverage, recall, and F1 are zero, precision is undefined, and both
distance summaries are undefined. With no music, alignment is undefined.
Nonfinite motion is an error. The quality diagnostics use root-relative joint
speed, RMS jerk, fraction of frames below 0.05 decoder units/s mean joint speed,
and distances between joint-speed distribution signatures within song.

All **24 of 24** historical dances were recovered. The per-bin reward and event
indices matched the archive exactly, and the saved reward, hit, coverage, and
distance summaries reproduced to floating-point tolerance. On the three songs
held out of historical RL, the full-position BAS changed from 0.218 to 0.225
and 100 ms F1 from 0.346 to 0.362. On the first 20 generated seconds of the
same songs, BAS changed from 0.261 to 0.239 and F1 from 0.404 to 0.373.
The endpoint comparison is sensitive to which interval is scored. These are
rescored outputs from the older greedy trainer and add zero new trajectories.

The stationary cached control has zero detected motion events. The released
bin rule still credits bins without music events. The offset control shifts
motion events without wraparound and scores all offsets on a fixed interior
interval. The detector locates motion speed minima. Beat channel 53 is a
tracked beat stream. The published paper, released implementations, and the
project's historical and midpoint generation protocols have different scopes.
[Historical tables](pilot_rescore.md) · [Per-case data](pilot_cases.csv) ·
[Paired plot](pilot_metric_changes.png) · [Controls](controls.json) ·
[Evaluator checks](evaluator_checks.json)

## Split and benchmark

The verified native full-track release contains ten usable music IDs. The
common adaptation panel is mHO5, mJB5, and mPO1. The validation panel is mBR0,
mJS3, and mKR2, all explored in the earlier pilot. The final test panel is
mLH4, mLO2, mMH3, and mWA0. These four song IDs were unused in the earlier
project runs. Music IDs are disjoint across groups, and the exact supervised
checkpoint's training membership is unknown. The smaller split gives fewer
independent songs than originally planned. The full-track release has one song
per genre, so genre composition varies between the splits.

The T4 screening run used four stochastic samples per condition over four
development song/start conditions. Across 24 within-condition pairs, bin and
BAS had 4 and 0 exact ties, 5 and 1 ties within 0.01 including exact ties,
and one pair ranked in opposite directions. There were zero duplicate code
sequences. Detected motion event counts ranged from 30 to 38 per 20 seconds.
Bin returns ranged from 0.760 to 0.873 and BAS from 0.077 to 0.317. The reward
screen measures spread and ordering. Its outputs do not establish better
perceptual timing.

The finite disposable update verified both action likelihood components, with
zero maximum observed recomputation error. The independent context checks
covered the start, the full-window transition, and the final query. Fingerprints
confirmed that only the intended head moved at each optimizer step and the
frozen modules were unchanged. Reload recovered both heads, optimizers and
RNG state and reproduced the sampled codes, likelihoods, and evaluation metrics.
The continuation was visually inspected through a contact sheet and rendered
with the original timing offset and audio. Reviewer count for blinded judgments
is zero.

| Completed setup stage | Completed continuations | Actions | Scored frames |
|---|---:|---:|---:|
| Historical CPU rescoring | 0 | 0 | 0 |
| Development screen and disposable update | 16 | 2,400 | 19,200 |
| Pre/post reload evaluation | 2 | 300 | 2,400 |

The successful benchmark took 74.74 worker seconds. Batch rollout took 19.69 s,
decode/scoring 0.58 s, and the checked update 28.57 s. Peak allocated GPU
memory was 2.37 GiB on a Tesla T4 under legacy PyTorch 1.6.0+cu101. The first
attempt stopped at the released-forward logit differential check. Using the
released head for the full projection resolved the discrepancy. That attempt
completed no scored continuations. Metered worker compute was **$0.02370593**
for both setup attempts. [Split](../../configs/midpoint_beat_split.json) ·
[Screen](benchmark/reward_screen.json) · [Benchmark manifest](benchmark/summary.json) ·
[Contact sheet](benchmark/smoke_contact_sheet.png) · [Costs](benchmark/costs.json)

## Completed training and compute

The prespecified endpoint is round 25 for both bin and BAS, with matched seed
IDs 42 and 43. Each round collects 16 continuations of 150 sampled upper/lower
pairs and 1,200 scored frames each. Round zero uses one shared frozen policy
validation panel. Validation is at rounds 0, 5, 10, 15, 20, and 25. Final test
samples each policy at two starts and three draws per song. The final
checkpoints are chosen by round number before exposure to test results.

| Completed phase | Continuations | Sampled action pairs | Scored frames |
|---|---:|---:|---:|
| Four training arms | 1,600 | 240,000 | 1,920,000 |
| Validation including frozen baseline | 252 | 37,800 | 302,400 |
| Final test including frozen baseline | 120 | 18,000 | 144,000 |
| Total | 1,972 | 295,800 | 2,366,400 |

The four arms completed 100 actor and 100 critic steps in total. All endpoint
checkpoints reloaded and their hashes were verified before test generation.
The behavior and update log probabilities agreed exactly for both action
branches over all logged rounds. Peak allocated GPU memory was 2.86 GiB.
The comparison completed in 5,553.38 worker seconds across six serial stages,
with no comparison-stage retries or failures. The Modal client timed out
reading final app logs after the entrypoint completed. All six stage manifests
and the expected result files were retrieved and verified independently.

Modal metered **$1.21255570** for the comparison and **$1.23626163** including
the two setup attempts. Within comparison worker time, the measured phase
durations correspond to about $1.0142 for training, $0.1096 for evaluation,
and $0.0310 for setup, checkpointing, and checks at the quoted resource rates.
The phase estimates exclude overhead outside the measured worker sections.
The comparison was projected at $1.23 and stayed within the approved $1.90
T4/CPU/RAM compute cap. App-level metering here excludes separately metered
storage and egress. [Frozen protocol](../../configs/midpoint_beat_frozen.json) ·
[Comparison cost ledger](comparison_costs.json)

## Held-out results

Each mean weights songs equally after averaging draws and starts, then weights
the two training seeds equally. F1 is averaged per clip after one-to-one matching.

| Policy | BAS | Precision 100 ms | Recall 100 ms | F1 100 ms | Events/s | Mean joint speed | RMS jerk | Within-song diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen supervised | 0.199 | 0.329 | 0.332 | 0.328 | 1.685 | 0.674 | 364.3 | 1.714 |
| Bin RL | 0.212 | 0.361 | 0.366 | 0.361 | 1.689 | 0.690 | 370.1 | 1.544 |
| BAS RL | 0.203 | 0.345 | 0.347 | 0.344 | 1.698 | 0.671 | 361.8 | 1.607 |

BAS RL changed BAS by **+0.0033** and F1 by **+0.0155** versus the frozen
policy. Seed 42 changed BAS by −0.0067 and seed 43 by +0.0133. Compared with
matched bin RL, BAS RL changed BAS by **−0.0087** and F1 by **−0.0173** after
averaging seeds. The 95% song-bootstrap intervals for those two BAS-versus-bin
contrasts were [−0.0207, −0.0009] and [−0.0419, +0.0018]. They describe
variation over the four observed music IDs conditional on these seeds.

On mMH3, the two objectives tied to four decimals in BAS and BAS RL gained
0.0082 F1. On mWA0, BAS RL lost 0.0270 BAS and 0.0543 F1 relative to bin RL.
Event density stayed within 0.013 events/s of the frozen mean for either arm.
Stationary fractions rounded to 0.001 or less for all three policies. Both
arms had lower within-song speed-signature diversity than the frozen policy.
These diagnostics describe motion statistics and cannot resolve audiovisual
naturalness or synchronization judgments.

The validation BAS ended above the shared baseline for all four runs. The
validation F1 averaged 0.383 for BAS RL and 0.366 for bin RL at round 25, while
the ranking reversed on the final test. Across development collections, the
first-to-last five-round mean BAS reward rose from 0.201 to 0.216 for seed 42
and fell from 0.210 to 0.200 for seed 43. Critic MSE declined in both BAS runs
with a larger residual than under the bin return scale. Reference-policy KL
remained below 0.001 nats per action pair across all arms. The mixed training
trajectories and limited policy drift leave open whether a different fixed
budget would change the outcome.

Two coded review panels use the first two test IDs, first start, first draw,
and seed 42, with common soundtrack and scale. Both panels contain 1,200
frames at 60 Hz and matching 20-second audio. Blinded review is pending with
zero reviewers so far.
[mLH4 clip](../../runs/midpoint_beats/review/mLH4_review.mp4) ·
[mLO2 clip](../../runs/midpoint_beats/review/mLO2_review.mp4)

[Seed-level results](comparison.md) · [Test cases](test_cases.csv) ·
[Per-song paired changes](test_song_deltas.csv) · [Learning curves](learning_curves.png) ·
[Coverage and paired changes](test_timing.png)

One follow-up is to repeat this frozen comparison on more music IDs and
independent training seeds. The result varied by song and seed, and adding
independent music and seed replication would test whether the advantage for
bin persists.

## Reproduce

From the project root in Conda environment `10703`:

```sh
python -m scripts.check_midpoint_beats
python -m scripts.rescore_midpoint_pilot
python -m scripts.midpoint_beat_controls
python -m scripts.summarize_midpoint_beats
python -m scripts.stage_midpoint_split --fetch
python scripts/check_modal_account.py
python -m scripts.freeze_midpoint_beats runs/midpoint_beats/midbeats-20261117T213508645736Z/benchmark
modal run -m infra.modal_midpoint_beats --stage comparison --approved-cap-usd 1.90
python -m scripts.summarize_midpoint_beats --run runs/midpoint_beats/<run-id>
python -m scripts.render_midpoint_beat_clips --run runs/midpoint_beats/<run-id> --fetch-audio
```

The paid comparison command requires authorization. To regenerate tables from
the completed run, use `midbeats-20261117T213957703365Z` as the run ID. The
round-25 checkpoints are stored on Modal volume `bailandopp-batch01` at
`runs/midbeats-20261117T213957703365Z/<arm>/last.pt`. The arm's
`summary.json` records its hash, trainable parameter names and frozen model
fingerprints. The same run's downloaded `test/` directory stores the test
codes, decoded motions and both selected-action log probabilities.
