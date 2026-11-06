# First batch: establish a runnable Bailando++ baseline and test two hypotheses

Status: step B passed on one T4. Run `smokeB-20261106T204747Z` produced a 20-second
mHO5 skeleton clip. Credit expiration is still not exposed by the billing API;
applied credits covered the run and billed cost is $0.
See [setup status](../reports/batch01/assets_and_environment.md). Updated November 6, 2026.
Working directory: `/Users/zhengfeili/2026_Fall/10703/10703_Project`.

**Aim:** establish the correct model, test music sensitivity, and expose the reward's
basic behavior for at most **$10 total Modal usage**. Target completion is one
focused working day after data/checkpoint access. This batch does not fine-tune
the model. It produces reusable inputs for the independent-update batch next.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full scope and source references.

## A. Asset and environment inventory — CPU/local first

- [x] Acquire the official repository under `third_party/Bailando` and pin commit
  `cc90b98bff81c9709570db413c9610c2562e27ca`. Work from its `tpami_bailandopp`
  entry points, retaining required original-Bailando assets.
- [x] Record availability, source, hash, stage, expected module, and load status
  for the supervised ++ music-Transformer policy, position VQ-VAE, ++ rotation
  decoder, and optional released RL policy. Download to persistent asset storage;
  inspect archive contents rather than assuming all checkpoints are included.
- [x] Locate two distinct songs A/B, audio, corresponding real motion, and the
  correct ++ music features. Verify that the policy uses the first 55 stored channels
  and nominal 60 Hz sampling; confirm lengths and beat-column meaning from the
  loader. Do not accidentally use original-Bailando 7.5 Hz features.
- [x] Select songs with at least 20 seconds of valid music/audio plus required
  context, and real motion covering seconds 2–6. Record IDs and crop timestamps.
  These are development songs, not the later held-out evaluation set. The selected
  real excerpts are 7.1 and 7.383 seconds; do not pad them to claim 20 seconds of real data.
  See `configs/batch01_data.json` and the data validation report.
- [ ] Record credit redemption/expiration, workspace, baseline usage, and a $10
  first-batch usage ceiling. Include CPU/RAM and setup charges in the ledger.
  Authentication is workspace `zhengfel`. A payment method is required before
  Modal will start a T4. Credits then covered the November 6 runs: metered
  cost $0.03039, billed cost $0. Credit expiration is still not returned by
  the billing API. The $10 first-batch ceiling includes CPU, RAM, and builds.
  Teammate instructions are in `docs/MODAL.md`.
- [x] Build the pinned environment and stage assets without a GPU. Use the
  project's environment; otherwise use `conda run -n common python` locally.
  Record compatibility edits separately from algorithm changes.
  The Modal image built on November 6. Legacy Python 3.8 / `torch==1.6.0+cu101`
  imported, and the Python 3.11 port has `torch==2.4.1+cu118`. The smoke test
  used the legacy interpreter. Compatibility notes are in `infra/COMPATIBILITY.md`.
- [x] Prepare Modal functions with one GPU container, timeout, no automatic
  retries, persistent outputs, and a distinct run ID. Avoid a perpetual GPU session.
  `gpu_smoke` used one T4, 900 seconds, `max_retries=0`, and volume
  `bailandopp-batch01`. Run ID: `smokeB-20261106T204747Z`.

**Text:** `reports/batch01/assets_and_environment.md`, including blockers.
**Visual:** one checkpoint/data readiness table.
**Gate:** a known pre-RL ++ checkpoint and the position decoder are available for
the intended baseline. If only released RL weights exist, label all subsequent
diagnostics accordingly and leave the supervised replication gate unresolved.

## B. One GPU smoke test — within the $5 setup allowance

- [x] On one T4 with a 15-minute timeout, load the position decoder and ++ policy;
  list every missing/unexpected key and justify any expected critic-only keys.
- [x] Encode a valid source pose sequence and save its first upper/lower code
  pair as initialization I. Record the encoder context: fixing I does not imply
  that every decoded first-frame coordinate is context-independent.
- [x] Generate approximately 20 seconds for song A, supplying the required music
  context/padding. Assert finite arrays and valid code ranges; save all timestamps.
- [x] Decode through the position path. Save raw reward input separately from
  global positions used for rendering. Check root velocity integration and units.
- [x] Measure cold-start/load time, generation/decode/reward time separately,
  peak allocated/reserved GPU memory, GPU type, and estimated/actual cost.
- [x] Produce a simple skeleton MP4 with synchronized source audio. If audio is
  inaccessible, label it silent and leave audio-sync validation pending.

**Text:** `reports/batch01/smoke_test.md`, with the exact command and versions.
**Visual:** one baseline clip and a duration/memory/cost table.
**Gate:** output is valid and temporal alignment is explainable. Do not solve a
software error by immediately renting a larger GPU. If memory is the blocker,
profile the failing operation and record a bounded larger-GPU test proposal.

## C. Minimal fixed-initialization music experiment

Execution and numerical diagnostics complete: `musicC-20261106T211618Z`.
See `reports/batch01/music_sensitivity.md` for the table, plots, and three clips.
A-repeat is exact. Human audiovisual review remains pending; steps D/E are not run.

Use the same checkpoint, I, generation mode, decoder, and output length throughout.
Reuse the smoke-test result instead of regenerating it.

| Run | Conditioning | Purpose |
|---|---|---|
| A-original | Song A | Baseline; reuse the step B smoke-test artifact |
| A-repeat | Same song A and settings | Measure reproducibility |
| B-swap | Song B | Test changing music while preserving I |
| A-shift | Song A advanced by eight native music frames | Test response to a controlled timing change |

There are **four generations total including the smoke test**, not a large sweep.
For A-shift, source frame t+8 becomes conditioning frame t. Preserve the same
frame-to-code phase, supply genuine surrounding context, shift the beat labels
and display audio consistently, and do not wrap the end of the song. Save the
transform and verify it with an impulse/timestamp check before inference.

Target 20 seconds of valid output per run. Compare the same central ten-second
interval (seconds 5–15 relative to each aligned output origin), excluding padding
and initialization transients. Record the actual common interval if the verified
loader introduces an offset. Compare changed runs against both their own music
beats and A's original beats. B-swap has no requirement to preserve A's motion style.

Calculate code disagreement, motion difference, beat timing/coverage, joint speed,
and jerk. A deterministic repeat should have identical codes; otherwise quantify
the discrepancy before attributing changes to music. Do not use four runs to
claim statistical significance or global music understanding.

**Text:** `reports/batch01/music_sensitivity.md`, answering whether motion and
timing respond, separately.
**Visual:** a four-row table, beat/velocity overlay, and A-versus-B/A-shift clips.
**Gate:** identical initial codes and coherent timestamp transformations are verified.

## D. Minimal reward audit — reuse cached motion

Use song A and the shared four-second interval **[2, 6) seconds**, native frames
**[120, 360)**, for every row below. This differs from the generated-only music
experiment's [5, 15) interval because the real excerpt is only 7.1 seconds long.
Use valid surrounding frames for derivatives and shifted controls, and exclude
loader padding. Score:

| Control | Construction |
|---|---|
| Real | Corresponding real dance, converted to the verified reward coordinates |
| Reconstructed | Encode/decode that real motion through the same position model |
| Generated | Cached A-original motion |
| Shifted | Advance generated motion by eight motion frames against unchanged music; use valid overlap |
| Repetitive | Tile a two-second generated segment; mark seams explicitly |
| Nearly stationary | Hold a valid pose and add tiny rigid yaw motion: sinusoidal amplitude 0.5 degrees at 0.5 Hz, fixed root |
| Stationary reference | Hold the same pose with no motion |

Construct low-motion controls in valid position coordinates, then convert to the
same reward input convention. Rigid yaw preserves bone lengths. Use a source pose
with nondegenerate body-normal calculations; flag non-finite rewards instead of
hiding them. Do not assume tiny movement is equivalent to zero movement for a
local-minimum beat detector. For the shifted control, make the common interval
valid for every table row and divisible by eight native frames.

Report mean total reward, beat/body components, music-beat-bin fraction, reward
on beat-containing bins, movement magnitude, and timing coverage. Display the
released ++ reward truth table alongside the measured table. It grants +5 on
no-music-beat bins, so test whether an apparently good total is dominated by those
bins. Any finite-input violation of the expected table requires investigation.

The released scoring helper uses CUDA. Score all controls in one short job, or
first verify a device-neutral adapter against the released implementation. Plot
and render locally afterward. Do not repeatedly regenerate motion for this audit.

**Text:** `reports/batch01/reward_audit.md`, distinguishing expected behavior,
observed successes/failures, and unresolved cases.
**Visual:** a seven-row control table, reward/beat timeline, and two annotated
examples selected to illustrate agreement or disagreement with motion quality.
**Gate:** every interpretation is supported by its saved input and reward output.

## E. Export one trace and close the batch

- [ ] Export one complete training-sized window and one later overlap window
  from A-original, including music padding, codes, decoded frames, beats, and mask.
- [ ] Draw the ++ pipeline including its music Transformer and the separate
  position-reward versus rotation-rendering paths. Record which native frames
  belong to one selected action and its reward; note music look-ahead.
- [ ] Reconcile spend against $10; list time/memory estimates and remaining credit.
- [ ] Produce `reports/batch01/summary.md` with a go/no-go decision for each gate,
  links to evidence, and any requested teammate checks. The decision is about
  valid execution and interpretation, not whether the hypothesis was confirmed.

**Text:** one-page batch summary and initial rollout walkthrough.
**Visual:** pipeline/time-alignment diagram plus a completion/cost scorecard.

## Work that can proceed in parallel

1. After asset identification: environment preparation and local source tracing.
2. After A-original exists: cached reward-control construction, metric/plot code,
   and teammate representation review while remaining GPU generations run.
3. After trace export: both teammates independently prepare the update calculation.

Keep GPU execution sequential during this batch. Teammate inputs are a
reconstruction example and a review of the rollout explanation; their own
representation experiment implementation remains separate.

## Next batch, only after valid first-batch evidence

Expand to six songs/two initializations; complete reward shift sweeps and the
rollout trace; independently verify actor and critic steps on disposable copies;
benchmark an entire update cycle and estimate epoch cost. These use the remaining
$20 exploration allowance. Then select the bounded RL pilot duration using the
main plan's cost rule.

Do not start long fine-tuning, new reward design, representation retraining,
large hyperparameter sweeps, or polished avatar rendering in this first batch.
If the $5 setup portion is exhausted without a valid rollout, stop paid debugging
and report the specific blocker before expanding GPU work.
