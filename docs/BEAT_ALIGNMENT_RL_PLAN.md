# Beat-alignment RL research plan

November 17, 2026. Status: proposed experiments, not completed training.
This is the active plan for the next research phase. The original PROJECT_PLAN.md
remains the record of the replication phase.

The [basic-beat implementation plan](BASIC_BEAT_IMPLEMENTATION_PLAN.md) specifies
the immediate reward/evaluation contracts, modules, and midpoint-sized experiment.

## Research objective

Learn a music-conditioned dance policy through reinforcement learning that improves
basic beat synchronization while retaining plausible motion and useful pretrained
behavior. Study how reward definition, reference regularization, and value-learning
schedule affect learning and generalization. Hierarchical rhythm is a later extension.

The central deliverable is a trained-policy comparison with learning curves,
independent training seeds, held-out music, ablations, and qualitative evaluation.
Metric auditing and pipeline work support that experiment; they are not its endpoint.

Research questions:

1. Does a graded temporal-alignment reward improve held-out synchronization more
   reliably than the released bin-based reward under the same RL algorithm?
2. Does reference-policy regularization improve the alignment–motion-quality
   tradeoff relative to alignment-only RL?
3. Do critic initialization and update allocation materially affect adaptation?
4. Which automatic improvements agree with perceived synchronization, and which
   exploit detector behavior or impair movement?

Hypotheses, not promised outcomes: graded rewards provide more useful distinctions
between near misses; reference regularization reduces harmful drift; critic warm-up
improves early advantage estimates. A controlled null or negative result remains a
research contribution. More generated samples alone do not establish improvement.

## 1. Formulation and relationship to prior work

State: the music conditioning available to the policy, generated upper/lower code
history, and temporal position. Action: the next upper/lower code pair. Transition:
append that pair and advance the music position. The frozen decoder maps the code
sequence to motion. Reward: a score computed from the generated motion and supplied
beat times. This is online sequence-generation RL with known computational dynamics,
not physical simulation and not purely offline RL.

Freeze the VQ-VAE, music encoder, and GPT base initially; train the actor head and
the value head. The pretrained supervised checkpoint is the BC initialization.
No representation retraining is required for the core experiment.

MusicRL motivates sequence-level reward optimization of an autoregressive token
policy, with reference regularization and evaluation beyond the optimized score.
Sequence Tutor motivates preserving a pretrained distribution during task-reward
optimization. We borrow these principles, not a claim of reproducing those systems.
E3D2 supplies a directly relevant warning: a dance alignment score can rise while
motion-quality metrics worsen.

- [MusicRL, ICML 2024](https://proceedings.mlr.press/v235/cideron24a.html)
- [MusicRL method](https://arxiv.org/html/2402.04229v1#S3.SS2)
- [Sequence Tutor, ICML 2017](https://proceedings.mlr.press/v70/jaques17a.html)
- [E3D2, reward ablation](https://arxiv.org/html/2312.11442v1#S6.SS2)

Course connections: policy gradients and baseline subtraction; Monte Carlo versus
TD value fitting; actor/critic scheduling; BC and distribution shift; KL-constrained
improvement; reward misspecification; generalization and experimental controls.

## 2. Data and benchmark expansion

First audit unique underlying music IDs in all available source archives. Multiple
dance recordings sharing music must stay in the same split. The current full-wave
subset has one music ID per genre; 40 dance sequences must not be described as
40 independent songs. Expansion needs additional music assets, not merely more
overlapping windows or starts of the current six songs.

Provisional target, conditional on the asset audit: at least 20 adaptation music
IDs, 6 validation IDs, and 10 untouched final-test IDs, covering multiple genres
and tempos. These are design targets, not verified available counts. Prefer genre
coverage across splits where possible; otherwise describe genre-shift evaluation
explicitly. If fewer IDs are accessible, freeze a smaller feasible split and state
the narrower scope rather than borrowing test songs for adaptation.

The current six-song pilot remains a regression/development set. Its inspected
RL-held-out songs are not a fresh final test. Keep music IDs from them out of the
new untouched test. Record known/unknown overlap with supervised pretraining;
held out from our RL is not necessarily held out from the pretrained model.

Use three evaluation layers:

| Layer | Purpose | Scope |
|---|---|---|
| Diagnostic controls | Validate timing and failure behavior | Shift, stationary, repetition, and jitter controls on several development songs |
| Expanded AIST++ benchmark | Primary trained-policy comparison | Frozen music-ID train/validation/test split, shared preprocessing and decoder |
| External music or second dataset | Generalization stretch | Only after core comparison; separately validate features, beats, and representation |

For published comparisons, add a protocol-matched 20-second evaluation on the
official test recordings only when their membership and preprocessing are verified.
Report that separately from the music-disjoint research split. Do not claim a paper
reproduction from matching the BAS formula alone. The paper reports 10 RL epochs;
the released YAML's 500 is not the paper's reported fine-tuning duration.

## 3. Basic-beat objectives

No downbeat, subdivision, or phrase-hierarchy term in the core study.

| Objective | Definition | Hypothesis / failure to inspect |
|---|---|---|
| R1: released coincidence | Original eight-frame music/dance-beat coincidence rule | Coarse bin boundaries and action-independent positive offset |
| R2: graded alignment | Published music-to-motion Gaussian BAS on decoded motion | Near-miss sensitivity, but possible proliferation of detected motion beats |
| R3: matched-event alignment | One-to-one music/motion event matching within a fixed tolerance, using event precision/recall F1 | Discourages excess events, but may penalize musically valid offbeat accents |

R3 is an experimental rhythmic preference, not a claim that every musical beat
requires exactly one motion accent. The existing detector identifies speed minima,
not verified foot strikes. Inspect both kinds of error in videos.

Before training, define detector coordinates, smoothing, difference-index convention,
crop context, tolerance, empty-event handling, and aggregation. Use a validated,
common detector/coordinate convention across objectives in the objective-comparison
suite. Keep an exact legacy path separately; changing the detector and the reward
together would confound a reward-only claim. The published BAS evaluation must retain
its own documented convention, even if the training detector differs.

If motion beats are absent, assign zero to alignment training scores and record the
failure explicitly; do not interpret an undefined timing distance as zero error.
Clips with no music beats need a declared exclusion/handling rule before selection.
Audit rewards on shifted controls using common valid interior intervals.

For the new sequence-level trainer, use fixed, documented scaling to [0,1] where
possible; R1 can be affinely mapped from its mean [-5,5] range. This is a controlled
variant, not bitwise legacy replication. Fix scaling from development data and do
not retune it on test results. Log raw and scaled values. Removing the no-beat offset
alone does not introduce a new quality preference on fixed-length matched music.

## 4. RL trainer and actor–critic schedule

Maintain two named paths:

**Legacy reference:** preserve the released greedy rollout, window scoring, masked
actor loss, shared Adam semantics, and five-actor/five-critic schedule. Retrain on
the expanded adaptation split from the same supervised checkpoint for comparison.
This provides continuity, but does not satisfy the categorical on-policy
score-function assumptions merely because rollouts are freshly generated.

**Main research trainer:** stochastic categorical rollouts with sequence-level
return and a learned value baseline (Monte Carlo actor–critic / REINFORCE with a
learned baseline). Begin with one actor update per fresh rollout batch. Fit the
critic on empirical returns, rather than introducing TD bootstrap error immediately.
Use separate actor/critic optimizers so inactive-head momentum does not silently
change the other head. Keep this convention fixed for the objective comparisons.

At collection, store exact action log probabilities, history, music extent, action
masks, and collection-policy identity. Validate recomputation under the same context.
Use temperature 1 initially; if a temperature is changed, the sampled distribution
must also be the distribution in the policy-gradient likelihood. Do not train the
score-function estimator on greedy actions. Gradient accumulation across chunks
must precede the single actor step; do not mutate the actor halfway through a
nominally on-policy rollout batch.

Train on 20-second generated continuations with a fixed context/burn-in convention.
Decode the scored continuation consistently, avoiding a different reward process
between training and evaluation. Terminal sequence reward gives every generated
action that return; this is noisy but conceptually clean. Define terminals and
padding masks explicitly. A sampled-return baseline targets this same sequence
reward, not a mismatched standalone decoder-window reward.

Compare schedules separately on one fixed basic objective:

| Variant | Change | Measurements |
|---|---|---|
| S0 | No warm-up; one actor and one critic pass per fresh batch | Learning curve and initial value error |
| S1 | Bounded critic-only warm-up with frozen actor, then S0 | Early advantage/value stability and quality |
| S2, conditional | More critic fitting per actor step, or shorter-horizon TD/GAE | Whether extra fitting/bootstrapping improves sample efficiency |

Use separate development trajectories for critic calibration. Freeze the baseline
used to form actor advantages before fitting it on the same outcomes, or use
cross-fitting; avoid a baseline overfit to its own sampled return. Compare both
matched actor-data budgets and actual compute; warm-up is additional work.
The local legacy wrapper ignores the upstream warm-up flag, so YAML edits alone
do not implement S1. Recheck checkpoint/optimizer restoration for any new trainer.

If Monte Carlo variance prevents learning, first inspect reward spread and critic
fit. A greedy-rollout self-critical baseline is a diagnostic alternative. PPO is
a conditional extension when reusing rollout batches is useful: store old log
probabilities, use a defined clipped surrogate, and measure KL/clip fraction. Do not
change optimizer, reward, and rollout distribution simultaneously and attribute
the resulting improvement to only one of them.

## 5. Reference regularization and main matrix

Primary regularizer: KL(pi_theta || pi_0) on rollout states, with pi_0 frozen.
Compute categorical KL for both heads under identical conditioning and average
over valid generated tokens. The sequence reward and KL normalization must be
documented so changing duration does not silently change the effective weight.
Use a correct regularized policy-gradient treatment, including the consequences
of future KL costs; a same-state auxiliary KL loss is a distinguishable surrogate,
not automatically the exact gradient of a trajectory-KL objective.

Uniform reference BC is a secondary regularizer ablation on the selected reward:
use a fixed adaptation-only buffer and a separately normalized NLL term. Hard BC
on greedy references preserves selected examples, not the full pretrained
distribution. Include a BC-only control if interpreting its contribution. Keep
balanced movement replay outside the required core matrix.

| Batch | Runs | Purpose |
|---|---:|---|
| Instrumentation | One tiny run per new code path | Verify rewards, gradients, frozen components, collection likelihoods, resume |
| Development screen | 3 objectives × 2 regularization settings × 1 seed = 6 short runs | Diagnose scales and failure modes, not rank methods conclusively |
| Main comparison | 3 objectives × {no KL, fixed selected KL} × 3 seeds = 18 runs | Estimate objective and regularization effects and their interaction |
| Legacy reference | 3 seeds | Compare the released training procedure with the new suite |
| Schedule ablation | S0 already represented; S1 × 3 seeds, optionally S2 × 3 | Isolate critic-schedule changes under a fixed objective and KL setting |
| BC alternative, optional | Selected objective with BC, plus BC-only control | Connect reference imitation to retention beyond initialization |

The fixed supervised policy needs no training seeds, but is evaluated with the
same initializations and sampling seeds as trained methods. Treat best-of-N as an
inference-cost reference, not the main project result.

Provisional scale: benchmark a collection batch of 16 independent 20-second
continuations, reducing physical GPU batch size through microbatching if needed.
Screen at 25 collection rounds. Choose a confirmation endpoint of 100–300 rounds
from measured learning and cost on development data, then freeze it before the
main comparison. At 300 rounds this is 4,800 continuations per run, approximately
715,200 generated code-pair actions if each continuation contains 149 new pairs.
This is generated-data volume, not GPU wall time. Log actual actions rather than
relying on this estimate. A seed must control critic initialization, training
sampling, and data order, not just evaluation randomness.

Use all three objectives in the main matrix if the screen passes correctness and
budget checks; if R3 is ill-defined or rewards degenerate motion, reduce to the two
prespecified core objectives and report why. Do not quietly omit unsuccessful runs.
Choose one KL value from a small development-only calibration; record the tuning
budget. A regularization-strength sweep is optional after the main result.

## 6. Evaluation and success criteria

For each test music ID, use four fixed valid starting-code pairs and three stochastic
draws per start: 12 dances per song/checkpoint. For a 10-song test set this is 120
dances per checkpoint, plus an optional four-start greedy panel (40 more). Match
all settings across methods. More starts/draws improve conditional estimates but
do not substitute for independent songs or training seeds.

At predeclared training budgets, evaluate on validation music. Lock the final budget
and method selection before test evaluation. Store intermediate checkpoints for
learning curves on validation, rather than repeatedly using the final test to steer
the study. If selecting a checkpoint instead of using a fixed endpoint, define the
validation selection rule in advance and apply it uniformly.

Evaluate every model against all of:

- Published BAS, released beat-bin hit rate, tolerance-based coverage at 50/100/150 ms,
  matched-event precision/recall/F1, timing distances in both directions, and counts.
- Movement speed, root travel, jerk, repetition/code coverage, and within-song
  diversity. Fix units and representation; no mixed position/rotation columns.
- Human synchronization, motion naturalness, and overall preference as separate
  questions, with randomized A/B order, ties, confidence, and synchronized audio.

Primary automated endpoint: matched-event F1 at 100 ms on untouched songs, with
BAS and 100 ms coverage mandatory companion metrics. This endpoint is itself an
imperfect, partly optimized proxy; call a gain an automatic timing improvement
unless human evidence supports a perceptual claim. Report per-song changes and
results at the other tolerances rather than selecting the most favorable threshold.

Before confirmation runs, define acceptable movement/jerk ranges from development
real/reconstructed/pretrained examples and review them visually. Do not invent
universal numerical quality limits. Treat the outcome as a tradeoff if timing rises
while motion deteriorates. A lower jerk from near-stationarity is not a quality gain.

Suggested small human panel: 30–50 prespecified paired excerpts spanning songs,
methods, and seeds, with at least three independent reviewers per pair if available.
Sample a representative panel for preference estimates; label any deliberately
selected disagreement examples separately. Use identical intervals for metric and
human comparisons. This is exploratory with a small participant count.

Aggregate starts and draws within song first. Report training-seed variability and
song-level deltas; uncertainty calculations should respect both levels. Do not treat
overlapping windows or thousands of generated clips as independent participants
or independent training runs. Three seeds remain a modest replication count.

Success levels:

1. Verified learning and a reproducible multi-seed comparison, even if effects are null.
2. Held-out timing improvement with no material movement-quality degradation.
3. Agreement with blinded perceived synchronization improvements.
4. Generalization to new music conditions or a second dataset.

## 7. Resources and execution gates

The historical five-epoch training plus evaluation cost $0.07236 excluding storage,
but this cannot price the new matrix: stochastic generation, expanded music,
reference passes, and longer contexts alter cost and memory. Benchmark collection,
decode/scoring, actor update, critic update, evaluation, and checkpoint storage
separately on the intended runtime. Estimate the full matrix from those measurements.

Use the dedicated 10703 environment locally and the validated remote runtime until
there is a documented reason to port. Keep immutable input/checkpoint hashes and
per-run configuration, split, seed, log probabilities, update counts, and costs.
Save model, optimizers, RNG, progress, and the necessary data-iterator state for
resumption. Verify numerical fixtures after compatibility changes.

Use the existing project credit envelope as a ceiling, not an assumption of current
available balance. Check actual remaining credits before execution, reserve 20% for
evaluation/reruns/storage, and set per-batch ceilings from the benchmark. If resources
are tight, reduce objective count or rollout budget before eliminating replication
seeds or the untouched evaluation. No large job is launched by writing this plan.

## 8. Milestones and midpoint narrative

Relative schedule begins when implementation starts; these are effort targets, not
verified course deadlines. Dataset access and human-review scheduling may dominate.

| Stage | Target | Deliverable and gate |
|---|---|---|
| A | First 2–3 working days | Music-ID inventory, frozen feasible splits, metric audit, cost benchmark |
| B | Next 2–3 working days | Validated stochastic actor–critic, reference KL, short development screen |
| C | Following week | Main matrix, validation learning curves, one focused schedule ablation |
| D | Following 2–3 working days | Locked test evaluation, reviewer panel, effect/tradeoff analysis |
| E | Only after C/D | Hierarchical rhythm or external-data stretch |

For the midpoint report, retain the completed procedural replication and negative
pilot result as completed work. Present this plan and exact loss definitions as
next work. Once the screen runs, add actual learning curves and verified update
counts, clearly labeling single-seed preliminary results. Do not imply the larger
benchmark or matrix is already available or complete.

## 9. Long-term rhythm hierarchy

Only after basic-beat RL and evaluation are stable:

1. **Mild downbeat emphasis.** Obtain and verify measure-position labels; current
   Librosa beat events do not identify downbeats. Add a small bonus for alignment
   on verified downbeats while retaining ordinary-beat credit. Normalize by total
   target weight so the experiment changes emphasis rather than just reward scale.
   Compare bonus zero versus a small fixed positive value before a larger sweep.
2. **Half-beat subdivisions.** Add weaker credit for selected midpoint accents.
   Keep event-count controls and inspect whether accents fit the audio instead of
   encouraging every possible slot. Audio onset strength or a specified rhythmic
   pattern can determine which subdivisions are appropriate.
3. **Quarter-beat / pattern-conditioned rhythm.** Study only if the motion detector
   and decoder can resolve it. Eight frames per action are about 133 ms; a quarter
   beat at 120 BPM is 125 ms. This is a control-resolution issue, not proof that the
   decoder cannot place sub-token events. Swing/triplets require different grids.

Do not label every fourth detected beat a downbeat without checking meter and phase.
Stronger downbeat emphasis is a controllable stylistic preference, not a universal
definition of good dance; backbeats and syncopation may be appropriate. Evaluate
rhythmic fit, perceived emphasis, and quality separately from the training bonus.
