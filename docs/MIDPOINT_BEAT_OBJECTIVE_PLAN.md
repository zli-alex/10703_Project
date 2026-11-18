# Midpoint: controlled beat-reward comparison

Status: implementation plan, November 17, 2026; no new runs or results implied.
This document narrows the midpoint scope of BASIC_BEAT_IMPLEMENTATION_PLAN.md.
That document remains the metric specification; its three-reward experiment is
an extension. The required midpoint experiment below has TWO training objectives.

## Research question and claim boundary

Does replacing bin-coincidence reward with continuous temporal-proximity reward
improve RL fine-tuning of a pretrained dance policy, when both use the same
trainer, detector, initialization, and generated-data budget?

Hypothesis: BAS gives useful credit for small timing improvements across bin
boundaries. Counter-hypotheses: BAS saturates, encourages extra motion events, or
improves its own detector score without perceptually better synchronization.
Success means an interpretable controlled result, not necessarily a positive gain.
Do not call automatic timing scores human preferences.

## Fixed scope

Required: frozen pretrained baseline; bin-rule RL; BAS RL; two training seeds;
BAS and event precision/recall/F1 evaluation; learning curves and quality checks.
Deferred: F1 as a training reward, GRPO/DPO, KL sweeps, GAE/schedule ablations,
hierarchical beats, new representations, and full benchmark-scale claims.

The existing greedy-rollout pilot is historical evidence, not a controlled
competitor to the new stochastic trainer. Position-path bin reward is a controlled
variant of the released rule, not exact reproduction of its raw-coordinate path.

## Phase 1: inspect and establish evaluator correctness

Read local instructions, the basic-beat specification, pilot reports, training and
evaluation scripts, and upstream reward/evaluator/exporter code. Inventory actual
cached artifacts, checkpoint hashes, music IDs, frame rates, and split membership.
Reuse existing modules where possible; proposed filenames are not required APIs.

Implement a CPU evaluator returning detected events and per-clip metrics:

- BAS: music-to-nearest-motion Gaussian score, sigma 50 ms.
- Original-rule bin reward and music-bin hit rate, with explicit bin origin.
- Many-to-one coverage at 50/100/150 ms.
- Maximum-cardinality one-to-one precision/recall/F1 at those tolerances;
  break matching ties by minimum total timing error.
- Both directions of timing distance; music/motion event counts and event density.

Preserve a legacy raw-coordinate adapter for reproducing pilot values, and a
verified exported-position adapter for the common objective/evaluation path.
Freeze smoothing, difference indexing, crop, boundary treatment, and time origin.
Use surrounding context for event detection where available; score only the declared
generated interval. Do not count the conditioning prefix as generated performance.
Document absent-beat and absent-motion handling explicitly; never silently drop
failed generations or turn undefined distances into perfect zeros.

Acceptance: analytic bin-boundary and timing-offset fixtures; exact-match,
missing/extra/duplicate-event and empty-set fixtures; stationary-motion diagnostic;
differential BAS comparison with upstream on valid fixtures; exporter agreement.
These checks protect scientific definitions and justify focused tests.

## Phase 2: rescore existing evidence and freeze data

Rescore the 24 historical generated dances if the inventory confirms availability.
Reproduce legacy columns before adding position-path scores. Report full-track and
prespecified 20-second panels separately; compare metrics on identical intervals.
Save per-case records, paired deltas, missing-case accounting, hashes, and plots.
This is new evaluation of old outputs, not additional RL training.

Audit splits at unique MUSIC identity, not recording filename. The old six-song
pilot set is already explored and may support development, not a fresh final test.
Distinguish held out from RL adaptation from held out from original pretraining;
the latter may be unknown.

Planning target, conditional on inventory: 10 adaptation, 4 validation, and 6
previously unused test music IDs. Do not invent availability. If insufficient,
freeze a smaller disjoint split before training and explicitly narrow the claim.
Stratify genres where possible; use the same starts and music across objective arms.
Keep test music untouched until the protocol is frozen.

## Phase 3: common RL trainer and smoke experiment

State = music conditioning plus sampled motion-code history. Action = upper/lower
code pair using the model's actual factorization. Transition appends sampled codes;
the fixed decoder maps codes to motion for scoring. The policy is initialized from
the same supervised checkpoint; freeze the encoder, base, and decoder according
to the verified adaptation setup. List trainable parameter names in the manifest.

Use fresh categorical rollouts, exact behavior-policy likelihoods, a learned value
baseline, sequence-terminal reward, undiscounted Monte Carlo returns, and one
actor update per fresh collection batch. Compute/detach actor advantages using
the pre-update critic; fit the critic with a separate optimizer. Reuse the same
critic initialization, optimizer settings, masks, start distribution, temperature,
and loss reduction across arms. Check dropout/eval modes so rollout and likelihood
recomputation describe the same policy. Never substitute greedy actions for
on-policy samples. No replay or multiple policy updates in this midpoint trainer.

Rewards on the SAME position events and generated interval:

    bin: (mean released-rule bin reward + 5) / 10
    BAS: mean Gaussian proximity over music events

Both lie in [0,1], but have different variances and song weighting. Log reward
variance, advantages, actor/critic losses, entropy, gradient norms, and reference
policy drift. Initial comparison uses KL coefficient zero; do not add KL to only
one arm. If smoke testing reveals collapse, revise a common protocol on development
data and restart both arms, recording why; do not quietly rescue selected runs.

Acceptance: finite rollout/update; correct upper/lower log probabilities and
conditioning; frozen weights unchanged; actor and critic change only as intended;
no inactive-head optimizer momentum updates; padding/prefix masks correct;
checkpoint reload reproduces evaluation. Inspect one actual decoded continuation.

Screen reward informativeness with four independent samples for each of four
development conditions (16 trajectories, reusable for both reward calculations).
Report ties, near-ties with a fixed tolerance, duplicate sequences, event counts,
and reward-ranking disagreements. Score variance alone is not evidence of quality.

## Phase 4: bounded training comparison

Provisional budget, freeze after one representative rollout/update benchmark:

| Setting | Midpoint target |
|---|---|
| Objectives | bin, BAS |
| Independent training seeds | 2 per objective; matched seed IDs |
| Collection rounds | 25 per run |
| Batch | 16 generated continuations per round |
| Scored continuation | 20 seconds, excluding conditioning |
| Training total | 400 continuations/run; 1,600 across four runs |
| Validation checkpoints | rounds 0, 5, 10, 15, 20, 25 |
| Validation panel | 4 songs x 2 fixed starts x 2 stochastic draws = 16/checkpoint |
| Final test panel | 6 songs x 2 fixed starts x 3 draws = 36/policy |

Round zero uses the shared frozen baseline. Cache it once when all generation
settings match. Five later checkpoints across four runs give 320 validation
continuations; the shared validation baseline adds 16. Four final policies plus
one shared test baseline give 180 test continuations. Thus the proposed core has
1,600 training + 336 validation + 180 test = 2,116 generated continuations, excluding
16 reward-screen samples, smoke/benchmark runs, retries, and optional greedy clips.
Validation/test generation is evaluation compute, never training data.

Use balanced adaptation-song sampling and a common schedule across arms. Count
actual generated actions/frames as well as rounds; fixed duration alone can hide
padding differences. Publish separate training, evaluation, and setup costs.
Benchmark runtime and GPU memory before committing; infer no cost from old window
updates alone. Respect documented existing spending authorization and remaining
budget. If new authorization is required, present the executable configuration,
measured projection, and hard cap before requesting it. Do not launch an unbounded
sweep. This planning document itself launches no paid work.

Use round 25 as the prespecified primary endpoint, avoiding selection on different
rewards. Validation curves diagnose behavior; test is scored only once after
protocol freeze. If compute requires fewer rounds, decide the common count before
the four runs. Preserve failed seeds and classify operational versus learning
failures. Do not stop an arm early because its result looks unfavorable.

## Phase 5: analysis and midpoint artifact

Primary outcome: held-out BAS change versus frozen baseline, plus BAS-RL minus
bin-RL. Key complementary outcome: matched-event F1 at 100 ms, accompanied by
precision and recall. Report coverage curves and event density to reveal score
gaming. Include movement magnitude, jerk, stationary fraction, and within-song
diversity using fixed preprocessing; these are diagnostics, not validated human
quality scores. Avoid pooling mostly cross-song pairs as conditional diversity.

Average draws/starts within song, then songs with equal weight. Show each training
seed, per-song paired changes, and descriptive uncertainty. Two seeds and six test
songs support preliminary findings, not robust algorithm rankings. If bootstrapping,
resample songs as clusters and distinguish song uncertainty from training-seed
uncertainty. Do not treat frames or draws as independent songs.

Produce one main table (baseline, bin RL, BAS RL), training/validation curves,
BAS-versus-F1 paired changes, and a small set of synchronized video comparisons.
Choose a prespecified representative subset for blinded synchronization and
naturalness review with ties; label deliberately chosen metric disagreements as
failure examples. Human review is useful supporting evidence, not a blocker for
the automated midpoint result. Report reviewer count and clip-selection method.

Interpretation rules:

- BAS and F1 rise with stable diagnostics: preliminary evidence of broader timing gain.
- BAS rises but F1 falls/event counts explode: objective-specific gain or exploitation.
- Training rises but held-out scores do not: weak transfer/possible overfitting.
- Neither rises: inspect reward spread, policy drift, critic fit, and budget before
  claiming the reward formulation cannot work.

Final report: motivation; MDP and policy-gradient formulation; original versus new
reward; implementation verification; actual train/evaluation counts and cost;
results and failure cases; limitations; one justified next experiment. Clearly
label pending work. A midpoint with real controlled fine-tuning and honest null
results meets this plan's scientific purpose.

## Deliverables and execution order

1. Verified evaluator and focused tests; cached-output rescore report.
2. Frozen split/config manifest and reward diagnostics.
3. Shared trainer, smoke evidence, measured compute estimate.
4. Four bounded runs, reloadable checkpoints, validation logs, final evaluation.
5. Per-case JSON/CSV, reproducible summary command, figures/clips, midpoint report.

Place reports under reports/midpoint_beats/ and configurations under configs/.
Keep large weights/media in the project's established artifact storage. Reuse
existing infrastructure; preserve historical outputs and user changes.
