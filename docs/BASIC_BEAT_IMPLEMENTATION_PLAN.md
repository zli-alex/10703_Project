# Basic-beat objectives: implementation and midpoint plan

Status: specification, November 17, 2026. No new RL results are claimed here.
Implements the first phase of [the research plan](BEAT_ALIGNMENT_RL_PLAN.md).

For the current midpoint, follow the narrower [two-objective execution plan](MIDPOINT_BEAT_OBJECTIVE_PLAN.md).
It supersedes the three-arm midpoint target below; F1 remains evaluation-only initially.

The [additional music-RL review](MUSIC_RL_LITERATURE_AND_REFINEMENTS.md) adds a
reward-informativeness screen and a conditional learned-baseline versus
leave-one-out group-baseline experiment. Local/GAE rewards and quality-aware DPO
remain optional; they do not expand the required objective matrix.

## 1. Separate reward, evaluator, and training algorithm

The paper does not directly train on its reported Beat Align Score (BAS). Its
beat reward is a binary coincidence test inside eight-frame bins. BAS is a
continuous nearest-event evaluation score. Our existing pilot reproduced the
released beat reward and added hit rate, timing distance, and 100 ms coverage;
it has not yet reproduced published BAS evaluation.

| Item | Paper / released implementation | Current pilot | Planned comparison |
|---|---|---|---|
| Training beat reward | +5 except -5 when a music-containing bin has no motion beat | Same rule | Bin reward versus BAS-based sequence reward versus matched-event F1 reward |
| Body consistency | Paper gives nonzero weight; selected released YAML uses weight zero | Weight zero | Keep zero across first objective comparison; report this scope |
| Reported alignment | BAS, Gaussian kernel sigma 3 frames | Not yet computed in the pilot | Compute beside coverage and event precision/recall |
| Motion representation | Paper has position and rotation variants | Fixed position decoder | Position throughout this phase |
| RL algorithm | Released greedy rollouts and TD-weighted actor update | Reproduced for five epochs | Keep legacy reference; compare new objectives under ONE common stochastic trainer |

This is a controlled extension to the released system, not a claim of exact paper
reproduction. In the paper, position BAS changes 0.2109 to 0.2597 and rotation BAS
changes 0.2264 to 0.2720 after actor-critic training (Tables 5 and 6). Their evaluation
uses 40 generated dances and 20-second excerpts. Our six-song pilot is a different
sample and protocol. Paper fine-tuning duration is 10 epochs, not the released
configuration's 500-epoch value.

## 2. Definitions to implement

Let M be supplied music-event times and D detected motion-event times, with all
timestamps in seconds. Retain integer frame indices as provenance. At 60 FPS,
eight frames are approximately 133 ms and sigma=3 frames means 50 ms.

**Legacy bin hit rate H:** number of music-containing bins also containing a
motion event, divided by number of music-containing bins. Multiple events in a
bin count once. If q is the fraction of all bins containing music events,

    mean legacy reward = 5 - 10 q (1 - H).

For a fixed music clip and bin grid, reward and H therefore give the same ranking.
Replacing reward with H is not a new within-clip alignment preference. Across
songs it changes weighting and critic scales, which must be distinguished from
an improved timing objective.

**Published BAS:** average over m in M of exp(-dist(m,D)^2/(2 sigma^2)), sigma=50 ms.
Higher means that music beats have nearby detected motion beats; it is not an
accuracy percentage. Do not reverse the direction of the nearest-neighbor query.

**Tolerance coverage C_tau:** fraction of music events with dist(m,D) <= tau.
Keep tau=100 ms as the existing headline and report 50/100/150 ms together.
This is recall-like, many-to-one nearest-neighbor coverage, not classification
accuracy and not necessarily the recall from one-to-one matching.

**Matched-event precision/recall/F1:** create edges between M and D within tau.
Find a maximum-cardinality one-to-one matching; among equal-cardinality matchings,
minimize summed absolute timing error for the timing summary. With k matched pairs,
P=k/|D|, R=k/|M|, F1=2k/(|M|+|D|). Do not use unconstrained nearest neighbors and
call the result one-to-one. Return the matching for visual inspection.

**Timing distances:** report music-to-motion and motion-to-music separately, with
both event counts. Include median and mean; optionally report percentiles. A
missing reference event set gives undefined distance, not zero.

These scores can disagree legitimately. For example, with a music event at frame 8:

| Motion event | Bin hit | BAS contribution | Within 100 ms |
|---|---:|---:|---:|
| Frame 7 (16.7 ms early, adjacent bin) | 0 | about 0.946 | Yes |
| Frame 15 (116.7 ms late, same bin) | 1 | about 0.066 | No |

These are analytic event fixtures, not observed dance results. For a single event
50 ms away, BAS is about 0.607; at 100 ms it is about 0.135, although both qualify
under inclusive 100 ms coverage. Thus BAS measures graded closeness while coverage
answers an easily interpreted threshold question. F1 adds a cost for excess
detected events but may penalize intentional offbeat movement.

## 3. Coordinates and detector contract

The existing reward detector operates on raw decoder channels, including root
displacement channels mixed with relative joint positions. The published evaluator
operates on exported pred_position arrays. Reconstruct the exporter convention
and verify it against cached output before calling a metric paper-compatible.
Do not feed raw decoder channels to the BAS evaluator as if they were all positions.

Use two explicitly labeled paths:

1. **Legacy diagnostic:** existing raw representation, detector, crop, and complete
   bins. Reproduce the saved rewards and pilot timing measures.
2. **Position evaluation:** reconstructed global joint positions with the released
   BAS speed-minimum detector, smoothing sigma=5 frames, and original difference
   indexing. Compute BAS, coverage, and F1 on the SAME detected events and interval.

For objective-only training comparisons, hold detector, coordinates, and interval
fixed across bin/BAS/F1 rewards. Call a bin reward on the position path a controlled
bin-rule variant, not the exact legacy reward path. Keep the legacy end-to-end run
as a separately labeled reference. A detector change is a separate ablation.

Evaluate full tracks for continuity with the pilot and a prespecified first-20-second
panel for paper-oriented comparison. Compute all compared scores on identical crops.
Record whether detection is performed after cropping or on a full track then sliced;
use the former for a released-evaluator-on-cropped-input comparison. The released
BAS script itself does not enforce a 20-second crop, so duration must be explicit
in our wrapper. Shift controls use shared valid interior support, never wraparound.

Empty cases: with music events but no motion events, set BAS/coverage/F1 to zero,
precision to null, recall to zero, and flag the condition. The original BAS helper
can fail on empty sets; our extension must be labeled. With no music events,
alignment scores are null and the clip is excluded from alignment aggregation
with counts disclosed. Invalid coordinates are errors, not zero scores.

## 4. Proposed modules and acceptance checks

The filenames below are proposed; they are not implemented by this document.

| Module | Responsibility | Acceptance evidence |
|---|---|---|
| scripts/beat_metrics.py | Coordinate adapters, detector, BAS, coverage, matching, bin metrics; CPU only | Exact/near-exact agreement with released nonempty fixtures; explicit empty handling |
| scripts/rescore_basic_beats.py | Read cached motions and manifests; produce per-case and paired CSV/JSON | Reproduce old pilot columns on legacy path; verify input hashes and all 24 cases |
| scripts/audit_basic_beats.py | Shift controls and event-level failure fixtures | Boundary example, missing/extra events, stationary, duplicate events, no-music cases |
| configs/basic_beat_experiment.json | Split IDs, detector/crop, objective, seed, budget, regularizer, decoding settings | No implicit defaults or train/evaluation music overlap |
| scripts/train_basic_beat_rl.py | Shared stochastic trainer with interchangeable sequence rewards | Likelihood/context agreement, finite updates, intended heads change, checkpoint reload |
| scripts/summarize_basic_beats.py | Learning curves, paired tables, representative examples | Song/seed-aware aggregation, stable rounding, no test-based checkpoint selection |

Tests are justified here because these definitions determine the scientific result.
Use event-level fixtures with known answers, upstream differential checks, and one
end-to-end rollout/update. No large test suite mirroring implementation is needed.

## 5. First RL experiment

Primary hypothesis: temporal proximity scoring provides a more useful learning
signal than bin membership, and one-to-one scoring reduces excessive accents.
Both hypotheses may fail; near-zero BAS values can also make return differences
small. Log return variance and the fraction of tied/near-identical reward batches.

Start with the common Monte Carlo actor-critic trainer from the research plan:
fresh categorical rollouts, sequence return, learned baseline, separate optimizers,
one actor update per collection batch, frozen encoder/base/decoder. All arms must
share initial checkpoint, music, start distribution, duration, sampling temperature,
update counts, and detector. Preserve exact conditioning in likelihood computation.
Record both upper/lower action probabilities.

Use normalized sequence rewards:

- Bin arm: (mean bin reward + 5)/10.
- BAS arm: sequence BAS in [0,1].
- F1 arm: matched-event F1 at 100 ms in [0,1].

These ranges do not imply equal variance or gradient magnitude. Record those
quantities and fix any further calibration using development data only. Use the
same KL setting across the first objective comparison (zero is the simplest
unregularized screen); add no-KL versus KL as the next factorial comparison.
Do not simultaneously tune a separate optimizer/schedule for each objective.

For a midpoint-sized training batch, aim for three objective arms with two training
seeds, 25 fresh collection rounds each, and 16 generated continuations per round
if the memory/cost benchmark supports it. This is 400 continuations per run and
six short runs, not the final benchmark. If F1 or the trainer is not ready, do
bin-versus-BAS with two seeds and state the reduced scope. A single-seed result
can be shown as preliminary, but must not support robustness claims.

Include frozen supervised evaluation and the already completed legacy pilot as
references. The historical pilot is not a matched-budget competitor to the new
trainer. For an end-to-end algorithm comparison, rerun legacy on the same expanded
adaptation split and disclose both generated-data and optimizer-update budgets.

After the screen, expand to three seeds and the frozen confirmation budget from
the main plan. Test critic warm-up under ONE fixed objective as a separate ablation.
Keep hierarchical beats and new representations outside this implementation phase.

## 6. Side-by-side report tables

Produce two tables to avoid conflating representation changes with metric changes.

**A. Cached pilot rescore (completed generation, newly evaluated):**

| Policy / split | Legacy reward | Legacy bin hit % | Position BAS | Position coverage 100 ms | Position P/R/F1 100 ms | Motion event count | Movement / jerk |
|---|---:|---:|---:|---:|---|---:|---|
| Supervised | pending rescore | | | | | | |
| Pilot epoch 5 | pending rescore | | | | | | |

Provide the same-event bin score in the downloadable case table too. Label legacy
versus position paths clearly; do not interpret their differences as pure metric
effects. Keep separate adaptation and RL-held-out rows.

**B. New RL comparison, shared position evaluation:**

| Training objective | BAS | Coverage 100 ms | P/R/F1 100 ms | Bin hit % | Quality / diversity | Reviewer synchronization preference |
|---|---:|---:|---|---:|---|---|
| Frozen supervised | pending | | | | | |
| Bin-rule RL | pending | | | | | |
| BAS RL | pending | | | | | |
| F1 RL | pending | | | | | |

Report mean and per-seed outcomes, with per-song deltas in the supporting file.
Average generations within song first. Show paired delta scatterplots, timing
coverage versus tolerance, and held-out/validation learning curves. Absolute BAS
and hit-rate values have different meanings and should not share a mislabeled axis.

Automatic metrics share detectors and therefore are not independent human validation.
Ask blinded reviewers separately about synchronization, naturalness, and overall
preference, allowing ties. Compute metrics on the same viewed interval. Representative
pairs estimate agreement; deliberately selected disagreements illustrate failure
cases and must be labeled as such.

## 7. Midpoint deliverable and execution order

This can be a strong midpoint report because the supplied template emphasizes
technical progress (13/20), resources (2/20), and updated plans (5/20). It is not
a guarantee of a grade; the report should clearly distinguish completed evidence
from planned training and benchmark expansion.

Recommended order:

1. Implement the shared evaluator and its analytic/differential checks.
2. Rescore the existing 24 cached dances and reward controls without new GPU inference.
3. Freeze the objective definitions and write the reward-versus-evaluation comparison.
   Before training, also measure within-condition reward spread and tied-group
   frequency on four independent samples per music/start condition; inspect whether
   reward differences reflect useful timing differences or detector artifacts.
4. Audit additional music IDs and benchmark a stochastic rollout/update batch.
5. Run the bounded objective screen, with validation learning curves and saved states.
6. Present actual results, disagreements, limitations, and the larger confirmation plan.

Midpoint content: a concise RL formulation; paper/released/pilot distinction; one
worked reward and actor update; paired metric table; reward sensitivity figure;
preliminary learning curves if completed; data/compute disclosures; and explicit
next hypotheses. The completed legacy training already supplies an RL component,
but a small new objective comparison would materially strengthen the revised topic.

Do not delay the midpoint report waiting for the full 18-run matrix. If time is
short, present the completed replication and validated rescore as progress and
identify new training as pending. Do not call rescoring alone an RL improvement.

## Sources

- [Bailando++ paper, Eq. 21/24 and Tables 5/6](https://linchunze.github.io/assets/papers/TPAMI_Bailendo%2B%2B.pdf)
- [Released beat reward](../third_party/Bailando/tpami_bailandopp/models/up_down_half_reward.py)
- [Released BAS evaluator](../third_party/Bailando/tpami_bailandopp/utils/beat_align_score.py)
- [Current pilot evaluator](../scripts/evaluate_pilot_gpu.py)
- [Pilot findings](../reports/batch03/rl_pilot.md)
- [MusicRL method](https://arxiv.org/html/2402.04229v1#S3.SS2)
