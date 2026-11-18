# Music-RL literature and refinements to basic-beat experiments

Reviewed November 17, 2026. These are proposed transfers, not new experimental
results. Companion to BASIC_BEAT_IMPLEMENTATION_PLAN.md.

## Evidence boundary

In music papers, alignment can mean text adherence, correct sung lyrics, harmonic
accompaniment, symbolic constraint satisfaction, or human preference. None is
identical to dance accents matching audio beats. Published improvements motivate
algorithms and experimental controls; they do not predict our effect size or prove
that a newer optimizer beats the proposed Monte Carlo actor-critic baseline.

The existing MusicRL and Sequence Tutor readings already justify sequence-level
policy optimization with a pretrained reference. Additional useful sources follow.

## Additional readings

### RL-Duet (AAAI 2020)

[Paper](https://arxiv.org/abs/2002.03082) ·
[Authors' project](https://labsites.rochester.edu/air/projects/RLDuet.html)

Uses actor-critic with generalized advantage estimation and learned rewards for
online symbolic accompaniment. Rewards assess compatibility within generated music
and between the human and machine parts. The target is coherent accompaniment,
not a dance beat-hit score.

Transfer: local contextual feedback and multistep advantage estimation can address
credit assignment. For dance, compare sequence-return learning with a carefully
defined beat-local reward after the sequence baseline works. A learned compatibility
model for music and motion would require new training data and independent validation.

Priority: medium. A useful GAE/credit-assignment ablation, not an immediate reason
to replace interpretable alignment scores with a learned model.

### Towards Hallucination-Free Music (2025 preprint)

[Full text](https://arxiv.org/html/2508.05011v1)

Compares DPO, PPO, and GRPO for reducing lyric-to-song content errors. Table 1
reports validation reward 0.771 for the initial model, 0.843 for DPO, 0.820 for
PPO, and 0.834 for GRPO. These are task-specific scores, and the training protocols
and compute differ. They are not an equal-budget proof of algorithm superiority.

GRPO replaces the learned critic with comparisons among multiple sampled outputs.
This motivates a critic-free comparator when our fresh critic is a suspected source
of instability. DPO motivates a stored-pair baseline, but should not displace online
RL as the core contribution of this course project.

Priority: high for a small group-baseline experiment; optional for DPO. The paper
does not establish that GRPO outperforms every simpler policy-gradient baseline.

### MR-FlowDPO (ICASSP 2026)

[Paper](https://arxiv.org/html/2512.10264v1) ·
[Publication](https://doi.org/10.1109/ICASSP55912.2026.11465005)

Optimizes flow-matching text-to-music generation with preference pairs formed from
multiple rewards: text alignment, production quality, and semantic consistency.
Its pair construction favors candidates that improve across reward axes, rather
than choosing winners using only one score. It also reports problems with extreme
values of a semantic consistency proxy, illustrating reward exploitation.

Transfer: rank dance candidates only when alignment improves clearly and movement
quality is acceptable. This can support optional DPO or filtered BC. Our weaker
"alignment improvement subject to quality tolerances" rule would be an adaptation,
not the paper's exact multi-axis strong-domination rule. Its flow-specific loss
cannot be copied directly into a categorical motion GPT.

Priority: high for evaluation/pair-selection discipline, medium for a preference
learning baseline. Automatically preferred does not mean human preferred.

### NotaGen (IJCAI 2025)

[Proceedings](https://ijcai.org/proceedings/2025/1134)

Introduces CLaMP-DPO for symbolic music quality and controllability after pretraining
and supervised fine-tuning. It offers another example of preference optimization
without a newly collected human-label dataset. It is not evidence of precise
temporal synchronization or a directly usable motion similarity model.

Priority: background for a DPO extension. Music-specific embedding rewards need
validation before being treated as dance-rhythm rewards.

### MusicRLVR / Which Constraints Are Missing? Ask the Verifier (2026 preprint)

[Full text](https://arxiv.org/html/2609.23665v1)

Studies symbolic constraint following using programmatic checks, graded credit,
and GRPO. In its reported matched diagnostic comparison, graded rewards reduce
the fraction of all-tied rollout groups early in training relative to binary
all-satisfied rewards. The paper distinguishes compliance from musical aesthetics.
Some other diagnostic comparisons change prompt mix, so use the matched comparison
when discussing causality. This is recent preprint evidence, not an established
general result for audio or dance generation.

Transfer: measure within-condition reward variation before choosing a group-based
optimizer. A reward that gives every sampled dance the same value supplies no
group-relative reward advantage. Our bin score is already an average of local
binary decisions, not the paper's all-or-nothing sequence criterion; do not assume
its reported tied-group rate applies here. Gaussian BAS can also nearly saturate
at zero when events are too far apart.

Priority: high for diagnostics; no need to reproduce its model or benchmark.

## Recommended revisions to the experiment plan

### 1. Add a reward-informativeness screen before RL

Sample four independent dances for each fixed music/start condition. Across a
representative development set, score bin reward, BAS, and matched-event F1. Log:

- within-condition reward standard deviation, range, and tied/near-tied group rate;
- identical token sequences and variability of detected event counts;
- agreement between reward rankings and movement-quality diagnostics;
- examples where nominally better alignment comes from jitter or excess accents.

Choose the numerical near-tie tolerance in advance and show raw distributions.
Higher reward variance is not inherently better: it can reflect noise or a broken
detector. Interpret the screen together with controlled shifts and video review.

### 2. Keep the current actor-critic as the primary algorithm

Retain the common stochastic Monte Carlo actor-critic trainer and objective-only
comparison. This preserves the course's value-learning component and avoids making
the plan depend on a more complex optimizer. Keep separate legacy reproduction.

Then add one critic-free comparison on a single frozen basic objective. Use four
rollouts per identical music/start condition and a leave-one-out baseline:

    A_i = R_i - mean(R_j for j != i).

Treat the baseline and rewards as detached. With independently sampled trajectories
under the same conditioning, other trajectories' rewards do not depend on this
trajectory's sampled actions. Use one actor update per fresh group initially, and
no group standard-deviation division. This is a simple group-baseline REINFORCE
method inspired by group-relative optimization, not an implementation of GRPO's
full clipped surrogate and normalization.

Match actor-critic collection to the same grouped layout: e.g., four music/start
conditions times four draws = 16 trajectories for both algorithms. Do not compare
16 distinct conditions against four conditions and attribute all differences to
the baseline. Count every rollout in both sample and wall-time budgets.

If batch reuse is needed later, implement actual GRPO/PPO with collection-policy
log probabilities, clipping, masks, and KL accounting. Report tied groups explicitly;
they can still have a KL update but no group-relative task-reward signal. Removing
the critic saves value fitting but multiple generations can dominate total cost.

### 3. Add local credit assignment only after the baseline is interpretable

For a finalized beat-centered interval, a proximity reward may be attributed when
the necessary motion context has been generated, allowing return-to-go or GAE.
However, the decoder and Gaussian-smoothed beat detector use temporal context.
Assigning a reward that depends on future actions to an earlier timestep and then
dropping it from later actions' returns would produce incorrect credit assignment.

First define a causal/finalized event rule, its delay, and boundary conditions.
Check whether local contributions sum to the terminal objective. If not, call it
a new reward objective rather than a pure variance-reduction change. F1 has global
count normalization and should not be assumed to decompose like per-beat BAS.
An ordinary partial-score difference is not automatically policy-invariant
potential-based shaping; terminal conventions and discounting matter.

Use critic warm-up and Monte Carlo-versus-GAE as separate experiments. Do not copy
a GAE lambda from a singing model and assume it is appropriate for 20-second dance.

### 4. Treat quality-aware preference optimization as a bounded optional branch

Generate candidates from adaptation music only, with identical music/start/duration
within each comparison. Select pairs where alignment differs by a meaningful
development-fixed margin and quality diagnostics pass reviewed tolerances. Keep
ambiguous tradeoffs out of automatic winner labels; inspect a sample with humans.

With categorical policy log likelihoods, a DPO loss is straightforward conceptually,
but requires exact history/music conditioning for both candidate sequences. Compare
to BC on the winning examples to determine whether pairwise negatives add value.
Count pair-generation compute. Audit pair yield and genre coverage: conservative
selection can discard most data or disproportionately select certain songs.

DPO is a preference-learning comparator, not evidence that online RL is unnecessary
or that our task has acquired genuine human preference supervision. Do not introduce
a learned reward model solely to replace a cheap, available beat calculation.

## Priority and evidence of effectiveness

Immediate: reward validation, reward-informativeness logging, reference KL, and the
existing objective comparison. These address known risks without a large scope change.

Next algorithm comparison: learned value baseline versus leave-one-out group
baseline on one fixed reward, three seeds if feasible. Hypothesis: removing critic
error helps early learning; counter-hypothesis: the value baseline is more sample
efficient and group rollouts are expensive. Evaluate both at matched generated
actions and report actual wall time. Two methods must have comparable quality
before claiming the one with higher alignment is more effective.

Conditional: local/GAE rewards if temporal credit assignment is limiting; DPO if
reliable, non-degenerate candidate pairs are available. Learned contextual rewards
require an independent generalization test and remain a larger extension.

Do not launch the Cartesian product of rewards, algorithms, schedules, regularizers,
and detectors. Establish the reward comparison first, then hold the selected reward
fixed on development data for one algorithm ablation. Keep the untouched test set
locked and preserve unsuccessful seeds. Hierarchical beats remain deferred.
