# Bailando++ objectives and midpoint review

Reviewed November 17, 2026. This is a proposed direction, not a record of newly completed training. It complements the existing project plan and course synthesis.

**Recommendation:** keep diversity as an outcome and a conditional extension. Make the main question **how to fine-tune a pretrained dance policy while retaining useful behavior**, using a controlled comparison of the released procedure and auxiliary imitation. Do not make project success depend on first demonstrating diversity collapse or on outperforming the published system.

## 1. What the evidence currently supports

The five-epoch pilot established a working training and evaluation pipeline: 35 actor-loss steps, 25 critic-loss steps, unchanged frozen components, checkpoint reload verification, and 12 paired song/start evaluations. It did not establish reward improvement, convergence, a reproduction of the paper's quantitative performance, or consistent conditional diversity loss. Correction after checking the paper: Section 4 reports **10 actor-critic fine-tuning epochs** at learning rate 1e-5. The upstream YAML specifies 500 epochs, but that is not the paper's reported RL duration. Our five epochs on three adaptation songs remain a small procedural pilot; epoch counts across different datasets are not equivalent update budgets.

| Group | Mean reward, supervised → epoch 5 | Music-beat-bin hit rate | Within-song distance across starts |
|---|---:|---:|---:|
| Adaptation | 3.315 → 3.277 | 25.51% → 24.19% | 1.123 → 1.305 (+16.2%) |
| Held out from RL | 3.368 → 3.367 | 25.14% → 25.19% | 2.281 → 2.761 (+21.0%) |

Sources: [pilot](../reports/batch03/rl_pilot.md), [paired results](../reports/batch03/paired_results.json), [new independent recheck](../reports/midpoint/diversity_recheck.json).

The diversity recheck recomputes descriptors from cached raw motions, rather than merely copying the group summary. It reproduces all existing pooled and within-song values within relative tolerance 1e-6. No new model generation or GPU work was needed.

**The diversity distinction matters.** Each clip is represented by the mean and standard deviation of each joint's speed, with root displacement channels removed, on a common 1,768-frame interval. Pooled pairwise distance decreased 13.4% on adaptation and 8.5% on RL-held-out songs. But 12 of the 15 pairs in each pooled group compare different songs; only three compare two starts of the same song. Cross-song-only distance decreased 16.9% and 12.6%, respectively. Within-song distance increased on three songs, decreased on two, and tied on one.

This is evidence that the definitions give different answers. It does not establish a semantic loss of style across songs or a semantic gain in within-song diversity. The descriptor loses movement order and phrasing, larger differences can arise from undesirable motion, and there is only one pair of starts per song. The two groups also contain different genres.

Distinguish three outcomes in future work:

1. **Within-song diversity across valid starts**, under a fixed distribution of initial code pairs.
2. **Within-song, fixed-start stochastic diversity**, requiring an actual stochastic decoder.
3. **Variety inside one dance**, such as excessive repetition versus coherent recurring phrases.

The released sampler is greedy. Changing a random seed does not create a new stochastic sample for fixed inputs in the deterministic evaluation path. Entropy of token probabilities, code occupancy, and motion diversity are separate measurements.

The prior audiovisual review adds useful, limited evidence. On mJB5 start 1, the user strongly preferred epoch 5 for stylistic consistency; on start 2, the user tentatively preferred supervised motion and timing. On mJS3 start 2, the user slightly preferred epoch 5's contrast between fast and slow passages but could not distinguish beat alignment or diversity. These are one reviewer's impressions of 16-second excerpts, not a blinded preference study or full-track labels. The teammate review remains open.

## 2. Objectives at different levels

These are alternatives and successive gates, not a requirement to implement every row.

| Level | Research question and minimum experiment | What it demonstrates | Completion criterion |
|---|---|---|---|
| Minimum, midpoint-ready | What does the released actor-critic procedure actually optimize, and what does its reward miss? Consolidate the trace, update checks, reward controls, and paired pilot. | BC initialization, TD targets, credit assignment, optimization state, reward specification, evaluation discipline. | Every claim traces to code or a result; separate procedural replication from improvement. Most evidence already exists. |
| Small DRL extension | Does inactive-head Adam momentum or an unfitted critic materially affect early adaptation? Test ONE first. | Gradient versus optimizer state; bootstrapping and baseline quality; actor/critic update schedules. | A controlled ablation with parameter/gradient checks and paired outcomes, including a null result. |
| Recommended core extension | Does auxiliary reference imitation reduce behavior drift while allowing reward adaptation? Compare pretrained, imitation-only, RL-only, and RL + uniform imitation. | Negative log-likelihood BC, empirical training distribution, regularization, reward/retention tradeoffs. | Verified losses and a matched-budget tradeoff comparison; improvement is an empirical question. |
| Diversity extension | Does movement-balanced imitation help beyond uniform imitation? | Coverage, sampling distributions, mode preservation, what a causal ablation can establish. | Fixed buffer and clusters; same replay-token budget and weights; coverage gains checked against timing and quality. |
| Higher-risk DRL extension | Does properly sampled categorical policy-gradient training behave differently from the greedy released procedure? Optionally add an entropy bonus. | On-policy distributions, log probabilities, likelihood ratios, exploration, maximum entropy. | Sampler/loss probability consistency and matched generation settings verified before comparisons. |

The small DRL extension is the safest next experiment. The reference-imitation comparison is the best main project extension because it directly addresses the TA's request and connects BC with RL. Movement balancing becomes worthwhile once the buffer has meaningful diversity and there is a reproducible retention or coverage problem to address. It can also be studied as coverage redistribution without claiming to repair collapse.

## 3. Concrete recommended experiment

Use one immutable supervised checkpoint, fixed encoders and decoder, the same adaptation music, and a frozen reference buffer from that checkpoint. Keep the original reward while first testing imitation; changing reward and replay together would make the cause of any change ambiguous.

For actor parameters theta, add a separately normalized imitation loss:

L_actor_total = L_released_actor + lambda * L_reference

L_reference = -mean over stored valid tokens [log pi_theta(a_upper | s) + log pi_theta(a_lower | s)].

Here s includes music and the stored code prefix. Train only the same actor head in all variants. Calling the general policy forward method can propagate gradients through the encoder/base, whereas the existing `actor` method explicitly computes them under `no_grad`; retain that freeze behavior and verify it.

Use reference data only in this supervised auxiliary term. Keep the RL term on the freshly collected rollout batch. This avoids adding an old-policy actor-critic replay problem. It does **not** make the existing greedy procedure an unbiased on-policy policy-gradient estimator.

The smallest useful comparison is:

| Condition | Purpose |
|---|---|
| Frozen supervised policy | Starting behavior and generation reference |
| Reference imitation only | Detect what imitation itself does, including sharpening or forgetting |
| Released RL procedure | Existing adaptation baseline |
| Same RL procedure + uniform imitation | Isolate the reference anchor |

Start with two nonzero lambda values selected using development data and logged RL/reference gradient scales; include lambda=0 through the RL baseline. Match RL rollout counts and actor/critic update counts between the RL arms. Report the added reference-token count and wall time rather than claiming identical compute. Use the same total number of actor updates for the imitation-only control and disclose its different objective. For uniform versus balanced replay, match both reference-token count and optimizer-step count.

A pilot can use eight valid paired upper/lower initializations drawn from adaptation demonstrations or an independently defined start pool, with the same starts for every method. Do not form arbitrary upper/lower combinations. Reuse the original two starts as a continuity check, not the whole diversity evaluation. Training seeds and generation/start variation answer different questions; aim for three training seeds for the final small comparison if feasible.

**Important BC distinction:** a buffer of the pretrained model's greedy outputs is self-imitation of selected sequences, not expert demonstration learning and not preservation of its full probability distribution. Its hard targets can sharpen probabilities. If full distribution retention becomes the objective, use cached reference probabilities and forward KL, KL(pi_0 || pi_theta), on specified reference states. For categorical heads this can be computed exactly by summing over each code vocabulary. Samples genuinely drawn from pi_0 give a Monte Carlo cross-entropy approximation; greedy samples do not. Do not introduce both hard BC and soft distillation as required core methods.

Evaluate reference NLL on a separate diagnostic buffer, within-song start diversity, code coverage/repetition, beat-bin hits, two-direction timing diagnostics, movement amount, jerk, and matched video judgments. Lower reference NLL alone is expected from the training objective and is not proof of better free-running motion. Distances and occupancy alone are not perceptual quality measures. Avoid small-sample FID as the headline result.

If lambda keeps behavior near the supervised baseline but prevents adaptation, report that tradeoff. If neither RL arm improves timing, do not call preservation of the baseline a successful beat-alignment improvement. Investigate training signal and update mechanics before scaling up the replay study.

## 4. Small DRL ablations grounded in observed behavior

**A. Inactive-head optimizer state (lowest implementation risk).** Replay the same saved minibatches from identical model and Adam states. Compare legacy zeroed inactive gradients with explicitly absent inactive gradients, keeping the active branch's loss identical. Measure head parameter deltas at the branch switch. For a longer run, apply the chosen optimizer convention consistently to all replay arms. A smaller update is not automatically a better policy. This experiment tests an observed mechanism; it should not be described as fixing proven performance harm.

**B. Critic warm-up (strongest actor-critic teaching connection).** The released setup uses a fresh critic and `pretrain_critic_epoch=0`, so the first five updates use advantages from an unfitted critic. Hold the actor fixed for a bounded critic-fitting phase, then compare early actor updates with the original schedule. Measure TD residuals, value scale, advantage signs, policy drift, and paired outcomes. Check return agreement on disjoint trajectories with a consistent reward/context/terminal definition. A training TD loss alone is not critic calibration.

The local `benchmark_cycle_gpu.train_cycle` currently chooses branches only from minibatch index; it does not honor the warm-up flag. Editing YAML alone would not implement this experiment. Warm-up also adds critic compute: report it, or include a matched-total-update control. Do not compare a window-decoded TD target to a full-track return as if they were the same reward process.

**C. Greedy versus stochastic training (more implementation work).** The source `sample` uses top-1 code selection, while the actor uses categorical negative log-likelihood weighted by TD error. The score-function theorem assumes actions sampled from the distribution appearing in the log probability. Greedy behavior has point-mass support and does not satisfy that assumption. Minibatch reuse after actor changes adds a separate behavior-policy mismatch. This does not invalidate faithful replication; it limits the theoretical label we should use.

A clean stochastic extension must record both upper/lower log probabilities under the actual collection distribution, its temperature, history, music extent, and masks. If reusing a collected batch across actor updates, use a defined surrogate/correction such as PPO with old log probabilities, or refresh data under a simple on-policy schedule. Top-k sampling changes support and is not equivalent to untruncated softmax sampling. First verify behavior and update contexts agree at included actor positions. The trace already shows some first-window discrepancies; many later-window discrepancies are masked out, so do not treat every mismatched position as a loss bug.

## 5. Other feasible directions, and what to defer

**Filtered or reward-weighted BC is a useful fallback comparison.** Generate several candidate trajectories on adaptation songs, rank within each song with a prespecified timing criterion plus movement/continuity checks, and train on selected trajectories while retaining an unfiltered reference portion. Compare with equally sized uniformly selected data. This connects directly to Lecture 13, but simple filtering can forget good modes and cannot exceed its candidate coverage by selection alone. A reward-weighted regression using a trustworthy advantage estimate is a further step, not automatically AWAC merely because its loss weights cross-entropy.

**Best-of-N is a cheap inference baseline.** For offline choreography generation, score complete candidate dances and choose one. It can test whether the frozen policy already generates better-scoring candidates than its single default output. Count N generations in the inference budget and evaluate with additional metrics/review. Choosing the maximum of the scoring reward mechanically improves that score relative to the same candidate set; it is not evidence of policy learning or perceptual improvement. The lecture's critic-based action selection differs from directly scoring a completed dance, and Bailando's existing critic is V(s), not Q(s,a).

**Reward refinement is justified but should be a separate branch.** Stationary motion tied generated motion in the four-second audit, and a repetitive splice scored higher despite a large jerk increase. Expand those controls to more songs/phases before introducing one movement or continuity term. Guard against the trivial stationary optimum of a smoothness penalty and against discouraging intentional repeated phrases. Phrase-level rhythm and style are interesting future goals but currently lack reliable annotations and independent evaluation.

For music-beat indicator B_t and motion-beat indicator D_t, the valid-input default reward is

r_t = 5 - 10 B_t + 10 B_t D_t.

On a fixed song, duration, and scoring grid, the first two terms are action-independent. Removing the +5 reward on no-music-beat bins changes the offset but does not change trajectory rankings or add a preference for natural movement. It may affect finite-sample critic fitting and the biased released update; do not market it as a new dance-quality objective. The positive baseline explains why mean reward is hard to interpret absolutely, not why the paired reward deltas are meaningless.

Defer a new VQ-VAE, a diffusion-model replacement, GAIL, a new preference reward trained from a handful of subjective reviews, full offline Q-learning, and a GFlowNet trajectory-balance comparison. These add different uncertainty sources before the current policy/reward question is resolved. Standard DAgger also requires corrective expert actions on learner states; the fixed dance data do not supply those. Querying the pretrained model would be teacher distillation, not access to a reliable human expert.

## 6. Course reading map

Page numbers below are one-based PDF pages, not printed slide numbers. Relevant pages were extracted and key BC, policy-gradient, warm-up, and filtering pages visually checked.

| Course source | Relevant material | Application and boundary |
|---|---|---|
| [Lecture 3](../../10703Slides/Lecture%2003%20-%20Policy%20Gradients,%20REINFORCE,%20and%20Actor-Critic%20Methods.pdf), pp. 48–63 | Baseline subtraction; advantages; Monte Carlo and TD fitting; actor-critic | Explain the detached TD error, warm-up, and why an action-independent reward offset is not a dance-quality signal. |
| [Lecture 8](../../10703Slides/Lecture%2008%20-%20PPO%20and%20Off-Policy%20Actor-Critic.pdf), pp. 4–8 and surrogate optimization section | AWR; KL direction; current-policy expectation; policy surrogates | Separate greedy rollout replication from a stochastic policy-gradient algorithm; use old-policy log probabilities if implementing PPO. |
| [Lecture 9](../../10703Slides/Lecture%2009%20-%20Off-Policy%20Actor-Critic%20and%20Maximum-Entropy%20RL.pdf), pp. 30–33 | Entropy bonus and soft rewards | Probability entropy can encourage exploration but does not guarantee decoded dance diversity or quality. Merely enabling the upstream coefficient is not a complete SAC implementation. |
| [Lecture 10](../../10703Slides/Lecture%2010%20-%20Imitation%20Learning.pdf), pp. 17–42, 76–83, 85–99 | Expert-state BC; distribution shift; DAgger; log-loss BC | Formulate reference imitation and distinguish expert data, generated references, and learner-state corrections. Neural greedy decoding does not automatically inherit the lecture's stochastic log-loss guarantees. |
| [Lecture 12](../../10703Slides/lec-12-offline-online.pdf), pp. 46–58, 63–67 | Policy regularization, BC warm-start, critic warm-up, offline-to-online adaptation | Motivate retention and critic calibration. Fresh generated dances plus computable reward make the project more than learning from a fixed offline dataset; there is still no physical dance simulator or corrective expert. |
| [Lecture 13](../../10703Slides/lec-13-gcps.pdf), pp. 25–43 | Filtered BC, coverage, forgetting, best-of-N, self-BC | Supply modest alternative baselines. Transfer these ideas to discrete autoregressive dance carefully; robot-control results do not guarantee the same result here. |

External cross-checks: the [authors' repository](https://github.com/lisiyao21/Bailando) identifies the ++ implementation and its relationship to the original release. [Bailando++ publisher abstract](https://xplorestaging.ieee.org/document/10264209/references) describes beat-alignment RL, rotation-domain pose modeling, and longer music context; our position-decoder pilot does not evaluate all of those contributions. [AWAC](https://arxiv.org/abs/2006.09359) combines dynamic programming with likelihood policy updates, motivating a future weighted-BC comparison but not a drop-in claim about our method.

## 7. Midpoint report: material to use now

Suggested title: **Bailando++ Fine-Tuning: Reward Alignment and Retention of Pretrained Behavior**.

Suggested updated project goal:

> We study how reward-based fine-tuning changes a pretrained music-conditioned dance policy. We first reproduce and audit the released actor-critic procedure, then test whether auxiliary imitation preserves useful pretrained behavior during adaptation. We evaluate musical timing, motion quality, and within-song movement diversity; movement-balanced imitation is a conditional extension.

Suggested abstract, using only completed work:

> We reproduced a short Bailando++ actor-critic fine-tuning procedure from the released supervised checkpoint and built a matched evaluation pipeline. We traced discrete actions through motion decoding and beat rewards, independently checked update calculations, and verified optimizer and checkpoint behavior. A five-epoch pilot on three adaptation songs changed generated sequences but did not improve mean reward on either adaptation songs or three songs held out from RL. A reward audit found that stationary and generated motion could tie, while a discontinuous repeated segment could score higher. Rechecking diversity showed that pooled cross-song variation and within-song variation across starting poses moved in opposite directions. These findings motivate controlled experiments on update stability and auxiliary imitation, without assuming that diversity collapse has already occurred.

The supplied template gives 13/20 points to progress, 2/20 to resources, and 5/20 to the updated plan. Allocate most space to the trace, verified implementation, and interpretable results. The template explicitly asks whether the formulation changed and why; this refinement fits that question.

| Template section | Evidence or text already available | Remaining work before submission |
|---|---|---|
| Overview and motivation | Updated goal above; original proposal and TA feedback | Explain the shift from assumed diversity loss to measuring adaptation/retention tradeoffs. |
| Progress and technical work, 13 points | Source/checkpoint pinning, legacy environment, reward controls, rollout trace, independent loss/gradient checks, shared Adam finding, pilot and paired evaluation | Include a compact pipeline diagram, one worked TD update, paired-results table, and diversity distinction. Do not imply new replay or warm-up results exist. |
| Takeaways box | Working procedure is established; average reward improvement is not; reward and diversity definitions require care | Tie each takeaway to a table or diagnostic. |
| Data/resources, 2 points | Six AIST++ full-track music inputs, three for adaptation and three excluded from RL, two evaluation starts, fixed position decoder, T4/legacy runtime | State that supervised-training exclusion is unverified and genre/group coincide. Include preprocessing/hash checks and compute scope. |
| Updated plan, 5 points | Tiered objectives and dates below | Choose one small ablation, core imitation comparison, and explicit stretch gate. |
| Contributions | Existing plan assigns Alex implementation/policy/reward work and teammate representation exploration/joint review | Confirm actual completed teammate contributions and names; do not infer them from planned assignments. |
| Reflection/feedback | Reward mismatch, optimizer-state effects, deterministic sampling, sparse evaluation | Ask staff whether the focused controlled study is sufficient scope and which retention/diversity definition best fits the hypothesis. |

For a worked update, the traced music-beat miss has r=-5, V(s)=1.043499, V(s')=1.343184, gamma=1, giving target -3.656816 and TD error -4.700315. These values come from the diagnostic's fresh critic, not a trained critic. Explain how negative advantage changes the selected codes' likelihood and where the actor mask applies.

Compute disclosure: the completed pilot records 205.89 T4 worker seconds for training and 114.31 for evaluation, or about 0.089 worker-hours combined. These include worker setup/verification and are not a whole-project GPU-hour total. The report records $0.07236 combined itemized training/evaluation charges, excluding storage. Earlier setup and diagnostic runs are additional. Use the historical measured charges as evidence, not a promise of future prices or credit availability.

## 8. Proposed milestones and stopping rules

These are planning targets, not verified course deadlines. The earlier proposal used December 7 for the final write-up; confirm the actual submission dates separately.

| Target | Deliverable | Stop/go decision |
|---|---|---|
| November 17–19 | Midpoint narrative, evidence table, existing clips, teammate review | Submit a truthful progress account even if no extension has run. |
| November 20–23 | One optimizer-isolation or critic-warm-up ablation; larger development start set | Establish the common training convention before comparing imitation methods. |
| November 24–28 | Verified reference loss; pretrained / BC-only / RL / RL+BC comparison | Use a short pilot and fixed later endpoint, e.g. 20 epochs, chosen before endpoint inspection; keep all seeds and failures. |
| November 29–December 3 | Replicate promising comparison; conditional balanced-replay experiment | Add balancing only with a usable buffer and a clearly defined coverage question. Otherwise deepen the core result. |
| December 4–7 | Locked evaluation, visual review, final analysis and writing | Report tradeoffs or null results instead of continuing an open-ended hyperparameter search. |

Reserve validation songs for choosing weights and schedules. The existing RL-held-out songs have already informed this review, so future experiments on them should be called exploratory evaluation. If possible, reserve new untouched music IDs for a final locked test; otherwise disclose the absence of a fresh test set. Never place evaluation-song trajectories in the reference training buffer. Song/start pairs and overlapping windows are not independent training seeds.

The existing timing suggests that modest experiments are computationally feasible, but correctness and interpretation are the bottlenecks. Benchmark added reference passes, snapshot storage, and additional starts before allocating longer runs. Preserve the legacy baseline separately rather than silently changing it.

Reproduce the new CPU-only diversity check with the project environment:

```sh
/Users/zhengfeili/opt/anaconda3/bin/conda run -n 10703 python scripts/recheck_pilot_diversity.py
```

Run from the project root. The script writes `reports/midpoint/diversity_recheck.json`, records input hashes, and checks agreement with the saved batch03 values.
