# Batch 01: baseline exploration and trace closure

Updated 2026-11-06T22:12:49.675812+00:00. Bailando++ released supervised policy,
position decoder, pinned source, two development songs. **Proceed to independent
update verification; do not yet claim readiness to modify the algorithm.**

The [training-cycle benchmark](../batch02/training_cycle.md) records the later
isolated update verification and shared-optimizer run.

| Gate | Status | Evidence / limit |
|---|---|---|
| A: source, weights, data, runtime | Pass | Verified hashes, strict loading, native60Hz features; credit expiration/balance unavailable |
| B: valid smoke rollout | Pass for execution | 20s clip, finite arrays, exact saved-array revalidation; human audiovisual judgment pending |
| C: fixed-initialization experiment | Pass for diagnostic execution | Exact repeat; swap/shift change motion; timing evidence is preliminary |
| D: reward audit | Pass | Seven controls,210 bins independently checked; stationary tie and repetition/smoothness mismatch documented |
| E: rollout/window trace | Pass | Two exported windows,56 original-context greedy matches, reward/TD/NLL checks, diagrams |
| One full update verified independently | Pending | No backward or optimizer step yet |
| Before/after RL comparison | Pending | No RL fine-tuning yet |
| Both teammates can explain update | Pending teammate review | Use walkthrough questions below |

## What we learned

- Released checkpoints execute with the intended legacy runtime. Their identity
  is fixed by hashes; the policy filename epoch400 differs from stored epoch440.
- Music changes motion with fixed initial codes, and repeat inference is exact.
- High reward need not mean activity or smoothness in the tested sample.
- Standalone training-window decoding can change rewards relative to a cached
  full-decode slice; one included action changed from +5 to −5.
- Sampling context and teacher-forced training context differ. Original-context
  replay matches; later-window actor masks exclude the early mismatching slots.
- One20-second rollout supplies only two minibatches at batch64, insufficient
  to reach the critic branch in the released five-actor/five-critic schedule.

## Runtime and cost scorecard

| Work | Measured worker time | Peak allocated GPU memory | Hardware |
|---|---:|---:|---|
| B: smoke | 17.93 s | 710.2 MiB | T4 |
| D: reconstruction and reward audit | 9.19 s | 260.1 MiB | T4 |
| E: two windows and56 context replays | 22.44 s | 792.0 MiB | T4 |

Worker times exclude complete cold start, image build, uploads, and downloads.
These are not training epoch estimates. Step C includes three sequential runs;
its individual timing files remain in `runs/batch01/musicC-20261106T211618Z/`.

| Stage / all listed apps | Metered app resources |
|---|---:|
| music | $0.021379 |
| reward | $0.003441 |
| smoke | $0.021128 |
| trace | $0.014556 |

Latest workspace summary: **$0.070391 metered,
$0.00 billed**. Applied credits: $0.07.
Using the entire workspace meter conservatively leaves approximately
**$9.93 under the $10 first-batch ceiling**.

Itemized project rows total $0.060505; this
does not exactly equal the separately returned workspace summary. Both snapshots
are preserved without forcing a reconciliation or assuming an unexplained item
is free. The table includes listed failed/setup apps, CPU, memory, GPU, and egress.
Billing can settle later. Remaining credit balance and expiration are unknown;
the approximate $250 grant is not a verified current balance.

[Sanitized cost ledger](batch01_costs.json) records these figures. Raw billing and
account snapshots are local-only; do not include them in a teammate Git commit.

## Teammate review and next batch

Both teammates should be able to explain:

1. Why30 codes produce28 rewarded actions and29 critic values.
2. Which history/music/code/frame range belongs to window20's final action.
3. Why that action has reward−5, bootstrap value1.343184, and target−3.656816
   in this diagnostic trace, and why the critic value is not a trained estimate.
4. Why actor masking differs from critic reduction, and why two minibatches
   per epoch would not test the critic update.
5. One supported baseline limitation, with its saved evidence and scope.

Next: independently calculate and execute one actor update and one critic update
on disposable copies using these exports. Check gradients, frozen modules,
optimizer membership, greedy selection, and parameter changes. Benchmark a full
schedule cycle before selecting a bounded RL pilot. Retain the untouched supervised
checkpoint and use identical evaluation conditions. The ready-to-modify criterion
remains unmet until this update verification and both teammate explanations pass.

## Read next

- [Rollout walkthrough](rollout_trace.md)
- [Reward audit](reward_audit.md)
- [Music sensitivity](music_sensitivity.md)
- [Smoke test](smoke_test.md)
- [First-batch checklist](../../docs/FIRST_BATCH.md)
