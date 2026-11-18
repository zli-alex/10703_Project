# Midpoint handoff

## Message to teammate

We finished the controlled Bailando++ midpoint experiment: bin reward versus
Gaussian BAS reward, using the same stochastic actor–critic trainer, two seeds,
and 25 rounds/run. Bin RL won on both test BAS and 100 ms F1 in both matched seeds.
Mean BAS/F1: frozen 0.199/0.328, bin 0.212/0.361, BAS 0.203/0.344.
This is a preliminary negative result for direct BAS optimization, not proof that
bin reward is generally better (only four fresh test songs and three adaptation
songs). Validation F1 ranked the objectives differently. Diversity decreased;
mode collapse has not been established. Blinded clip review remains pending.

Training/evaluation completed 1,972 continuations; setup added 18. Metered compute
including setup was $1.2363, excluding storage/egress. Evaluator, exact action
likelihoods, frozen weights, and checkpoint reload checks passed.

Suggested division: I investigate extending the original four runs to round 100;
you investigate critic calibration and a small critic-warm-up comparison. Neither
follow-up is launched or implemented by this handoff. The last-five-round BAS
advantages remain about -1 despite returns near 0.2: the critic is substantially
overpredicting. This is a diagnostic hypothesis, not a proven explanation of the
reward ranking. Let's freeze the follow-up protocol and compute cap before runs.

Start with reports/midpoint_beats/MIDPOINT_REPORT.md and comparison.md. The core
trainer is scripts/midpoint_actor_critic.py. The agent prompt follows below.

## Agent pickup prompt

Continue this Bailando++ course project from the committed midpoint experiment.
Read applicable AGENTS.md, README.md, docs/TEAMMATE_HANDOFF.md,
reports/midpoint_beats/MIDPOINT_REPORT.md, configs/midpoint_beat_frozen.json,
scripts/midpoint_actor_critic.py, and scripts/run_midpoint_beats_gpu.py.
Resolve paths from your checkout; do not depend on the author's absolute paths.

Your proposed track is CRITIC CALIBRATION, not extending the original runs.
First review logs and reproduce the calibration diagnosis without new GPU work.
Compute early/late return, advantage, implied value, gradient, and entropy summaries.
Check that the baseline is action-independent and uses exactly the sampled context.
Do not infer systematic policy-gradient bias merely from negative mean advantages.

Prepare a controlled critic-warm-up experiment from the SAME supervised policy:
collect an adaptation-only pool, score identical trajectories under both rewards,
fit only each critic, assess on separate adaptation draws against a constant-return
predictor, then use the existing 25-round common actor–critic training. Preserve
both objective arms and two seeds; do not tune separate actor schedules. Choose
and document the fixed warm-up budget on development evidence; count its compute
and rollouts separately. If warm-up fails calibration checks, investigate before
launching four runs. Keep cold-start completed runs as references and disclose the
additional warm-up budget. Coordinate any shared trainer changes with the teammate
running longer training; prefer separate configs and output directories.

Add fixed-condition multiple-draw diversity diagnostics if missing: motion
distances, duplicate sequences/subsequences, upper/lower entropy, code usage,
event density, and movement magnitude. Lower speed-signature diversity alone is
not mode collapse. Keep greedy and sampled evaluation distinct. Do not tune on
the four previously evaluated test songs; label any reuse as follow-up evaluation.

Use the project Conda environment 10703, not system Python; recreate it from
environment.yml if needed. Remote jobs use the established legacy image. Run
`python -m scripts.check_midpoint_beats` after preparing assets; upstream/cache
checks require files excluded from Git. Pure source checkout is not a full data
or checkpoint distribution. Consult docs/MODAL.md for authentication and access.

Implement and verify independent local work first. Confirm available Modal access
and existing spending authorization before paid jobs; this prompt does not provide
a new spending cap. Prepare a concrete bounded command and measured estimate if
approval is required. Do not reuse the completed run's output directories or
overwrite frozen configs/results. Preserve historical evidence and failed runs.
Deliver code/config changes, verification, diagnostics, exact compute counts,
and a report distinguishing completed evidence from hypotheses.

## Artifact access and reproducibility

- Completed run ID: midbeats-20261117T213957703365Z.
- Modal volume: bailandopp-batch01; checkpoints live at
  runs/<run-id>/<arm>/last.pt for bin-42, bin-43, bas-42, bas-43.
- Benchmark run: midbeats-20261117T213508645736Z.
- Hashes and protocol: configs/midpoint_beat_frozen.json and completed stage manifests.
- Large weights, cached motions, datasets, videos, upstream checkout, credentials,
  and account billing snapshots are not distributed by this commit.
- Use the existing access mechanism with the volume owner; do not exchange tokens.
- Reports contain original local artifact paths; those paths must be restored or
  adjusted in another checkout. Follow report reproduction instructions selectively:
  launching the comparison again is paid training, not merely viewing results.
- Existing validation/test summaries and plots are committed for immediate review.

## Commit scope

Include source scripts and infrastructure dependencies, runtime requirements,
configuration/split manifests, methodological plans, text/JSON/CSV reports, and
compact figures. Include the historical pilot dependencies needed by midpoint
imports and documentation. Exclude videos, model/data arrays, checkpoints, local
runtime caches, credentials and raw billing snapshots. No push or teammate message
is performed by preparing this handoff.
