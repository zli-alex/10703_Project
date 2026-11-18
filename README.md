# 10703_Project

Bailando++ policy/reward exploration and RL replication for 10-703.

- [Current midpoint execution plan](docs/MIDPOINT_BEAT_OBJECTIVE_PLAN.md): bounded bin-versus-BAS RL comparison, evaluation, and report.
- [New-agent kickoff prompt](docs/MIDPOINT_BEAT_AGENT_PROMPT.md): copyable implementation handoff.

- [Beat-alignment RL research plan](docs/BEAT_ALIGNMENT_RL_PLAN.md): the next phase, with objective comparisons, expanded evaluation, actor–critic ablations, and later downbeat/subdivision extensions.
- [Project plan](docs/PROJECT_PLAN.md): modules, dependencies, evidence, evaluation, and Modal budget.
- [First batch](docs/FIRST_BATCH.md): the initial setup and small diagnostic experiments.
- [Setup status](reports/batch01/assets_and_environment.md): completed checks and remaining prerequisites.
- [Checkpoint inventory](reports/batch01/checkpoint_inventory.md): verified weights, hashes, and the filename/epoch discrepancy.
- [Data validation](reports/batch01/data_validation.md): two prepared development pairs, native features, and timeline plots.
- [Smoke test](reports/batch01/smoke_test.md): the completed T4 rollout, clip, timing, and cost.
- [Batch summary](reports/batch01/summary.md): completed gates, limitations, spending, and next checks.
- [Rollout trace](reports/batch01/rollout_trace.md): two training windows, action/frame alignment, rewards, masks, and bootstrap values.
- [Reward audit](reports/batch01/reward_audit.md) and [music sensitivity](reports/batch01/music_sensitivity.md): diagnostic experiments.
- [Training-cycle benchmark](reports/batch02/training_cycle.md): shared Adam behavior, checkpoint reload, and T4 timings.
- [Pilot split and readiness](reports/batch02/pilot_readiness.md): validated songs, batch-size 64 benchmark, and five-epoch budget.
- [Five-epoch RL pilot](reports/batch03/rl_pilot.md): checkpoint verification, paired results, and synchronized clips.
- [Modal setup](docs/MODAL.md): how a teammate authenticates, reuses the volume, and launches a run.

Project root: `/Users/zhengfeili/2026_Fall/10703/10703_Project`.
All subsequent project code, configuration, and reports belong here. Large datasets,
weights, and remote run artifacts will use persistent storage and stay out of Git.

The five-epoch RL pilot and matched evaluation are complete. The epoch-5
checkpoint reloaded successfully. Mean reward fell slightly on adaptation songs
and was nearly flat on songs held out from RL. The paired clips are ready for
human audiovisual review.

Target: **Bailando++ (TPAMI 2023)**, using the `tpami_bailandopp` implementation
inside the official Bailando repository. Preserve original-Bailando components
where the ++ configuration explicitly depends on them.

## Local environment

Use the dedicated Conda environment `10703`, defined by [environment.yml](environment.yml),
for project tooling. It uses Python 3.11 with Modal, PyYAML, requests, gdown,
NumPy, SciPy, Matplotlib, and SoundFile.
This is the local orchestration environment; Modal uses the legacy runtime
specified under `infra/runtime/` (Python 3.8, PyTorch 1.6.0+cu101).
Installed pip dependency versions are recorded in
[requirements-local.lock.txt](requirements-local.lock.txt).

```sh
source /Users/zhengfeili/opt/anaconda3/etc/profile.d/conda.sh
conda activate 10703
cd /Users/zhengfeili/2026_Fall/10703/10703_Project
```

For noninteractive execution:

```sh
/Users/zhengfeili/opt/anaconda3/bin/conda run -n 10703 python --version
```

The preliminary `.venv` is unused and ignored by Git. Do not use `/usr/bin/python3`.

The paired review clips and timing caveats are in the pilot report.
