# 10703_Project

Bailando++ policy/reward exploration and RL replication for 10-703.

- [Project plan](docs/PROJECT_PLAN.md): modules, dependencies, evidence, evaluation, and Modal budget.
- [First batch](docs/FIRST_BATCH.md): the initial setup and small diagnostic experiments.
- [Setup status](reports/batch01/assets_and_environment.md): completed checks and remaining prerequisites.
- [Checkpoint inventory](reports/batch01/checkpoint_inventory.md): verified weights, hashes, and the filename/epoch discrepancy.
- [Data validation](reports/batch01/data_validation.md): two prepared development pairs, native features, and timeline plots.
- [Smoke test](reports/batch01/smoke_test.md): the completed T4 rollout, clip, timing, and cost.
- [Modal setup](docs/MODAL.md): how a teammate authenticates, reuses the volume, and launches a run.

Project root: `/Users/zhengfeili/2026_Fall/10703/10703_Project`.
All subsequent project code, configuration, and reports belong here. Large datasets,
weights, and remote run artifacts will use persistent storage and stay out of Git.

Current status: step B passed. Run `smokeB-20261106T204747Z` loaded the
supervised policy and position decoder on a T4 with PyTorch 1.6.0 and generated
a 20-second mHO5 skeleton clip. Workspace metered cost is about $0.030 and
billed cost is $0. See [smoke test](reports/batch01/smoke_test.md).

Target: **Bailando++ (TPAMI 2023)**, using the `tpami_bailandopp` implementation
inside the official Bailando repository. Preserve original-Bailando components
where the ++ configuration explicitly depends on them.

## Local environment

Use the dedicated Conda environment `10703`, defined by [environment.yml](environment.yml),
for project tooling. It uses Python 3.11 with Modal, PyYAML, requests, gdown,
NumPy, SciPy, Matplotlib, and SoundFile.
This is the local orchestration environment; the historical CUDA/PyTorch runtime
for Bailando++ will be specified separately for Modal.
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

Latest: step B saved-array revalidation passed; step C three comparisons completed.
See [music sensitivity](reports/batch01/music_sensitivity.md). Reward audit (D) and
rollout trace (E) remain pending.
