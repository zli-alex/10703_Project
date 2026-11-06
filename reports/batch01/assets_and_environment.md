# First-step status: source, assets, and environment

Updated November 6, 2026. The T4 smoke test has run. See `docs/MODAL.md` for teammate setup.

## Readiness table

| Item | Status | Evidence / remaining action |
|---|---|---|
| Project plan and first-batch checklist | Complete | Saved under `docs/`; all future project work uses this repository. |
| Official source checkout | Complete | `third_party/Bailando`, detached at `cc90b98bff81c9709570db413c9610c2562e27ca`; clean checkout, GIF assets excluded. |
| ++ configuration and code inspection | Complete for setup inventory | Identified entry points, dependencies, expected asset paths, and static compatibility issues below. |
| Existing Conda installation | Located | `/Users/zhengfeili/opt/anaconda3/bin/conda`; previous shell simply did not put it on PATH. |
| Dedicated local environment | Complete and verified | `10703`, Python 3.11.18; all four setup-tool imports pass and `pip check` finds no broken requirements. |
| Supervised ++ checkpoint | Downloaded and metadata-verified | Expected file is present; saved epoch is 440 despite filename `epoch_400.pt`. See checkpoint inventory. |
| Position VQ-VAE checkpoint | Downloaded and metadata-verified | `SepVQVAER`, stored epoch 500; expected codebook sizes and shared configuration verified. |
| Rotation decoder checkpoint | Archive presence verified; download deferred | Not needed for first position-only clip. |
| Motion/features/audio | Two development pairs downloaded and validated | Native 60 Hz features preserved; real motion is 7.1/7.383 s. Shared reward-audit interval is [2, 6) s. See data validation report. |
| Modal login and cycle usage | Verified for workspace `zhengfel` | Token created November 6, 2026. A payment method is required for T4. Credits covered the runs: metered $0.03039, billed $0. |
| Credit expiration | Not exposed by the API | Applied credits are visible. Remaining balance and expiration are not. The launch gate allows work while metered cost is under $4. |
| Remote GPU environment | Built and used | Legacy Python 3.8 / PyTorch 1.6.0+cu101 ran the smoke test on one T4. A PyTorch 2.4.1 port is also in the image and was not needed. |
| Model loading and inference | Passed | Run `smokeB-20261106T204747Z`: strict loads, 20-second mHO5 clip. See the smoke-test report. |

## Findings from the actual checkout

The source is available and clean; model weights and processed data are not
bundled with it. The ++ `data` directory contains only a placeholder.

The ++ package does not contain `sep_vqvae` or `sep_vqvae_root`. Those modules
are taken from the original Bailando tree by `infra/compat/overlay.py`. Upstream
files are not edited. The loader, YAML, and checkpoint-prefix handling are
recorded in [infra/COMPATIBILITY.md](../../infra/COMPATIBILITY.md). The T4 run
used that assembly with PyTorch 1.6.0.

The local computer is Apple Silicon. The local Python environment is for
orchestration and asset inspection. It is not evidence that the original
PyTorch 1.6/CUDA implementation runs; that check belongs to the Modal runtime.

## Asset sources and current limits

- [Official ++ pretrained archive](https://drive.google.com/file/d/1jepf20XuplVSGEfwaPaqwniyl4styvqa/view)
- [Original-Bailando pretrained archive](https://drive.google.com/file/d/1Fi0TIiBV6EQAQrBU0IOnlke2Nu4IcutC/view)
- [Preprocessed data archive linked by the original README](https://drive.google.com/file/d/1EGJeBE1fE59ByjxR_-ipwV6Dz-Cx-stT/view)

The web reader could not access the Drive pages, but direct HTTP access worked.
Archive directories were read with byte-range requests, and the two required
checkpoints were selectively extracted. Their CRCs, SHA256 hashes, metadata, and
storage byte lengths are verified. A T4 load with PyTorch 1.6.0 then accepted
every tensor. See [checkpoint inventory](checkpoint_inventory.md) and
[the smoke test](smoke_test.md).

## Next actions

1. A teammate joins workspace `zhengfel` and follows [docs/MODAL.md](../../docs/MODAL.md).
2. Step C reuses this smoke clip as A-original. Do not regenerate it.
3. Credit expiration is still unknown. Check the Modal billing page before
   spending beyond the recorded few cents.

## Verified local environment

Location: `/Users/zhengfeili/opt/anaconda3/envs/10703`.

| Component | Installed version |
|---|---|
| Python | 3.11.18 |
| Modal | 1.6.3 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| gdown | 6.5.1 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| SoundFile | 0.14.0 |

Setup-tool imports and the scientific data-validation script succeeded under
`conda run -n 10703 python`.
`python -m pip check` reported no broken requirements. The top-level versions are
pinned in `environment.yml`; the pip dependency snapshot is recorded in
`requirements-local.lock.txt`. These checks do not contact Modal or run a GPU job.

The initial `.venv` contains only pip; its interrupted dependency install did not
complete. It is now unused and ignored. The user's requested Conda environment
supersedes it. Nothing has been committed or pushed.
