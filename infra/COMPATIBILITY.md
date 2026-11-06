"""Compatibility changes for the Bailando++ smoke runtime.

Upstream commit `cc90b98bff81c9709570db413c9610c2562e27ca` is not modified.
These changes make that checkout importable and let one GPU job call the
released sampling function. They do not change the policy, reward, or decoder.

## Package assembly

`tpami_bailandopp/models/__init__.py` imports `sep_vqvae` and `sep_vqvae_root`.
Those modules are not in the ++ tree. They are in `models/` and depend on
`vqvae.py` and `vqvae_root.py`. The ++ copies of `encdec.py`, `bottleneck.py`,
`resnet.py`, and `models/utils` are byte-identical to the original copies, so
the inherited modules can share them.

`infra/compat/overlay.py` builds a temporary `models` package from symlinks plus
a replacement `__init__.py`. The replacement imports `SepVQVAER`,
`CrossCondGPT2MW`, and `CrossCondGPT2MWAC` only. The upstream init also imports
`SepVQVAERmix`, which pulls in `smplx` and `pytorch3d`. Those are unused on the
position-decoder path.

## Configuration loading

Upstream entry points call `yaml.load` without a loader. The smoke script reads
the released YAML with `yaml.safe_load` and wraps it in `EasyDict`. Checkpoint
objects are still loaded with `torch.load` because that is the released
serialization. No training entry point or Slurm `srun` script is used.

## Weight loading

Saved state keys have a `module.` prefix from `DataParallel`. The smoke script
strips that prefix and calls the underlying module, which is what
`vqvae.module` / `gpt.module` do inside the released eval loop. It records every
missing and unexpected key. `strict=False` is used only after that list is
checked, and only for the actor-critic class, whose new `critic_net` parameters
are absent from the supervised checkpoint. Generation uses `CrossCondGPT2MW`,
the class stored in the checkpoint, loaded with `strict=True`.

## Data loading

`load_test_data_aist` is copied into `infra/compat/released_test_loader.py`
because importing `utils/functional.py` requires OpenCV and SMPL. The function
body is unchanged. The 20-second clip is produced by giving that loader the
first 1,200 native music frames, then calling `CrossCondGPT2MW.sample` with the
same music slicing as `mc_gpt_all.eval`.

## Runtimes

The Modal image installs two interpreters:

- `/opt/legacy`: Python 3.8 and `torch==1.6.0+cu101`, the version named in the
  ++ README. This is tried first.
- The image Python 3.11 with `torch==2.4.1+cu118`. This is a compatibility port.
  A successful run on it is not an exact PyTorch 1.6 reproduction.

The same container may fall back from a legacy infrastructure failure to the
port. It does not start a second GPU and it does not replace a completed but
invalid rollout.

`scripts/smoke_infer.py --device cpu` replaces `Tensor.cuda` with a no-op so a
Mac can construct the codebook buffers. That replacement is not used on the T4.

The released test loader always appends one extra `music_move` block before the
right-hand zeros. For this 1,200-frame crop and `music_move` 1, the padded music
length is 1,281 rather than 1,280. Integer division in `sample` still yields
150 codes and 1,200 decoded frames. The extra frame is preserved, not removed.
