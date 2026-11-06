"""Build an importable ``models`` package without editing upstream source.

The ++ ``models/__init__.py`` imports ``sep_vqvae`` and ``sep_vqvae_root``, which
live only in the original Bailando package, and it also imports the rotation
decoder, which requires ``smplx`` and ``pytorch3d``. Those inherited position
modules use the same ``encdec``, ``bottleneck``, ``resnet``, and ``utils``
sources as the ++ tree (byte-identical at the pinned commit).

This overlay symlinks the ++ package, adds the four missing position modules,
and installs a narrower ``__init__.py`` so inference can import ``SepVQVAER``
and the music-window policies. Upstream files are not modified.
"""
from __future__ import annotations

import os
from pathlib import Path


INHERITED = (
    "sep_vqvae.py",
    "sep_vqvae_root.py",
    "vqvae.py",
    "vqvae_root.py",
)

INIT_SOURCE = '''\
"""Compatibility package init. Not upstream source.

Imports only the position VQ-VAE and the ++ music-window policies.
The rotation/SMPL modules stay out of this import path.
"""
from .sep_vqvae_root import SepVQVAER
from .cross_cond_gpt2_music_window import CrossCondGPT2MW
from .cross_cond_gpt2_music_window_ac import CrossCondGPT2MWAC

__all__ = ["SepVQVAER", "CrossCondGPT2MW", "CrossCondGPT2MWAC"]
'''


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def build_overlay(overlay_dir: Path, bailando_root: Path) -> Path:
    plus_models = bailando_root / "tpami_bailandopp" / "models"
    original_models = bailando_root / "models"
    if not plus_models.is_dir() or not original_models.is_dir():
        raise FileNotFoundError(f"Bailando checkout is incomplete under {bailando_root}")

    models_dir = overlay_dir / "models"
    if models_dir.exists():
        for child in models_dir.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                raise RuntimeError(f"Refusing to replace non-file overlay entry {child}")
    models_dir.mkdir(parents=True, exist_ok=True)

    for source in sorted(plus_models.iterdir()):
        if source.name == "__init__.py":
            continue
        dest = models_dir / source.name
        os.symlink(source.resolve(), dest)

    for name in INHERITED:
        source = original_models / name
        if not source.is_file():
            raise FileNotFoundError(source)
        dest = models_dir / name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(source.resolve(), dest)

    init_path = models_dir / "__init__.py"
    init_path.write_text(INIT_SOURCE)
    return models_dir
