"""Load the supervised Bailando++ policy and run one 20-second smoke rollout.

Generation calls CrossCondGPT2MW.sample, which selects the top-1 code at every
step. The position decoder is SepVQVAER. Raw decoder coordinates are saved
separately from root-velocity integration and from global joint positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from infra.compat.overlay import build_overlay
from infra.compat.released_test_loader import load_test_data_aist


EXIT_INFRA = 2
EXIT_INVALID = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"Hash mismatch for {path}: {actual}")


def stage_example(cfg: dict, root: Path, work: Path) -> dict:
    features_path = resolve(root, cfg["external_features"])
    motion_path = resolve(root, cfg["motion_record"])
    with features_path.open() as handle:
        feature_obj = json.load(handle)
    frames = int(cfg["native_music_frames"])
    music = feature_obj["music_array"]
    if feature_obj.get("id") != cfg["song_id"]:
        raise SystemExit(f"Feature id is {feature_obj.get('id')}")
    if len(music) < frames or len(music[0]) != 438:
        raise SystemExit(f"Unexpected music shape {(len(music), len(music[0]))}")
    cropped = music[:frames]

    external = work / "external"
    motion_dir = work / "motion"
    external.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)
    with (external / f"{cfg['song_id']}.json").open("w") as handle:
        json.dump({"id": cfg["song_id"], "music_array": cropped}, handle)
    staged_motion = motion_dir / motion_path.name
    if staged_motion.exists() or staged_motion.is_symlink():
        staged_motion.unlink()
    os.symlink(motion_path.resolve(), staged_motion)

    loader = cfg["released_loader"]
    music_data, dance_data, names = load_test_data_aist(
        str(motion_dir),
        rotmat=loader["rotmat"],
        move=loader["move"],
        external_wav=str(external),
        external_wav_rate=loader["external_wav_rate"],
        music_normalize=False,
        wav_padding=loader["wav_padding_frames"],
    )
    if names != [motion_path.name] or dance_data is None:
        raise SystemExit(f"Loader returned unexpected records: {names}")
    music_np = music_data[0]
    dance_np = dance_data[0]
    expected = cfg["expected_shapes"]
    if music_np.shape[0] != expected["padded_music_frames"]:
        raise SystemExit(f"Padded music length {music_np.shape[0]}")
    if dance_np.shape != (expected["padded_motion_frames"], 72):
        raise SystemExit(f"Padded dance shape {dance_np.shape}")
    if not (music_np[: loader["wav_padding_frames"]] == 0).all():
        raise SystemExit("Left music padding is not zero")
    if not (music_np[-loader["wav_padding_frames"] :] == 0).all():
        raise SystemExit("Right music padding is not zero")
    native = music_np[loader["wav_padding_frames"] : loader["wav_padding_frames"] + frames]
    reloaded = __import__("numpy").array(cropped)
    if native.shape != reloaded.shape or not (native == reloaded).all():
        raise SystemExit("Cropped music did not round-trip through the released loader")
    return {
        "music": music_np,
        "dance": dance_np,
        "name": names[0],
        "native_frames": frames,
    }


def prepare_cond(music_np, cfg: dict):
    import torch

    loader = cfg["released_loader"]
    n_music = cfg["policy_feature_slice"][1]
    music = torch.from_numpy(music_np).unsqueeze(0)
    music_ds_rate = loader["music_ds_rate"]
    music = music[:, :, : n_music // music_ds_rate].contiguous().float()
    batch, length, channels = music.size()
    music = music.view(batch, length // music_ds_rate, channels * music_ds_rate)
    drop = loader["ds_rate"] // loader["music_relative_rate"]
    return music[:, drop:]


def torch_load(path: Path):
    import inspect
    import torch

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def strip_module_prefix(state: dict) -> tuple[dict, str]:
    keys = list(state)
    prefixed = [key.startswith("module.") for key in keys]
    if all(prefixed):
        return {key[len("module.") :]: value for key, value in state.items()}, "stripped_module_prefix"
    if any(prefixed):
        raise SystemExit("Checkpoint mixes DataParallel prefixes; refusing to guess")
    return dict(state), "no_prefix"


def key_report(model, state: dict) -> dict:
    model_keys = set(model.state_dict())
    state_keys = set(state)
    missing = sorted(model_keys - state_keys)
    unexpected = sorted(state_keys - model_keys)
    return {"missing": missing, "unexpected": unexpected, "model_keys": len(model_keys), "state_keys": len(state_keys)}


def load_strict(model, state: dict) -> dict:
    import torch

    report = key_report(model, state)
    if report["missing"] or report["unexpected"]:
        report["loaded"] = False
        return report
    incompatible = model.load_state_dict(state, strict=True)
    leftover = list(getattr(incompatible, "missing_keys", [])) + list(getattr(incompatible, "unexpected_keys", []))
    if leftover:
        raise SystemExit(f"strict load reported keys after an empty diff: {leftover}")
    report["loaded"] = True
    report["strict"] = True
    with torch.no_grad():
        for key, value in state.items():
            loaded = model.state_dict()[key].detach().cpu()
            if loaded.shape != value.shape or not torch.equal(loaded, value.detach().cpu()):
                raise SystemExit(f"Loaded tensor differs from checkpoint: {key}")
    report["reloaded_tensors_match_checkpoint"] = True
    return report


def load_actor_critic(ac_model, supervised_state: dict) -> dict:
    import torch

    report = key_report(ac_model, supervised_state)
    missing = report["missing"]
    critic = [key for key in missing if key.startswith("critic_net.")]
    other = [key for key in missing if not key.startswith("critic_net.")]
    report["critic_missing"] = critic
    report["non_critic_missing"] = other
    report["critic_justified"] = not report["unexpected"] and not other and bool(critic)
    if not report["critic_justified"]:
        report["loaded"] = False
        report["justification"] = (
            "Refused to load. Missing keys must be exactly the new critic parameters, "
            "and the supervised checkpoint must not contain unexpected keys."
        )
        return report
    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in ac_model.state_dict().items()
        if name.startswith("critic_net.")
    }
    incompatible = ac_model.load_state_dict(supervised_state, strict=False)
    still_missing = sorted(set(incompatible.missing_keys) - set(critic))
    if list(incompatible.unexpected_keys) or still_missing:
        raise SystemExit(f"Actor-critic load changed the reviewed key sets: {incompatible}")
    max_delta = 0.0
    with torch.no_grad():
        for name, tensor in ac_model.state_dict().items():
            loaded = tensor.detach().cpu()
            if name.startswith("critic_net."):
                max_delta = max(max_delta, float((loaded - before[name]).abs().max()))
            else:
                source = supervised_state[name].detach().cpu()
                if not torch.equal(loaded, source):
                    raise SystemExit(f"Actor tensor changed unexpectedly: {name}")
    report["loaded"] = True
    report["strict"] = False
    report["strict_false_reason"] = "critic_net parameters are newly initialized and absent from the supervised checkpoint"
    report["critic_tensors_initialized"] = len(before)
    report["critic_parameter_names"] = sorted(name for name, _ in ac_model.named_parameters() if name.startswith("critic_net."))
    report["critic_max_abs_change_during_load"] = max_delta
    report["used_for_generation"] = False
    return report


def import_models(work: Path, bailando: Path):
    overlay = build_overlay(work / "overlay", bailando)
    parent = str(overlay.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules.pop("models", None)
    import models

    return models


def use_cpu_placeholders() -> None:
    """Let a CPU-only inspection construct modules that call Tensor.cuda during init.

    The T4 path does not call this. Upstream still allocates those buffers on CUDA.
    """
    import torch

    def tensor_cuda(self, *_args, **_kwargs):
        return self

    def module_cuda(self, *_args, **_kwargs):
        return self

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda


def versions() -> dict:
    import numpy
    import torch

    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "numpy": numpy.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return info


def integrate_root(raw):
    import torch

    integrated = raw.clone()
    integrated[:, 0, :3] = 0
    for index in range(1, integrated.size(1)):
        integrated[:, index, :3] = integrated[:, index - 1, :3] + raw[:, index - 1, :3]
    return integrated


def global_positions(integrated_numpy):
    import numpy as np

    joints = integrated_numpy.reshape(integrated_numpy.shape[0], 24, 3).copy()
    root = joints[:, 0, :].copy()
    joints = joints + root[:, None, :]
    joints[:, 0, :] = root
    return joints


def encode_first(vqvae, pose):
    quants = vqvae.encode(pose)
    if not isinstance(quants, tuple) or len(quants) != 2:
        raise SystemExit(f"Unexpected encode return type: {type(quants)}")
    initial = tuple(quants[branch][0][:, :1].clone() for branch in range(2))
    full = tuple(quants[branch][0].detach().cpu().clone() for branch in range(2))
    return initial, full


def context_probe(vqvae, pose) -> dict:
    import torch

    total = pose.size(1)
    horizons = [8, 16, 32, 64, 128, 256, total]
    rows = []
    with torch.no_grad():
        reference, _ = encode_first(vqvae, pose)
        reference_pair = [int(reference[0][0, 0]), int(reference[1][0, 0])]
        for horizon in horizons:
            trial = pose.clone()
            if horizon < total:
                trial[:, horizon:, :] = trial[:, horizon - 1 : horizon, :]
            try:
                initial, _ = encode_first(vqvae, trial)
                pair = [int(initial[0][0, 0]), int(initial[1][0, 0])]
                rows.append({"future_frames_held_from": horizon, "first_codes": pair, "matches_full_sequence": pair == reference_pair})
            except Exception as exc:
                rows.append({"future_frames_held_from": horizon, "error": str(exc)})
    return {
        "description": (
            "The encoder is convolutional, so the first code can depend on later frames. "
            "Each trial keeps the prefix and repeats its last frame through the rest of the loader sequence."
        ),
        "full_sequence_first_codes": reference_pair,
        "trials": rows,
    }


def music_timing(cfg: dict) -> dict:
    loader = cfg["released_loader"]
    pad = loader["wav_padding_frames"]
    drop = loader["ds_rate"] // loader["music_relative_rate"]
    return {
        "native_origin": "feature frame 0 is the audio/file origin; no fitted offset",
        "loader_left_zero_frames": pad,
        "loader_extra_repeated_frame": 1,
        "loader_repeat_note": "Released padding always appends one music_move block, here one copy of the last real frame, before the right zeros.",
        "frames_dropped_before_sample": drop,
        "generated_code_index": "initial code is index 0; sample step k appends code k+1",
        "aligned_native_music_frames_for_generated_code_k_plus_1": "[8 + 8k, 16 + 8k)",
        "decoded_motion_frames_for_code_index": "[8 * index, 8 * index + 8)",
        "music_transformer_context": (
            "window_size 11 and downsample_rate 8 keep a +/- 40-frame attention band. "
            "The loader's 40 zero frames on each side supply that band at the ends."
        ),
        "init_code_motion_frames": "[0, 8)",
        "init_code_source": "first upper/lower code of the full padded real dance, not of an 8-frame crop",
    }


def run(args) -> None:
    import numpy as np
    import torch

    root = Path(args.project).resolve()
    cfg = load_config(Path(args.config))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    policy_path = resolve(root, cfg["policy"]["path"])
    decoder_path = resolve(root, cfg["position_decoder"]["path"])
    require_hash(policy_path, cfg["policy"]["sha256"])
    require_hash(decoder_path, cfg["position_decoder"]["sha256"])
    staged = stage_example(cfg, root, work)

    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed_all(int(cfg["seed"]))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device_name = "cuda" if args.device == "cuda" else "cpu"
    if device_name == "cpu":
        use_cpu_placeholders()
    elif not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    device = torch.device(device_name)

    started = time.perf_counter()
    models = import_models(work, root / "third_party" / "Bailando")
    policy_ckpt = torch_load(policy_path)
    decoder_ckpt = torch_load(decoder_path)
    read_seconds = time.perf_counter() - started
    if int(policy_ckpt["epoch"]) != cfg["policy"]["stored_epoch"]:
        raise SystemExit(f"Policy stored epoch is {policy_ckpt['epoch']}")
    if int(decoder_ckpt["epoch"]) != cfg["position_decoder"]["stored_epoch"]:
        raise SystemExit(f"Decoder stored epoch is {decoder_ckpt['epoch']}")

    from easydict import EasyDict

    def as_easy(value):
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return EasyDict({key: as_easy(item) for key, item in value.items()})
        return value

    def section(config_obj, name):
        if hasattr(config_obj, name):
            return getattr(config_obj, name)
        return config_obj[name]

    policy_cfg = as_easy(section(policy_ckpt["config"], "structure_generate"))
    decoder_cfg = as_easy(section(decoder_ckpt["config"], "structure"))
    if policy_cfg.name != "CrossCondGPT2MW" or decoder_cfg.name != "SepVQVAER":
        raise SystemExit("Checkpoint config class does not match the baseline manifest")

    build_started = time.perf_counter()
    decoder = models.SepVQVAER(decoder_cfg).eval()
    policy = models.CrossCondGPT2MW(policy_cfg).eval()
    decoder_state, decoder_prefix = strip_module_prefix(decoder_ckpt["model"])
    policy_state, policy_prefix = strip_module_prefix(policy_ckpt["model"])
    decoder_report = load_strict(decoder, decoder_state)
    policy_report = load_strict(policy, policy_state)
    if not decoder_report["loaded"] or not policy_report["loaded"]:
        (output / "loading_report.json").write_text(json.dumps({
            "decoder": decoder_report, "policy": policy_report,
        }, indent=2))
        raise SystemExit(EXIT_INVALID)

    rl_yaml_path = root / "third_party" / "Bailando" / "tpami_bailandopp" / "configs" / "actor_critic_music_trans_400.yaml"
    import copy
    import yaml
    rl_yaml = yaml.safe_load(rl_yaml_path.read_text())
    ac_cfg = as_easy(copy.deepcopy(dict(policy_cfg)))
    ac_cfg.critic_net = as_easy(rl_yaml["structure_generate"]["critic_net"])
    ac_cfg.name = "CrossCondGPT2MWAC"
    actor_critic = models.CrossCondGPT2MWAC(ac_cfg).eval()
    ac_report = load_actor_critic(actor_critic, policy_state)
    build_seconds = time.perf_counter() - build_started

    loading = {
        "policy_file": str(policy_path),
        "policy_sha256": cfg["policy"]["sha256"],
        "policy_filename_epoch": cfg["policy"]["filename_epoch"],
        "policy_stored_epoch": int(policy_ckpt["epoch"]),
        "policy_prefix": policy_prefix,
        "policy_load": {key: value for key, value in policy_report.items() if key != "model_keys"},
        "decoder_file": str(decoder_path),
        "decoder_sha256": cfg["position_decoder"]["sha256"],
        "decoder_stored_epoch": int(decoder_ckpt["epoch"]),
        "decoder_prefix": decoder_prefix,
        "decoder_load": {key: value for key, value in decoder_report.items() if key != "model_keys"},
        "actor_critic_diagnostic": ac_report,
        "generation_class": "CrossCondGPT2MW",
        "versions": versions(),
    }
    (output / "loading_report.json").write_text(json.dumps(loading, indent=2))
    if args.load_only:
        timing = {"read_checkpoints_seconds": read_seconds, "build_and_load_seconds": build_seconds, "load_only": True}
        (output / "timing.json").write_text(json.dumps(timing, indent=2))
        print(json.dumps({"status": "load_only", "output": str(output)}))
        return

    decoder.to(device)
    policy.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    pose = torch.from_numpy(staged["dance"]).unsqueeze(0).float().to(device)
    cond = prepare_cond(staged["music"], cfg).to(device)
    expected = cfg["expected_shapes"]
    if cond.size(1) != expected["cond_frames_after_drop8"] or cond.size(2) != 55:
        raise SystemExit(f"Conditioning shape {tuple(cond.shape)}")

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        encode_started = time.perf_counter()
        initial, full_codes = encode_first(decoder, pose)
        probe = context_probe(decoder, pose)
        if device.type == "cuda":
            torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - encode_started
        memory_after_encode = memory_stats()

        sample_started = time.perf_counter()
        generated = policy.sample(initial, cond=cond, shift=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sample_seconds = time.perf_counter() - sample_started
        memory_after_sample = memory_stats()

        decode_started = time.perf_counter()
        raw = decoder.decode(generated)
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started

    upper = generated[0][0].detach().cpu().numpy()[0]
    lower = generated[1][0].detach().cpu().numpy()[0]
    raw_numpy = raw.detach().cpu().numpy()[0]
    integrated = integrate_root(raw.detach()).cpu().numpy()[0]
    positions = global_positions(integrated)
    memory_after_decode = memory_stats()

    validation_failures = validate(upper, lower, raw_numpy, integrated, positions, cond.detach().cpu().numpy()[0], cfg)
    init_pair = [int(initial[0][0, 0]), int(initial[1][0, 0])]
    if int(upper[0]) != init_pair[0] or int(lower[0]) != init_pair[1]:
        validation_failures.append("generated sequence does not start with the encoded initial codes")

    np.savez(
        output / "codes.npz",
        upper=upper.astype(np.int64),
        lower=lower.astype(np.int64),
        real_upper=full_codes[0].numpy()[0].astype(np.int64),
        real_lower=full_codes[1].numpy()[0].astype(np.int64),
    )
    np.save(output / "motion_raw_reward.npy", raw_numpy.astype(np.float32))
    np.save(output / "motion_root_integrated.npy", integrated.astype(np.float32))
    np.save(output / "motion_global_positions.npy", positions.astype(np.float32))
    np.save(output / "music_cond.npy", cond.detach().cpu().numpy()[0].astype(np.float32))

    render_started = time.perf_counter()
    from scripts.render_skeleton import render_skeleton_mp4

    audio = resolve(root, cfg["audio"])
    render_info = render_skeleton_mp4(positions, output / "skeleton_audio.mp4", audio, frame_rate=cfg["frame_rate_hz"])
    render_seconds = time.perf_counter() - render_started

    manifest = {
        "song_id": cfg["song_id"],
        "record_id": cfg["record_id"],
        "init_upper_lower": init_pair,
        "code_length": int(upper.size),
        "decoded_frames": int(raw_numpy.shape[0]),
        "duration_seconds": raw_numpy.shape[0] / cfg["frame_rate_hz"],
        "encoder_context": probe,
        "music_timing": music_timing(cfg),
        "root_integration": {
            "formula": "integrated[0, :3] = 0; integrated[t, :3] = integrated[t-1, :3] + raw[t-1, :3]",
            "source": "mc_gpt_all.eval global_vel block",
            "non_root_channels_unchanged": True,
            "raw_reward_coordinates": "motion_raw_reward.npy",
            "integrated_before_joint_translation": "motion_root_integrated.npy",
            "global_positions": "motion_global_positions.npy",
        },
        "validation_failures": validation_failures,
        "render": render_info,
        "finite_raw": bool(np.isfinite(raw_numpy).all()),
        "finite_positions": bool(np.isfinite(positions).all()),
        "code_min_max": [int(min(upper.min(), lower.min())), int(max(upper.max(), lower.max()))],
        "unique_codes": {"upper": int(len(set(upper.tolist()))), "lower": int(len(set(lower.tolist())))},
    }
    timing = {
        "read_checkpoints_seconds": read_seconds,
        "build_and_load_seconds": build_seconds,
        "encode_and_context_seconds": encode_seconds,
        "sample_seconds": sample_seconds,
        "decode_seconds": decode_seconds,
        "render_seconds": render_seconds,
        "function_compute_seconds": time.perf_counter() - started,
        "memory": {
            "after_encode": memory_after_encode,
            "after_sample": memory_after_sample,
            "after_decode": memory_after_decode,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps({"status": "invalid" if validation_failures else "ok", "output": str(output), "init": init_pair}))
    if validation_failures:
        raise SystemExit(EXIT_INVALID)


def memory_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def validate(upper, lower, raw, integrated, positions, cond, cfg) -> list[str]:
    import numpy as np

    failures = []
    expected = cfg["expected_shapes"]
    limit = cfg["codebook_size"]
    if upper.shape != (expected["total_codes_including_init"],) or lower.shape != upper.shape:
        failures.append(f"code length {upper.shape} {lower.shape}")
    if raw.shape != (expected["decoded_frames"], 72):
        failures.append(f"raw shape {raw.shape}")
    if integrated.shape != raw.shape or positions.shape != (raw.shape[0], 24, 3):
        failures.append("integrated or global shape mismatch")
    if cond.shape != (expected["cond_frames_after_drop8"], 55):
        failures.append(f"cond shape {cond.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(integrated).all() or not np.isfinite(positions).all() or not np.isfinite(cond).all():
        failures.append("non-finite array")
    if upper.min() < 0 or lower.min() < 0 or upper.max() >= limit or lower.max() >= limit:
        failures.append("code outside [0, 512)")
    if not np.array_equal(integrated[:, 3:], raw[:, 3:]):
        failures.append("integration changed non-root channels")
    if not np.allclose(integrated[0, :3], 0):
        failures.append("integrated frame-0 root is not zero")
    velocity_sum = np.cumsum(raw[:, :3], axis=0)
    velocity_sum = np.vstack([np.zeros((1, 3)), velocity_sum[:-1]])
    if not np.allclose(integrated[:, :3], velocity_sum):
        failures.append("root integration does not match the released prefix sum")
    reconstructed_root = positions[:, 0, :]
    relative = positions - reconstructed_root[:, None, :]
    relative[:, 0, :] = reconstructed_root
    if not np.allclose(relative.reshape(raw.shape), integrated, rtol=1e-5, atol=1e-5):
        failures.append("global positions do not invert to the integrated pose")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--runtime-label", default="")
    args = parser.parse_args()
    try:
        run(args)
    except SystemExit as exc:
        if exc.code in (EXIT_INFRA, EXIT_INVALID, 0, None):
            raise
        print(f"infrastructure failure: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    except Exception:
        traceback.print_exc()
        raise SystemExit(EXIT_INFRA)


if __name__ == "__main__":
    main()
