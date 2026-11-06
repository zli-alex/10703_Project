"""One T4 smoke test. The local entrypoint refuses to launch without a fresh account gate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import modal

PROJECT = Path(__file__).resolve().parents[1]
ACCOUNT_REPORT = PROJECT / "reports" / "batch01" / "modal_account.json"
VOLUME_NAME = "bailandopp-batch01"
APP_NAME = "bailandopp-batch01-smoke"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("ubuntu:20.04", add_python="3.11")
    .run_commands(
        "ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime",
        "printf 'Etc/UTC\\n' > /etc/timezone",
        env={"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "DEBCONF_NONINTERACTIVE_SEEN": "true"},
    )
    .apt_install(
        "python3.8",
        "python3.8-venv",
        "python3-distutils",
        "ffmpeg",
        "wget",
        "ca-certificates",
        "libgomp1",
        "git",
        "tzdata",
        env={"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "DEBCONF_NONINTERACTIVE_SEEN": "true"},
    )
    .run_commands(
        "/usr/bin/python3.8 -m venv /opt/legacy",
        "/opt/legacy/bin/pip install --upgrade 'pip==24.0'",
        "/opt/legacy/bin/pip install 'numpy==1.19.5' 'PyYAML==5.4.1' 'easydict==1.9' 'tqdm==4.64.1' 'Pillow==8.4.0' 'future==0.18.2'",
        "/opt/legacy/bin/pip install --no-deps 'https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp38-cp38-linux_x86_64.whl'",
        "/opt/legacy/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)'",
    )
    .pip_install(
        "numpy==1.26.4",
        "PyYAML==6.0.2",
        "easydict==1.13",
        "tqdm==4.66.5",
        "pillow==10.4.0",
    )
    .pip_install("torch==2.4.1+cu118", index_url="https://download.pytorch.org/whl/cu118")
    .add_local_dir(PROJECT / "infra", remote_path="/root/project/infra", ignore=["**/__pycache__/**"])
    .add_local_dir(PROJECT / "scripts", remote_path="/root/project/scripts", ignore=["**/__pycache__/**"])
    .add_local_dir(PROJECT / "configs", remote_path="/root/project/configs")
    .add_local_dir(
        PROJECT / "third_party" / "Bailando",
        remote_path="/root/project/third_party/Bailando",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)

RETRIES = modal.Retries(max_retries=0, initial_delay=1.0)
ASSETS = {
    "assets/policy/epoch_400.pt": "assets/bailandopp/cc_gpt_music_trans/ckpt/epoch_400.pt",
    "assets/decoder/epoch_500.pt": "assets/bailando_original/experiments/sep_vqvae_root/ckpt/epoch_500.pt",
    "assets/motion/gHO_sBM_cAll_d20_mHO5_ch01.json": "assets/preprocessed_data/data/aistpp_test_full_wav/gHO_sBM_cAll_d20_mHO5_ch01.json",
    "assets/features/mHO5.json": "data/batch01/music_feat_60fps/mHO5.json",
    "assets/audio/mHO5.wav": "assets/audio/mHO5.wav",
}


def _resolved_config() -> dict:
    config = json.loads(Path("/root/project/configs/smoke_batch01.json").read_text())
    config["policy"]["path"] = "/vol/assets/policy/epoch_400.pt"
    config["position_decoder"]["path"] = "/vol/assets/decoder/epoch_500.pt"
    config["motion_record"] = "/vol/assets/motion/gHO_sBM_cAll_d20_mHO5_ch01.json"
    config["external_features"] = "/vol/assets/features/mHO5.json"
    config["audio"] = "/vol/assets/audio/mHO5.wav"
    return config


def _execute(runtime: str, output: str, device: str, load_only: bool) -> int:
    python = "/opt/legacy/bin/python" if runtime == "legacy" else sys.executable
    remote_config = Path(output) / "resolved_config.json"
    Path(output).mkdir(parents=True, exist_ok=True)
    remote_config.write_text(json.dumps(_resolved_config()))
    command = [
        python,
        "/root/project/scripts/smoke_infer.py",
        "--config", str(remote_config),
        "--project", "/root/project",
        "--output", output,
        "--device", device,
        "--runtime-label", runtime,
    ]
    if load_only:
        command.append("--load-only")
    log_path = Path(output) / f"{runtime}_{device}.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


@app.function(image=image, cpu=2.0, memory=8192, timeout=900, retries=RETRIES, volumes={"/vol": volume})
def cpu_load(run_id: str) -> dict:
    output = f"/vol/runs/{run_id}/cpu_load"
    attempts = []
    selected = None
    for runtime in ("legacy", "modern"):
        code = _execute(runtime, f"{output}/{runtime}", "cpu", True)
        attempts.append({"runtime": runtime, "exit_code": code})
        if code == 0:
            selected = runtime
            break
        if code != 2:
            break
    volume.commit()
    return {"run_id": run_id, "attempts": attempts, "selected_runtime": selected}


@app.function(image=image, gpu="T4", cpu=2.0, memory=8192, timeout=900, retries=RETRIES, volumes={"/vol": volume})
def gpu_smoke(run_id: str, runtime: str) -> dict:
    output = f"/vol/runs/{run_id}/gpu"
    started = datetime.now(timezone.utc).isoformat()
    code = _execute(runtime, output, "cuda", False)
    used = runtime
    fallback = None
    if code == 2 and runtime == "legacy":
        fallback = _execute("modern", output + "_modern", "cuda", False)
        if fallback == 0:
            used = "modern"
            code = 0
    volume.commit()
    return {
        "run_id": run_id,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "requested_runtime": runtime,
        "used_runtime": used,
        "exit_code": code,
        "fallback_exit_code": fallback,
        "output": output if used == runtime else output + "_modern",
    }


def _gate_is_fresh() -> bool:
    if not ACCOUNT_REPORT.exists():
        return False
    record = json.loads(ACCOUNT_REPORT.read_text())
    if not record.get("gate_passed"):
        return False
    checked = datetime.fromisoformat(record["checked_at_utc"])
    return datetime.now(timezone.utc) - checked < timedelta(hours=6)


def _upload_assets() -> None:
    marker = "\n".join(f"{remote} {local}" for remote, local in sorted(ASSETS.items()))
    try:
        existing = b"".join(volume.read_file("assets/SHA256SUMS")).decode()
    except Exception:
        existing = ""
    if existing == marker:
        print("Volume assets already match the local manifest.")
        return
    with volume.batch_upload() as batch:
        for remote, local in ASSETS.items():
            batch.put_file(PROJECT / local, remote)
        marker_path = PROJECT / "runs" / "batch01" / "volume_manifest.txt"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(marker)
        batch.put_file(marker_path, "assets/SHA256SUMS")


def _download(remote_dir: str, local_dir: Path) -> None:
    names = [
        "loading_report.json", "timing.json", "manifest.json", "codes.npz",
        "motion_raw_reward.npy", "motion_root_integrated.npy", "motion_global_positions.npy",
        "music_cond.npy", "skeleton_audio.mp4", "skeleton_audio_silent.mp4",
        "legacy_cuda.log", "modern_cuda.log", "resolved_config.json",
    ]
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        remote = f"{remote_dir}/{name}"
        try:
            payload = b"".join(volume.read_file(remote))
        except Exception:
            continue
        (local_dir / name).write_bytes(payload)


@app.local_entrypoint()
def main(run_id: str = "") -> None:
    if not _gate_is_fresh():
        raise SystemExit("Account gate is missing or older than 6 hours. Run scripts/check_modal_account.py first.")
    if not run_id:
        run_id = "smokeB-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"Run ID: {run_id}")
    _upload_assets()
    load_result = cpu_load.remote(run_id)
    print(json.dumps(load_result))
    if not load_result["selected_runtime"]:
        raise SystemExit("CPU load did not succeed on either runtime. GPU job was not started.")
    gpu_result = gpu_smoke.remote(run_id, load_result["selected_runtime"])
    print(json.dumps(gpu_result))
    local_dir = PROJECT / "runs" / "batch01" / run_id
    _download(gpu_result["output"], local_dir)
    (local_dir / "modal_calls.json").write_text(json.dumps({"cpu_load": load_result, "gpu": gpu_result}, indent=2))
    if gpu_result["exit_code"] != 0:
        raise SystemExit(f"GPU smoke failed with exit {gpu_result['exit_code']}")
