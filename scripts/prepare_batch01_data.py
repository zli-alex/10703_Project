"""Validate two registered AIST development pairs and prepare native features.

Uses downloaded records/audio only. Does not launch Modal, resample features,
pad real motion, execute model inference, or tune temporal alignment.
"""
from __future__ import annotations

import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import soundfile as sf

FPS = 60
AUDIT_START, AUDIT_END = 2, 6
SOURCES = [
    ("A", "mHO5", "gHO_sBM_cAll_d20_mHO5_ch01"),
    ("B", "mJB5", "gJB_sBM_cAll_d08_mJB5_ch01"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    baseline_path = ROOT / "configs/baseline_assets.json"
    baseline = json.loads(baseline_path.read_text())
    assert baseline["dataset"]["aist_application_status"].startswith("submitted_confirmed")
    report_dir = ROOT / "reports/batch01"
    feature_dir = ROOT / "data/batch01/music_feat_60fps"
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    excluded = set((ROOT / "third_party/Bailando/ignore_list.txt").read_text().splitlines())
    downloaded = json.loads((report_dir / "downloaded_assets.json").read_text())
    source_hashes = {row["path"]: row["sha256"] for row in downloaded}
    rows, pairs, plot_data = [], [], []
    for label, song, record_id in SOURCES:
        relative = f"assets/preprocessed_data/data/aistpp_test_full_wav/{record_id}.json"
        source_path = ROOT / relative
        assert source_path.name not in excluded, "Record is on upstream exclusion list"
        record = json.loads(source_path.read_text())
        assert record["id"] == record_id and record_id.split("_")[-2] == song
        assert sha256(source_path) == source_hashes[relative], "Downloaded source hash changed"
        music = np.asarray(record["music_array"], dtype=np.float64)
        motion = np.asarray(record["dance_array"], dtype=np.float64)
        assert music.ndim == motion.ndim == 2
        assert music.shape[1] == 438 and motion.shape[1] == 72
        assert np.isfinite(music).all() and np.isfinite(motion).all()
        assert set(np.unique(music[:, 53])).issubset({0.0, 1.0})
        assert music[:, 53].sum() > 0
        assert len(motion) >= AUDIT_END * FPS
        assert len(music) >= 20 * FPS + 40, "Insufficient 20-second generation plus music context"
        assert (AUDIT_END - AUDIT_START) * FPS % 8 == 0

        audio_path = ROOT / f"assets/audio/{song}.wav"
        audio, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
        audio_info = sf.info(audio_path)
        assert sample_rate == 48000 and audio.shape[1] == 2
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) <= 1
        audio_seconds = len(audio) / sample_rate
        feature_duration_error = len(music) / FPS - audio_seconds
        assert abs(feature_duration_error) <= 1 / FPS + 1e-9, "Audio/features clock mismatch"

        # Preserve all original feature channels; the policy selects columns 0:55.
        external_path = feature_dir / f"{song}.json"
        external_path.write_text(json.dumps({"id": song, "music_array": record["music_array"]}) + "\n")
        restored = np.asarray(json.loads(external_path.read_text())["music_array"])
        assert np.array_equal(restored, music), "Feature export altered values"

        points = motion.reshape(-1, 24, 3)
        root_relative = points - points[:, :1, :]
        speed = np.linalg.norm(np.diff(root_relative, axis=0), axis=2).mean(axis=1) * FPS
        audio_blocks = len(audio) // (sample_rate // FPS)
        rms = np.sqrt(np.mean(audio[:audio_blocks * 800].reshape(audio_blocks, 800, 2)**2, axis=(1, 2)))
        crop = slice(AUDIT_START * FPS, AUDIT_END * FPS)
        beat_frames = np.flatnonzero(music[:, 53])
        row = {
            "case": label, "song_id": song, "record_id": record_id,
            "audio_sample_rate_hz": sample_rate, "audio_channels": audio.shape[1],
            "audio_seconds": round(audio_seconds, 6),
            "music_frames": len(music), "music_channels_stored": music.shape[1],
            "music_channels_consumed": 55, "feature_rate_hz": FPS,
            "feature_duration_minus_audio_ms": round(feature_duration_error * 1000, 4),
            "motion_frames": len(motion), "motion_channels": motion.shape[1],
            "motion_seconds": round(len(motion) / FPS, 6),
            "full_song_music_beats": int(music[:, 53].sum()),
            "audit_seconds": AUDIT_END - AUDIT_START,
            "audit_music_beats": int(music[crop, 53].sum()),
            "audit_mean_joint_speed_source_units_per_s": float(speed[AUDIT_START * FPS: AUDIT_END * FPS - 1].mean()),
            "arrays_finite": True, "source_hash_verified": True,
            "on_upstream_exclusion_list": False,
        }
        rows.append(row)
        pairs.append({
            **row, "split": "development_not_held_out",
            "motion_record": relative, "audio": str(audio_path.relative_to(ROOT)),
            "audio_source": f"https://aistdancedb.ongaaccel.jp/v1.0.0/audio/wav/{song}.wav",
            "audio_sha256": sha256(audio_path), "audio_bytes": audio_path.stat().st_size,
            "audio_subtype": audio_info.subtype,
            "motion_record_sha256": source_hashes[relative],
            "external_features": str(external_path.relative_to(ROOT)),
            "external_features_sha256": sha256(external_path),
            "feature_values_preserved_exactly": True,
            "time_origin": "native record/audio origin; no fitted offset",
            "alignment_status": "clock/length consistency verified; perceptual synchronization still requires clip review",
        })
        plot_data.append((song, row, music, rms, speed, beat_frames))

    manifest = {
        "purpose": "Small Bailando++ development sample; not a generalization evaluation set",
        "frame_rate_hz": FPS, "policy_feature_slice": [0, 55],
        "music_beat_column_zero_based": 53,
        "generation_target_seconds": 20,
        "music_intervention_scoring_seconds": [5, 15],
        "real_motion_audit_seconds": [AUDIT_START, AUDIT_END],
        "real_motion_audit_frames": [AUDIT_START * FPS, AUDIT_END * FPS],
        "motion_dir": "assets/preprocessed_data/data/aistpp_test_full_wav",
        "external_wav": "data/batch01/music_feat_60fps",
        "rl_music_root": "assets/preprocessed_data/data/aistpp_test_full_wav",
        "compatibility_note": "External 60 Hz arrays are copied exactly from the authors' full-record JSONs. Separate external-feature files used for training are not available for bitwise comparison. No resampling or modern feature extraction was used.",
        "local_validation_versions": {name: version(name) for name in ["numpy", "scipy", "matplotlib", "soundfile"]},
        "pairs": pairs,
    }
    (ROOT / "configs/batch01_data.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (report_dir / "data_validation.json").write_text(json.dumps({"checks_passed": True, "pairs": pairs}, indent=2) + "\n")
    with (report_dir / "data_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.5), constrained_layout=True)
    for col, (song, row, music, rms, speed, beats) in enumerate(plot_data):
        ax = axes[0, col]
        durations = [row["audio_seconds"], len(music) / FPS, (len(speed) + 1) / FPS]
        ax.barh([2, 1, 0], durations, color=["#087f8c", "#52b4ae", "#7557a6"], height=0.55)
        ax.set_yticks([2, 1, 0], ["Audio", "Music features", "Real motion"])
        ax.axvspan(AUDIT_START, AUDIT_END, color="#f6bd60", alpha=0.35, label="Real-motion audit: 2–6 s")
        ax.axvline(20, color="#59636d", linestyle="--", linewidth=1.2, label="Generation target: 20 s")
        ax.set_xlim(0, 34)
        ax.set_title(f"{song} · available data")
        ax.set_xlabel("Seconds from record origin")
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, col]
        time_audio = (np.arange(len(rms)) + 0.5) / FPS
        time_features = np.arange(len(music)) / FPS
        ax.plot(time_audio, rms / max(rms.max(), 1e-12), color="#087f8c", label="Audio RMS (normalized)")
        ax.plot(time_features, music[:, 52] / max(music[:, 52].max(), 1e-12), color="#8bbfba", alpha=0.8, label="Stored onset strength (normalized)")
        for beat in beats / FPS:
            if AUDIT_START <= beat < AUDIT_END:
                ax.axvline(beat, color="#c97912", alpha=0.65, linewidth=0.8)
        ax.set_xlim(AUDIT_START, AUDIT_END)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title("Audio and supplied music-beat markers")
        ax.set_ylabel("Relative amplitude")
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[2, col]
        ax.plot((np.arange(len(speed)) + 0.5) / FPS, gaussian_filter1d(speed, 5), color="#7557a6", label="Root-relative joint speed (smoothed)")
        for beat in beats / FPS:
            if AUDIT_START <= beat < AUDIT_END:
                ax.axvline(beat, color="#c97912", alpha=0.65, linewidth=0.8)
        ax.set_xlim(AUDIT_START, AUDIT_END)
        ax.set_ylim(bottom=0)
        ax.set_title("Real motion over the same 4-second interval")
        ax.set_xlabel("Seconds from record origin")
        ax.set_ylabel("Source position units / second")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Bailando++ first-batch data validation\nNative 60 Hz features · two development songs · real motion is shorter than the audio", fontsize=15)
    fig.savefig(report_dir / "data_timeline.png", dpi=170)
    fig.savefig(report_dir / "data_timeline.pdf")
    plt.close(fig)

    baseline["dataset"].update({
        "download_status": "two_motion_records_and_matching_wavs_downloaded",
        "candidates_validated": True,
        "native_60hz_embedded_features_verified": True,
        "data_manifest": "configs/batch01_data.json",
        "validation_report": "reports/batch01/data_validation.md",
        "note": "Original 438-channel native features are preserved; policy consumes first 55. Shared real-motion audit is 2–6 s; music-only generation remains 20 s. Perceptual synchronization and model loading remain pending.",
    })
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    print(json.dumps({"checks_passed": True, "summary": rows}, indent=2))


if __name__ == "__main__":
    main()
