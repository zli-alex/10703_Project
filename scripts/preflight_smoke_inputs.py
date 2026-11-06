"""Check the song A crop and released padding before any GPU job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from infra.compat.released_test_loader import load_test_data_aist


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cfg = json.loads((PROJECT / "configs" / "smoke_batch01.json").read_text())
    data = json.loads((PROJECT / "configs" / "batch01_data.json").read_text())
    song = next(item for item in data["pairs"] if item["song_id"] == "mHO5")
    for path, expected in (
        (cfg["policy"]["path"], cfg["policy"]["sha256"]),
        (cfg["position_decoder"]["path"], cfg["position_decoder"]["sha256"]),
        (song["audio"], song["audio_sha256"]),
        (song["external_features"], song["external_features_sha256"]),
        (song["motion_record"], song["motion_record_sha256"]),
    ):
        actual = sha256_file(PROJECT / path)
        if actual != expected:
            raise SystemExit(f"Hash mismatch {path}: {actual}")

    with (PROJECT / cfg["external_features"]).open() as handle:
        features = json.load(handle)
    frames = cfg["native_music_frames"]
    music = np.array(features["music_array"])
    if music.shape != (song["music_frames"], song["music_channels_stored"]):
        raise SystemExit(f"Feature shape {music.shape}")
    cropped = features["music_array"][:frames]
    work = PROJECT / "runs" / "batch01" / "_preflight"
    external = work / "external"
    motion_dir = work / "motion"
    external.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)
    with (external / "mHO5.json").open("w") as handle:
        json.dump({"id": "mHO5", "music_array": cropped}, handle)
    motion_name = Path(cfg["motion_record"]).name
    link = motion_dir / motion_name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to((PROJECT / cfg["motion_record"]).resolve())

    loader = cfg["released_loader"]
    music_data, dance_data, names = load_test_data_aist(
        str(motion_dir), False, loader["move"], str(external),
        loader["external_wav_rate"], False, loader["wav_padding_frames"],
    )
    padded = music_data[0]
    dance = dance_data[0]
    expected = cfg["expected_shapes"]
    if padded.shape[0] != expected["padded_music_frames"] or dance.shape != (expected["padded_motion_frames"], 72):
        raise SystemExit(f"Unexpected loader shapes {padded.shape} {dance.shape}")
    native = padded[loader["wav_padding_frames"]:loader["wav_padding_frames"] + frames]
    if not np.array_equal(native, np.array(cropped)):
        raise SystemExit("Cropped features changed in the loader round-trip")
    sentinel_frame = 100
    original = float(native[sentinel_frame, 0])
    native_probe = padded.copy()
    native_probe[loader["wav_padding_frames"] + sentinel_frame, 0] = 12345.0
    cond = native_probe[loader["ds_rate"]:]
    found = np.where(cond[:, 0] == 12345.0)[0]
    expected_index = loader["wav_padding_frames"] + sentinel_frame - loader["ds_rate"]
    if found.tolist() != [expected_index]:
        raise SystemExit(f"Impulse landed at {found.tolist()}, expected {expected_index}")
    if not np.isfinite(dance).all() or not np.isfinite(padded).all():
        raise SystemExit("Non-finite staged arrays")
    print(json.dumps({
        "status": "ok",
        "names": names,
        "padded_music": list(padded.shape),
        "padded_dance": list(dance.shape),
        "impulse_cond_index": int(found[0]),
        "sentinel_replaced": original,
        "expected_codes": expected["total_codes_including_init"],
        "expected_frames": expected["decoded_frames"],
    }, indent=2))


if __name__ == "__main__":
    main()
