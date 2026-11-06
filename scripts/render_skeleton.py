"""Draw a 24-joint stick figure and mux it with source audio.

The projection uses the first two coordinates as x and y, matching the axis
choice in the released 2D visualizer. The camera subtracts the current root so
the body stays in frame. Saved global positions are not recentered.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


EDGES = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
)


def render_skeleton_mp4(
    joints_global: np.ndarray,
    output_path: Path,
    audio_path: Path | None,
    frame_rate: int = 60,
    size: tuple[int, int] = (640, 480),
) -> dict:
    if joints_global.ndim != 3 or joints_global.shape[1:] != (24, 3):
        raise ValueError(f"Expected [T, 24, 3] joints, got {joints_global.shape}")
    if not np.isfinite(joints_global).all():
        raise ValueError("Skeleton joints are not finite")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + "_silent.mp4")
    width, height = size
    relative = joints_global - joints_global[:, :1, :]
    span = float(np.percentile(np.abs(relative[:, :, :2]), 99))
    span = max(span, 1e-4)
    scale = 0.38 * min(width, height) / span

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(frame_rate), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index, pose in enumerate(relative):
            image = Image.new("RGB", size, (18, 18, 22))
            draw = ImageDraw.Draw(image)
            points = []
            for joint in pose:
                x = width * 0.5 + float(joint[0]) * scale
                y = height * 0.42 - float(joint[1]) * scale
                points.append((x, y))
            order = np.argsort(pose[:, 2])
            for a, b in EDGES:
                color = (220, 230, 240) if min(a, b) == 0 else (120, 190, 255)
                draw.line([points[a], points[b]], fill=color, width=3)
            for joint_index in order:
                x, y = points[int(joint_index)]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 210, 120))
            seconds = index / frame_rate
            draw.text((12, 12), f"t={seconds:05.2f}s  frame {index}", fill=(230, 230, 230))
            process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg skeleton encode failed with status {return_code}")

    muxed = False
    if audio_path is not None:
        mux = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_path), "-i", str(audio_path),
            "-t", f"{joints_global.shape[0] / frame_rate:.6f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_path),
        ]
        mux_result = subprocess.run(mux, check=False)
        muxed = mux_result.returncode == 0
        if not muxed:
            output_path.write_bytes(silent_path.read_bytes())
    else:
        output_path.write_bytes(silent_path.read_bytes())

    probe = _probe(output_path)
    return {
        "path": str(output_path),
        "silent_path": str(silent_path),
        "frames": int(joints_global.shape[0]),
        "frame_rate_hz": frame_rate,
        "audio_muxed": muxed,
        "display_recenters_current_root": True,
        "projection": "orthographic x right, y up; z used only for draw order",
        "ffprobe": probe,
    }


def _probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,duration,nb_frames,r_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {"available": False, "stderr": result.stderr[-500:]}
    return {"available": True, "json": result.stdout}
