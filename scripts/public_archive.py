"""List or selectively extract the authors' public ZIPs using HTTP byte ranges.

Never falls back to an unbounded full-file download. ZIP CRC and extracted SHA256
are checked/recorded without unpickling checkpoints. Run from the project root.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import time
import zipfile

import requests


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int, block_size: int = 4 * 1024**2):
        self.url, self.size, self.block_size = url, size, block_size
        self.pos = 0
        self.cache_start = -1
        self.cache = b""
        self.transferred = 0
        self.session = requests.Session()
        self.session.headers["Accept-Encoding"] = "identity"

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}
        target = bases[whence] + offset
        if target < 0:
            raise ValueError("Negative seek")
        self.pos = target
        return target

    def read(self, count=-1):
        if count is None or count < 0:
            count = self.size - self.pos
        remaining = min(count, max(0, self.size - self.pos))
        parts = []
        while remaining:
            start = self.pos // self.block_size * self.block_size
            if start != self.cache_start:
                end = min(start + self.block_size, self.size) - 1
                with self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"},
                    stream=True, timeout=(15, 90),
                ) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor exact range: " + str(response.headers.get("Content-Range")))
                    data = response.raw.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete or oversized byte-range response")
                self.cache_start, self.cache = start, data
                self.transferred += len(data)
            offset = self.pos - self.cache_start
            chunk = self.cache[offset: offset + remaining]
            parts.append(chunk)
            self.pos += len(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    def close(self):
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", choices=["bailandopp", "bailando_original", "preprocessed_data"])
    parser.add_argument("--member", action="append", default=[], help="Exact ZIP member to extract; otherwise only list")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/public_assets.json").read_text())
    asset = config["archives"][args.archive]
    url = "https://drive.usercontent.google.com/download?id=" + asset["drive_id"] + "&export=download&confirm=t"
    report_dir = project / "reports/batch01"
    report_dir.mkdir(parents=True, exist_ok=True)
    destination = project / "assets" / args.archive
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    with RangeReader(url, asset["reported_bytes"]) as remote, zipfile.ZipFile(remote) as archive:
        members = [
            {"name": info.filename, "bytes": info.file_size,
             "compressed_bytes": info.compress_size, "crc32": f"{info.CRC:08x}",
             "method": info.compress_type, "directory": info.is_dir()}
            for info in archive.infolist()
        ]
        (report_dir / f"{args.archive}_index.json").write_text(json.dumps(
            {"source": asset, "members": members}, indent=2) + "\n")
        print(json.dumps({"archive": args.archive, "members": len(members),
                          "expanded_bytes": sum(m["bytes"] for m in members)}), flush=True)
        for name in args.member:
            info = archive.getinfo(name)
            target = (destination / name).resolve()
            if not target.is_relative_to(destination.resolve()) or info.is_dir():
                raise ValueError("Invalid output member")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Refusing a ZIP symlink")
            if info.file_size > shutil.disk_usage(destination).free - 2 * 1024**3:
                raise RuntimeError("Insufficient free space with 2 GiB reserve")
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            if partial.exists():
                raise FileExistsError(f"Inspect existing partial file before retrying: {partial}")
            digest = hashlib.sha256()
            written = 0
            last_progress = time.monotonic()
            with archive.open(info) as source, partial.open("xb") as output:
                while chunk := source.read(1024**2):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if time.monotonic() - last_progress > 20:
                        print(f"{name}: {written}/{info.file_size} bytes", flush=True)
                        last_progress = time.monotonic()
            if written != info.file_size:
                raise RuntimeError("Extracted length mismatch")
            partial.rename(target)
            result = {"archive": args.archive, "member": name,
                      "path": str(target.relative_to(project)), "bytes": written,
                      "sha256": digest.hexdigest(), "zip_crc_verified": True}
            results.append(result)
            print(json.dumps(result), flush=True)
            digest_path = report_dir / "downloaded_assets.json"
            with (report_dir / ".asset_inventory.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                previous = json.loads(digest_path.read_text()) if digest_path.exists() else []
                replacement = digest_path.with_suffix(".json.tmp")
                replacement.write_text(json.dumps(previous + [result], indent=2) + "\n")
                replacement.replace(digest_path)
        print(json.dumps({"network_bytes": remote.transferred,
                          "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
