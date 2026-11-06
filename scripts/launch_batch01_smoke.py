"""Local preflight, account gate, then one Modal smoke invocation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
CONDA = "/Users/zhengfeili/opt/anaconda3/bin/conda"
PYTHON = [CONDA, "run", "--no-capture-output", "-n", "10703", "python"]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def modal_json(args: list[str]) -> dict:
    completed = subprocess.run(
        [CONDA, "run", "--no-capture-output", "-n", "10703", "modal", *args],
        cwd=PROJECT, check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr[-500:], "stdout": completed.stdout[-500:]}
    text = completed.stdout
    start = min(index for index in (text.find("{"), text.find("[")) if index >= 0)
    return json.loads(text[start:])


def main() -> None:
    run(PYTHON + ["scripts/preflight_smoke_inputs.py"])
    run(PYTHON + ["scripts/check_modal_account.py"])
    before = modal_json(["billing", "summary", "--json"])
    run([CONDA, "run", "--no-capture-output", "-n", "10703", "modal", "run", "infra/modal_app.py"])
    after = modal_json(["billing", "summary", "--json"])
    report = modal_json(["billing", "report", "--for", "today", "--show-resources", "--json"])
    record = {"before": before, "after": after, "today": report}
    destination = PROJECT / "runs" / "batch01" / "billing_around_launch.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"billing_record": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
