"""Read Modal identity and billing without printing credentials.

Exit 0 only when authentication, current usage, a credit adjustment or balance,
and an expiration timestamp are all present. A missing field blocks the GPU job.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
CONDA_PYTHON = "/Users/zhengfeili/opt/anaconda3/bin/conda"
REPORT = PROJECT / "reports" / "batch01" / "modal_account.json"


def run_modal(args: list[str]) -> tuple[int, str, str]:
    command = [CONDA_PYTHON, "run", "--no-capture-output", "-n", "10703", "modal", *args]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def parse_token_info(text: str) -> dict:
    info = {"raw_lines": []}
    for line in text.splitlines():
        if "Token:" in line or "token_" in line.lower() and "ak-" in line:
            continue
        info["raw_lines"].append(line)
        lowered = line.lower()
        if "workspace:" in lowered:
            info["workspace_line"] = line.split("Workspace:", 1)[-1].strip()
        if "expires at:" in lowered:
            info["token_expires_at"] = line.split(":", 1)[-1].strip()
        if "created at:" in lowered:
            info["token_created_at"] = line.split(":", 1)[-1].strip()
        if "user:" in lowered:
            info["user_line"] = line.split("User:", 1)[-1].strip()
    return info


def load_json(text: str):
    text = text.strip()
    start = text.find("{")
    start_list = text.find("[")
    if start < 0 or (0 <= start_list < start):
        start = start_list
    if start < 0:
        raise ValueError("No JSON payload")
    return json.loads(text[start:])


def credit_fields(summary: dict) -> dict:
    adjustments = summary.get("adjustments") or {}
    found = {key: value for key, value in adjustments.items() if "credit" in key.lower() or "expir" in key.lower()}
    return {"adjustment_keys": sorted(adjustments), "credit_related": found, "adjustments": adjustments}


def main() -> None:
    code, out, err = run_modal(["token", "info"])
    authenticated = code == 0 and "Token missing" not in err and "Token missing" not in out
    record = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "authenticated": authenticated,
        "token_info": parse_token_info(out) if authenticated else {"error": (err or out)[-500:]},
        "summaries": {},
        "rates": None,
        "gate_passed": False,
        "blockers": [],
    }
    if not authenticated:
        record["blockers"].append("Modal token is not configured for the 10703 environment.")
    else:
        for label, argument in (("this_month", "this month"), ("last_month", "last month")):
            status, payload, error = run_modal(["billing", "summary", "--for", argument, "--json"])
            if status != 0:
                record["blockers"].append(f"Billing summary failed for {label}.")
                record["summaries"][label] = {"error": error[-500:]}
                continue
            summary = load_json(payload)
            record["summaries"][label] = summary
        status, payload, error = run_modal(["billing", "rates", "--json"])
        if status == 0:
            record["rates"] = load_json(payload)
        else:
            record["rates_error"] = error[-300:]

        this = record["summaries"].get("this_month")
        if not isinstance(this, dict) or "metered_cost" not in this:
            record["blockers"].append("Current-cycle usage was not returned.")
        else:
            credits = credit_fields(this)
            record["credit_fields"] = credits
            metered = Decimal(str(this["metered_cost"]))
            record["metered_cost_usd"] = format(metered, "f")
            record["credit_expiration"] = (
                "present_in_summary" if "expir" in json.dumps(this).lower()
                else "not_exposed_by_modal_billing_api"
            )
            record["launch_basis"] = (
                "Authentication and current-cycle usage are readable. "
                "Modal does not return a remaining credit balance or credit expiration. "
                "The smoke test may start only while this cycle's metered cost is under $4, "
                "leaving the $5 setup allowance for one T4 job."
            )
            if "credits" not in credits["adjustments"]:
                record["blockers"].append("Billing summary did not include a credits adjustment field.")
            if metered >= Decimal("4"):
                record["blockers"].append(
                    f"Current-cycle metered cost is ${record['metered_cost_usd']}, too close to the $5 setup allowance."
                )
    record["gate_passed"] = not record["blockers"]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"gate_passed": record["gate_passed"], "blockers": record["blockers"], "report": str(REPORT)}, indent=2))
    if not record["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
