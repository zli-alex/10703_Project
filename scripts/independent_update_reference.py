"""Independent arithmetic verification of M4 update objectives.

Uses only NumPy and SciPy special functions; no PyTorch or released loss functions.
Reads saved trace NPZs and policy/decoder hashes from baseline_assets.json.

Outputs reports/batch01/update_reference.json.

Usage (project 10703 Conda env):
    python scripts/independent_update_reference.py traceE-20261106T221045Z
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "batch01"
RUNS_DIR = ROOT / "runs" / "batch01"
BASELINE = json.loads((ROOT / "configs" / "baseline_assets.json").read_text())
CFG = json.loads((ROOT / "configs" / "update_batch01.json").read_text())


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(path: Path, expected: str) -> dict:
    actual = sha256_file(path)
    return {"path": str(path), "expected": expected, "actual": actual, "match": actual == expected}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else CFG["source_trace"]
    run_dir = RUNS_DIR / run_name

    # 1. Verify artifact hashes
    hash_checks = {}
    critic_pt = run_dir / "critic_initial.pt"
    hash_checks["critic_initial"] = check_hash(critic_pt, CFG["critic_initial_sha256_expected"])
    policy_path = ROOT / "assets" / "bailandopp" / "cc_gpt_music_trans" / "ckpt" / "epoch_400.pt"
    hash_checks["policy"] = check_hash(policy_path, CFG["policy_sha256_expected"])
    decoder_path = ROOT / "assets" / "bailando_original" / "experiments" / "sep_vqvae_root" / "ckpt" / "epoch_500.pt"
    hash_checks["decoder"] = check_hash(decoder_path, CFG["decoder_sha256_expected"])

    for label, r in hash_checks.items():
        status = "OK" if r["match"] else "MISMATCH"
        print(f"Hash {label}: {status}  {r['actual'][:16]}...")

    # 2. Load windows
    windows = {}
    for idx in CFG["window_starts"]:
        npz_path = run_dir / f"window_{idx:03d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing trace NPZ: {npz_path}")
        windows[idx] = np.load(npz_path)
        hash_checks[f"window_{idx:03d}_npz"] = {"path": str(npz_path), "sha256": sha256_file(npz_path)}

    # 3. Independent NLL, TD, and loss per window
    gamma = CFG["gamma"]
    window_results = []
    all_nll = []
    all_delta = []
    all_mask = []
    all_td_max_abs_err = []
    all_nll_max_abs_err = []

    for idx, d in windows.items():
        logits_up = d["actor_logits_up"].astype(np.float64)   # (28, 512)
        logits_down = d["actor_logits_down"].astype(np.float64)
        targets = d["actor_target"]                            # (28, 2)
        rewards = d["rewards"].astype(np.float64)             # (28,)
        values = d["values"].astype(np.float64)               # (29,)
        mask = d["mask"].astype(np.float64)                   # (28,)
        saved_td = d["td_targets"].astype(np.float64)         # (28,)
        saved_nll = d["actor_nll"].astype(np.float64)         # (28,)

        # 3a. Cross-entropy per step: CE = logsumexp(logits) - logits[target]
        ce_up = logsumexp(logits_up, axis=-1) - logits_up[np.arange(28), targets[:, 0]]
        ce_down = logsumexp(logits_down, axis=-1) - logits_down[np.arange(28), targets[:, 1]]
        indep_nll = ce_up + ce_down

        # 3b. TD targets and errors
        indep_td = rewards + gamma * values[1:]   # stop_gradient of sum
        delta = indep_td - values[:-1]            # gradient on values[:-1] only

        # 3c. Actor and critic losses for this window
        actor_loss_w = float(np.sum(mask * indep_nll * delta) / mask.sum())
        critic_loss_w = float(np.mean(delta ** 2))

        # 3d. Agreement vs saved
        td_err = np.abs(indep_td - saved_td)
        nll_err = np.abs(indep_nll - saved_nll)
        td_max = float(td_err.max())
        nll_max = float(nll_err.max())
        all_td_max_abs_err.append(td_max)
        all_nll_max_abs_err.append(nll_max)

        atol = CFG["atol"]
        rtol = CFG["rtol"]
        # Use np.allclose(atol=1e-6, rtol=1e-5) per PROJECT_PLAN.md specification.
        # Window 20 NLL max error 1.84e-6 exceeds pure atol but passes combined rtol check
        # (saved values ~0-5, rtol=1e-5 -> relative tolerance covers the gap).
        td_pass = bool(np.allclose(indep_td, saved_td, atol=atol, rtol=rtol))
        nll_pass = bool(np.allclose(indep_nll, saved_nll, atol=atol, rtol=rtol))

        all_nll.append(indep_nll)
        all_delta.append(delta)
        all_mask.append(mask)

        window_results.append({
            "window_start": int(idx),
            "actor_loss": actor_loss_w,
            "critic_loss": critic_loss_w,
            "mask_sum": int(mask.sum()),
            "td_max_abs_err_vs_saved": td_max,
            "nll_max_abs_err_vs_saved": nll_max,
            "td_agreement_pass": td_pass,
            "nll_agreement_pass": nll_pass,
        })
        print(f"Window {idx}: actor_loss={actor_loss_w:.8f}  critic_loss={critic_loss_w:.8f}  "
              f"td_err={td_max:.3e}  nll_err={nll_max:.3e}  "
              f"TD={'PASS' if td_pass else 'FAIL'}  NLL={'PASS' if nll_pass else 'FAIL'}")

    # 4. Stacked-batch losses (batch of 2)
    all_nll_cat = np.concatenate(all_nll)         # (56,)
    all_delta_cat = np.concatenate(all_delta)      # (56,)
    all_mask_cat = np.concatenate(all_mask)        # (56,)

    batch_actor_loss = float(np.sum(all_mask_cat * all_nll_cat * all_delta_cat) / all_mask_cat.sum())
    batch_critic_loss = float(np.mean(all_delta_cat ** 2))
    print(f"\nStacked batch (2 windows): actor_loss={batch_actor_loss:.8f}  critic_loss={batch_critic_loss:.8f}")

    # 5. Schedule table: epoch 1 and 2, batches 0-9
    actor_iter = CFG["actor_iter"]
    critic_iter = CFG["critic_iter"]
    pretrain_epoch = CFG["pretrain_critic_epoch"]
    schedule = []
    for epoch in [1, 2]:
        for batch_i in range(actor_iter + critic_iter):
            is_actor_epoch = (epoch >= pretrain_epoch)
            is_actor_batch = (batch_i % (actor_iter + critic_iter)) < actor_iter
            branch = "actor" if (is_actor_epoch and is_actor_batch) else "critic"
            schedule.append({"epoch": epoch, "batch_i": batch_i, "branch": branch})
    print("\nSchedule (epoch 1-2, batch 0-9):")
    for s in schedule[:10]:
        print(f"  epoch={s['epoch']} batch={s['batch_i']} -> {s['branch']}")

    # 6. Greedy replay check on saved arrays
    greedy_checks = {}
    for idx, d in windows.items():
        replay = d["replay_greedy"]     # (28, 2) - original context greedy
        actor_target = d["actor_target"]  # (28, 2)
        actor_window_greedy = d["actor_window_greedy"]  # (28, 2) - teacher-forced
        orig_match = int(np.all(replay == actor_target))
        teacher_mismatch = int(np.any(actor_window_greedy != actor_target, axis=1).sum())
        greedy_checks[f"window_{idx}"] = {
            "original_context_greedy_matches_all_28": bool(orig_match),
            "teacher_forced_mismatch_count": teacher_mismatch,
        }
        print(f"Window {idx}: orig-context matches={orig_match==1} teacher-forced mismatches={teacher_mismatch}/28")

    # 7. Worked example: window 20, local action 27
    w20 = windows[20]
    j = 27
    wa_upper = int(w20["actor_target"][j, 0])
    wa_lower = int(w20["actor_target"][j, 1])
    wa_reward = float(w20["rewards"][j])
    wa_value = float(w20["values"][j])
    wa_bootstrap = float(w20["values"][j + 1])
    wa_saved_nll = float(w20["actor_nll"][j])
    wa_mask = float(w20["mask"][j])

    indep_nll_w20 = (logsumexp(w20["actor_logits_up"].astype(np.float64), axis=-1)
                     - w20["actor_logits_up"].astype(np.float64)[j, wa_upper]
                     + logsumexp(w20["actor_logits_down"].astype(np.float64), axis=-1)
                     - w20["actor_logits_down"].astype(np.float64)[j, wa_lower])
    wa_indep_nll = float(indep_nll_w20[j])  # single step
    wa_indep_td = wa_reward + gamma * wa_bootstrap
    wa_indep_delta = wa_indep_td - wa_value
    wa_nll_err = abs(wa_indep_nll - wa_saved_nll)

    worked_example = {
        "window": 20,
        "local_action": 27,
        "target_upper": wa_upper,
        "target_lower": wa_lower,
        "reward": wa_reward,
        "value_current": wa_value,
        "value_next": wa_bootstrap,
        "indep_td_target": wa_indep_td,
        "saved_td_target": float(w20["td_targets"][j]),
        "indep_delta": wa_indep_delta,
        "indep_nll": wa_indep_nll,
        "saved_nll": wa_saved_nll,
        "nll_max_abs_err": wa_nll_err,
        "mask": wa_mask,
        "this_action_actor_contribution": wa_mask * wa_indep_nll * wa_indep_delta,
    }
    print(f"\nWorked action (w20/j27): td={wa_indep_td:.8f}  delta={wa_indep_delta:.8f}  "
          f"nll={wa_indep_nll:.8f}  nll_err={wa_nll_err:.3e}")

    # 8. Write JSON
    output = {
        "run_name": run_name,
        "batch_description": CFG["batch_description"],
        "hash_checks": hash_checks,
        "per_window_losses": window_results,
        "stacked_batch": {
            "num_windows": len(windows),
            "actor_loss": batch_actor_loss,
            "critic_loss": batch_critic_loss,
            "note": "diagnostic batch of 2; not released batch64",
        },
        "schedule": schedule,
        "greedy_checks": greedy_checks,
        "worked_example": worked_example,
        "tolerances": {"atol": CFG["atol"], "rtol": CFG["rtol"]},
        "all_hash_checks_pass": all(r["match"] for r in hash_checks.values() if isinstance(r, dict) and "match" in r),
        "all_td_checks_pass": all(w["td_agreement_pass"] for w in window_results),
        "all_nll_checks_pass": all(w["nll_agreement_pass"] for w in window_results),
    }
    out_path = REPORT_DIR / "update_reference.json"
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\nWrote {out_path}")

    all_pass = output["all_hash_checks_pass"] and output["all_td_checks_pass"] and output["all_nll_checks_pass"]
    print(f"Local reference overall: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
