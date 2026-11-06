"""Write the smoke-test report from a completed run directory."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]


def mib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 2):.1f} MiB"


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    timing = json.loads((run_dir / "timing.json").read_text())
    loading = json.loads((run_dir / "loading_report.json").read_text())
    account_path = PROJECT / "reports" / "batch01" / "modal_account.json"
    cost = json.loads((run_dir / "cost.json").read_text()) if (run_dir / "cost.json").exists() else {}
    memory = timing["memory"]["after_decode"]
    policy = loading["policy_load"]
    decoder = loading["decoder_load"]
    critic = loading["actor_critic_diagnostic"]
    lines = [
        "# Supervised Bailando++ smoke test",
        "",
        f"Run directory: `{run_dir.relative_to(PROJECT)}`.",
        "",
        "Generation used `CrossCondGPT2MW.sample` with top-1 selection, the released music padding, and the position decoder. The actor-critic class was loaded only to account for its new critic parameters.",
        "",
        "## Loading",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Policy file / stored epoch | `epoch_400.pt` / {loading['policy_stored_epoch']} |",
        f"| Policy SHA256 | `{loading['policy_sha256']}` |",
        f"| Policy missing keys | {len(policy['missing'])} |",
        f"| Policy unexpected keys | {len(policy['unexpected'])} |",
        f"| Decoder missing keys | {len(decoder['missing'])} |",
        f"| Decoder unexpected keys | {len(decoder['unexpected'])} |",
        f"| Critic tensors absent from the supervised file | {critic.get('critic_tensors_initialized', 'not loaded')} |",
        f"| Critic max change while loading actor weights | {critic.get('critic_max_abs_change_during_load', 'n/a')} |",
        f"| Runtime | {loading['versions'].get('torch')} / {loading['versions'].get('gpu_name', 'cpu')} |",
        "",
        "## Timing, memory, and cost",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Checkpoint read | {timing['read_checkpoints_seconds']:.2f} s |",
        f"| Build and load | {timing['build_and_load_seconds']:.2f} s |",
        f"| Encode and context probe | {timing['encode_and_context_seconds']:.2f} s |",
        f"| Sample | {timing['sample_seconds']:.2f} s |",
        f"| Decode | {timing['decode_seconds']:.2f} s |",
        f"| Render | {timing['render_seconds']:.2f} s |",
        f"| Peak GPU allocated | {mib(memory.get('peak_allocated_bytes'))} |",
        f"| Peak GPU reserved | {mib(memory.get('peak_reserved_bytes'))} |",
        f"| Metered cost delta | {cost.get('metered_delta_usd', 'pending')} |",
        f"| Billed cost delta | {cost.get('billed_delta_usd', 'pending')} |",
        "",
        "## Output",
        "",
        f"- Initial upper/lower codes: `{manifest['init_upper_lower']}`.",
        f"- Codes: {manifest['code_length']} including initialization. Decoded frames: {manifest['decoded_frames']} ({manifest['duration_seconds']:.3f} s).",
        f"- Code range: {manifest['code_min_max']}. Unique upper/lower codes: {manifest['unique_codes']['upper']} / {manifest['unique_codes']['lower']}.",
        f"- Validation failures: {manifest['validation_failures'] or 'none'}.",
        f"- Clip: `{manifest['render']['path']}`. Audio muxed: {manifest['render']['audio_muxed']}.",
        "",
        "Raw reward coordinates are `motion_raw_reward.npy`. Root integration follows the released prefix sum and is stored in `motion_root_integrated.npy`. Global joints used for the video are `motion_global_positions.npy`. The video recenters the current root for display.",
        "",
        "The displayed clock starts at the native audio origin. Generated code k+1 is aligned with native music frames `[8+8k, 16+8k)`, and the music Transformer also attends across its +/- 40-frame band. Perceptual beat alignment is not established by the container duration check.",
        "",
        f"Account record: `{account_path.relative_to(PROJECT)}`.",
        "",
    ]
    destination = PROJECT / "reports" / "batch01" / "smoke_test.md"
    destination.write_text("\n".join(lines))
    print(destination)


if __name__ == "__main__":
    main()
