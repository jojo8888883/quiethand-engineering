#!/usr/bin/env python3
"""Prepare the single A800 Autopilot job for M3-v1.2 temporal semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
REMOTE = Path("/llm_jzm/dty_user/quiethand_m3")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-manifest-sha256", required=True)
    parser.add_argument("--input-sha256", required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def main() -> int:
    cfg = args()
    local_input = ROOT / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_INPUT.json"
    local_protocol = ROOT / "refine-logs/QUIETHAND_M3_PROTOCOL_V1_2_20260830_172950.md"
    local_runner = ROOT / "scripts/quiethand/m3_v1_2/run_temporal_semantic.py"
    local_contract = ROOT / "quiethand/m3_v1_2_contract.py"
    local_common = ROOT / "scripts/quiethand/m3_jobs/runtime_common.py"
    local_resource = ROOT / "artifacts/quiethand/m3/QH_M3_CHECKPOINT_LANDING.json"
    if sha(local_input) != cfg.input_sha256 or sha(local_resource) != cfg.resource_manifest_sha256:
        raise SystemExit("requested frozen input/resource digest does not match")
    remote_by_local = {
        local_input: REMOTE / local_input.relative_to(ROOT),
        local_protocol: REMOTE / local_protocol.relative_to(ROOT),
        local_runner: REMOTE / local_runner.relative_to(ROOT),
        local_contract: REMOTE / local_contract.relative_to(ROOT),
        local_common: REMOTE / local_common.relative_to(ROOT),
        local_resource: REMOTE / local_resource.relative_to(ROOT),
    }
    checks = "\n".join(f"{sha(local)}  {remote}" for local, remote in remote_by_local.items())
    job_id = "qh-m3-v12-temporal-semantic"
    job_dir = ROOT / "artifacts/quiethand/m3_v1_2/a800_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    remote_job = REMOTE / "jobs" / job_id
    output = REMOTE / "results_v1_2" / "temporal_semantic"
    command = [
        str(REMOTE / "envs/semantic/bin/python"), "-I", str(REMOTE / local_runner.relative_to(ROOT)),
        "--workspace", str(REMOTE), "--input", str(REMOTE / local_input.relative_to(ROOT)),
        "--input-sha256", cfg.input_sha256,
        "--model", str(REMOTE / "external_data/quiethand_m3_resources/semantic"),
        "--resource-root", str(REMOTE / "external_data/quiethand_m3_resources"),
        "--resource-manifest", str(REMOTE / local_resource.relative_to(ROOT)),
        "--resource-manifest-sha256", cfg.resource_manifest_sha256,
        "--output", str(output),
    ]
    run = (
        "#!/usr/bin/env bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n"
        "export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        "sha256sum --check --strict <<'QH_M3_V12_FILES'\n" + checks +
        "\nQH_M3_V12_FILES\nexec " + " ".join(quote(part) for part in command) + "\n"
    )
    run_path = job_dir / "run.sh"
    run_path.write_text(run, encoding="utf-8")
    run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR)
    context = {
        "schema": "quiethand.m3_v1_2.a800_job_context.v1", "job_id": job_id,
        "adapter": "temporal_semantic", "calibration_video_count": 30,
        "input_sha256": cfg.input_sha256, "resource_manifest_sha256": cfg.resource_manifest_sha256,
        "training_allowed": False, "evaluation_results_allowed": False,
    }
    (job_dir / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "id": job_id, "display_name": "QuietHand M3-v1.2 complete-action semantics",
        "managed": True, "priority": 80, "cwd": str(REMOTE), "command": ["bash", str(remote_job / "run.sh")],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"},
        "gpu": {
            "allowed": list(range(8)), "mode": "idle-first-shared", "idle_memory_mib": 500,
            "expected_peak_mib": 73728, "reserve_mib": 4096, "warning_free_mib": 6144,
            "stop_free_mib": 4096, "pressure_checks": 2,
            "foreign_owner_policy": "coexist-until-pressure", "xla_memory_fraction": "0.30", "settle_seconds": 5,
        },
        "preflight": {
            "paths": [str(REMOTE / "envs/semantic/bin/python"), str(remote_job / "run.sh"), str(REMOTE / local_input.relative_to(ROOT))],
            "disk_min_free_gib": {"/llm_jzm": 15},
            "commands": [[str(REMOTE / "envs/semantic/bin/python"), "-I", "-c", "import torch; assert torch.cuda.is_available()"]],
            "timeout_seconds": 120,
        },
        "execution": {"resource_retry_seconds": 60, "oom_retry_seconds": 120, "max_oom_requeues": 1},
        "completion": {"type": "artifact", "path": str(output / "audit.json")},
        "progress": {"status_path": str(output / "status.json")},
        "science": {"context_path": str(remote_job / "context.json"), "protocol_path": str(REMOTE / local_protocol.relative_to(ROOT))},
    }
    (job_dir / "job.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
