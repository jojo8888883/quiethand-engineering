#!/usr/bin/env python3
"""Prepare the authorized evaluation semantic job for A800 Autopilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
LOCAL_GATE = ROOT / "artifacts/quiethand/m3_5/evaluation_router_blind"
REMOTE_WORKSPACE = Path("/mnt/sdc/dty_user/quiethand_m3_eval")
RESOURCE_WORKSPACE = Path("/llm_jzm/dty_user/quiethand_m3")
JOB_ID = "qh-m3-eval-router-semantic"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    local_input = LOCAL_GATE / "QH_M3_EVALUATION_TEMPORAL_INPUT.json"
    local_protocol = LOCAL_GATE / "PLAN.md"
    local_rule = LOCAL_GATE / "router_rule.json"
    local_runner = ROOT / "scripts/quiethand/m3_eval_router/run_temporal_semantic.py"
    local_common = ROOT / "scripts/quiethand/m3_jobs/runtime_common.py"
    local_contract = ROOT / "quiethand/m3_v1_2_contract.py"
    local_resource = ROOT / "artifacts/quiethand/m3/QH_M3_CHECKPOINT_LANDING.json"
    payload = json.loads(local_input.read_text(encoding="utf-8"))
    if (
        payload.get("evaluation_event_count") != 30
        or payload.get("evaluation_model_results_authorized") is not True
        or payload.get("evaluation_model_results_opened") is not False
        or payload.get("training_performed") is not False
    ):
        raise SystemExit("evaluation input authorization envelope is invalid")

    job_dir = LOCAL_GATE / "a800_jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=False)
    remote_job = REMOTE_WORKSPACE / "jobs" / JOB_ID
    remote_input = REMOTE_WORKSPACE / local_input.relative_to(ROOT)
    remote_protocol = REMOTE_WORKSPACE / local_protocol.relative_to(ROOT)
    remote_rule = REMOTE_WORKSPACE / local_rule.relative_to(ROOT)
    remote_runner = REMOTE_WORKSPACE / local_runner.relative_to(ROOT)
    remote_common = REMOTE_WORKSPACE / local_common.relative_to(ROOT)
    remote_contract = REMOTE_WORKSPACE / local_contract.relative_to(ROOT)
    remote_resource = RESOURCE_WORKSPACE / local_resource.relative_to(ROOT)
    resource_root = RESOURCE_WORKSPACE / "external_data/quiethand_m3_resources"
    python = RESOURCE_WORKSPACE / "envs/semantic/bin/python"
    output = REMOTE_WORKSPACE / "results/temporal_semantic"
    checks = {
        remote_input: sha(local_input),
        remote_protocol: sha(local_protocol),
        remote_rule: sha(local_rule),
        remote_runner: sha(local_runner),
        remote_common: sha(local_common),
        remote_contract: sha(local_contract),
        remote_resource: sha(local_resource),
    }
    check_text = "\n".join(f"{digest}  {path}" for path, digest in checks.items())
    command = [
        str(python), "-I", str(remote_runner),
        "--workspace", str(REMOTE_WORKSPACE),
        "--input", str(remote_input),
        "--input-sha256", sha(local_input),
        "--model", str(resource_root / "semantic"),
        "--resource-root", str(resource_root),
        "--resource-manifest", str(remote_resource),
        "--resource-manifest-sha256", sha(local_resource),
        "--output", str(output),
    ]
    run = (
        "#!/usr/bin/env bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n"
        "export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        "sha256sum --check --strict <<'QH_M3_EVAL_FILES'\n"
        + check_text
        + "\nQH_M3_EVAL_FILES\nexec "
        + " ".join(quote(part) for part in command)
        + "\n"
    )
    run_path = job_dir / "run.sh"
    run_path.write_text(run, encoding="utf-8")
    run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR)
    context = {
        "schema": "quiethand.m3_5.evaluation_router_job_context.v1",
        "job_id": JOB_ID,
        "adapter": "temporal_semantic",
        "evaluation_event_count": 30,
        "evaluation_results_authorized": True,
        "training_allowed": False,
        "input_sha256": sha(local_input),
        "protocol_sha256": sha(local_protocol),
        "router_rule_sha256": sha(local_rule),
        "resource_manifest_sha256": sha(local_resource),
    }
    write_json(job_dir / "context.json", context)
    manifest = {
        "schema_version": 1,
        "id": JOB_ID,
        "display_name": "QuietHand 30-video evaluation semantic blind test",
        "managed": True,
        "priority": 80,
        "cwd": str(REMOTE_WORKSPACE),
        "command": ["bash", str(remote_job / "run.sh")],
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        },
        "gpu": {
            "allowed": list(range(8)),
            "mode": "idle-first-shared",
            "idle_memory_mib": 500,
            "expected_peak_mib": 73728,
            "reserve_mib": 4096,
            "warning_free_mib": 6144,
            "stop_free_mib": 4096,
            "pressure_checks": 2,
            "foreign_owner_policy": "coexist-until-pressure",
            "xla_memory_fraction": "0.30",
            "settle_seconds": 5,
        },
        "preflight": {
            "paths": [str(python), str(remote_job / "run.sh"), str(remote_input)],
            "disk_min_free_gib": {"/mnt/sdc": 20},
            "commands": [[str(python), "-I", "-c", "import torch; assert torch.cuda.is_available()"]],
            "timeout_seconds": 120,
        },
        "execution": {
            "resource_retry_seconds": 60,
            "oom_retry_seconds": 120,
            "max_oom_requeues": 1,
        },
        "completion": {"type": "artifact", "path": str(output / "audit.json")},
        "progress": {"status_path": str(output / "status.json")},
        "science": {
            "context_path": str(remote_job / "context.json"),
            "protocol_path": str(remote_protocol),
        },
    }
    write_json(job_dir / "job.json", manifest)
    print(job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

