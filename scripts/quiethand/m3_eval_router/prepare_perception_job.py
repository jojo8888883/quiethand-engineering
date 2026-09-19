#!/usr/bin/env python3
"""Prepare the evaluation SAM2/FoundationPose job after identity completion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "artifacts/quiethand/m3_5/evaluation_router_blind"
REMOTE = Path("/mnt/sdc/dty_user/quiethand_m3_eval")
RESOURCE_WORKSPACE = Path("/llm_jzm/dty_user/quiethand_m3")
RESOURCE_ROOT = RESOURCE_WORKSPACE / "external_data/quiethand_m3_resources"
RESOURCE_MANIFEST = RESOURCE_WORKSPACE / "artifacts/quiethand/m3/QH_M3_CHECKPOINT_LANDING.json"
REPOSITORY_PARENT = RESOURCE_WORKSPACE / "external_repos/quiethand_m3"
REPOSITORY_BINDING = RESOURCE_WORKSPACE / "artifacts/quiethand/m3/QH_M3_REPOSITORY_BINDING.json"
JOB_ID = "qh-m3-eval-router-perception"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-output", type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def remote_local(path: Path) -> Path:
    return REMOTE / path.relative_to(ROOT)


def main() -> int:
    cfg = arguments()
    identity = cfg.identity_output.resolve()
    audit = json.loads((identity / "audit.json").read_text(encoding="utf-8"))
    input_path = GATE / "QH_M3_EVALUATION_PERCEPTION_INPUT.json"
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    event_ids = {event["event_id"] for event in input_manifest["events"]}
    identity_files = sorted((identity / "events").glob("*.json"))
    if (
        audit.get("status") != "COMPLETE"
        or audit.get("event_count") != 30
        or audit.get("evaluation_event_count") != 30
        or audit.get("evaluation_results_opened") is not True
        or audit.get("training_performed") is not False
        or {path.stem for path in identity_files} != event_ids
    ):
        raise SystemExit("identity output is not a complete one-to-one evaluation dependency")
    job_dir = GATE / "a800_jobs" / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=False)
    runner = ROOT / "scripts/quiethand/m3_v1_2/run_perception.py"
    common = ROOT / "scripts/quiethand/m3_jobs/runtime_common.py"
    contract = ROOT / "quiethand/m3_v1_2_contract.py"
    object_identity = ROOT / "quiethand/object_identity.py"
    protocol = GATE / "PLAN.md"
    rule = GATE / "router_rule.json"
    resource_local = ROOT / RESOURCE_MANIFEST.relative_to(RESOURCE_WORKSPACE)
    binding_local = ROOT / REPOSITORY_BINDING.relative_to(RESOURCE_WORKSPACE)
    remote_identity = REMOTE / "results/identity_binding"
    checks = {
        remote_local(input_path): sha(input_path),
        remote_local(runner): sha(runner),
        remote_local(common): sha(common),
        remote_local(contract): sha(contract),
        remote_local(object_identity): sha(object_identity),
        remote_local(protocol): sha(protocol),
        remote_local(rule): sha(rule),
        RESOURCE_MANIFEST: sha(resource_local),
        REPOSITORY_BINDING: sha(binding_local),
        remote_identity / "audit.json": sha(identity / "audit.json"),
    }
    checks.update({remote_identity / "events" / path.name: sha(path) for path in identity_files})
    python = RESOURCE_WORKSPACE / "envs/perception/bin/python"
    output = REMOTE / "results/perception"
    command = [
        str(python), "-I", str(remote_local(runner)),
        "--workspace", str(REMOTE),
        "--input", str(remote_local(input_path)),
        "--input-sha256", sha(input_path),
        "--semantic-output", str(REMOTE / "results/temporal_semantic"),
        "--identity-output", str(remote_identity),
        "--sam-repository", str(REPOSITORY_PARENT / "sam2"),
        "--sam-config", "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "--sam-checkpoint", str(RESOURCE_ROOT / "segmentation/sam2.1_hiera_base_plus.pt"),
        "--foundation-repository", str(REPOSITORY_PARENT / "FoundationPose"),
        "--repository-parent", str(REPOSITORY_PARENT),
        "--repository-binding", str(REPOSITORY_BINDING),
        "--resource-root", str(RESOURCE_ROOT),
        "--resource-manifest", str(RESOURCE_MANIFEST),
        "--resource-manifest-sha256", sha(resource_local),
        "--output", str(output),
    ]
    check_text = "\n".join(f"{digest}  {path}" for path, digest in checks.items())
    run_text = (
        "#!/usr/bin/env bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n"
        "export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        "sha256sum --check --strict <<'QH_M3_EVAL_FILES'\n"
        + check_text
        + "\nQH_M3_EVAL_FILES\nexec "
        + " ".join(quote(part) for part in command)
        + "\n"
    )
    run_path = job_dir / "run.sh"
    run_path.write_text(run_text, encoding="utf-8")
    run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR)
    remote_job = REMOTE / "jobs" / JOB_ID
    write_json(job_dir / "context.json", {
        "schema": "quiethand.m3_5.evaluation_router_job_context.v1",
        "job_id": JOB_ID,
        "adapter": "segmentation_object_state",
        "evaluation_event_count": 30,
        "evaluation_results_authorized": True,
        "training_allowed": False,
        "input_sha256": sha(input_path),
        "identity_audit_sha256": sha(identity / "audit.json"),
        "protocol_sha256": sha(protocol),
        "router_rule_sha256": sha(rule),
    })
    write_json(job_dir / "job.json", {
        "schema_version": 1,
        "id": JOB_ID,
        "display_name": "QuietHand evaluation SAM2 and FoundationPose",
        "managed": True,
        "priority": 80,
        "cwd": str(REMOTE),
        "command": ["bash", str(remote_job / "run.sh")],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"},
        "gpu": {
            "allowed": list(range(8)), "mode": "idle-first-shared", "idle_memory_mib": 500,
            "expected_peak_mib": 49152, "reserve_mib": 4096, "warning_free_mib": 8192,
            "stop_free_mib": 4096, "pressure_checks": 2, "foreign_owner_policy": "coexist-until-pressure",
            "xla_memory_fraction": "0.30", "settle_seconds": 5,
        },
        "preflight": {
            "paths": [str(python), str(remote_job / "run.sh"), str(remote_local(input_path)), str(remote_identity / "audit.json")],
            "disk_min_free_gib": {"/mnt/sdc": 20, "/llm_jzm": 8},
            "commands": [[str(python), "-I", "-c", "import torch; assert torch.cuda.is_available()"]],
            "timeout_seconds": 120,
        },
        "execution": {"resource_retry_seconds": 60, "oom_retry_seconds": 120, "max_oom_requeues": 1},
        "completion": {"type": "artifact", "path": str(output / "audit.json")},
        "progress": {"status_path": str(output / "status.json")},
        "science": {"context_path": str(remote_job / "context.json"), "protocol_path": str(remote_local(protocol))},
    })
    print(job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
