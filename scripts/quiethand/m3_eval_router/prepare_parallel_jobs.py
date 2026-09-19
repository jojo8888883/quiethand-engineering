#!/usr/bin/env python3
"""Prepare evaluation HaWoR and RGB/CAD binding jobs for A800 Autopilot."""

from __future__ import annotations

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


def create_job(job_id: str, adapter: str, runner: Path, input_path: Path, python: Path, output: Path, peak_mib: int, command_tail: list[str], dependencies: list[Path]) -> None:
    job_dir = GATE / "a800_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    remote_job = REMOTE / "jobs" / job_id
    protocol = GATE / "PLAN.md"
    rule = GATE / "router_rule.json"
    checks = {remote_local(input_path): sha(input_path), remote_local(runner): sha(runner), remote_local(protocol): sha(protocol), remote_local(rule): sha(rule)}
    checks.update({remote_local(path): sha(path) for path in dependencies})
    checks[RESOURCE_MANIFEST] = sha(ROOT / RESOURCE_MANIFEST.relative_to(RESOURCE_WORKSPACE))
    if adapter == "raw_hand_metric":
        checks[REPOSITORY_BINDING] = sha(ROOT / REPOSITORY_BINDING.relative_to(RESOURCE_WORKSPACE))
    command = [
        str(python), "-I", str(remote_local(runner)),
        "--workspace", str(REMOTE),
        "--input", str(remote_local(input_path)),
        "--input-sha256", sha(input_path),
        *command_tail,
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
    write_json(job_dir / "context.json", {
        "schema": "quiethand.m3_5.evaluation_router_job_context.v1",
        "job_id": job_id,
        "adapter": adapter,
        "evaluation_event_count": 30,
        "evaluation_results_authorized": True,
        "training_allowed": False,
        "input_sha256": sha(input_path),
        "protocol_sha256": sha(protocol),
        "router_rule_sha256": sha(rule),
    })
    write_json(job_dir / "job.json", {
        "schema_version": 1,
        "id": job_id,
        "display_name": f"QuietHand evaluation {adapter}",
        "managed": True,
        "priority": 80,
        "cwd": str(REMOTE),
        "command": ["bash", str(remote_job / "run.sh")],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"},
        "gpu": {
            "allowed": list(range(8)),
            "mode": "idle-first-shared",
            "idle_memory_mib": 500,
            "expected_peak_mib": peak_mib,
            "reserve_mib": 4096,
            "warning_free_mib": 8192,
            "stop_free_mib": 4096,
            "pressure_checks": 2,
            "foreign_owner_policy": "coexist-until-pressure",
            "xla_memory_fraction": "0.30",
            "settle_seconds": 5,
        },
        "preflight": {
            "paths": [str(python), str(remote_job / "run.sh"), str(remote_local(input_path))],
            "disk_min_free_gib": {"/mnt/sdc": 20, "/llm_jzm": 8},
            "commands": [[str(python), "-I", "-c", "import torch; assert torch.cuda.is_available()"]],
            "timeout_seconds": 120,
        },
        "execution": {"resource_retry_seconds": 60, "oom_retry_seconds": 120, "max_oom_requeues": 1},
        "completion": {"type": "artifact", "path": str(output / "audit.json")},
        "progress": {"status_path": str(output / "status.json")},
        "science": {"context_path": str(remote_job / "context.json"), "protocol_path": str(remote_local(protocol))},
    })


def main() -> int:
    common = ROOT / "scripts/quiethand/m3_jobs/runtime_common.py"
    raw_runner = ROOT / "scripts/quiethand/m3_5_fusion/run_raw_hand_metric.py"
    raw_helper = ROOT / "scripts/quiethand/m3_v1_2/run_raw_hand.py"
    raw_input = GATE / "QH_M3_EVALUATION_RAW_HAND_INPUT.json"
    create_job(
        "qh-m3-eval-router-raw-hand",
        "raw_hand_metric",
        raw_runner,
        raw_input,
        RESOURCE_WORKSPACE / "envs/raw_hand/bin/python",
        REMOTE / "results/raw_hand_metric",
        32768,
        [
            "--repository", str(REPOSITORY_PARENT / "HaWoR"),
            "--repository-parent", str(REPOSITORY_PARENT),
            "--repository-binding", str(REPOSITORY_BINDING),
            "--resource-root", str(RESOURCE_ROOT),
            "--resource-manifest", str(RESOURCE_MANIFEST),
            "--resource-manifest-sha256", sha(ROOT / RESOURCE_MANIFEST.relative_to(RESOURCE_WORKSPACE)),
            "--checkpoint", str(RESOURCE_ROOT / "raw_hand/hawor/checkpoints/hawor.ckpt"),
            "--detector", str(RESOURCE_ROOT / "raw_hand/external/detector.pt"),
            "--mano-left", str(RESOURCE_WORKSPACE / "external_data/mano_v1_2/models/MANO_LEFT.pkl"),
            "--mano-right", str(RESOURCE_WORKSPACE / "external_data/mano_v1_2/models/MANO_RIGHT.pkl"),
        ],
        [common, raw_helper],
    )
    identity_runner = ROOT / "scripts/quiethand/m3_eval_router/run_identity_binding.py"
    identity_input = GATE / "identity_binding/input.json"
    object_identity = ROOT / "quiethand/object_identity.py"
    create_job(
        "qh-m3-eval-router-identity",
        "rgb_cad_identity",
        identity_runner,
        identity_input,
        RESOURCE_WORKSPACE / "envs/semantic/bin/python",
        REMOTE / "results/identity_binding",
        24576,
        ["--model", str(RESOURCE_ROOT / "semantic")],
        [common, object_identity],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
