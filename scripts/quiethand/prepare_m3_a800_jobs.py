#!/usr/bin/env python3
"""Generate exact A800 Autopilot manifests after the 26-file closure exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LOCAL_OUTPUT = ROOT / "artifacts" / "quiethand" / "m3" / "a800_jobs"
REMOTE = Path("/llm_jzm/dty_user/quiethand_m3")
PROTOCOL = REMOTE / "refine-logs" / "QUIETHAND_M3_PROTOCOL_V1_1_20260820_122944.md"
RESOURCE_MANIFEST = REMOTE / "artifacts" / "quiethand" / "m3" / "QH_M3_CHECKPOINT_LANDING.json"
RESOURCE_ROOT = REMOTE / "external_data" / "quiethand_m3_resources"
REPOSITORY_PARENT = REMOTE / "external_repos" / "quiethand_m3"
REPOSITORY_BINDING = REMOTE / "artifacts" / "quiethand" / "m3" / "QH_M3_REPOSITORY_BINDING.json"
LOCAL_PROTOCOL = ROOT / PROTOCOL.relative_to(REMOTE)
LOCAL_RESOURCE_MANIFEST = ROOT / RESOURCE_MANIFEST.relative_to(REMOTE)
LOCAL_REPOSITORY_BINDING = ROOT / REPOSITORY_BINDING.relative_to(REMOTE)
RUNTIME_COMMON = REMOTE / "scripts" / "quiethand" / "m3_jobs" / "runtime_common.py"
ADAPTER_CONTRACT = REMOTE / "quiethand" / "m3_adapter_contract.py"
INPUTS = {
    "semantic": REMOTE / "artifacts" / "quiethand" / "m3" / "QH_M3_SEMANTIC_INPUT.json",
    "raw_hand": REMOTE / "artifacts" / "quiethand" / "m3" / "QH_M3_RAW_HAND_INPUT.json",
    "perception": REMOTE / "artifacts" / "quiethand" / "m3" / "QH_M3_SEGMENTATION_OBJECT_INPUT.json",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-manifest-sha256", required=True)
    parser.add_argument("--semantic-control-index", type=int, default=0)
    parser.add_argument("--raw-control-index", type=int, default=0)
    parser.add_argument("--perception-control-index", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def command(adapter: str, output: Path, resource_sha: str, event_index: int | None) -> list[str]:
    common = [
        "--workspace", str(REMOTE),
        "--resource-root", str(RESOURCE_ROOT),
        "--resource-manifest", str(RESOURCE_MANIFEST),
        "--resource-manifest-sha256", resource_sha,
        "--output", str(output),
    ]
    if event_index is not None:
        common.extend(["--event-index", str(event_index)])
    if adapter == "semantic":
        return [
            str(REMOTE / "envs" / "semantic" / "bin" / "python"), "-I",
            str(REMOTE / "scripts" / "quiethand" / "m3_jobs" / "run_semantic.py"),
            "--input", str(INPUTS[adapter]),
            "--model", str(RESOURCE_ROOT / "semantic"),
            *common,
        ]
    shared = [
        "--repository-parent", str(REPOSITORY_PARENT),
        "--repository-binding", str(REPOSITORY_BINDING),
    ]
    if adapter == "raw_hand":
        return [
            str(REMOTE / "envs" / "raw_hand" / "bin" / "python"), "-I",
            str(REMOTE / "scripts" / "quiethand" / "m3_jobs" / "run_raw_hand.py"),
            "--input", str(INPUTS[adapter]),
            "--repository", str(REPOSITORY_PARENT / "HaWoR"),
            *shared,
            "--checkpoint", str(RESOURCE_ROOT / "raw_hand" / "hawor" / "checkpoints" / "hawor.ckpt"),
            "--detector", str(RESOURCE_ROOT / "raw_hand" / "external" / "detector.pt"),
            "--mano-left", str(REMOTE / "external_data" / "mano_v1_2" / "models" / "MANO_LEFT.pkl"),
            "--mano-right", str(REMOTE / "external_data" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl"),
            *common,
        ]
    if adapter == "perception":
        return [
            str(REMOTE / "envs" / "perception" / "bin" / "python"), "-I",
            str(REMOTE / "scripts" / "quiethand" / "m3_jobs" / "run_segmentation_object.py"),
            "--input", str(INPUTS[adapter]),
            "--semantic-output", str(REMOTE / "results" / "semantic_calibration"),
            "--sam-repository", str(REPOSITORY_PARENT / "sam2"),
            "--sam-config", "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "--sam-checkpoint", str(RESOURCE_ROOT / "segmentation" / "sam2.1_hiera_base_plus.pt"),
            "--foundation-repository", str(REPOSITORY_PARENT / "FoundationPose"),
            *shared,
            *common,
        ]
    raise ValueError("unsupported adapter")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def frozen_files(adapter: str) -> dict[Path, str]:
    runner_name = {
        "semantic": "run_semantic.py",
        "raw_hand": "run_raw_hand.py",
        "perception": "run_segmentation_object.py",
    }[adapter]
    remote_runner = REMOTE / "scripts" / "quiethand" / "m3_jobs" / runner_name
    local_by_remote = {
        remote_runner: ROOT / remote_runner.relative_to(REMOTE),
        RUNTIME_COMMON: ROOT / RUNTIME_COMMON.relative_to(REMOTE),
        ADAPTER_CONTRACT: ROOT / ADAPTER_CONTRACT.relative_to(REMOTE),
        INPUTS[adapter]: ROOT / INPUTS[adapter].relative_to(REMOTE),
        RESOURCE_MANIFEST: LOCAL_RESOURCE_MANIFEST,
        PROTOCOL: LOCAL_PROTOCOL,
    }
    if adapter != "semantic":
        local_by_remote[REPOSITORY_BINDING] = LOCAL_REPOSITORY_BINDING
    return {remote: sha256_file(local) for remote, local in local_by_remote.items()}


def manifest(job_id: str, display: str, script: Path, output: Path, context: Path, adapter: str, control: bool) -> dict[str, object]:
    peaks = {"semantic": 43008, "raw_hand": 32768, "perception": 49152}
    env_python = REMOTE / "envs" / adapter.replace("perception", "perception") / "bin" / "python"
    return {
        "schema_version": 1,
        "id": job_id,
        "display_name": display,
        "managed": True,
        "priority": 80 if control else 60,
        "cwd": str(REMOTE),
        "command": ["bash", str(script)],
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"},
        "gpu": {
            "allowed": list(range(8)),
            "mode": "idle-first-shared",
            "idle_memory_mib": 500,
            "expected_peak_mib": peaks[adapter],
            "reserve_mib": 4096,
            "warning_free_mib": 8192,
            "stop_free_mib": 4096,
            "pressure_checks": 2,
            "foreign_owner_policy": "coexist-until-pressure",
            "xla_memory_fraction": "0.30",
            "settle_seconds": 5,
        },
        "preflight": {
            "paths": [str(env_python), str(script), str(RESOURCE_MANIFEST), str(INPUTS[adapter])],
            "disk_min_free_gib": {"/llm_jzm": 20},
            "commands": [[str(env_python), "-I", "-c", "import torch; assert torch.cuda.is_available()"]],
            "timeout_seconds": 120,
        },
        "execution": {"resource_retry_seconds": 60, "oom_retry_seconds": 120, "max_oom_requeues": 1},
        "completion": {"type": "artifact", "path": str(output / "audit.json")},
        "progress": {"status_path": str(output / "status.json")},
        "science": {"context_path": str(context), "protocol_path": str(PROTOCOL)},
    }


def main() -> int:
    args = arguments()
    if _SHA.fullmatch(args.resource_manifest_sha256) is None:
        raise SystemExit("resource manifest SHA-256 must be 64 lowercase hex characters")
    indices = {
        "semantic": args.semantic_control_index,
        "raw_hand": args.raw_control_index,
        "perception": args.perception_control_index,
    }
    if any(not 0 <= value < 150 for value in indices.values()):
        raise SystemExit("every control index must be within 0..149")
    if not LOCAL_RESOURCE_MANIFEST.is_file() or sha256_file(LOCAL_RESOURCE_MANIFEST) != args.resource_manifest_sha256:
        raise SystemExit("the complete local resource manifest does not match the requested SHA-256")
    if LOCAL_OUTPUT.exists():
        raise SystemExit(f"destination already exists: {LOCAL_OUTPUT}")
    LOCAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{LOCAL_OUTPUT.name}.", dir=LOCAL_OUTPUT.parent))
    try:
        jobs = []
        for adapter in ("semantic", "raw_hand", "perception"):
            checks = frozen_files(adapter)
            check_text = "\n".join(f"{digest}  {path}" for path, digest in checks.items())
            for control in (True, False):
                suffix = f"control_i{indices[adapter]:03d}" if control else "calibration"
                job_id = f"qh-m3-{adapter.replace('_', '-')}-{suffix.replace('_', '-')}"
                remote_dir = REMOTE / "jobs" / job_id
                result_name = "semantic_calibration" if adapter == "semantic" and not control else f"{adapter}_{suffix}"
                output = REMOTE / "results" / result_name
                context = remote_dir / "context.json"
                run_script = remote_dir / "run.sh"
                local_dir = staging / job_id
                local_dir.mkdir()
                cmd = command(adapter, output, args.resource_manifest_sha256, indices[adapter] if control else None)
                script_text = (
                    "#!/usr/bin/env bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n"
                    "export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
                    "sha256sum --check --strict <<'QH_M3_FROZEN_FILES'\n"
                    + check_text
                    + "\nQH_M3_FROZEN_FILES\nexec "
                    + " ".join(shell_quote(part) for part in cmd)
                    + "\n"
                )
                local_script = local_dir / "run.sh"
                local_script.write_text(script_text, encoding="utf-8")
                local_script.chmod(local_script.stat().st_mode | stat.S_IXUSR)
                context_value = {
                    "schema": "quiethand.m3.a800_job_context.v1",
                    "job_id": job_id,
                    "adapter": adapter,
                    "run_mode": "positive_control" if control else "full_calibration",
                    "event_index": indices[adapter] if control else None,
                    "resource_manifest_sha256": args.resource_manifest_sha256,
                    "frozen_file_sha256": {str(path): digest for path, digest in checks.items()},
                    "training_allowed": False,
                    "evaluation_results_allowed": False,
                }
                write_json(local_dir / "context.json", context_value)
                manifest_value = manifest(job_id, f"QuietHand M3 {adapter} {suffix}", run_script, output, context, adapter, control)
                write_json(local_dir / "job.json", manifest_value)
                jobs.append({"job_id": job_id, "adapter": adapter, "control": control, "local_dir": (LOCAL_OUTPUT / job_id).relative_to(ROOT).as_posix()})
        write_json(staging / "index.json", {"schema": "quiethand.m3.a800_job_index.v1", "resource_manifest_sha256": args.resource_manifest_sha256, "jobs": jobs})
        os.replace(staging, LOCAL_OUTPUT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[artifact] {LOCAL_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
