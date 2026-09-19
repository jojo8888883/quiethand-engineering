#!/usr/bin/env python3
"""Prepare parallel HaWoR and SAM2/FoundationPose v1.2 A800 jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
REMOTE = Path("/llm_jzm/dty_user/quiethand_m3")
RESOURCE_ROOT = REMOTE / "external_data/quiethand_m3_resources"
RESOURCE_MANIFEST = REMOTE / "artifacts/quiethand/m3/QH_M3_CHECKPOINT_LANDING.json"
REPOSITORY_PARENT = REMOTE / "external_repos/quiethand_m3"
REPOSITORY_BINDING = REMOTE / "artifacts/quiethand/m3/QH_M3_REPOSITORY_BINDING.json"
PROTOCOL = REMOTE / "refine-logs/QUIETHAND_M3_PROTOCOL_V1_2_20260830_172950.md"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-manifest-sha256", required=True)
    parser.add_argument("--raw-input-sha256", required=True)
    parser.add_argument("--perception-input-sha256", required=True)
    parser.add_argument("--identity-output", type=Path, required=True,
                        help="absolute remote directory containing explicit CAD binding results")
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    cfg = args()
    resource_local = ROOT / RESOURCE_MANIFEST.relative_to(REMOTE)
    binding_local = ROOT / REPOSITORY_BINDING.relative_to(REMOTE)
    protocol_local = ROOT / PROTOCOL.relative_to(REMOTE)
    common_local = ROOT / "scripts/quiethand/m3_jobs/runtime_common.py"
    contract_local = ROOT / "quiethand/m3_v1_2_contract.py"
    if sha(resource_local) != cfg.resource_manifest_sha256:
        raise SystemExit("resource manifest digest changed")
    root = ROOT / "artifacts/quiethand/m3_v1_2/a800_jobs"
    root.mkdir(parents=True, exist_ok=True)
    specifications = {
        "raw_hand": {
            "job_id": "qh-m3-v12-raw-hand", "runner": ROOT / "scripts/quiethand/m3_v1_2/run_raw_hand.py",
            "input": ROOT / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_RAW_HAND_INPUT.json",
            "input_sha": cfg.raw_input_sha256, "env": "raw_hand", "peak": 32768,
        },
        "perception": {
            "job_id": "qh-m3-v12-perception", "runner": ROOT / "scripts/quiethand/m3_v1_2/run_perception.py",
            "input": ROOT / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_PERCEPTION_INPUT.json",
            "input_sha": cfg.perception_input_sha256, "env": "perception", "peak": 49152,
        },
    }
    for adapter, spec in specifications.items():
        if sha(spec["input"]) != spec["input_sha"]:
            raise SystemExit(f"{adapter} input digest changed")
        job_dir = root / spec["job_id"]
        job_dir.mkdir(exist_ok=False)
        remote_job = REMOTE / "jobs" / spec["job_id"]
        output = REMOTE / "results_v1_2" / adapter
        remote_runner = REMOTE / spec["runner"].relative_to(ROOT)
        remote_input = REMOTE / spec["input"].relative_to(ROOT)
        checks = {
            remote_runner: sha(spec["runner"]), remote_input: spec["input_sha"],
            REMOTE / common_local.relative_to(ROOT): sha(common_local),
            REMOTE / contract_local.relative_to(ROOT): sha(contract_local),
            RESOURCE_MANIFEST: cfg.resource_manifest_sha256,
            REPOSITORY_BINDING: sha(binding_local), PROTOCOL: sha(protocol_local),
        }
        python = REMOTE / f"envs/{spec['env']}/bin/python"
        command = [str(python), "-I", str(remote_runner), "--workspace", str(REMOTE),
                   "--input", str(remote_input), "--input-sha256", spec["input_sha"],
                   "--repository-parent", str(REPOSITORY_PARENT), "--repository-binding", str(REPOSITORY_BINDING)]
        if adapter == "raw_hand":
            command += [
                "--repository", str(REPOSITORY_PARENT / "HaWoR"),
                "--checkpoint", str(RESOURCE_ROOT / "raw_hand/hawor/checkpoints/hawor.ckpt"),
                "--detector", str(RESOURCE_ROOT / "raw_hand/external/detector.pt"),
                "--mano-left", str(REMOTE / "external_data/mano_v1_2/models/MANO_LEFT.pkl"),
                "--mano-right", str(REMOTE / "external_data/mano_v1_2/models/MANO_RIGHT.pkl"),
            ]
        else:
            command += [
                "--semantic-output", str(REMOTE / "results_v1_2/temporal_semantic"),
                "--identity-output", str(cfg.identity_output),
                "--sam-repository", str(REPOSITORY_PARENT / "sam2"),
                "--sam-config", "configs/sam2.1/sam2.1_hiera_b+.yaml",
                "--sam-checkpoint", str(RESOURCE_ROOT / "segmentation/sam2.1_hiera_base_plus.pt"),
                "--foundation-repository", str(REPOSITORY_PARENT / "FoundationPose"),
            ]
        command += ["--resource-root", str(RESOURCE_ROOT), "--resource-manifest", str(RESOURCE_MANIFEST),
                    "--resource-manifest-sha256", cfg.resource_manifest_sha256, "--output", str(output)]
        check_text = "\n".join(f"{digest}  {path}" for path, digest in checks.items())
        run_text = (
            "#!/usr/bin/env bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n"
            "export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
            "sha256sum --check --strict <<'QH_M3_V12_FILES'\n" + check_text +
            "\nQH_M3_V12_FILES\nexec " + " ".join(quote(item) for item in command) + "\n"
        )
        run_path = job_dir / "run.sh"
        run_path.write_text(run_text, encoding="utf-8")
        run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR)
        write_json(job_dir / "context.json", {
            "schema": "quiethand.m3_v1_2.a800_job_context.v1", "job_id": spec["job_id"],
            "adapter": adapter, "input_sha256": spec["input_sha"],
            "training_allowed": False, "evaluation_results_allowed": False,
        })
        manifest = {
            "schema_version": 1, "id": spec["job_id"], "display_name": f"QuietHand M3-v1.2 {adapter}",
            "managed": True, "priority": 60, "cwd": str(REMOTE), "command": ["bash", str(remote_job / "run.sh")],
            "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"},
            "gpu": {"allowed": list(range(8)), "mode": "idle-first-shared", "idle_memory_mib": 500,
                    "expected_peak_mib": spec["peak"], "reserve_mib": 4096, "warning_free_mib": 8192,
                    "stop_free_mib": 4096, "pressure_checks": 2, "foreign_owner_policy": "coexist-until-pressure",
                    "xla_memory_fraction": "0.30", "settle_seconds": 5},
            "preflight": {"paths": [str(python), str(remote_job / "run.sh"), str(remote_input)],
                          "disk_min_free_gib": {"/llm_jzm": 8},
                          "commands": [[str(python), "-I", "-c", "import torch; assert torch.cuda.is_available()"]],
                          "timeout_seconds": 120},
            "execution": {"resource_retry_seconds": 60, "oom_retry_seconds": 120, "max_oom_requeues": 1},
            "completion": {"type": "artifact", "path": str(output / "audit.json")},
            "progress": {"status_path": str(output / "status.json")},
            "science": {"context_path": str(remote_job / "context.json"), "protocol_path": str(PROTOCOL)},
        }
        write_json(job_dir / "job.json", manifest)
        print(job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
