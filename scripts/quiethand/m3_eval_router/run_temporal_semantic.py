#!/usr/bin/env python3
"""Run frozen Qwen3-VL on the authorized 30-video evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (HERE.parent / "m3_jobs", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runtime_common import M3JobError, atomic_json, load_exact_json, sha256_file, status, verify_bound_files, verify_resource_manifest
from quiethand.m3_v1_2_contract import M3V12ContractError, validate_candidate


MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--resource-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    cfg = arguments()
    progress = cfg.output / "status.json"
    started = time.monotonic()
    total = 30
    try:
        manifest = load_exact_json(cfg.input, cfg.input_sha256)
        events = manifest.get("events")
        if (
            manifest.get("schema") != "quiethand.m3_v1_2.evaluation_temporal_input.v1"
            or manifest.get("adapter") != "temporal_semantic"
            or manifest.get("event_count") != total
            or manifest.get("evaluation_event_count") != total
            or manifest.get("evaluation_model_results_authorized") is not True
            or manifest.get("evaluation_model_results_opened") is not False
            or manifest.get("training_performed") is not False
            or not isinstance(events, list)
            or len(events) != total
            or any(not isinstance(row, dict) or not str(row.get("event_id", "")).startswith("qh-m3-v12-eval-") for row in events)
        ):
            raise M3JobError("input is not the explicitly authorized 30-video evaluation envelope")
        verified_files = verify_bound_files(cfg.workspace, events)
        verified_resource_bytes = verify_resource_manifest(
            cfg.resource_manifest, cfg.resource_manifest_sha256, cfg.resource_root
        )
        expected_model = cfg.resource_root / "semantic"
        if cfg.model.is_symlink() or not cfg.model.is_dir() or not cfg.model.samefile(expected_model):
            raise M3JobError("semantic model path is not the verified resource")
        status(progress, "loading_model", 0, total)
        processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        ).eval()
        results = cfg.output / "events"
        results.mkdir(parents=True, exist_ok=True)
        observed = invalid = 0
        for ordinal, event in enumerate(events, start=1):
            event_id = event["event_id"]
            content = [
                {"type": "image", "url": str((cfg.workspace / relative).resolve())}
                for relative in event["ordered_rgb_frames"]
            ]
            content.append({"type": "text", "text": event["prompt"]})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=640, use_cache=True
                )
            raw = processor.decode(
                generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            try:
                candidate = validate_candidate(raw)
                result_status, failure = "observed", None
                observed += 1
            except M3V12ContractError as exc:
                candidate = None
                result_status, failure = "semantic_invalid", str(exc)
                invalid += 1
            atomic_json(results / f"{event_id}.json", {
                "schema": "quiethand.m3_v1_2.evaluation_temporal_semantic_result.v1",
                "event_id": event_id,
                "adapter": "temporal_semantic",
                "status": result_status,
                "candidate": candidate,
                "raw_text": raw,
                "failure_reason": failure,
                "training_performed": False,
                "evaluation_results_opened": True,
            })
            status(progress, "running", ordinal, total)
        atomic_json(cfg.output / "audit.json", {
            "schema": "quiethand.m3_v1_2.evaluation_adapter_job_audit.v1",
            "status": "COMPLETE",
            "adapter": "temporal_semantic",
            "event_count": total,
            "evaluation_event_count": total,
            "observed_count": observed,
            "invalid_count": invalid,
            "input_sha256": cfg.input_sha256,
            "model_revision": MODEL_REVISION,
            "verified_input_file_count": verified_files,
            "resource_manifest_sha256": cfg.resource_manifest_sha256,
            "verified_resource_bytes": verified_resource_bytes,
            "runner_sha256": sha256_file(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "training_performed": False,
            "evaluation_results_opened": True,
        })
        status(progress, "complete", total, total)
    except Exception as exc:
        status(progress, "failed", 0, total, f"{type(exc).__name__}: {exc}")
        print(f"[hold] HOLD_EVALUATION_INCOMPLETE: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

