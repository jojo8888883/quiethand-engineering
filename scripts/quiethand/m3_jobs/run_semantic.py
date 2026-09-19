#!/usr/bin/env python3
"""Frozen Qwen3-VL calibration-only semantic adapter."""

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_common import M3JobError, atomic_json, load_exact_json, sha256_file, status, verify_bound_files, verify_resource_manifest  # noqa: E402
from quiethand.m3_adapter_contract import M3AdapterContractError, validate_semantic_candidate  # noqa: E402


INPUT_SHA256 = "88a75ba3a7fd146b22aa82c1926f0253968d45449ecd0bfe10c00c0227cd8473"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--resource-manifest-sha256", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-index", type=int)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    progress = args.output / "status.json"
    started = time.monotonic()
    try:
        manifest = load_exact_json(args.input, INPUT_SHA256)
        events = manifest.get("events")
        if (
            manifest.get("adapter") != "semantic"
            or manifest.get("event_count") != 150
            or manifest.get("evaluation_event_count") != 0
            or manifest.get("evaluation_model_results_opened") is not False
            or not isinstance(events, list)
            or len(events) != 150
        ):
            raise M3JobError("semantic input envelope is not calibration-only")
        if args.event_index is not None:
            if not 0 <= args.event_index < 150:
                raise M3JobError("positive-control event index is outside calibration")
            events = [events[args.event_index]]
        total = len(events)
        run_mode = "positive_control" if args.event_index is not None else "full_calibration"
        verified_files = verify_bound_files(args.workspace, events)
        verified_resource_bytes = verify_resource_manifest(args.resource_manifest, args.resource_manifest_sha256, args.resource_root)
        expected_model = args.resource_root / "semantic"
        if args.model.is_symlink() or not args.model.is_dir() or not expected_model.is_dir() or not args.model.samefile(expected_model):
            raise M3JobError("semantic model path is not the verified resource directory")
        status(progress, "loading_model", 0, total)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        ).eval()
        results_dir = args.output / "events"
        results_dir.mkdir(parents=True, exist_ok=True)
        valid = invalid = 0
        for ordinal, event in enumerate(events, start=1):
            event_id = event["event_id"]
            content = [
                {"type": "image", "url": str((args.workspace / relative).resolve())}
                for relative in event["ordered_rgb_frames"]
            ]
            content.append({"type": "text", "text": event["prompt"]})
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            raw_text = processor.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
            try:
                candidate = validate_semantic_candidate(raw_text)
                result_status = "observed"
                failure_reason = None
                valid += 1
            except M3AdapterContractError as exc:
                candidate = None
                result_status = "semantic_invalid"
                failure_reason = str(exc)
                invalid += 1
            atomic_json(
                results_dir / f"{event_id}.json",
                {
                    "schema": "quiethand.m3.semantic_result.v1",
                    "event_id": event_id,
                    "adapter": "semantic",
                    "status": result_status,
                    "candidate": candidate,
                    "raw_text": raw_text,
                    "failure_reason": failure_reason,
                },
            )
            status(progress, "running", ordinal, total)
        audit = {
            "schema": "quiethand.m3.adapter_job_audit.v1",
            "status": "COMPLETE",
            "adapter": "semantic",
            "event_count": total,
            "input_event_count": 150,
            "run_mode": run_mode,
            "event_index": args.event_index,
            "observed_count": valid,
            "invalid_count": invalid,
            "input_sha256": INPUT_SHA256,
            "model_revision": MODEL_REVISION,
            "verified_input_file_count": verified_files,
            "resource_manifest_sha256": args.resource_manifest_sha256,
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
            "evaluation_results_opened": False,
        }
        atomic_json(args.output / "audit.json", audit)
        status(progress, "complete", total, total)
    except Exception as exc:
        status(progress, "failed", 0, 1 if args.event_index is not None else 150, f"{type(exc).__name__}: {exc}")
        print(f"[hold] HOLD_ADAPTER_INTEGRATION: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
