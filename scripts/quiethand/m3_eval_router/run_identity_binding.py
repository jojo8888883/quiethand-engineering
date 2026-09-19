#!/usr/bin/env python3
"""Bind blinded evaluation RGB regions to candidate CAD entities with Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[3]
M3_JOBS = ROOT / "scripts/quiethand/m3_jobs"
for directory in (ROOT, M3_JOBS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from runtime_common import M3JobError, atomic_json, load_exact_json, sha256_file, status  # noqa: E402
from quiethand.object_identity import ROLES, ObjectIdentityError, entity_catalog, parse_visual_binding  # noqa: E402


EXPECTED_EVENTS = 30
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    cfg = arguments()
    progress = cfg.output / "status.json"
    started = time.monotonic()
    try:
        manifest = load_exact_json(cfg.input, cfg.input_sha256)
        events = manifest.get("events")
        if (
            manifest.get("schema") != "quiethand.m3_5.evaluation_identity_input.v1"
            or manifest.get("event_count") != EXPECTED_EVENTS
            or manifest.get("evaluation_event_count") != EXPECTED_EVENTS
            or manifest.get("evaluation_results_opened") is not True
            or manifest.get("training_performed") is not False
            or not isinstance(events, list)
            or len(events) != EXPECTED_EVENTS
        ):
            raise M3JobError("identity input is not the frozen 30-event evaluation set")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        status(progress, "loading_model", 0, EXPECTED_EVENTS)
        processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        ).eval()
        for ordinal, event in enumerate(events, start=1):
            catalog = entity_catalog(event["object_state"])
            content = []
            for relative in event["identity_images"]:
                image_path = cfg.workspace / relative
                if not image_path.is_file():
                    raise M3JobError(f"identity image missing: {relative}")
                content.append({"type": "image", "url": str(image_path)})
            for entity_id, entity in sorted(catalog.items()):
                reference = cfg.workspace / entity["reference_image"]
                if not reference.is_file():
                    raise M3JobError(f"CAD reference missing: {entity['reference_image']}")
                content += [{"type": "text", "text": "CAD entity " + entity_id}, {"type": "image", "url": str(reference)}]
            content.append({"type": "text", "text": manifest["prompt"]})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=400)
            raw = processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            atomic_json(cfg.output / "raw" / f"{event['event_id']}.json", {"raw_text": raw})
            try:
                binding = parse_visual_binding(raw, event)
            except (json.JSONDecodeError, ObjectIdentityError) as exc:
                binding = {
                    "event_id": event["event_id"],
                    "role_to_entity": {role: None for role in ROLES},
                    "reasons": {role: str(exc) for role in ROLES},
                    "source": "qwen_rgb_cad_visual_match",
                    "status": "invalid_model_answer",
                    "native_pose_used": False,
                    "human_corrections_used": False,
                }
            atomic_json(cfg.output / "events" / f"{event['event_id']}.json", binding)
            status(progress, "running", ordinal, EXPECTED_EVENTS)
        results = [json.loads((cfg.output / "events" / f"{event['event_id']}.json").read_text(encoding="utf-8")) for event in events]
        atomic_json(cfg.output / "audit.json", {
            "schema": "quiethand.m3_5.evaluation_identity_job_audit.v1",
            "status": "COMPLETE",
            "event_count": EXPECTED_EVENTS,
            "evaluation_event_count": EXPECTED_EVENTS,
            "invalid_event_count": sum(result.get("status") == "invalid_model_answer" for result in results),
            "unresolved_region_count": sum(value is None for result in results for value in result["role_to_entity"].values()),
            "model_revision": MODEL_REVISION,
            "input_sha256": cfg.input_sha256,
            "runner_sha256": sha256_file(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
            "native_pose_used": False,
            "human_corrections_used": False,
            "training_performed": False,
            "evaluation_results_opened": True,
        })
        status(progress, "complete", EXPECTED_EVENTS, EXPECTED_EVENTS)
    except Exception as exc:
        status(progress, "failed", 0, EXPECTED_EVENTS, f"{type(exc).__name__}: {exc}")
        print(f"[hold] HOLD_IDENTITY_BINDING: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
