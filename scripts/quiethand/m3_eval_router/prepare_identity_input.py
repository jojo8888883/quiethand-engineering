#!/usr/bin/env python3
"""Build blinded RGB-region/CAD-sheet inputs for evaluation identity binding."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
M3_5 = ROOT / "scripts/quiethand/m3_5_fusion"
M3_JOBS = ROOT / "scripts/quiethand/m3_jobs"
for directory in (ROOT, M3_5, M3_JOBS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from repair_object_identity import PROMPT, cad_sheet  # noqa: E402
from runtime_common import atomic_json  # noqa: E402
from quiethand.object_identity import ROLES, entity_catalog  # noqa: E402


EXPECTED_EVENTS = 30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--perception-input", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    cfg = arguments()
    manifest = json.loads(cfg.perception_input.read_text(encoding="utf-8"))
    semantic_audit = json.loads((cfg.semantic_output / "audit.json").read_text(encoding="utf-8"))
    events = manifest.get("events")
    if (
        manifest.get("schema") != "quiethand.m3_5.evaluation_perception_input.v1"
        or manifest.get("event_count") != EXPECTED_EVENTS
        or manifest.get("evaluation_event_count") != EXPECTED_EVENTS
        or manifest.get("evaluation_results_opened") is not True
        or not isinstance(events, list)
        or len(events) != EXPECTED_EVENTS
        or semantic_audit.get("status") != "COMPLETE"
        or semantic_audit.get("event_count") != EXPECTED_EVENTS
        or semantic_audit.get("evaluation_results_opened") is not True
    ):
        raise SystemExit("identity dependencies are not the completed 30-event evaluation set")
    if cfg.output_root.exists():
        raise SystemExit(f"destination exists: {cfg.output_root}")
    inputs = cfg.output_root / "inputs"
    cad_root = inputs / "cads"
    cad_root.mkdir(parents=True)
    output_events = []
    for ordinal, source_event in enumerate(events, start=1):
        event = copy.deepcopy(source_event)
        event_id = event["event_id"]
        candidate_path = cfg.semantic_output / "events" / f"{event_id}.json"
        semantic = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate = semantic.get("candidate") if semantic.get("status") == "observed" else None
        folder = inputs / event_id
        folder.mkdir()
        rgb = Image.open(cfg.workspace / event["segmentation"]["ordered_rgb_frames"][0]).convert("RGB")
        rgb.resize((960, 540)).save(folder / "scene.png")
        for index, role in enumerate(ROLES):
            box = None if candidate is None else candidate["start_sample_boxes_0_1000"][role]
            if box is None:
                crop = Image.new("RGB", (256, 256), "white")
                ImageDraw.Draw(crop).text((10, 120), "MISSING REGION", fill="black")
            else:
                x1, y1, x2, y2 = box
                crop = rgb.crop((int(x1 * rgb.width / 1000), int(y1 * rgb.height / 1000), int(x2 * rgb.width / 1000), int(y2 * rgb.height / 1000)))
                crop.thumbnail((640, 640))
            crop.save(folder / f"region_{index}.png")
        event["identity_images"] = [(folder / name).relative_to(cfg.workspace).as_posix() for name in ("scene.png", "region_0.png", "region_1.png")]
        catalog = entity_catalog(event["object_state"])
        event["object_state"]["entities"] = []
        for entity_id, entity in sorted(catalog.items()):
            sheet = cad_root / f"{entity_id}.png"
            if not sheet.exists():
                cad_sheet(cfg.workspace / entity["mesh_path"], entity_id, sheet)
            event["object_state"]["entities"].append({**entity, "reference_image": sheet.relative_to(cfg.workspace).as_posix()})
        event.pop("materialized_file_binding", None)
        output_events.append(event)
        print(f"[identity-input] {ordinal}/{EXPECTED_EVENTS} {event_id}", flush=True)
    atomic_json(cfg.output_root / "input.json", {
        "schema": "quiethand.m3_5.evaluation_identity_input.v1",
        "event_count": EXPECTED_EVENTS,
        "evaluation_event_count": EXPECTED_EVENTS,
        "evaluation_results_opened": True,
        "training_performed": False,
        "prompt": PROMPT,
        "events": output_events,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
