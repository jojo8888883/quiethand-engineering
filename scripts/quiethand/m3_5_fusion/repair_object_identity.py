#!/usr/bin/env python3
"""Prepare, visually bind, and rerun only the object states of selected calibration clips.

Each command uses its existing environment: local CPU / semantic / perception.
No native pose, native mask, action name, or human correction enters the matcher.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/quiethand/m3_jobs"))
from runtime_common import atomic_json, atomic_npy, status
from quiethand.object_identity import ROLES, ObjectIdentityError, entity_catalog, parse_visual_binding, resolve_role_entity

BASE = Path("artifacts/quiethand/m3_v1_2")
REPAIR = Path("artifacts/quiethand/m3_5/identity_repair")
PROMPT = """Match two numbered RGB regions to the physical objects in the CAD sheets.
The first image is the scene. Then come region_0, region_1, and labelled CAD sheets.
Each CAD sheet shows one object from several directions in neutral grey; ignore
colour differences between a rendering and the RGB. Match physical shape, not
left/right order, filename order, action role, or which object seems to be a tool.
Each CAD entity can be assigned at most once. If a crop is missing, not an object,
or visually indistinguishable among candidates, use null; do not guess by order.
Return ONLY JSON with exactly region_0 and region_1. Each value must have entity_id
(an exact supplied CAD id or null) and reason (a short visual explanation).
"""


def read(path):
    return json.loads(path.read_text())


def cad_sheet(path, entity_id, destination):
    """Deterministic mesh rendering; no learned generation or native pose."""
    import trimesh
    mesh = trimesh.load(path, process=False)
    points = np.asarray(mesh.vertices)
    points = points - (points.min(0) + points.max(0)) / 2
    triangles = points[np.asarray(mesh.faces)]
    sheet = Image.new("RGB", (900, 640), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 8), entity_id + " | neutral CAD, arbitrary views", fill="black")
    directions = [(1, 1, 1), (-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)]
    scale = 240 / max(float(np.linalg.norm(points, axis=1).max()) * 2, 1e-8)
    for index, direction in enumerate(directions):
        z = np.asarray(direction, float)
        z /= np.linalg.norm(z)
        up = np.array([0., 0., 1.]) if abs(z[2]) < .9 else np.array([0., 1., 0.])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        projected = triangles @ np.stack([x, y, z], axis=1)
        normal = np.cross(projected[:, 1] - projected[:, 0], projected[:, 2] - projected[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        shade = (100 + 130 * np.abs(normal @ np.array([.25, .35, .9]))).clip(60, 235).astype(int)
        origin = np.array([150 + index % 3 * 300, 185 + index // 3 * 300])
        uv = projected[:, :, :2] * [scale, -scale] + origin
        for face in np.argsort(projected[:, :, 2].mean(1)):
            value = int(shade[face])
            draw.polygon([tuple(p) for p in uv[face]], fill=(value, value, value))
        draw.text((index % 3 * 300 + 15, index // 3 * 300 + 40), f"view {index + 1}", fill="black")
    sheet.save(destination)


def prepare(cfg):
    import imageio_ffmpeg
    from quiethand.m3_materialization import scale_obj_cm_to_m
    workspace = cfg.workspace
    plan = read(workspace / BASE / "QH_M3_V1_2_TEMPORAL_PLAN.json")
    if plan["calibration_video_count"] != 30 or plan["evaluation_video_count"] != 0:
        raise ValueError("expected the existing 30/0 calibration plan")
    selected = {r["event_id"] for r in plan["events"]} if cfg.all_calibration else set(cfg.event_ids)
    rows = [r for r in plan["events"] if r["event_id"] in selected]
    if len(rows) != len(selected):
        raise ValueError("selection must be existing calibration IDs only")
    old = {e["event_id"]: e for e in read(workspace / BASE / "QH_M3_V1_2_PERCEPTION_INPUT.json")["events"]}
    sampled = {e["event_id"]: e for e in read(workspace / BASE / "QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json")["events"]}
    output = workspace / cfg.output_root
    if (output / "input.json").exists():
        raise ValueError("prepared input already exists; do not overwrite an issued run")
    destination = output / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / "cads"
    assets.mkdir(exist_ok=True)
    events = []
    for row in rows:
        eid = row["event_id"]
        folder = destination / eid
        folder.mkdir(exist_ok=True)
        event = copy.deepcopy(old[eid])
        geometry = event["object_state"]
        geometry.pop("tool_mesh_path"); geometry.pop("target_mesh_path")
        geometry["entities"] = []
        # These are the CAD identities actually used by the OLD pose runner,
        # not semantic answers. They are never included in the Qwen prompt.
        event["previous_role_to_entity"] = {
            role: "cad_" + Path(row["source_binding"][f"{role}_mesh"]["relative_path"]).stem.removesuffix("_cm")
            for role in ROLES}
        # Only CAD files are taken from source_binding. Role keys are discarded.
        mesh_paths = sorted({row["source_binding"][f"{r}_mesh"]["relative_path"] for r in ROLES})
        for relative in mesh_paths:
            source = workspace / "external_data/taco_v1" / relative
            identity = "cad_" + source.stem.removesuffix("_cm")
            mesh_path = assets / f"{identity}.obj"
            reference_path = assets / f"{identity}.png"
            if not mesh_path.exists():
                mesh_path.write_bytes(scale_obj_cm_to_m(source)[0])
            if not reference_path.exists():
                cad_sheet(mesh_path, identity, reference_path)
            geometry["entities"].append({"entity_id": identity,
                "mesh_path": mesh_path.relative_to(workspace).as_posix(),
                "reference_image": reference_path.relative_to(workspace).as_posix()})
        # Original RGB only, at the exact first geometry frame used by SAM2.
        frame = sampled[eid]["frame_indices"][0]
        video = workspace / "external_data/taco_v1" / row["source_binding"]["rgb"]["relative_path"]
        payload = subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1",
            "-i", str(video), "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        rgb = Image.frombytes("RGB", (1920, 1080), payload)
        rgb.resize((960, 540)).save(folder / "scene.png")
        candidate = read(workspace / BASE / "results/temporal_semantic/events" / f"{eid}.json")["candidate"]
        for index, role in enumerate(ROLES):
            box = candidate["start_sample_boxes_0_1000"][role]
            if box is None:
                crop = Image.new("RGB", (256, 256), "white")
                ImageDraw.Draw(crop).text((10, 120), "MISSING REGION", fill="black")
            else:
                x1, y1, x2, y2 = box
                crop = rgb.crop((int(x1 * 1.92), int(y1 * 1.08), int(x2 * 1.92), int(y2 * 1.08)))
                crop.thumbnail((640, 640))
            crop.save(folder / f"region_{index}.png")
        event["identity_images"] = [(folder / name).relative_to(workspace).as_posix()
            for name in ("scene.png", "region_0.png", "region_1.png")]
        event["source_frame_indices"] = sampled[eid]["frame_indices"]
        event.pop("materialized_file_binding")  # Old binding includes replaced mesh paths.
        events.append(event)
        print(f"Prepared {len(events)}/{len(rows)} {eid}", flush=True)
    atomic_json(output / "input.json", {"event_count": len(events), "evaluation_event_count": 0,
        "evaluation_results_opened": False, "events": events, "prompt": PROMPT})
    print(f"Prepared {len(events)} calibration event(s); old model outputs unchanged.")


def bind(cfg, manifest):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model_path = cfg.workspace / "external_data/quiethand_m3_resources/semantic"
    output = cfg.workspace / cfg.output_root / "binding"
    started = time.monotonic()
    status(output / "status.json", "loading_model", 0, len(manifest["events"]))
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
    for ordinal, event in enumerate(manifest["events"], 1):
        cached_path = None if cfg.reuse_root is None else cfg.workspace / cfg.reuse_root / "binding/events" / (event["event_id"] + ".json")
        if cached_path is not None and cached_path.is_file():
            binding = read(cached_path)
            for role in ROLES:
                resolve_role_entity(event, binding, role)
            atomic_json(output / "events" / (event["event_id"] + ".json"), dict(binding, reused_from=cached_path.relative_to(cfg.workspace).as_posix()))
            status(output / "status.json", "running", ordinal, len(manifest["events"]))
            continue
        content = [{"type": "image", "url": str(cfg.workspace / p)} for p in event["identity_images"]]
        for entity in entity_catalog(event["object_state"]).values():
            content += [{"type": "text", "text": "CAD entity " + entity["entity_id"]},
                        {"type": "image", "url": str(cfg.workspace / entity["reference_image"])}]
        content.append({"type": "text", "text": manifest["prompt"]})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=400)
        raw = processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        atomic_json(output / "raw" / (event["event_id"] + ".json"), {"raw_text": raw})
        try:
            binding = parse_visual_binding(raw, event)
        except (json.JSONDecodeError, ObjectIdentityError) as exc:
            # An invalid model answer is explicit missing identity, never the
            # legacy same-name mapping. One bad clip must not erase the batch.
            binding = {"event_id": event["event_id"], "role_to_entity": {r: None for r in ROLES},
                "reasons": {r: str(exc) for r in ROLES}, "source": "qwen_rgb_cad_visual_match",
                "status": "invalid_model_answer", "native_pose_used": False, "human_corrections_used": False}
        atomic_json(output / "events" / (event["event_id"] + ".json"), binding)
        status(output / "status.json", "running", ordinal, len(manifest["events"]))
    results = [read(output / "events" / (e["event_id"] + ".json")) for e in manifest["events"]]
    atomic_json(output / "audit.json", {"status": "COMPLETE", "event_count": len(manifest["events"]),
        "reused_event_count": sum("reused_from" in r for r in results),
        "invalid_event_count": sum(r.get("status") == "invalid_model_answer" for r in results),
        "unresolved_region_count": sum(v is None for r in results for v in r["role_to_entity"].values()),
        "elapsed_seconds": time.monotonic() - started,
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "native_pose_used": False, "human_corrections_used": False,
        "training_performed": False, "evaluation_results_opened": False})
    status(output / "status.json", "complete", len(manifest["events"]), len(manifest["events"]))


def rerun(cfg, manifest):
    sys.path.insert(0, str(ROOT / "scripts/quiethand/m3_v1_2"))
    from run_perception import _load_foundation, _poses, _artifact
    workspace = cfg.workspace
    output = workspace / cfg.output_root / "perception"
    started = time.monotonic()
    foundation = None
    counts = {"reused_existing_pose": 0, "reused_first_repair": 0, "rerun_pose": 0, "abstain": 0, "invalid": 0}
    total = len(manifest["events"])
    for ordinal, event in enumerate(manifest["events"], 1):
        eid = event["event_id"]
        folder = output / "events" / eid
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "object_state.json").is_file() and (folder / "segmentation.json").is_file():
            # Autopilot may resume after shared-GPU pressure; completed events
            # belong to this fixed input and must not be recomputed.
            for item in read(folder / "object_state.json")["items"].values():
                counts[item["pose_origin"] if item["status"] == "observed" else item["status"]] += 1
            status(output / "status.json", "running", ordinal, total)
            continue
        binding = read(workspace / cfg.output_root / "binding/events" / f"{eid}.json")
        old = workspace / "results_v1_2/perception"
        segments = read(old / "events" / eid / "segmentation.json")
        old_objects = read(old / "events" / eid / "object_state.json")
        cached_root = None if cfg.reuse_root is None else workspace / cfg.reuse_root / "perception"
        cached_path = None if cached_root is None else cached_root / "events" / eid / "object_state.json"
        cached = read(cached_path) if cached_path is not None and cached_path.is_file() else None
        rgbd = None
        objects, reused_segments = {}, {}
        for role in ROLES:
            segment = segments["items"][role]
            entity = resolve_role_entity(event, binding, role)
            reused_segments[role] = copy.deepcopy(segment)
            if segment["status"] == "observed":
                mask_path = folder / f"{role}_mask.npy"
                if not mask_path.exists():
                    mask_path.hardlink_to(old / segment["artifact"]["relative_path"])
                reused_segments[role]["artifact"]["relative_path"] = mask_path.relative_to(output).as_posix()
            if segment["status"] != "observed" or entity is None:
                objects[role] = {"status": "abstain", "artifact": None, "failure_reason": "object_identity_unresolved" if entity is None else "segmentation_unavailable"}
                counts["abstain"] += 1
                continue
            pose_path = folder / f"{role}_pose.npy"
            cached_item = None if cached is None else cached["items"][role]
            if cached_item is not None and cached_item["status"] == "observed" and cached_item["entity_id"] == entity["entity_id"]:
                source = cached_root / cached_item["artifact"]["relative_path"]
                if not pose_path.exists():
                    pose_path.hardlink_to(source)
                origin = "reused_first_repair"
            elif event["previous_role_to_entity"][role] == entity["entity_id"]:
                old_item = old_objects["items"][role]
                if old_item["status"] != "observed":
                    objects[role] = dict(old_item, entity_id=entity["entity_id"], pose_origin="existing_unavailable")
                    counts[old_item["status"]] += 1
                    continue
                source = old / old_item["artifact"]["relative_path"]
                if not pose_path.exists():
                    pose_path.hardlink_to(source)
                origin = "reused_existing_pose"
            else:
                if foundation is None:
                    foundation = _load_foundation(workspace / "external_repos/quiethand_m3/FoundationPose")
                if rgbd is None:
                    rgbs = [np.asarray(Image.open(workspace / p).convert("RGB")) for p in event["segmentation"]["ordered_rgb_frames"]]
                    depths = [np.load(workspace / p, allow_pickle=False) for p in event["object_state"]["ordered_depth_frames"]]
                    intrinsic = np.load(workspace / event["object_state"]["intrinsic_path"], allow_pickle=False)
                    rgbd = (rgbs, depths, intrinsic)
                masks = np.load(old / segment["artifact"]["relative_path"], allow_pickle=False)
                try:
                    poses = _poses(*foundation, workspace / entity["mesh_path"], *rgbd, masks)
                    atomic_npy(pose_path, poses)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    objects[role] = {"status": "invalid", "artifact": None, "failure_reason": str(exc), "entity_id": entity["entity_id"], "pose_origin": "rerun_failed"}
                    counts["invalid"] += 1
                    continue
                source = pose_path
                origin = "rerun_pose"
            counts[origin] += 1
            objects[role] = {"status": "observed", "entity_id": entity["entity_id"],
                "mesh_path": entity["mesh_path"], "identity_source": binding["source"], "failure_reason": None,
                "pose_origin": origin, "source_artifact": source.relative_to(workspace).as_posix(),
                "artifact": _artifact(output, pose_path, dtype="float32", shape=[15, 4, 4], unit="m_SE3", frame_id="camera")}
        atomic_json(folder / "object_state.json", {"schema": "quiethand.object_state.entity_bound.v1", "event_id": eid, "adapter": "object_state", "items": objects})
        atomic_json(folder / "segmentation.json", {"schema": "quiethand.m3.adapter_result.v1", "event_id": eid, "adapter": "segmentation", "items": reused_segments})
        status(output / "status.json", "running", ordinal, total)
    atomic_json(output / "audit.json", {"status": "COMPLETE", "event_count": total,
        "execution_counts": counts,
        "object_observed_count": sum(read(output / "events" / e["event_id"] / "object_state.json")["items"][r]["status"] == "observed" for e in manifest["events"] for r in ROLES),
        "elapsed_seconds": time.monotonic() - started, "segmentation_reused": True,
        "native_pose_used": False, "native_masks_used": False, "human_corrections_used": False,
        "training_performed": False, "evaluation_results_opened": False})
    status(output / "status.json", "complete", total, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "bind", "rerun"])
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True, help="workspace-relative directory for this run")
    parser.add_argument("--reuse-root", type=Path, help="workspace-relative completed identity repair to reuse")
    parser.add_argument("--all-calibration", action="store_true")
    parser.add_argument("--event-ids", nargs="+")
    cfg = parser.parse_args()
    if cfg.command == "prepare":
        if bool(cfg.event_ids) == cfg.all_calibration:
            parser.error("prepare needs either --event-ids or --all-calibration")
        return prepare(cfg)
    manifest = read(cfg.workspace / cfg.output_root / "input.json")
    if manifest["evaluation_event_count"] != 0 or manifest["evaluation_results_opened"] is not False:
        raise ValueError("repair is calibration-only")
    selected = {e["event_id"] for e in manifest["events"]}
    plan = read(cfg.workspace / BASE / "QH_M3_V1_2_TEMPORAL_PLAN.json")
    if not selected or not selected <= {e["event_id"] for e in plan["events"]}:
        raise ValueError("repair contains non-calibration event")
    {"bind": bind, "rerun": rerun}[cfg.command](cfg, manifest)


if __name__ == "__main__":
    main()
