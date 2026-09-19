#!/usr/bin/env python3
"""Package one existing calibration record and its H-RDT input gaps; no inference."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/quiethand/downstream_requirements"
V12 = ROOT / "artifacts/quiethand/m3_v1_2"
M35 = ROOT / "artifacts/quiethand/m3_5"
REVISION = "8be02cb5e631898dfab12f9b6bbb77e32999547a"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = read_json(M35 / "QH_M3_5_RAW_HAND_METRIC_INPUT.json")
    assert manifest["evaluation_event_count"] == 0
    assert manifest["evaluation_model_results_opened"] is False
    event = manifest["events"][0]
    event_id = event["event_id"]
    plan = read_json(V12 / "QH_M3_V1_2_TEMPORAL_PLAN.json")
    source = next(x for x in plan["events"] if x["event_id"] == event_id)
    semantic_path = V12 / "results/temporal_semantic/events" / f"{event_id}.json"
    semantic = read_json(semantic_path)
    hand_root = M35 / "results/raw_hand_metric"
    hand_path = hand_root / "events" / f"{event_id}.json"
    hand = read_json(hand_path)
    assert hand["event_id"] == semantic["event_id"] == event_id
    indices = np.asarray(event["source_frame_indices"], dtype=np.int64)
    sample_rgb = OUT / "sample_rgb"
    sample_rgb.mkdir(parents=True, exist_ok=True)

    # Decode only this calibration RGB video, never an evaluation or native hand file.
    video = ROOT / "external_data/taco_v1" / source["source_rgb"]["relative_path"]
    stream = imageio_ffmpeg.read_frames(str(video))
    video_meta = next(stream)
    selected = {int(index): i for i, index in enumerate(indices)}
    frame_count = 0
    for frame_index, pixels in enumerate(stream):
        frame_count += 1
        if frame_index in selected:
            image = Image.frombytes("RGB", tuple(video_meta["size"]), pixels)
            image.save(sample_rgb / f"rgb_{selected[frame_index]:02d}.png")
    fps = float(Fraction(source["source_rate"]))
    assert abs(video_meta["fps"] - fps) < 1e-6
    assert frame_count == source["source_frame_count"]
    assert np.all(np.diff(indices) > 0) and indices[-1] < frame_count
    assert len(indices) == len(event["ordered_rgb_frames"]) == 15
    frames = []
    for i, (index, rgb, bound) in enumerate(zip(
        indices, event["ordered_rgb_frames"], event["materialized_file_binding"]["rgb"]
    )):
        assert bound["source_frame_index"] == index and bound["relative_path"] == rgb
        local_rgb = sample_rgb / f"rgb_{i:02d}.png"
        with Image.open(local_rgb) as im:
            im.load()
            assert list(im.size) == event["camera_model"]["image_size_px"]
            assert im.mode == "RGB"
        frames.append({"sample_index": i, "source_frame_index": int(index),
                       "timestamp_seconds": float(index / fps), "rgb_path": str(local_rgb),
                       "original_inference_frame_path": rgb,
                       "rgb_provenance": "decoded from original video at the manifest frame index; original inference PNG is not local; pixel identity not asserted"})

    arrays = {"source_frame_indices": indices, "timestamps_seconds": indices / fps,
              "intrinsic_px": np.asarray(event["camera_model"]["intrinsic_px"], dtype=np.float64)}
    for side in ("left", "right"):
        item = hand["items"][side]
        assert item["status"] == "observed"
        artifact = item["artifact"]
        assert artifact["unit"] == "m" and artifact["frame_id"] == "camera"
        assert artifact["frame_indices"] == list(range(len(indices)))
        vertices = np.load(hand_root / artifact["relative_path"], allow_pickle=False)
        assert vertices.shape == (len(indices), 778, 3)
        assert vertices.dtype == np.float32 and np.isfinite(vertices).all()
        arrays[f"{side}_vertices_camera_m"] = vertices

    # Dataset calibration exists; preserve it as a reference, not an invented world transform.
    extrinsic_path = ROOT / "external_data/taco_v1" / source["source_binding"]["extrinsic"]["relative_path"]
    extrinsic = np.load(extrinsic_path, allow_pickle=False)
    assert extrinsic.shape == (frame_count, 4, 4) and np.isfinite(extrinsic).all()

    # Mirror the pinned loader's explicit index arithmetic, not an idealized description.
    index = int(indices[0])
    stride, horizon = 3, 16
    max_index = frame_count - 2
    action_end = min(index + horizon * stride, max_index + 1)
    target_indices = list(range(index + 1, action_end + 1, stride))
    while len(target_indices) < horizon:
        target_indices.append(target_indices[-1] if target_indices else index + 1)
    target_indices = target_indices[:horizon]
    image_start = max(index - 1 * stride + 1, 0)
    image_indices = list(range(image_start, index + 1, stride))[-1:]
    assert target_indices == list(range(20, 66, 3))
    assert image_indices == [17]

    gaps = [
        {"field": "states[1,48] and actions[16,48]", "status": "not_exportable_from_current_artifacts",
         "reason": "Only 15 camera-frame MANO meshes were persisted. Wrist rotations, canonical wrist/joint mapping and temporally dense world-frame actions were not exported."},
        {"field": "temporally aligned hand states", "status": "sparse",
         "reason": "The 15 samples have 6/7-frame gaps. They cannot be relabeled as a contiguous 30 Hz sequence."},
        {"field": "stationary coordinate frame", "status": "not_yet_converted",
         "reason": "TACO camera extrinsics exist as dataset calibration; their direction/axes and the hand joint convention must be established before conversion. No camera trajectory has been estimated from ordinary RGB here."},
        {"field": "task language embedding", "status": "not_generated",
         "reason": "Raw Qwen action/object candidates are retained, not certified task instructions. No T5 embedding or human correction-derived label was created."},
        {"field": "train-split normalization statistics", "status": "not_computed",
         "reason": "A single calibration example does not supply production training statistics. No evaluation data was used."},
    ]
    sample = {
        "event_id": event_id, "selection": "first event in the frozen calibration manifest",
        "package_kind": "existing_evidence_and_target_schema_mapping",
        "ready_for_hrdt_training": False,
        "reason": "Required target tensors do not yet exist; this is not a substitute training format.",
        "consumer": {"name": "H-RDT EgoDex pretraining", "revision": REVISION,
                     "repo": "https://github.com/HongzheBi/H_RDT"},
        "source": {"dataset": "TACO", "split": "calibration", "video_path": str(video),
                   "frame_count": frame_count, "fps": fps, "image_size_px": list(video_meta["size"])},
        "frames": frames,
        "geometry": {"file": "sample_arrays.npz", "unit": "m", "frame": "per-frame camera",
                     "hand_arrays": ["left_vertices_camera_m", "right_vertices_camera_m"],
                     "intrinsic_key": "intrinsic_px", "prediction_source": str(hand_path)},
        "available_camera_calibration": {"path": str(extrinsic_path), "shape": list(extrinsic.shape),
                                         "used_to_transform_hands": False, "source": "dataset calibration, not RGB-estimated"},
        "raw_language_candidate": {"source": str(semantic_path), "candidate": semantic["candidate"],
                                   "accepted_as_training_instruction": False},
        "target_layout": {"states": [1, 48], "actions": [16, 48],
                          "per_hand": "wrist position(3), rotation first column then second column(6), thumb/index/middle/ring/little tip positions(15)",
                          "hand_order": ["left", "right"], "language_embedding": ["L", 4096],
                          "joint_frame": "stationary EgoDex ARKit origin; upstream code uses stored transforms without camera conversion"},
        "consumer_sampling_example": {"anchor_source_frame": index, "image_source_frames_in_pinned_code": image_indices,
            "target_source_frames": target_indices, "target_times_seconds": [x / fps for x in target_indices],
            "target_frames_with_existing_mesh": sorted(set(target_indices) & set(indices.tolist())),
            "warning": "Pinned source uses image index t-2 for image_history=1, stride=3. This offset is reported, not silently adopted as desired alignment."},
        "gaps": gaps,
        "fields_not_consumed_by_this_training_path": ["object masks", "object meshes", "object 6D poses", "active/support role", "contact label", "C-OAME"],
        "training_run": False, "model_inference_run": False, "evaluation_opened": False,
        "human_corrections_used_as_labels": False, "native_hand_or_object_labels_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "sample_arrays.npz", **arrays)
    (OUT / "sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Verify saved values survive packaging and that no target tensors were fabricated.
    with np.load(OUT / "sample_arrays.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, value in arrays.items():
            np.testing.assert_array_equal(saved[name], value)
    saved_sample = read_json(OUT / "sample.json")
    assert saved_sample["ready_for_hrdt_training"] is False
    verification = {"status": "PASS_EVIDENCE_PACKAGE_ONLY", "event_id": event_id,
                    "decoded_video_frames": frame_count, "fps": fps,
                    "matched_rgb_and_left_right_mesh_samples": len(indices),
                    "camera_calibration_available": True,
                    "target_temporal_mesh_coverage": f"{len(set(target_indices) & set(indices.tolist()))}/{horizon}",
                    "array_round_trip_exact": True, "hrdt_loader_executed": False,
                    "model_or_loss_executed": False, "ready_for_hrdt_training": False,
                    "note": "Source path through model loss was traced statically; no training success is claimed."}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
