#!/usr/bin/env python3
"""Create the calibration-only blinded human semantic audit package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_RAW_HAND_INPUT.json"
INPUT_SHA256 = "5ade716e9a7b0b400db72abdc7d16bed9e3476d7f09023bd287581237ed47703"
OUTPUT = ROOT / "artifacts" / "quiethand" / "m3" / "human_audit_calibration"
ROLE = "active|support|both|neither|unknown"
CONTACT = "tool|target|both|neither|unknown"
EVIDENCE = "visible|ambiguous|not_visible"


class HumanAuditPackageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_order(events: list[dict[str, object]], salt: str) -> list[dict[str, object]]:
    return sorted(
        events,
        key=lambda event: hashlib.sha256(f"{salt}\0{event['event_id']}".encode()).digest(),
    )


def write_sheet(path: Path, events: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, dialect="excel-tab", lineterminator="\n")
        writer.writerow(
            [
                "audit_index",
                "event_id",
                "video_relative_path",
                "left_role",
                "right_role",
                "left_contact",
                "right_contact",
                "evidence",
                "comment_optional",
            ]
        )
        for index, event in enumerate(events, start=1):
            event_id = event["event_id"]
            writer.writerow([index, event_id, f"videos/{event_id}.mp4", "", "", "", "", "", ""])


def main() -> int:
    temporary: Path | None = None
    try:
        if OUTPUT.exists():
            raise HumanAuditPackageError("human audit package destination already exists")
        if sha256_file(INPUT) != INPUT_SHA256:
            raise HumanAuditPackageError("frozen raw-hand manifest changed")
        manifest = json.loads(INPUT.read_text(encoding="utf-8"))
        events = manifest.get("events")
        if (
            manifest.get("adapter") != "raw_hand"
            or manifest.get("event_count") != 150
            or manifest.get("evaluation_event_count") != 0
            or manifest.get("evaluation_model_results_opened") is not False
            or not isinstance(events, list)
            or len(events) != 150
        ):
            raise HumanAuditPackageError("human audit source is not the exact calibration envelope")
        temporary = Path(tempfile.mkdtemp(prefix=".human_audit_calibration.", dir=OUTPUT.parent))
        videos = temporary / "videos"
        videos.mkdir()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        video_rows = []
        for ordinal, event in enumerate(events, start=1):
            event_id = event.get("event_id")
            frames = event.get("ordered_rgb_frames")
            if not isinstance(event_id, str) or not isinstance(frames, list) or len(frames) != 15:
                raise HumanAuditPackageError("human audit event identity or frame coverage is invalid")
            expected = [f"artifacts/quiethand/m3/calibration_inputs/{event_id}/rgb_{index:02d}.png" for index in range(15)]
            if frames != expected:
                raise HumanAuditPackageError("human audit frame order changed")
            source_pattern = ROOT / "artifacts" / "quiethand" / "m3" / "calibration_inputs" / event_id / "rgb_%02d.png"
            output_video = videos / f"{event_id}.mp4"
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "30",
                    "-start_number",
                    "0",
                    "-i",
                    str(source_pattern),
                    "-frames:v",
                    "15",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    "-movflags",
                    "+faststart",
                    str(output_video),
                ],
                check=True,
            )
            if not output_video.is_file() or output_video.stat().st_size <= 0:
                raise HumanAuditPackageError("ffmpeg did not create a non-empty blinded video")
            video_rows.append(
                {
                    "event_id": event_id,
                    "relative_path": f"videos/{event_id}.mp4",
                    "bytes": output_video.stat().st_size,
                    "sha256": sha256_file(output_video),
                    "frame_count": 15,
                    "display_fps": 30,
                    "duration_seconds": 0.5,
                }
            )
            if ordinal % 10 == 0:
                print(f"[package] {ordinal}/150", flush=True)
        write_sheet(temporary / "annotator_a.tsv", audit_order(events, "QH-M3-HUMAN-A"))
        write_sheet(temporary / "annotator_b.tsv", audit_order(events, "QH-M3-HUMAN-B"))
        (temporary / "README.md").write_text(
            "# QuietHand M3 calibration blind audit\n\n"
            "Each annotator independently labels every row without viewing the other sheet or any model/native output. "
            "Play the linked 0.5-second clip as often as needed. Fill only these exact enums:\n\n"
            f"- left_role/right_role: `{ROLE}`\n"
            f"- left_contact/right_contact: `{CONTACT}`\n"
            f"- evidence: `{EVIDENCE}`\n\n"
            "Use `unknown` and `ambiguous` when visible evidence is insufficient. Do not infer support from low motion alone, "
            "and do not label force, friction, pressure, normals or millimetre contact. Leave every event_id and path unchanged.\n",
            encoding="utf-8",
        )
        package_manifest = {
            "schema": "quiethand.m3.human_audit_package.v1",
            "status": "READY_FOR_TWO_INDEPENDENT_ANNOTATORS",
            "source_input_sha256": INPUT_SHA256,
            "event_count": 150,
            "calibration_sequence_count": 30,
            "evaluation_event_count": 0,
            "model_or_native_outputs_included": False,
            "split_or_action_metadata_included": False,
            "video_encoding": "15 exact ordered source frames displayed at 30 fps; H.264 is human-viewing-only",
            "annotator_order_rule": "sha256(salt || NUL || opaque_event_id)",
            "videos": video_rows,
            "scientific_result": False,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, OUTPUT)
        temporary = None
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError, HumanAuditPackageError) as exc:
        print(f"[hold] HOLD_HUMAN_AUDIT_PACKAGE: {exc}", file=sys.stderr)
        return 3
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    print(f"[artifact] {OUTPUT / 'manifest.json'}")
    print(f"[sha256] {sha256_file(OUTPUT / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
