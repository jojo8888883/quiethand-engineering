#!/usr/bin/env python3
"""Refine the selected spoon pose translation from same-mask RGB-D depth."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/quiethand/m3_jobs'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_depth_translation import refine_translation_once
from repair_entity_regions import initialization_mask, read
from runtime_common import atomic_json, atomic_npy, status

EVENT_ID = 'qh-m3-v12-cal-594fc82381d222917ad15a7b'
ENTITY_ID = 'cad_200'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path,
                        default=Path('artifacts/quiethand/m3_5/entity_region_repair'))
    cfg = parser.parse_args()
    workspace, output = cfg.workspace, cfg.workspace / cfg.output_root
    manifest = read(output / 'input.json')
    if manifest['evaluation_event_count'] != 0 or manifest['evaluation_results_opened'] is not False:
        raise ValueError('calibration-only repair')
    event = next(e for e in manifest['events'] if e['event_id'] == EVENT_ID)
    entity = next(e for e in event['entities'] if e['entity_id'] == ENTITY_ID)
    source = output / 'spoon_mask_aware_tracking'
    source_audit = read(source / 'audit.json')
    if source_audit['status'] != 'COMPLETE' or source_audit['native_poses_used'] is not False:
        raise ValueError('prediction-side source pose is unavailable')
    poses = np.load(source / 'pose.npy', allow_pickle=False)
    folder = output / 'events' / EVENT_ID / ENTITY_ID
    located = read(folder / 'locate.json')
    masks = np.load(folder / 'mask.npy', allow_pickle=False).astype(bool)
    masks[0] = initialization_mask(masks[0], located['location'])
    depths = [np.load(workspace / path, allow_pickle=False) for path in event['depth_frames']]
    intrinsic = np.load(workspace / event['intrinsic_path'], allow_pickle=False)

    import trimesh
    mesh = trimesh.load(workspace / entity['mesh_path'], process=False)
    result = output / 'spoon_rgbd_depth_refinement'
    result.mkdir(exist_ok=True)
    status(result / 'status.json', 'depth_refinement', 0, 15)
    started = time.monotonic()
    refined, records = [], []
    for index, (pose, depth, mask) in enumerate(zip(poses, depths, masks, strict=True)):
        value, record = refine_translation_once(
            np.asarray(mesh.vertices), pose.astype(np.float64), intrinsic, depth, mask)
        refined.append(value)
        records.append(dict(frame_index=index, **record))
        status(result / 'status.json', 'depth_refinement', index + 1, 15)
    refined = np.stack(refined).astype(np.float32)
    elapsed = time.monotonic() - started
    atomic_npy(result / 'pose.npy', refined)
    atomic_json(result / 'audit.json', dict(
        status='COMPLETE', event_id=EVENT_ID, entity_id=ENTITY_ID,
        source_frame_indices=event['source_frame_indices'],
        pose_path=(result / 'pose.npy').relative_to(workspace).as_posix(),
        source_pose_path=(source / 'pose.npy').relative_to(workspace).as_posix(),
        method='one robust same-pixel mask-interior RGB-D translation refinement',
        estimator='median observed-minus-projected-visible-surface 3D residual',
        iterations=1, orientation_changed=False, frame_records=records,
        coordinate_frame='camera', unit='m_SE3', elapsed_seconds=elapsed,
        evaluation_results_opened=False, native_masks_used=False,
        native_poses_used=False, human_corrections_used=False,
        temporal_smoothing_used=False, frame_specific_rules_used=False,
        model_changed=False))
    status(result / 'status.json', 'complete', 15, 15)


if __name__ == '__main__':
    main()
