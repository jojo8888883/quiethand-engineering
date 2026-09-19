#!/usr/bin/env python3
"""Track one spoon using temporal and current-mask pose candidates."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/quiethand/m3_jobs'))
sys.path.insert(0, str(ROOT / 'scripts/quiethand/m3_v1_2'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_pose_consistency import projected_mask_iou, select_source
from repair_entity_regions import initialization_mask, read
from run_perception import _load_foundation, _valid_pose
from runtime_common import atomic_json, atomic_npy, status

EVENT_ID = 'qh-m3-v12-cal-594fc82381d222917ad15a7b'
ENTITY_ID = 'cad_200'


def internal_pose(estimator):
    return estimator.pose_last.detach().cpu().numpy().reshape(4, 4).copy()


def set_internal_pose(estimator, pose):
    import torch
    estimator.pose_last = torch.as_tensor(pose, device='cuda', dtype=torch.float)


def mask_aware_track(estimator, vertices, rgbs, depths, intrinsic, masks, progress_path):
    if not (len(rgbs) == len(depths) == len(masks) == 15):
        raise ValueError('expected exactly 15 aligned RGB-D-mask frames')
    for index, (depth, mask) in enumerate(zip(depths, masks, strict=True)):
        if mask.sum() < 4 or ((depth >= 0.001) & mask).sum() < 4:
            raise ValueError(f'frame {index} has fewer than four valid masked depth pixels')

    first = estimator.register(
        K=intrinsic, rgb=rgbs[0], depth=depths[0], ob_mask=masks[0], iteration=5)
    values = [first]
    selected_internal = internal_pose(estimator)
    first_iou = projected_mask_iou(vertices, first, intrinsic, masks[0])
    decisions = [dict(frame_index=0, selected='register', track_iou=None,
                      register_iou=first_iou)]
    status(progress_path, 'pose', 1, 15)

    for index in range(1, 15):
        set_internal_pose(estimator, selected_internal)
        track = estimator.track_one(
            rgb=rgbs[index], depth=depths[index], K=intrinsic, iteration=2)
        track_internal = internal_pose(estimator)

        register = estimator.register(
            K=intrinsic, rgb=rgbs[index], depth=depths[index],
            ob_mask=masks[index], iteration=5)
        register_internal = internal_pose(estimator)
        track_iou = projected_mask_iou(vertices, track, intrinsic, masks[index])
        register_iou = projected_mask_iou(vertices, register, intrinsic, masks[index])
        selected = select_source(track_iou, register_iou)
        if selected == 'track':
            values.append(track)
            selected_internal = track_internal
        else:
            values.append(register)
            selected_internal = register_internal
        set_internal_pose(estimator, selected_internal)
        decisions.append(dict(frame_index=index, selected=selected,
                              track_iou=track_iou, register_iou=register_iou))
        status(progress_path, 'pose', index + 1, 15)

    poses = np.stack(values).astype(np.float32)
    if poses.shape != (15, 4, 4) or not all(
            _valid_pose(pose.astype(np.float64)) for pose in poses):
        raise ValueError('FoundationPose output is non-finite or not SE(3)')
    return poses, decisions


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
    folder = output / 'events' / EVENT_ID / ENTITY_ID
    located, verified = read(folder / 'locate.json'), read(folder / 'verify.json')
    if located['status'] != 'located' or verified['status'] != 'accepted':
        raise ValueError('spoon localization/mask check is not available')

    masks = np.load(folder / 'mask.npy', allow_pickle=False).astype(bool)
    masks[0] = initialization_mask(masks[0], located['location'])
    result = output / 'spoon_mask_aware_tracking'
    result.mkdir(exist_ok=True)
    status(result / 'status.json', 'pose', 0, 15)
    rgbs = [np.asarray(Image.open(workspace / p).convert('RGB')) for p in event['rgb_frames']]
    depths = [np.load(workspace / p, allow_pickle=False) for p in event['depth_frames']]
    intrinsic = np.load(workspace / event['intrinsic_path'], allow_pickle=False)

    FoundationPose, scorer, refiner, context = _load_foundation(
        workspace / 'external_repos/quiethand_m3/FoundationPose')
    import trimesh
    mesh = trimesh.load(workspace / entity['mesh_path'], process=False)
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir='/tmp/quiethand-foundationpose-debug', debug=0, glctx=context)
    started = time.monotonic()
    poses, decisions = mask_aware_track(
        estimator, np.asarray(mesh.vertices), rgbs, depths, intrinsic, masks,
        result / 'status.json')
    elapsed = time.monotonic() - started
    atomic_npy(result / 'pose.npy', poses)
    record = dict(
        status='COMPLETE', event_id=EVENT_ID, entity_id=ENTITY_ID,
        source_frame_indices=event['source_frame_indices'],
        pose_path=(result / 'pose.npy').relative_to(workspace).as_posix(),
        method='online mask-aware selection between temporal track and current-frame register',
        selection_rule='select track when projected CAD convex-hull IoU with current SAM mask is not worse; otherwise select register',
        decisions=decisions, coordinate_frame='camera', unit='m_SE3',
        elapsed_seconds=elapsed, evaluation_results_opened=False,
        native_masks_used=False, native_poses_used=False,
        human_corrections_used=False, temporal_smoothing_used=False,
        frame_specific_rules_used=False, model_changed=False)
    atomic_json(result / 'audit.json', record)
    status(result / 'status.json', 'complete', 15, 15)


if __name__ == '__main__':
    main()
