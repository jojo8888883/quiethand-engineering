#!/usr/bin/env python3
"""Register the diagnosed spoon independently on every sampled frame."""
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
from runtime_common import atomic_json, atomic_npy, status
from repair_entity_regions import initialization_mask, read
from run_perception import _load_foundation, _valid_pose

EVENT_ID = 'qh-m3-v12-cal-594fc82381d222917ad15a7b'
ENTITY_ID = 'cad_200'


def register_each_frame(estimator, rgbs, depths, intrinsic, masks):
    if not (len(rgbs) == len(depths) == len(masks) == 15):
        raise ValueError('expected exactly 15 aligned RGB-D-mask frames')
    values = []
    for index, (rgb, depth, mask) in enumerate(zip(rgbs, depths, masks, strict=True)):
        if mask.sum() < 4 or ((depth >= 0.001) & mask).sum() < 4:
            raise ValueError(f'frame {index} has fewer than four valid masked depth pixels')
        values.append(estimator.register(
            K=intrinsic, rgb=rgb, depth=depth, ob_mask=mask, iteration=5))
    poses = np.stack(values).astype(np.float32)
    if poses.shape != (15, 4, 4) or not all(
            _valid_pose(pose.astype(np.float64)) for pose in poses):
        raise ValueError('FoundationPose output is non-finite or not SE(3)')
    return poses


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
    result = output / 'spoon_per_frame_registration'
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
    poses = register_each_frame(estimator, rgbs, depths, intrinsic, masks)
    elapsed = time.monotonic() - started
    atomic_npy(result / 'pose.npy', poses)
    record = dict(
        status='COMPLETE', event_id=EVENT_ID, entity_id=ENTITY_ID,
        source_frame_indices=event['source_frame_indices'],
        pose_path=(result / 'pose.npy').relative_to(workspace).as_posix(),
        method='independent FoundationPose register on every sampled frame',
        first_mask_rule='actual_sam_mask AND same_localization_box',
        later_mask_rule='actual_sam_mask for the same entity and frame',
        mask_pixels=[int(mask.sum()) for mask in masks],
        coordinate_frame='camera', unit='m_SE3', elapsed_seconds=elapsed,
        evaluation_results_opened=False, native_masks_used=False,
        native_poses_used=False, human_corrections_used=False,
        temporal_smoothing_used=False, model_changed=False)
    atomic_json(result / 'audit.json', record)
    status(result / 'status.json', 'complete', 15, 15)


if __name__ == '__main__':
    main()
