#!/usr/bin/env python3
"""Independently run FoundationPose's learned RGB-D refiner from frozen poses."""
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
from repair_entity_regions import read
from run_perception import _load_foundation, _valid_pose
from runtime_common import atomic_json, atomic_npy, status

EVENT_ID = 'qh-m3-v12-cal-594fc82381d222917ad15a7b'
ENTITY_ID = 'cad_200'
FOUNDATION_REPOSITORY_REVISION = 'a1b694b83e633c2cb6115b9063d940a687759392'


def external_to_centered_pose(external_pose, transform_to_centered_mesh):
    external_pose = np.asarray(external_pose, dtype=np.float64)
    transform_to_centered_mesh = np.asarray(transform_to_centered_mesh, dtype=np.float64)
    if external_pose.shape != (4, 4) or transform_to_centered_mesh.shape != (4, 4):
        raise ValueError('pose transforms must be 4x4')
    return external_pose @ np.linalg.inv(transform_to_centered_mesh)


def rotation_change_deg(before, after):
    relative = after[:3, :3] @ before[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


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
    source = output / 'spoon_rgbd_depth_refinement'
    source_audit = read(source / 'audit.json')
    if (source_audit['status'] != 'COMPLETE'
            or source_audit['native_poses_used'] is not False
            or source_audit['method'] != 'one robust same-pixel mask-interior RGB-D translation refinement'):
        raise ValueError('frozen prediction-side source pose is unavailable')
    poses = np.load(source / 'pose.npy', allow_pickle=False)
    if poses.shape != (15, 4, 4):
        raise ValueError('expected exactly15 frozen source poses')
    rgbs = [np.asarray(Image.open(workspace / path).convert('RGB'))
            for path in event['rgb_frames']]
    depths = [np.load(workspace / path, allow_pickle=False)
              for path in event['depth_frames']]
    intrinsic = np.load(workspace / event['intrinsic_path'], allow_pickle=False)

    repository = workspace / 'external_repos/quiethand_m3/FoundationPose'
    FoundationPose, scorer, refiner, context = _load_foundation(repository)
    import torch
    import trimesh
    mesh = trimesh.load(workspace / entity['mesh_path'], process=False)
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir='/tmp/quiethand-foundationpose-debug', debug=0, glctx=context)
    transform_to_centered = estimator.get_tf_to_centered_mesh().detach().cpu().numpy()

    result = output / 'spoon_foundation_refinement'
    result.mkdir(exist_ok=True)
    status(result / 'status.json', 'foundation_rgbd_refinement', 0, 15)
    started = time.monotonic()
    refined, records = [], []
    for index, (source_pose, rgb, depth) in enumerate(zip(poses, rgbs, depths, strict=True)):
        internal = external_to_centered_pose(source_pose, transform_to_centered)
        estimator.pose_last = torch.as_tensor(internal, device='cuda', dtype=torch.float)
        value = estimator.track_one(
            rgb=rgb, depth=depth, K=intrinsic, iteration=2).astype(np.float64)
        if not _valid_pose(value):
            raise ValueError(f'FoundationPose returned invalid SE3 at frame {index}')
        refined.append(value)
        records.append({
            'frame_index': index,
            'rotation_change_from_source_deg': rotation_change_deg(source_pose, value),
            'translation_change_from_source_m': float(
                np.linalg.norm(value[:3, 3] - source_pose[:3, 3])),
        })
        status(result / 'status.json', 'foundation_rgbd_refinement', index + 1, 15)
    refined = np.stack(refined).astype(np.float32)
    elapsed = time.monotonic() - started
    atomic_npy(result / 'pose.npy', refined)
    atomic_json(result / 'audit.json', dict(
        status='COMPLETE', event_id=EVENT_ID, entity_id=ENTITY_ID,
        source_frame_indices=event['source_frame_indices'],
        pose_path=(result / 'pose.npy').relative_to(workspace).as_posix(),
        source_pose_path=(source / 'pose.npy').relative_to(workspace).as_posix(),
        method='independent learned FoundationPose RGB-D render refinement from frozen poses',
        foundationpose_revision=FOUNDATION_REPOSITORY_REVISION,
        foundationpose_operation='track_one', refinement_iterations=2,
        global_registration_used=False, scorer_used=False,
        source_pose_reset_before_every_frame=True, frame_chaining_used=False,
        frame_records=records, coordinate= 'RGB and rendered/observed XYZ crop refinement',
        coordinate_frame='camera', unit='m_SE3', elapsed_seconds=elapsed,
        evaluation_results_opened=False, native_masks_used=False,
        native_poses_used=False, human_corrections_used=False,
        temporal_smoothing_used=False, frame_specific_rules_used=False,
        post_result_tuning_used=False, model_changed=False, gpu_used=True))
    status(result / 'status.json', 'complete', 15, 15)


if __name__ == '__main__':
    main()
