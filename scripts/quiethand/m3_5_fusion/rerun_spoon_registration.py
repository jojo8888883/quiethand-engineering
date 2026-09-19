#!/usr/bin/env python3
"""Rerun only the diagnosed spoon pose with a localized initialization mask."""
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
from run_perception import _load_foundation, _poses

EVENT_ID = 'qh-m3-v12-cal-594fc82381d222917ad15a7b'
ENTITY_ID = 'cad_200'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path,
                        default=Path('artifacts/quiethand/m3_5/entity_region_repair'))
    cfg = parser.parse_args()
    workspace, output = cfg.workspace, cfg.workspace/cfg.output_root
    manifest = read(output/'input.json')
    if manifest['evaluation_event_count'] != 0 or manifest['evaluation_results_opened'] is not False:
        raise ValueError('calibration-only repair')
    event = next(e for e in manifest['events'] if e['event_id'] == EVENT_ID)
    entity = next(e for e in event['entities'] if e['entity_id'] == ENTITY_ID)
    folder = output/'events'/EVENT_ID/ENTITY_ID
    located, verified = read(folder/'locate.json'), read(folder/'verify.json')
    if located['status'] != 'located' or verified['status'] != 'accepted':
        raise ValueError('spoon localization/mask check is not available')
    masks = np.load(folder/'mask.npy', allow_pickle=False)
    filtered = initialization_mask(masks[0], located['location'])
    removed = masks[0] & ~filtered
    if not removed.any():
        raise ValueError('diagnosed external mask pixels are absent')
    result = output/'spoon_registration'
    result.mkdir(exist_ok=True)
    status(result/'status.json', 'pose', 0, 1)
    atomic_npy(result/'initialization_mask.npy', filtered)
    rgb = [np.asarray(Image.open(workspace/p).convert('RGB')) for p in event['rgb_frames']]
    depth = [np.load(workspace/p, allow_pickle=False) for p in event['depth_frames']]
    intrinsic = np.load(workspace/event['intrinsic_path'], allow_pickle=False)
    started = time.monotonic()
    foundation = _load_foundation(workspace/'external_repos/quiethand_m3/FoundationPose')
    pose_masks = masks.copy()
    pose_masks[0] = filtered
    poses = _poses(*foundation, workspace/entity['mesh_path'], rgb, depth, intrinsic, pose_masks)
    elapsed = time.monotonic()-started
    atomic_npy(result/'pose.npy', poses)
    record = dict(status='COMPLETE', event_id=EVENT_ID, entity_id=ENTITY_ID,
        source_frame_indices=event['source_frame_indices'], pose_path=(result/'pose.npy').relative_to(workspace).as_posix(),
        initialization_mask_path=(result/'initialization_mask.npy').relative_to(workspace).as_posix(),
        original_first_mask_pixels=int(masks[0].sum()), retained_first_mask_pixels=int(filtered.sum()),
        removed_first_mask_pixels=int(removed.sum()), rule='actual_sam_mask AND same_localization_box_on_first_frame',
        coordinate_frame='camera', unit='m_SE3', elapsed_seconds=elapsed,
        evaluation_results_opened=False, native_masks_used=False, native_poses_used=False,
        human_corrections_used=False, model_changed=False)
    atomic_json(result/'audit.json', record)
    status(result/'status.json', 'complete', 1, 1)


if __name__ == '__main__':
    main()
