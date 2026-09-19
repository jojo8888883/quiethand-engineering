#!/usr/bin/env python3
"""Post-hoc visual and native-pose QA for RGB-D spoon depth refinement."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from build_identity_batch_preview import decode_rgb, draw_cloud
from diagnose_trajectories import trajectory_errors, transform_points

EID = 'qh-m3-v12-cal-594fc82381d222917ad15a7b'
FRAMES = [24, 33, 41, 50, 58, 67, 75, 84, 93, 101, 110, 118, 127, 135, 144]
SELECTED = (0, 3, 4, 7, 11, 14)


def main():
    taco = ROOT / 'external_data/taco_v1'
    mask_aware = ROOT / 'artifacts/quiethand/m3_5/entity_region_repair/spoon_mask_aware_tracking'
    result = ROOT / 'artifacts/quiethand/m3_5/entity_region_repair/spoon_rgbd_depth_refinement'
    mesh = trimesh.load(
        ROOT / 'artifacts/quiethand/m3_5/identity_batch/inputs/cads/cad_200.obj',
        process=False)
    vertices = np.asarray(mesh.vertices)
    sample = vertices[::max(1, len(vertices) // 900)]
    K = np.loadtxt(taco / 'Egocentric_Camera_Parameters/(put out, spoon, plate)/20231024_056/egocentric_intrinsic.txt')
    cameras = np.load(taco / 'Egocentric_Camera_Parameters/(put out, spoon, plate)/20231024_056/egocentric_frame_extrinsic.npy')[FRAMES]
    reference = cameras @ np.load(
        taco / 'Object_Poses/(put out, spoon, plate)/20231024_056/tool_200.npy')[FRAMES]
    poses = {
        'mask_aware': np.load(mask_aware / 'pose.npy'),
        'rgbd_refined': np.load(result / 'pose.npy'),
        'native_QA': reference,
    }
    rgbs = decode_rgb(
        ROOT / 'artifacts/quiethand/m3_v1_2/result_preview/source_videos' / f'{EID}.mp4',
        FRAMES)
    canvas = Image.new('RGB', (2560, len(SELECTED) * 380), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(SELECTED):
        for col, label in enumerate(('source', 'mask_aware', 'rgbd_refined', 'native_QA')):
            draw.text((col * 640 + 8, row * 380 + 4),
                      f'{label} | frame {FRAMES[index]}', fill='black')
            image = Image.fromarray(rgbs[index]).copy()
            if label != 'source':
                pose = poses[label][index]
                cloud = sample @ pose[:3, :3].T + pose[:3, 3]
                draw_cloud(image, cloud, K, '#ff8a22', 1)
            canvas.paste(image, (col * 640, row * 380 + 20))
    canvas.save(result / 'qa_geometry.jpg', quality=93)

    all_frames = Image.new('RGB', (1920, 3 * 236), 'white')
    all_draw = ImageDraw.Draw(all_frames)
    for index, (image_array, pose) in enumerate(zip(rgbs, poses['rgbd_refined'], strict=True)):
        col, row = index % 5, index // 5
        image = Image.fromarray(image_array).resize((384, 216))
        scaled_K = K.copy(); scaled_K[0] *= 384 / 640; scaled_K[1] *= 216 / 360
        cloud = sample @ pose[:3, :3].T + pose[:3, 3]
        draw_cloud(image, cloud, scaled_K, '#ff8a22', 1)
        all_draw.text((col * 384 + 5, row * 236 + 3),
                      f'frame {FRAMES[index]}', fill='black')
        all_frames.paste(image, (col * 384, row * 236 + 20))
    all_frames.save(result / 'all_frames.jpg', quality=93)

    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    centers = np.broadcast_to(center, (15, 1, 3))
    ref = transform_points(reference, centers)[:, 0]
    metrics = {}
    for label in ('mask_aware', 'rgbd_refined'):
        values = trajectory_errors(transform_points(poses[label], centers)[:, 0], ref)
        position = np.asarray(values['position_error_m'])
        metrics[label] = {
            'mean_position_error_m': values['mean_position_error_m'],
            'middle_frames_50_to_93_mean_position_error_m': float(position[3:9].mean()),
            'max_position_error_m': float(position.max()),
            'mean_displacement_error_m': values['mean_displacement_error_m'],
            'position_error_m': values['position_error_m'],
        }
    audit = json.loads((result / 'audit.json').read_text())
    audit.update(native_reference_usage='post_hoc_QA_only', qa_metrics=metrics,
                 qa_image='qa_geometry.jpg', all_frames_image='all_frames.jpg',
                 visually_inspected=False)
    (result / 'audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
