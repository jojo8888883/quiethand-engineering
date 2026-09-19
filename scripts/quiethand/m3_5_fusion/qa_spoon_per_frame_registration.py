#!/usr/bin/env python3
"""Post-hoc visual and native-pose QA for per-frame spoon registration."""
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
    initial = ROOT / 'artifacts/quiethand/m3_5/entity_region_repair/spoon_registration'
    result = ROOT / 'artifacts/quiethand/m3_5/entity_region_repair/spoon_per_frame_registration'
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
        'initial_only': np.load(initial / 'pose.npy'),
        'per_frame': np.load(result / 'pose.npy'),
        'native_QA': reference,
    }
    rgbs = decode_rgb(
        ROOT / 'artifacts/quiethand/m3_v1_2/result_preview/source_videos' / f'{EID}.mp4',
        FRAMES)
    masks = np.load(
        ROOT / 'artifacts/quiethand/m3_5/entity_region_repair/events' / EID / 'cad_200/mask.npy')
    canvas = Image.new('RGB', (2560, len(SELECTED) * 380), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(SELECTED):
        for col, label in enumerate(('source', 'initial_only', 'per_frame', 'native_QA')):
            draw.text((col * 640 + 8, row * 380 + 4),
                      f'{label} | frame {FRAMES[index]}', fill='black')
            image = Image.fromarray(rgbs[index]).copy()
            if label != 'source':
                pose = poses[label][index]
                cloud = sample @ pose[:3, :3].T + pose[:3, 3]
                draw_cloud(image, cloud, K, '#ff8a22', 1)
            canvas.paste(image, (col * 640, row * 380 + 20))
    canvas.save(result / 'qa_geometry.jpg', quality=93)

    mask_canvas = Image.new('RGB', (1920, 760), 'white')
    mask_draw = ImageDraw.Draw(mask_canvas)
    for slot, index in enumerate(SELECTED):
        col, row = slot % 3, slot // 3
        image = rgbs[index].copy()
        small = np.asarray(Image.fromarray(masks[index]).resize(
            (640, 360), Image.Resampling.NEAREST))
        image[small] = (.5 * image[small] + .5 * np.array([255, 0, 255])).astype(np.uint8)
        mask_draw.text((col * 640 + 8, row * 380 + 4),
                       f'SAM mask | frame {FRAMES[index]}', fill='black')
        mask_canvas.paste(Image.fromarray(image), (col * 640, row * 380 + 20))
    mask_canvas.save(result / 'mask_qa.jpg', quality=93)

    center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    centers = np.broadcast_to(center, (15, 1, 3))
    ref = transform_points(reference, centers)[:, 0]
    metrics = {}
    for label in ('initial_only', 'per_frame'):
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
    audit.update(
        native_reference_usage='post_hoc_QA_only', qa_metrics=metrics,
        qa_image='qa_geometry.jpg', mask_qa_image='mask_qa.jpg', visually_inspected=False)
    (result / 'audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
