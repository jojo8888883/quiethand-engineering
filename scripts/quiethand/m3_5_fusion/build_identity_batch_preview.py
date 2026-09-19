#!/usr/bin/env python3
"""Show entity-bound calibration geometry and unchanged-rule fusion, without GT."""
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from build_fusion_preview import item_array, load_json, atomic_json

BATCH = ROOT / 'artifacts/quiethand/m3_5/identity_batch'
BASE = ROOT / 'artifacts/quiethand/m3_v1_2'
HAND = ROOT / 'artifacts/quiethand/m3_5/results/raw_hand_metric'
ROLES = ('tool', 'target')
HAND_COLORS = {'left': '#00d9ed', 'right': '#cd8bff'}
ENTITY_COLORS = ('#7ded66', '#ffa840')


def decode_rgb(video, frames):
    expression = '+'.join(f'eq(n\\,{f})' for f in frames)
    raw = subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin',
        '-threads', '1', '-i', str(video), '-vf', f'select={expression},scale=640:360',
        '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    if len(raw) != len(frames) * 640 * 360 * 3:
        raise ValueError('RGB source frame count mismatch')
    return np.frombuffer(raw, np.uint8).reshape(len(frames), 360, 640, 3)


def projected(points, intrinsic):
    points = np.asarray(points)
    good = np.isfinite(points).all(axis=1) & (points[:, 2] > .001)
    uvz = points[good] @ intrinsic.T
    return uvz[:, :2] / uvz[:, 2:] / 3


def draw_cloud(image, points, intrinsic, color, radius):
    draw = ImageDraw.Draw(image)
    for x, y in projected(points, intrinsic):
        if 0 <= x < 640 and 0 <= y < 360:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def draw_axes(image, center, pose, intrinsic):
    local = np.vstack([center, center + np.eye(3) * .04])
    uv = projected(local @ pose[:3, :3].T + pose[:3, 3], intrinsic)
    if len(uv) != 4 or not ((uv >= [-640, -360]) & (uv <= [1280, 720])).all():
        return
    draw = ImageDraw.Draw(image)
    for endpoint, color in zip(uv[1:], ('#ff4050', '#3aed80', '#409bff')):
        draw.line([tuple(uv[0]), tuple(endpoint)], fill=color, width=2)


def sample_spans(frames, values, threshold=.03):
    """Runs over actual sampled timestamps, not interpolated contact duration."""
    if len(frames) != len(values):
        raise ValueError('sample timing mismatch')
    result, run = [], []
    for frame, value in zip(frames, values):
        if value is not None and value <= threshold:
            run.append(frame)
        elif run:
            result.append({'first_frame': run[0], 'last_frame': run[-1], 'sample_count': len(run)})
            run = []
    if run:
        result.append({'first_frame': run[0], 'last_frame': run[-1], 'sample_count': len(run)})
    return result


def main():
    manifest = load_json(BATCH / 'input.json')
    binding_audit = load_json(BATCH / 'binding/audit.json')
    pose_audit = load_json(BATCH / 'perception/audit.json')
    fusion = load_json(BATCH / 'fusion_data/data.json')
    plan = load_json(BASE / 'QH_M3_V1_2_TEMPORAL_PLAN.json')
    events = {e['event_id']: e for e in manifest['events']}
    sources = {e['event_id']: e for e in plan['events']}
    if manifest['event_count'] != 30 or manifest['evaluation_event_count'] != 0:
        raise ValueError('batch must remain calibration 30 / evaluation 0')
    if {r['event_id'] for r in fusion['records']} != set(events):
        raise ValueError('fusion and batch event identities differ')
    if pose_audit['status'] != 'COMPLETE' or pose_audit['event_count'] != 30:
        raise ValueError('pose batch incomplete')
    output = BATCH / 'preview'
    output.mkdir(exist_ok=True)
    mesh_cache = {}
    for ordinal, record in enumerate(fusion['records'], 1):
        eid = record['event_id']
        event = events[eid]
        frames = event['source_frame_indices']
        source = sources[eid]['source_binding']
        K = np.loadtxt(ROOT / 'external_data/taco_v1' / source['intrinsic']['relative_path'])
        rgb = decode_rgb(ROOT / 'external_data/taco_v1' / source['rgb']['relative_path'], frames)
        hand_doc = load_json(HAND / 'events' / f'{eid}.json')
        hands = {s: item_array(HAND, hand_doc, s)[0] for s in HAND_COLORS}
        old_root = BASE / 'results/perception'
        new_root = BATCH / 'perception'
        old_doc = load_json(old_root / 'events' / eid / 'object_state.json')
        new_doc = load_json(new_root / 'events' / eid / 'object_state.json')
        binding = load_json(BATCH / 'binding/events' / f'{eid}.json')
        entities = {e['entity_id']: e for e in event['object_state']['entities']}
        colors = dict(zip(sorted(entities), ENTITY_COLORS))
        meshes = {}
        for entity_id, entity in entities.items():
            if entity_id not in mesh_cache:
                mesh_cache[entity_id] = np.asarray(trimesh.load(ROOT / entity['mesh_path'], process=False).vertices)
            meshes[entity_id] = mesh_cache[entity_id]
        old = {event['previous_role_to_entity'][role]: item_array(old_root, old_doc, role)[0] for role in ROLES}
        new = {item['entity_id']: item_array(new_root, new_doc, role)[0]
            for role, item in new_doc['items'].items() if item['status'] == 'observed'}
        for poses in list(old.values()) + list(new.values()):
            if poses is not None and (poses.shape != (15, 4, 4) or not np.isfinite(poses).all()):
                raise ValueError(f'{eid}: invalid pose shape or nonfinite geometry')
        images = output / 'frames' / eid
        images.mkdir(parents=True, exist_ok=True)
        for index, pixels in enumerate(rgb):
            for version, poses_by_entity in (('before', old), ('after', new)):
                image = Image.fromarray(pixels).copy()
                for side, vertices in hands.items():
                    if vertices is not None:
                        draw_cloud(image, vertices[index], K, HAND_COLORS[side], .7)
                for entity_id, poses in poses_by_entity.items():
                    if poses is None:
                        continue
                    vertices = meshes[entity_id]
                    sampled = vertices[::max(1, len(vertices)//900)]
                    pose = poses[index]
                    draw_cloud(image, sampled @ pose[:3, :3].T + pose[:3, 3], K, colors[entity_id], 1)
                    draw_axes(image, (vertices.min(0)+vertices.max(0))/2, pose, K)
                image.save(images / f'{index:02d}_{version}.jpg', quality=88)
        record['source_frame_indices'] = frames
        record['video_url'] = f'/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{eid}.mp4'
        record['identities'] = []
        for role in ROLES:
            item = new_doc['items'][role]
            entity_id = binding['role_to_entity'][role]
            record['identities'].append({'role': role, 'name_zh': record['semantic'][role+'_zh'],
                'before': event['previous_role_to_entity'][role], 'after': entity_id,
                'color': colors.get(entity_id), 'pose_status': item['status'],
                'pose_origin': item.get('pose_origin'), 'failure_reason': item.get('failure_reason'),
                'reason': binding.get('reasons', {}).get(role),
                'crop_url': f'../inputs/{eid}/region_{ROLES.index(role)}.png',
                'cad_url': None if entity_id is None else '../inputs/cads/'+entity_id+'.png'})
        record['entity_colors'] = colors
        for side in HAND_COLORS:
            for pair in record['hands'][side]['pairs'].values():
                pair['close_sample_spans'] = sample_spans(frames, pair['distance_m'])
        atomic_json(output / 'records' / f'{eid}.json', record)
        print(f'Preview geometry {ordinal}/30 {eid}', flush=True)
    report = {'event_count': 30, 'evaluation_event_count': 0, 'evaluation_results_opened': False,
        'binding': binding_audit, 'poses': pose_audit, 'fusion': load_json(BATCH / 'fusion_data/audit.json'),
        'geometry_images': 900, 'native_pose_used': False, 'human_corrections_used_for_fusion': False,
        'records': fusion['records']}
    atomic_json(output / 'data.json', report)
    (output / 'index.html').write_text((HERE / 'identity_batch_preview.html').read_text(), encoding='utf-8')
    print('READY: ' + str(output / 'index.html'))


if __name__ == '__main__':
    main()
