#!/usr/bin/env python3
"""Inspect existing calibration identity/pose results; no prediction or repair."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from diagnose_trajectories import trajectory_errors, transform_points

BATCH = ROOT / 'artifacts/quiethand/m3_5/identity_batch'
BASE = ROOT / 'artifacts/quiethand/m3_v1_2'
OUT = BATCH / 'diagnosis'
FONT = ImageFont.truetype('/System/Library/Fonts/Monaco.ttf', 14)


def read(path):
    return json.loads(path.read_text())


def paste_fit(canvas, path, xy, size):
    with Image.open(path) as source:
        image = source.convert('RGB')
        image.thumbnail(size)
        canvas.paste(image, (xy[0] + (size[0]-image.width)//2, xy[1]))


def main():
    OUT.mkdir(exist_ok=True)
    manifest = read(BATCH / 'input.json')
    assert manifest['event_count'] == 30 and manifest['evaluation_event_count'] == 0
    events = {x['event_id']: x for x in manifest['events']}
    plan = read(BASE / 'QH_M3_V1_2_TEMPORAL_PLAN.json')
    assert plan['calibration_video_count'] == 30 and plan['evaluation_video_count'] == 0
    sources = {x['event_id']: x['source_binding'] for x in plan['events']}
    preview = read(BATCH / 'preview/data.json')['records']
    native = ROOT / 'external_data/taco_v1'
    records, changed = [], []
    for ordinal, record in enumerate(preview, 1):
        eid = record['event_id']
        event, source = events[eid], sources[eid]
        frames = event['source_frame_indices']
        old_root, new_root = BASE / 'results/perception', BATCH / 'perception'
        old = read(old_root / 'events' / eid / 'object_state.json')['items']
        new = read(new_root / 'events' / eid / 'object_state.json')['items']
        cameras = np.load(native / source['extrinsic']['relative_path'], allow_pickle=False)[frames]
        metrics = []
        for role in ('tool', 'target'):
            identity = event['previous_role_to_entity'][role]
            entity = next(x for x in event['object_state']['entities'] if x['entity_id'] == identity)
            mesh = trimesh.load(ROOT / entity['mesh_path'], process=False)
            center = (mesh.bounds[0] + mesh.bounds[1]) / 2
            centers = np.broadcast_to(center, (15, 1, 3))
            reference = cameras @ np.load(native / source[role+'_pose']['relative_path'], allow_pickle=False)[frames]
            ref_center = transform_points(reference, centers)[:, 0]
            new_items = [x for x in new.values() if x.get('entity_id') == identity and x['status'] == 'observed']
            after = new_items[0] if new_items else None
            pairs = [('before', old[role], old_root), ('after', after, new_root)]
            row = {'entity_id': identity, 'after_origin': None if after is None else after['pose_origin']}
            arrays = []
            for name, item, folder in pairs:
                if item is None or item['status'] != 'observed':
                    row[name] = None
                    continue
                pose = np.load(folder / item['artifact']['relative_path'], allow_pickle=False)
                assert pose.shape == (15, 4, 4)
                values = trajectory_errors(transform_points(pose, centers)[:, 0], ref_center)
                row[name] = {k: values[k] for k in ('mean_position_error_m', 'mean_displacement_error_m', 'position_error_m')}
                arrays.append(pose)
            row['poses_exactly_unchanged'] = len(arrays) == 2 and bool(np.array_equal(*arrays))
            metrics.append(row)
        records.append({'ordinal': ordinal, 'event_id': eid, 'source_frame_indices': frames,
                        'entities': metrics})
        if any(x.get('pose_origin') == 'rerun_pose' for x in new.values()):
            changed.append((ordinal, record))

    report = {'purpose': 'existing calibration diagnosis, native pose only a measurement reference',
              'model_calls': 0, 'evaluation_events': 0, 'predictions_modified': False,
              'records': records}
    (OUT / 'measurements.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

    # Compact visual evidence: the actual crops followed by the CAD assigned to them.
    for page in range((len(changed)+4)//5):
        canvas = Image.new('RGB', (1000, 930), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), 'Actual crop 0 | Assigned CAD 0 | Actual crop 1 | Assigned CAD 1', fill='black', font=FONT)
        for slot, (ordinal, record) in enumerate(changed[page*5:page*5+5]):
            y = 40 + slot*175
            draw.text((10,y), f"#{ordinal:02d}", fill='black', font=FONT)
            eid = record['event_id']
            for index, item in enumerate(record['identities']):
                x = index*500
                draw.text((x+65,y), 'region_'+str(index)+' -> '+str(item['after']), fill='black', font=FONT)
                paste_fit(canvas, BATCH/'inputs'/eid/f'region_{index}.png', (x,y+23),(240,140))
                paste_fit(canvas, BATCH/'inputs/cads'/f"{item['after']}.png", (x+250,y+23),(240,140))
        canvas.save(OUT/f'bindings_{page+1}.jpg', quality=93)

    # Three timestamps for actual visual inspection of representative cases.
    for ordinal in (3, 13, 28, 2, 29):
        record = preview[ordinal-1]
        eid = record['event_id']
        canvas = Image.new('RGB',(1280,1140),'white')
        draw = ImageDraw.Draw(canvas)
        for row, index in enumerate((0,7,14)):
            y = row*380
            for column, version in enumerate(('before','after')):
                draw.text((column*640+10,y+3),f'#{ordinal} {version} frame {record["source_frame_indices"][index]}',fill='black',font=FONT)
                paste_fit(canvas,BATCH/'preview/frames'/eid/f'{index:02d}_{version}.jpg',(column*640,y+20),(640,360))
        canvas.save(OUT/f'timeline_{ordinal:02d}.jpg',quality=93)
    mask_rows = []
    for ordinal in (3, 4, 13, 26):
        record = preview[ordinal-1]
        eid = record['event_id']
        event = events[eid]
        source = Image.open(BATCH/'inputs'/eid/'scene.png').convert('RGB').resize((640,360))
        segment = read(BASE/'results/perception/events'/eid/'segmentation.json')['items']
        canvas = Image.new('RGB',(1280,390),'white')
        draw = ImageDraw.Draw(canvas)
        masks=[]
        for index, role in enumerate(('tool','target')):
            path = BASE/'results/perception'/segment[role]['artifact']['relative_path']
            mask = np.load(path, mmap_mode='r', allow_pickle=False)[0]
            masks.append(mask)
            rgb = np.asarray(source).copy()
            small = np.asarray(Image.fromarray(mask).resize((640,360),Image.Resampling.NEAREST))
            rgb[small] = (.5*rgb[small]+.5*np.array([255,0,255])).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(index*640,30))
            draw.text((index*640+10,5),f'#{ordinal} SAM {role} / region_{index}',fill='black',font=FONT)
        iou = float(np.logical_and(*masks).sum()/max(1,np.logical_or(*masks).sum()))
        mask_rows.append({'ordinal':ordinal,'first_frame_masks_identical':bool(np.array_equal(*masks)),'first_frame_masks_iou':iou})
        canvas.save(OUT/f'masks_{ordinal:02d}.jpg',quality=93)
    (OUT/'mask_checks.json').write_text(json.dumps(mask_rows,indent=2)+'\n')
    for r in records:
        cells=[]
        for e in r['entities']:
            vals=[None if e[k] is None else round(e[k]['mean_position_error_m']*100,2) for k in ('before','after')]
            cells.append(f"{e['entity_id']} {vals} cm unchanged={e['poses_exactly_unchanged']}")
        print(f"{r['ordinal']:02d}: "+'; '.join(cells))


if __name__ == '__main__':
    main()
