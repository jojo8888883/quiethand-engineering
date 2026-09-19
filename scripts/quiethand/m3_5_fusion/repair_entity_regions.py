#!/usr/bin/env python3
"""Localize each CAD, segment that object, inspect the mask, then estimate its pose.

No tool/target slot or prior mask supplies physical identity. Each stage retains
the same CAD id, source frames, and camera; old predictions remain untouched.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/quiethand/m3_jobs'))
from runtime_common import atomic_json, atomic_npy, status

LOCATE = '''Locate ONLY the physical object whose shape matches TARGET_CAD in SCENE.
Ignore CAD rendering colour. Do not use image order or hand/tool roles as identity.
Return only a JSON object, beginning with { and ending with }. No Markdown,
no code fences and no surrounding explanation. Copy this exact entity_id:
{"entity_id": @ENTITY_ID@, "visible": true or false,
"name_zh": "short Chinese object name", "box": [x1,y1,x2,y2] or null,
"positive_points": [[x,y]], "negative_points": [[x,y]], "reason": "visual reason"}.
All coordinates are SCENE image coordinates normalized to 0..1000, x left-to-right,
y top-to-bottom. The box must enclose the entire visible target, not its hand.
Give 1 to 3 positive points clearly on the visible target surface, never on a hand,
background or occluder. Give up to 3 negative points on nearby hands/other objects
if needed. If the object cannot be identified, visible=false, box=null, points=[].
Do not force a match when the target is absent or indistinguishable.
'''
VERIFY = '''Check the exact physical object TARGET_CAD against the proposed MASK.
TARGET_CAD is a neutral geometry rendering, not a photograph. Match its shape;
ignore differences in colour, material, texture and motion-capture marker dots.
SCENE is original RGB. MASK is RGB with everything outside the actual SAM mask
greyed out. OVERLAY shows that same mask in magenta for spatial context.
Return only a JSON object, beginning with { and ending with }. No Markdown,
no code fences and no surrounding explanation. Copy this exact entity_id:
{"entity_id": @ENTITY_ID@, "matches": true or false,
"reason": "short visual explanation"}. True requires that the mask covers the
visible target itself, not a hand, another object, or just a small handle fragment.
Reject a mask that includes a substantial visible part of another object, even
if it also covers the target. Inspect the whole highlighted region for this.
Occlusion is allowed; don't demand the invisible surface. If uncertain use false.
Judge the actual highlighted pixels, not the intended box or the file name.
'''


def read(path):
    return json.loads(path.read_text())


def prompt_for(template, identity):
    return template.replace('@ENTITY_ID@', json.dumps(identity))


def parse_location(raw, identity):
    value = json.loads(raw)
    if value.get('entity_id') != identity or type(value.get('visible')) is not bool:
        raise ValueError('location must identify this exact CAD and visibility')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('visual reason required')
    if not isinstance(value.get('name_zh'), str):
        raise ValueError('object name required')
    if not value['visible']:
        if value.get('box') is not None or value.get('positive_points') != [] or value.get('negative_points') != []:
            raise ValueError('invisible entity must have no localization')
        return value
    box = np.asarray(value.get('box'), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1000):
        raise ValueError('invalid scene-normalized box')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('box must have positive area')
    for key, minimum in [('positive_points', 1), ('negative_points', 0)]:
        points = value.get(key)
        if not isinstance(points, list) or not minimum <= len(points) <= 3:
            raise ValueError('wrong number of segmentation points')
        if not points:
            continue
        points = np.asarray(points, dtype=float)
        if points.shape != (len(points), 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1000):
            raise ValueError('invalid normalized point')
        if key == 'positive_points' and (np.any(points < box[:2]) or np.any(points > box[2:])):
            raise ValueError('positive point outside object box')
    return value


def sam_prompt(location, width, height):
    scale = np.array([width, height], dtype=np.float32) / 1000
    positives, negatives = location['positive_points'], location['negative_points']
    return {'box': np.asarray(location['box'], np.float32) * np.tile(scale, 2),
            'points': np.asarray(positives + negatives, np.float32) * scale,
            'labels': np.asarray([1]*len(positives) + [0]*len(negatives), np.int32)}


def initialization_mask(mask, location):
    """Keep only pixels inside the same whole-object box used to prompt SAM."""
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('initialization mask must be a 2D boolean array')
    height, width = mask.shape
    x1, y1, x2, y2 = sam_prompt(location, width, height)['box']
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    result = np.zeros_like(mask)
    result[top:bottom, left:right] = mask[top:bottom, left:right]
    if result.sum() < 4:
        raise ValueError('localized initialization mask has fewer than four pixels')
    return result


def parse_mask_check(raw, identity):
    value = json.loads(raw)
    if value.get('entity_id') != identity or type(value.get('matches')) is not bool:
        raise ValueError('mask check must identify this exact CAD and boolean verdict')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('mask check needs visual evidence')
    return value


def labelled_images(images, instruction):
    content = []
    for label, path in images:
        content.extend([{'type': 'text', 'text': label}, {'type': 'image', 'url': str(path)}])
    content.append({'type': 'text', 'text': instruction})
    return content


def segment_entity(predictor, paths, location):
    sys.path.insert(0, str(ROOT / 'scripts/quiethand/m3_v1_2'))
    from run_perception import _temporary_jpeg_aliases
    with Image.open(paths[0]) as first:
        width, height = first.size
    aliases = _temporary_jpeg_aliases(paths)
    directory = next(aliases)
    state = None
    try:
        state = predictor.init_state(video_path=str(directory), offload_video_to_cpu=True, offload_state_to_cpu=False)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                       **sam_prompt(location, width, height))
        masks = []
        for frame, ids, logits in predictor.propagate_in_video(state):
            if frame != len(masks) or list(map(int, ids)) != [1]:
                raise ValueError('SAM changed entity or frame order')
            masks.append((logits[0, 0] > 0).detach().cpu().numpy().astype(bool))
        result = np.stack(masks)
        if result.shape != (len(paths), height, width):
            raise ValueError('SAM output frame/shape mismatch')
        return result
    finally:
        if state is not None:
            predictor.reset_state(state)
        try:
            next(aliases)
        except StopIteration:
            pass


def render_mask(rgb_path, mask, folder):
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    isolated = np.full_like(rgb, 210)
    isolated[mask] = rgb[mask]
    overlay = rgb.copy()
    overlay[mask] = (.5*rgb[mask] + .5*np.array([255, 0, 255])).astype(np.uint8)
    for name, pixels in [('mask_rgb.png', isolated), ('mask_overlay.png', overlay)]:
        image = Image.fromarray(pixels)
        image.thumbnail((960, 540))
        image.save(folder/name)


def qwen_loader(workspace):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model_path = workspace/'external_data/quiethand_m3_resources/semantic'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0}).eval()
    def infer(content):
        inputs = processor.apply_chat_template([{'role':'user','content':content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
        inputs.pop('token_type_ids', None)
        with torch.inference_mode():
            result = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        return processor.decode(result[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return infer


def run(cfg):
    workspace = cfg.workspace
    output = workspace/cfg.output_root
    manifest = read(output/'input.json')
    if manifest['evaluation_event_count'] != 0 or manifest['evaluation_results_opened'] is not False:
        raise ValueError('calibration-only repair')
    calibration = read(workspace/'artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json')
    known = {e['event_id'] for e in calibration['events']}
    if not {e['event_id'] for e in manifest['events']} <= known:
        raise ValueError('unknown calibration event')
    infer, sam, foundation = None, None, None
    started = time.monotonic()
    items = [(event, entity) for event in manifest['events'] for entity in event['entities']]
    for ordinal, (event, entity) in enumerate(items, 1):
        identity = entity['entity_id']
        folder = output/'events'/event['event_id']/identity
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder/(cfg.stage+'.json')
        if result_path.exists():
            continue
        status(output/'status.json', cfg.stage, ordinal-1, len(items))
        scene = workspace/event['rgb_frames'][0]
        cad = workspace/entity['reference_image']
        record = {'event_id':event['event_id'], 'entity_id':identity,
                  'source_frame_indices':event['source_frame_indices']}
        if cfg.stage == 'locate':
            if infer is None:
                infer = qwen_loader(workspace)
            raw = infer(labelled_images([('SCENE',scene), ('TARGET_CAD '+identity,cad)], prompt_for(LOCATE,identity)))
            atomic_json(folder/'locate_raw.json', {'raw_text':raw})
            try:
                value = parse_location(raw, identity)
                record.update(status='located' if value['visible'] else 'not_visible', location=value)
            except (ValueError, TypeError) as exc:
                record.update(status='invalid', reason=str(exc))
        elif cfg.stage == 'segment':
            located = read(folder/'locate.json')
            if located['status'] != 'located':
                record.update(status='unavailable',reason='localization_'+located['status'])
            else:
                sys.path.insert(0,str(ROOT/'scripts/quiethand/m3_v1_2'))
                from run_perception import _load_sam
                if sam is None:
                    sam = _load_sam(workspace/'external_repos/quiethand_m3/sam2',
                        'configs/sam2.1/sam2.1_hiera_b+.yaml',workspace/'external_data/quiethand_m3_resources/segmentation/sam2.1_hiera_base_plus.pt')
                masks = segment_entity(sam,[workspace/p for p in event['rgb_frames']],located['location'])
                atomic_npy(folder/'mask.npy',masks)
                render_mask(scene,masks[0],folder)
                record.update(status='segmented',mask_path=(folder/'mask.npy').relative_to(workspace).as_posix(),
                              pixel_counts=masks.sum(axis=(1,2)).tolist())
        elif cfg.stage == 'verify':
            segmented = read(folder/'segment.json')
            if segmented['status'] != 'segmented':
                record.update(status='unavailable',reason=segmented['reason'])
            else:
                if infer is None:
                    infer = qwen_loader(workspace)
                raw = infer(labelled_images([('SCENE',scene),('TARGET_CAD '+identity,cad),
                    ('MASK',folder/'mask_rgb.png'),('OVERLAY',folder/'mask_overlay.png')],prompt_for(VERIFY,identity)))
                atomic_json(folder/'verify_raw.json',{'raw_text':raw})
                try:
                    value = parse_mask_check(raw,identity)
                    record.update(status='accepted' if value['matches'] else 'rejected',check=value)
                except (ValueError,TypeError) as exc:
                    record.update(status='invalid',reason=str(exc))
        elif cfg.stage == 'pose':
            verified = read(folder/'verify.json')
            if verified['status'] != 'accepted':
                record.update(status='unavailable',reason='mask_check_'+verified['status'])
            else:
                sys.path.insert(0,str(ROOT/'scripts/quiethand/m3_v1_2'))
                from run_perception import _load_foundation, _poses
                if foundation is None:
                    foundation = _load_foundation(workspace/'external_repos/quiethand_m3/FoundationPose')
                rgb = [np.asarray(Image.open(workspace/p).convert('RGB')) for p in event['rgb_frames']]
                depth = [np.load(workspace/p,allow_pickle=False) for p in event['depth_frames']]
                intrinsic = np.load(workspace/event['intrinsic_path'],allow_pickle=False)
                masks = np.load(folder/'mask.npy',allow_pickle=False)
                try:
                    poses = _poses(*foundation,workspace/entity['mesh_path'],rgb,depth,intrinsic,masks)
                    atomic_npy(folder/'pose.npy',poses)
                    record.update(status='observed',pose_path=(folder/'pose.npy').relative_to(workspace).as_posix(),
                                  unit='m_SE3',coordinate_frame='camera',mesh_path=entity['mesh_path'])
                except RuntimeError as exc:
                    if 'out of memory' in str(exc).lower():
                        raise
                    record.update(status='invalid',reason=str(exc))
        atomic_json(result_path,record)
    results=[read(output/'events'/e['event_id']/a['entity_id']/(cfg.stage+'.json')) for e,a in items]
    counts={key:sum(x['status']==key for x in results) for key in sorted({x['status'] for x in results})}
    atomic_json(output/(cfg.stage+'_audit.json'),{'stage':cfg.stage,'status':'COMPLETE',
        'event_count':len(manifest['events']),'entity_count':len(items),'counts':counts,
        'elapsed_seconds':time.monotonic()-started,'evaluation_results_opened':False,
        'native_masks_used':False,'native_poses_used':False,'human_corrections_used':False})
    status(output/'status.json',cfg.stage+'_complete',len(items),len(items))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['locate','segment','verify','pose'])
    parser.add_argument('--workspace',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path,required=True)
    run(parser.parse_args())
