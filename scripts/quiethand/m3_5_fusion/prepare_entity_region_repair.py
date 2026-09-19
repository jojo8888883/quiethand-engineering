#!/usr/bin/env python3
"""Prepare the four diagnosed calibration cases; do not launch inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLD = Path('artifacts/quiethand/m3_5/identity_batch')
NEW = Path('artifacts/quiethand/m3_5/entity_region_repair')
SERVER = Path('/llm_jzm/dty_user/quiethand_m3')
SELECTED = (3, 4, 13, 28)


def main():
    source = json.loads((ROOT/OLD/'input.json').read_text())
    page = json.loads((ROOT/OLD/'preview/data.json').read_text())
    catalog = {e['event_id']: e for e in source['events']}
    events = []
    for ordinal in SELECTED:
        view = page['records'][ordinal-1]
        old = catalog[view['event_id']]
        frames = old['source_frame_indices']
        rgb = old['segmentation']['ordered_rgb_frames']
        depth = old['object_state']['ordered_depth_frames']
        assert frames == view['source_frame_indices'] and len(frames) == len(rgb) == len(depth) == 15
        assert len(set(frames)) == 15 and frames == sorted(frames)
        entities = old['object_state']['entities']
        assert len({e['entity_id'] for e in entities}) == len(entities)
        events.append(dict(event_id=old['event_id'], source_frame_indices=frames,
            rgb_frames=rgb, depth_frames=depth, intrinsic_path=old['object_state']['intrinsic_path'],
            entities=[{k: entity[k] for k in ('entity_id','mesh_path','reference_image')} for entity in entities]))
    # Deliberately exclude old semantic slots, native poses/masks and human reviews.
    manifest = dict(schema_version=1, event_count=len(events), evaluation_event_count=0,
        evaluation_results_opened=False, events=events)
    job = json.loads((ROOT/OLD/'job.json').read_text())
    job.update(id='qh-m3-5-entity-regions-4', display_name='QuietHand four-case entity localization and mask repair',
        command=['timeout','--signal=TERM','3600','bash',str(SERVER/NEW/'run.sh')])
    job['preflight']['paths'] = [str(SERVER/p) for p in (
        'envs/semantic/bin/python','envs/perception/bin/python',NEW/'input.json',NEW/'run.sh',
        'scripts/quiethand/m3_5_fusion/repair_entity_regions.py',
        'external_data/quiethand_m3_resources/semantic/config.json',
        'external_data/quiethand_m3_resources/segmentation/sam2.1_hiera_base_plus.pt')]
    job['completion']['path'] = str(SERVER/NEW/'pose_audit.json')
    job['progress']['status_path'] = str(SERVER/NEW/'status.json')
    job['science'] = dict(context_path=str(SERVER/NEW/'input.json'), protocol_path=str(SERVER/NEW/'WORK_LOG.md'))
    (ROOT/NEW).mkdir(parents=True, exist_ok=True)
    for name, data in [('input.json',manifest),('job.json',job)]:
        (ROOT/NEW/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(prepared_events=len(events), entities=sum(len(e['entities']) for e in events),
                         original_page_numbers=SELECTED, job_submitted=False)))


if __name__ == '__main__':
    main()
