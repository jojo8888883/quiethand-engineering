#!/usr/bin/env python3
"""Local batch completion: reuse local masks, recompute fusion, render geometry."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BATCH = ROOT / 'artifacts/quiethand/m3_5/identity_batch'
BASE = ROOT / 'artifacts/quiethand/m3_v1_2'


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(BATCH / 'input.json')
    audit = read(BATCH / 'perception/audit.json')
    if audit['status'] != 'COMPLETE' or audit['event_count'] != 30 or audit['evaluation_results_opened']:
        raise ValueError('batch inference must be complete and calibration-only')
    materialization = {e['event_id']: e for e in read(BASE / 'QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json')['events']}
    hand_input = read(ROOT / 'artifacts/quiethand/m3_5/QH_M3_5_RAW_HAND_METRIC_INPUT.json')
    hands = {e['event_id']: e for e in hand_input['events']}
    for event in manifest['events']:
        eid = event['event_id']
        if event['source_frame_indices'] != materialization[eid]['frame_indices']:
            raise ValueError(f'{eid}: source sampling differs')
        if hands[eid]['ordered_rgb_frames'] != event['segmentation']['ordered_rgb_frames'] or hands[eid]['source_frame_indices'] != event['source_frame_indices']:
            raise ValueError(f'{eid}: hand and object RGB sampling differ')
        old_root = BASE / 'results/perception'
        old = read(old_root / 'events' / eid / 'segmentation.json')
        new_root = BATCH / 'perception'
        new = read(new_root / 'events' / eid / 'segmentation.json')
        for role, item in new['items'].items():
            if item['status'] != 'observed':
                continue
            source = old_root / old['items'][role]['artifact']['relative_path']
            target = new_root / item['artifact']['relative_path']
            if not target.exists():
                target.hardlink_to(source)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run([sys.executable, str(HERE / 'build_fusion_preview.py'), '--workspace', str(ROOT),
        '--hand-results', str(ROOT / 'artifacts/quiethand/m3_5/results/raw_hand_metric'),
        '--perception-results', str(BATCH / 'perception'), '--output', str(BATCH / 'fusion_data')], check=True, env=env)
    subprocess.run([sys.executable, str(HERE / 'build_identity_batch_preview.py')], check=True, env=env)


if __name__ == '__main__':
    main()
