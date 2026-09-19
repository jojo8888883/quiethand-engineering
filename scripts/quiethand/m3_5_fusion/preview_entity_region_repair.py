#!/usr/bin/env python3
"""Agent-review sheets from four-case outputs, with no native pose input."""
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from build_identity_batch_preview import decode_rgb, draw_cloud, draw_axes

NEW=ROOT/'artifacts/quiethand/m3_5/entity_region_repair'
OLD=ROOT/'artifacts/quiethand/m3_5/identity_batch'
BASE=ROOT/'artifacts/quiethand/m3_v1_2'


def read(path):
    return json.loads(path.read_text())


def put(canvas,image,xy,size):
    image=image.copy(); image.thumbnail(size)
    canvas.paste(image,xy)


def main():
    manifest=read(NEW/'input.json')
    plan={e['event_id']:e for e in read(BASE/'QH_M3_V1_2_TEMPORAL_PLAN.json')['events']}
    old_records=read(OLD/'preview/data.json')['records']
    ranks={r['event_id']:i for i,r in enumerate(old_records,1)}
    output=NEW/'preview';output.mkdir(exist_ok=True)
    summaries=[]
    for event in manifest['events']:
        eid=event['event_id'];rank=ranks[eid];frames=event['source_frame_indices']
        assert old_records[rank-1]['source_frame_indices']==frames
        rgb=decode_rgb(BASE/'result_preview/source_videos'/f'{eid}.mp4',frames)
        K=np.loadtxt(ROOT/'external_data/taco_v1'/plan[eid]['source_binding']['intrinsic']['relative_path'])
        colors=dict(zip(sorted(e['entity_id'] for e in event['entities']),('#7ded66','#ffa840')))
        geometry=Image.new('RGB',(1920,1140),'white');gd=ImageDraw.Draw(geometry)
        masks=Image.new('RGB',(1440,400*len(event['entities'])),'white');md=ImageDraw.Draw(masks)
        new_poses={};vertices={};statuses={}
        for i,entity in enumerate(event['entities']):
            identity=entity['entity_id'];folder=NEW/'events'/eid/identity
            loc=read(folder/'locate.json');check=read(folder/'verify.json');pose=read(folder/'pose.json')
            statuses[identity]=dict(localization=loc,mask_check=check,pose=pose)
            if pose['status']=='observed':
                assert pose['source_frame_indices']==frames and pose['entity_id']==identity
                spoon_audit=NEW/'spoon_registration/audit.json'
                spoon_per_frame_audit=NEW/'spoon_per_frame_registration/audit.json'
                spoon_mask_aware_audit=NEW/'spoon_mask_aware_tracking/audit.json'
                spoon_rgbd_depth_audit=NEW/'spoon_rgbd_depth_refinement/audit.json'
                spoon_foundation_audit=NEW/'spoon_foundation_refinement/audit.json'
                if identity=='cad_200' and eid=='qh-m3-v12-cal-594fc82381d222917ad15a7b' and spoon_foundation_audit.is_file():
                    replacement=read(spoon_foundation_audit)
                    assert replacement['gate_outcome']=='TERMINAL_FULL_PASS' and replacement['source_frame_indices']==frames
                    new_poses[identity]=np.load(ROOT/replacement['pose_path'],allow_pickle=False)
                    statuses[identity]['displayed_pose']='foundation_learned_rgbd_refinement'
                elif identity=='cad_200' and eid=='qh-m3-v12-cal-594fc82381d222917ad15a7b' and spoon_rgbd_depth_audit.is_file():
                    replacement=read(spoon_rgbd_depth_audit)
                    assert replacement['status']=='COMPLETE' and replacement['source_frame_indices']==frames
                    new_poses[identity]=np.load(ROOT/replacement['pose_path'],allow_pickle=False)
                    statuses[identity]['displayed_pose']='same_mask_rgbd_depth_refinement'
                elif identity=='cad_200' and eid=='qh-m3-v12-cal-594fc82381d222917ad15a7b' and spoon_mask_aware_audit.is_file():
                    replacement=read(spoon_mask_aware_audit)
                    assert replacement['status']=='COMPLETE' and replacement['source_frame_indices']==frames
                    new_poses[identity]=np.load(ROOT/replacement['pose_path'],allow_pickle=False)
                    statuses[identity]['displayed_pose']='mask_aware_temporal_selection'
                elif identity=='cad_200' and eid=='qh-m3-v12-cal-594fc82381d222917ad15a7b' and spoon_per_frame_audit.is_file():
                    replacement=read(spoon_per_frame_audit)
                    assert replacement['status']=='COMPLETE' and replacement['source_frame_indices']==frames
                    new_poses[identity]=np.load(ROOT/replacement['pose_path'],allow_pickle=False)
                    statuses[identity]['displayed_pose']='per_frame_registration'
                elif identity=='cad_200' and eid=='qh-m3-v12-cal-594fc82381d222917ad15a7b' and spoon_audit.is_file():
                    replacement=read(spoon_audit)
                    assert replacement['status']=='COMPLETE' and replacement['source_frame_indices']==frames
                    new_poses[identity]=np.load(ROOT/replacement['pose_path'],allow_pickle=False)
                    statuses[identity]['displayed_pose']='localized_initialization_rerun'
                else:
                    new_poses[identity]=np.load(ROOT/pose['pose_path'],allow_pickle=False)
                    statuses[identity]['displayed_pose']='four_case_pose'
                vertices[identity]=np.asarray(trimesh.load(ROOT/entity['mesh_path'],process=False).vertices)
            md.text((8,i*400+5),f'#{rank} {identity} | CAD | localization | actual SAM mask ({check["status"]})',fill='black')
            put(masks,Image.open(ROOT/entity['reference_image']),(0,i*400+35),(470,350))
            scene=Image.fromarray(rgb[0]).copy();sd=ImageDraw.Draw(scene)
            if loc['status']=='located':
                value=loc['location'];x1,y1,x2,y2=value['box']
                sd.rectangle((x1*.64,y1*.36,x2*.64,y2*.36),outline='#ff3050',width=2)
                for key,color in [('positive_points','#00ff50'),('negative_points','#ff3030')]:
                    for x,y in value[key]:
                        x*=.64;y*=.36;sd.ellipse((x-4,y-4,x+4,y+4),fill=color)
            put(masks,scene,(480,i*400+35),(470,350))
            if (folder/'mask_overlay.png').exists():
                put(masks,Image.open(folder/'mask_overlay.png'),(960,i*400+35),(470,350))
        for row,index in enumerate((0,7,14)):
            for col,version in enumerate(('before','after','new')):
                label={'before':'original v1.2','after':'previous batch','new':'this repair'}[version]
                gd.text((col*640+8,row*380+5),f'#{rank} {label} | source frame {frames[index]}',fill='black')
                if version!='new':
                    picture=Image.open(OLD/'preview/frames'/eid/f'{index:02d}_{version}.jpg').convert('RGB')
                else:
                    picture=Image.fromarray(rgb[index]).copy()
                    for identity,poses in new_poses.items():
                        mesh=vertices[identity];pose=poses[index]
                        sampled=mesh[::max(1,len(mesh)//900)]
                        draw_cloud(picture,sampled@pose[:3,:3].T+pose[:3,3],K,colors[identity],1)
                        draw_axes(picture,(mesh.min(0)+mesh.max(0))/2,pose,K)
                    if not new_poses:
                        ImageDraw.Draw(picture).text((8,8),'NO ACCEPTED OBJECT POSES',fill='red')
                put(geometry,picture,(col*640,row*380+20),(640,360))
        geometry.save(output/f'geometry_{rank:02d}.jpg',quality=93)
        masks.save(output/f'masks_{rank:02d}.jpg',quality=93)
        summaries.append(dict(ordinal=rank,event_id=eid,source_frame_indices=frames,entities=statuses))
    (output/'data.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    observations={
        3:'托盘贴回实物，刀不再落到左手上；但刀的后段姿态仍不贴合。',
        4:'砧板明显改善；木勺学习式RGB-D细化已通过冻结的全部工程门。第58帧中心误差4.19→1.14 cm、旋转误差18.32→6.04度；15帧投影都留在木勺上。当前展示这一通过版。',
        13:'碗不再飞到木柄旁；但碗/木铲分割仍混合，模型给的通过判断不可靠。',
        28:'两个壶的身份不再互换；黄壶中段和末段仍有偏移。'
    }
    sections=[]
    for record in summaries:
        rank=record['ordinal']
        descriptions=[observations[rank]]+[f'{k}：'+('有新姿态' if v['pose']['status']=='observed' else '未获得新姿态') for k,v in record['entities'].items()]
        spoon=''
        if rank==4 and (NEW/'spoon_registration/qa_geometry.jpg').is_file():
            spoon='<p><strong>当前通过版：</strong>原画面 → 上一RGB-D平移版 → FoundationPose学习式RGB-D细化 → 原生参考（仅事后检查）。第58帧中心误差4.19→1.14 cm、旋转18.32→6.04度。</p><img src="../spoon_foundation_refinement/qa_geometry.jpg"><p>当前通过版全部15帧投影：均留在真实木勺上。</p><img src="../spoon_foundation_refinement/all_frames.jpg"><p>此前RGB-D平移版：第58帧降至4.19 cm，但旋转没有修正。</p><img src="../spoon_rgbd_depth_refinement/qa_geometry.jpg"><p>局部6D ICP失败对照：内部点云残差虽下降，但第58帧旋转18.32→28.66度，因此未采用。</p><img src="../spoon_rgbd_se3_refinement/qa_geometry.jpg"><p>更早的首帧、独立逐帧与mask-aware定位过程</p><img src="../spoon_registration/qa_geometry.jpg"><img src="../spoon_per_frame_registration/qa_geometry.jpg"><img src="../spoon_mask_aware_tracking/qa_geometry.jpg"><p>同一组时刻的实际SAM mask</p><img src="../spoon_per_frame_registration/mask_qa.jpg">'
        sections.append(f'<section><h2>原网页第 {rank} 条</h2><p>{html.escape("；".join(descriptions))}</p><p>CAD → 新定位 → 实际分割</p><img src="masks_{rank:02d}.jpg"><p>原始版本 → 上轮版本 → 本次修复；三行分别为前、中、后帧</p><img src="geometry_{rank:02d}.jpg">{spoon}</section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuietHand 四例修复对照</title><style>body{font:16px/1.6 system-ui;max-width:1800px;margin:auto;padding:20px;background:#f4f6f8}section{background:white;padding:18px;margin:20px 0}img{width:100%;height:auto}</style><h1>四条典型问题：定位、分割与姿态</h1><p>本次只重算物体，手部输出与原评语未改；右列暂不画手，便于看物体。绿色/橙色物体编号跨版本保持一致。没有新姿态不代表修复成功。</p>'''+''.join(sections)+'</html>'
    (output/'index.html').write_text(page)
    print(json.dumps(dict(event_count=len(summaries),geometry_sheets=len(summaries),mask_sheets=len(summaries),native_pose_used=False)))


if __name__=='__main__':
    main()
