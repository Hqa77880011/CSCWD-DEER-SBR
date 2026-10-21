#!/usr/bin/env python3
"""Train-only source coverage audit; geometry does not certify weather provenance.

This report deliberately cannot authorize positive weather labels: proving that
a point moved/disappeared is distinct from proving its physical source/type.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from opencood.project.config import load_config,file_sha256
from opencood.project.fixed_gt import load_manifest
from opencood.project.data_io import read_pcd
from opencood.project.data_geometry import prepare_points
from opencood.project.supervision import mutual_nn_targets
from project_tools.run_experiments import read_hypes,source_tree_hash,data_contract_metadata,supervision_contract


def audit_provenance(config, samples):
    """Replay the actual input augmentation; never certifies natural Fog labels."""
    from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
    d=config['data'];start=time.perf_counter();source_before=source_tree_hash()
    dataset=ProjectV2XRDataset(read_hypes(config),root=d['root'],split='train',training=True,
        track='W',seed=config['experiment']['seed'],paired=False,augment=True,cache_bytes=0,
        supervision_source='provenance_v1',controlled_corruption=d['controlled_corruption'],
        gt_manifest=str(Path(d['manifest_dir'])/'eval_gt_manifest.json'),matching=d['matching'])
    indices=np.linspace(0,len(dataset)-1,min(samples,len(dataset)),dtype=int).tolist()
    rows=[];fog_masked=True
    for index in indices:
        sample=dataset[int(index)]
        for target,meta in zip(sample['train_targets'],sample['ego']['metadata']):
            stats=meta['supervision']
            target_digest=hashlib.sha256()
            for key,value in sorted(target.items()):
                target_digest.update(key.encode());target_digest.update(str(value.shape).encode())
                target_digest.update(value.astype('<f4').tobytes())
            rows.append({'sample_id':sample['ego']['sample_id'],'agent_id':meta['agent_id'],
                         'sample_index':int(index),'epoch':0,'weather':meta['weather'],
                         'target_sha256':target_digest.hexdigest(),
                         'perturbation_applied':meta['controlled_corruption'],**stats})
            if meta['weather']=='fog':
                fog_masked &= all(not np.any(target[key]) for key in ('miss_valid','spur_valid','deg_valid'))
    exact=[row for row in rows if row.get('source')=='provenance_v1']
    fields=('normal_points','observed_points','removed_points','injected_points','miss_valid_cells',
            'spur_valid_cells','positive_miss_cells','positive_spur_cells','joint_positive_cells')
    sums={key:sum(row[key] for row in exact) for key in fields}
    verified=bool(exact) and all(row['source_ids_verified'] for row in exact)
    ready=(samples==32 and len(indices)==32 and verified and fog_masked and sums['positive_miss_cells']>0 and sums['positive_spur_cells']>0
           and source_tree_hash()==source_before)
    contract=supervision_contract(config)
    contract.update(source_sha256=source_before,audit_samples=len(indices),sample_indices=indices)
    return {'schema':'project_provenance_supervision_audit_v2','split':'train','dataset_id':'v2xr',
        'supervision_source':'provenance_v1','samples':len(indices),'agents':len(rows),
        'seed':config['experiment']['seed'],'epoch':0,'sample_indices':indices,
        'scope':'deterministic evenly spaced train scenes and all retained real agents, epoch 0',
        'controlled_corruption':d['controlled_corruption'],**sums,
        'unmatched_semantics_verified':False,'formal_positive_supervision_ready':ready,
        'evidence':{'kind':'controlled_source_ids','source_ids_verified':verified,
            'normal_train_only':True,'all_natural_fog_labels_masked':fog_masked,
            'no_snow_or_validation_test':True,'no_teacher_network':True},
        'contract':contract,
        'limitations':['Labels certify only the registered controlled deletion/injection, not natural Fog/Snow physics.',
            'Real Fog labels remain unknown; no unmatched-point or point-count-difference positive labels.',
            'No model training, checkpoint selection or AP measurement; sampled coverage is not an exhaustive audit.'],
        'seconds':time.perf_counter()-start,'rows':rows}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='cscwd_configs/project_v4_W_AB.yaml')
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--output',default='/root/autodl-tmp/CSCWD/setup/pair_supervision_audit.json')
    args=p.parse_args()
    if not 1<=args.samples<=8083:p.error('samples must be between 1 and 8083')
    config=load_config(args.config);d=config['data'];root=Path(d['root'])
    if d['supervision_source']=='provenance_v1':
        if config['experiment']['track']!='W':p.error('provenance audit requires the W training protocol')
        report=audit_provenance(config,args.samples)
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
        return
    manifest_path=Path(d['manifest_dir'])/'eval_gt_manifest.json'
    manifest=load_manifest(manifest_path);hypes=read_hypes(config)
    bounds=hypes['preprocess']['cav_lidar_range'];voxel=hypes['preprocess']['args']['voxel_size']
    matching=d.get('matching',dict(max_distance=.002,ambiguity_margin=.001,unmatched_min_distance=.05))
    ids=sorted(sid for sid,sample in manifest['samples'].items() if sample['split']=='train')
    ids=[ids[i] for i in np.linspace(0,len(ids)-1,min(args.samples,len(ids)),dtype=int)]
    rows=[];start=time.perf_counter()
    for sid in ids:
        sample=manifest['samples'][sid]
        base=root/sample['sequence']/sample['ego_id']/sample['timestamp']
        clean=read_pcd(str(base)+'.pcd');fog=read_pcd(str(base)+'_fog_0.060.pcd')
        # Ego cloud is already in the ego coordinate system; retain the exact
        # training self-filter, FOV, range and grid, with identity augmentation.
        clean=prepare_points(clean,np.eye(4),sample['camera'],np.eye(4),bounds)
        fog=prepare_points(fog,np.eye(4),sample['camera'],np.eye(4),bounds)
        target=mutual_nn_targets(clean[:,:3],fog[:,:3],bounds,voxel,allow_unmatched=False,**matching)
        candidates=mutual_nn_targets(clean[:,:3],fog[:,:3],bounds,voxel,allow_unmatched=True,**matching)
        stats=target['stats'];candidate=candidates['stats']
        rows.append({'sample_id':sid,'agent_id':sample['ego_id'],**stats,
                     'geometric_unmatched_candidate_miss_cells':candidate['positive_miss_cells'],
                     'geometric_unmatched_candidate_spur_cells':candidate['positive_spur_cells']})
    sums={key:sum(row[key] for row in rows) for key in ('clean_points','observed_points','matched_points',
          'miss_valid_cells','spur_valid_cells','positive_miss_cells','positive_spur_cells',
          'geometric_unmatched_candidate_miss_cells','geometric_unmatched_candidate_spur_cells')}
    report={'schema':'pair_supervision_coverage_v1','split':'train','dataset_id':'v2xr',
        'samples':len(rows),'scope':'deterministic evenly spaced train frames, ego only, normal versus fog',
        'matching':matching,'grid':{'bounds':bounds,'voxel_size':voxel},**sums,
        'unmatched_semantics_verified':False,'formal_positive_supervision_ready':False,
        'evidence':{'kind':'geometry_only','does_not_certify_unmatched_weather_types':True},
        'limitations':['No weather generator or source IDs have certified unmatched-point semantics.',
                      'Geometric candidate counts are not accepted positive labels.',
                      'Only conservative reciprocal unique matches supply negative labels; unknown cells are masked.',
                      'No Snow or validation/test data used. No model training or AP measurement.'],
        'source_hashes':{name:file_sha256(Path(d['manifest_dir'])/(name+'.json')) for name in
                         ('data_manifest','split_manifest','weather_manifest','protocol_manifest','eval_gt_manifest')},
        'contract':supervision_contract(config),
        'seconds':time.perf_counter()-start,'rows':rows}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
