"""V2X-R project dataset: fixed GT, no synthetic padding points, no RGB reads.

Legacy datasets are kept for diagnostic reproduction. This path isolates model
inputs from training-only labels/clean pairs and preserves every legal sample.
"""
import copy,json,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.project.data_io import ByteLRU
from opencood.project.geometry import pose_to_world,transform_points
from opencood.project.data_geometry import (augmentation_matrix,augment_boxes,prepare_points,
    bev_grid,observation_maps,point_fov_mask)
from opencood.project.anchor_targets import AnchorTargets
from opencood.project.weather import weather_for,augmentation_seed
from opencood.project.supervision import (normal_targets,mutual_nn_targets,controlled_corruption as corrupt_points,
    controlled_corruption_config,provenance_targets)
from opencood.project.fixed_gt import load_manifest
from opencood.project.config import file_sha256

class ProjectV2XRDataset(Dataset):
    def __init__(self,hypes,*,root='/root/autodl-tmp/CSCWD/datasets/v2xr',split='train',
                 training=True,track='C',weather='normal',seed=17,epoch=0,paired=False,
                 augment=True,cache_bytes=0,supervision_source='none',disabled_modalities=(),
                 gt_manifest=None,supervision_audit=None,matching=None,controlled_corruption=None):
        self.hypes=copy.deepcopy(hypes);self.root=Path(root).resolve();self.split=split
        self.training=bool(training);self.track=track;self.weather=weather;self.seed=int(seed)
        if split not in ('train','validate','test'):raise ValueError('Invalid split')
        if training and split!='train':raise ValueError('Training only uses train split')
        if training and weather=='snow':raise ValueError('Snow is final evaluation only')
        if paired and not training:raise ValueError('Clean paired inputs are training-only')
        if not training and augment:augment=False
        if supervision_source not in ('none','mutual_nn_v1','provenance_v1'):
            raise ValueError('Unknown supervision source')
        if paired and supervision_source!='mutual_nn_v1':
            raise ValueError('paired is only a normal-point-array correspondence supervision option')
        self.paired=bool(paired);self.augment=bool(augment);self.supervision_source=supervision_source
        self.controlled_corruption=controlled_corruption_config(controlled_corruption)
        self.epoch_value=mp.Value('q',int(epoch));self.cache=ByteLRU(cache_bytes)
        self.disabled=set(disabled_modalities)
        if not self.disabled <= {'lidar','radar'}:raise ValueError('Unknown modality override')
        self.allow_unmatched=False
        self.matching=dict(max_distance=0.002,ambiguity_margin=0.001,unmatched_min_distance=0.05)
        if matching is not None:
            if set(matching)!=set(self.matching):raise ValueError('Unexpected pairing threshold keys')
            self.matching.update(matching)
        if not (0<self.matching['max_distance']<self.matching['unmatched_min_distance'] and self.matching['ambiguity_margin']>=0):
            raise ValueError('Invalid pairing thresholds')
        if supervision_audit and supervision_source=='mutual_nn_v1':
            audit=json.loads(Path(supervision_audit).read_text())
            if not audit.get('unmatched_semantics_verified') or audit.get('split')!='train':
                raise ValueError('Invalid supervision correspondence audit')
            self.allow_unmatched=True
        path=Path(gt_manifest) if gt_manifest else self.root/'eval_gt_manifest.json'
        self.manifest=load_manifest(path)
        for name,expected in self.manifest['source_manifests'].items():
            if file_sha256(self.root/name)!=expected:
                raise ValueError(f'Dataset root does not match fixed GT source: {name}')
        data_manifest=json.loads((self.root/'data_manifest.json').read_text())
        self.cache.expected_sha256={str(self.root/item['path']):item['sha256']
            for frame in data_manifest['frames'] if frame['split']==split
            for item in frame['modalities'].values()}
        del data_manifest
        self.sample_ids=sorted(sid for sid,s in self.manifest['samples'].items() if s['split']==split)
        if not self.sample_ids:raise ValueError('No manifest samples for split')
        self.bounds=self.hypes['preprocess']['cav_lidar_range']
        self.voxel=self.hypes['preprocess']['args']['voxel_size']
        self.grid=bev_grid(self.bounds,self.voxel)
        self.pre_processor=build_preprocessor(self.hypes['preprocess'],train=training)
        self.post_processor=build_postprocessor(self.hypes['postprocess'],dataset='opv2v',train=training)
        self.anchor_box=self.post_processor.generate_anchor_box().astype(np.float32)
        ta=self.hypes['postprocess']['target_args']
        self.assigner=AnchorTargets(self.anchor_box,ta['pos_threshold'],ta['neg_threshold'])
        self.max_agents=int(self.hypes.get('train_params',{}).get('max_cav',5))
        self.comm_range=float(self.hypes.get('comm_range',70.))
    def __len__(self):return len(self.sample_ids)
    def set_epoch(self,epoch):self.epoch_value.value=int(epoch)
    def _voxelize(self,points,rng):
        if len(points)==0:
            return {'voxel_features':np.empty((0,self.hypes['preprocess']['args']['max_points_per_voxel'],4),np.float32),
                    'voxel_coords':np.empty((0,3),np.int32),'voxel_num_points':np.empty((0,),np.int32)}
        # Point order can affect voxel caps; the independent stream is identical
        # across variants, and cannot depend on model RNG consumption.
        if self.training:points=points[rng.permutation(len(points))]
        output=self.pre_processor.preprocess(np.ascontiguousarray(points,dtype=np.float32))
        return {key:np.array(value,copy=True) for key,value in output.items()}
    def __getitem__(self,index):
        sid=self.sample_ids[index];sample=self.manifest['samples'][sid]
        seq=sample['sequence'];timestamp=sample['timestamp'];epoch=self.epoch_value.value
        # Manifest sequence is root-relative split/sequence. Accept only this
        # documented layout; never silently switch to an unrelated directory.
        if not str(seq).startswith(self.split+'/'):seq=self.split+'/'+str(seq)
        ego_id=str(sample['ego_id']);agent_ids=list(map(str,sample['agent_ids']))
        agent_ids=[ego_id]+[x for x in agent_ids if x!=ego_id]
        for agent in agent_ids:
            source=sample['source_annotations'][agent]
            if self.cache.expected_sha256.get(str(self.root/source['path']))!=source['sha256']:
                raise ValueError('GT annotation and dataset manifest sources disagree')
        ann={a:self.cache.get(self.root/seq/a/(timestamp+'.yaml'),'yaml') for a in agent_ids}
        ego_world=pose_to_world(ann[ego_id]['lidar_pose']);world_to_ego=np.linalg.inv(ego_world)
        selected=[]
        for agent in agent_ids:
            position=np.asarray(ann[agent]['lidar_pose'][:3]);ego_pos=np.asarray(ann[ego_id]['lidar_pose'][:3])
            if agent==ego_id or np.linalg.norm(position[:2]-ego_pos[:2])<=self.comm_range:selected.append(agent)
        selected=selected[:self.max_agents]
        rng=np.random.default_rng(augmentation_seed(sid,epoch,self.seed))
        aug=augmentation_matrix(rng,self.training and self.augment)
        boxes=np.asarray([g['box'] for g in sample['gt']],dtype=np.float64).reshape(-1,7)
        train_boxes=augment_boxes(boxes,aug) if self.training else boxes.copy()
        current=[];targets=[];metadata=[]
        for agent in selected:
            a=ann[agent];base=self.root/seq/agent/timestamp
            transform=world_to_ego@pose_to_world(a['lidar_pose'])
            domain=weather_for(sid,agent,epoch,self.seed,self.track) if self.training else self.weather
            if domain not in ('normal','fog','snow'):raise ValueError('Unknown weather')
            suffix={'normal':'.pcd','fog':'_fog_0.060.pcd','snow':'_snow_2.5_2.0.pcd'}[domain]
            valid=np.array(['lidar' not in self.disabled,'radar' not in self.disabled],dtype=bool)
            lidar=self.cache.get(str(base)+suffix) if valid[0] else np.empty((0,4),np.float32)
            radar=self.cache.get(str(base)+'_radar.pcd') if valid[1] else np.empty((0,4),np.float32)
            lidar=prepare_points(lidar,transform,sample['camera'],aug,self.bounds)
            radar=prepare_points(radar,transform,sample['camera'],aug,self.bounds)
            source_reference=None;source_ids=None;corruption_applied=False
            if self.training and self.track=='W' and domain=='normal' and self.supervision_source=='provenance_v1' and valid[0]:
                source_reference=lidar
                corruption_rng=np.random.default_rng(augmentation_seed(sid+'/'+agent,epoch,self.seed,stream='controlled_corruption'))
                def legal_ghosts(points):
                    # Check the original sensor self mask, original ego camera
                    # and augmented range without applying geometry twice.
                    ego_xyz=transform_points(points[:,:3],np.linalg.inv(aug))
                    agent_xyz=transform_points(ego_xyz,np.linalg.inv(transform))
                    own=((agent_xyz[:,0]>=-1.95)&(agent_xyz[:,0]<=2.95)
                         &(agent_xyz[:,1]>=-1.1)&(agent_xyz[:,1]<=1.1))
                    bounds=np.asarray(self.bounds)
                    return (~own & point_fov_mask(ego_xyz,sample['camera'])
                            & np.isfinite(points).all(1)
                            & (points[:,:3]>=bounds[:3]).all(1)&(points[:,:3]<bounds[3:]).all(1))
                lidar,source_ids,corruption_applied=corrupt_points(
                    source_reference,corruption_rng,self.controlled_corruption,legal_mask=legal_ghosts)
            origins=[transform[:3,3],transform_points(np.array([a.get('radar_pose',a['lidar_pose'])[:3]]),world_to_ego)[0]]
            origins=[transform_points(np.asarray(o)[None],aug)[0] for o in origins]
            maps=observation_maps(lidar,radar,valid,origins,sample['camera'],aug,self.bounds,self.voxel,self.grid)
            point_rng=np.random.default_rng(augmentation_seed(sid+'/'+agent,epoch,self.seed,stream='point_order'))
            entry={'lidar':self._voxelize(lidar,point_rng),'radar':self._voxelize(radar,point_rng),'maps':maps,'valid':valid}
            current.append(entry)
            reference=None
            need_reference=self.training and self.supervision_source=='mutual_nn_v1' and domain=='fog'
            if need_reference:
                reference=self.cache.get(str(base)+'.pcd') if valid[0] else np.empty((0,4),np.float32)
                reference=prepare_points(reference,transform,sample['camera'],aug,self.bounds)
            target=normal_targets(lidar,self.bounds,self.voxel)
            stats={}
            if source_reference is not None:
                target=provenance_targets(source_reference,lidar,source_ids,self.bounds,self.voxel)
                stats=target.pop('stats');stats['perturbation_applied']=corruption_applied
            if domain!='normal':
                if self.training and self.supervision_source=='mutual_nn_v1':
                    target=mutual_nn_targets(reference[:,:3],lidar[:,:3],self.bounds,self.voxel,allow_unmatched=self.allow_unmatched,**self.matching)
                    stats=target.pop('stats')
                else:
                    target={k:np.zeros_like(v) for k,v in target.items()}
            if not valid[0]:target={k:np.zeros_like(v) for k,v in target.items()}
            target['normal_mask']=np.full((1,1,1),float(domain=='normal' and not corruption_applied),np.float32)
            targets.append(target)
            metadata.append({'agent_id':agent,'weather':domain,'lidar_points':len(lidar),'radar_points':len(radar),
                             'modality_valid':valid.tolist(),'sensor_origins':[o.tolist() for o in origins],
                             'controlled_corruption':corruption_applied,'supervision':stats})
        labels=self.assigner(train_boxes)
        target_diagnostics=copy.deepcopy(self.assigner.last_diagnostics)
        result={'ego':{'agents':current,'label_dict':labels,'target_diagnostics':target_diagnostics,'sample_id':sid,
                       'gt_boxes':boxes.astype(np.float32),'gt_ids':[g['gt_id'] for g in sample['gt']],
                       'training_boxes':train_boxes.astype(np.float32),'camera':sample['camera'],
                       'metadata':metadata,'augmentation':aug.astype(np.float32)},
                'train_targets':targets if self.training else None}
        return result
    def _collate_inputs(self,items):
        agents=[a for item in items for a in item['agents']]
        processed={m:self.pre_processor.collate_batch([a[m] for a in agents]) for m in ('lidar','radar')}
        return {'processed_lidar':processed['lidar'],'processed_radar':processed['radar'],
                'record_len':torch.tensor([len(item['agents']) for item in items],dtype=torch.long),
                'batch_size':len(agents),
                'pairwise_t_matrix':torch.eye(4).reshape(1,1,1,4,4).repeat(len(items),self.max_agents,self.max_agents,1,1),
                'observation_maps':{k:torch.from_numpy(np.stack([a['maps'][k] for a in agents])) for k in agents[0]['maps']},
                'modality_valid':torch.from_numpy(np.stack([a['valid'] for a in agents]))}
    def collate_batch(self,batch):
        items=[x['ego'] for x in batch];ego=self._collate_inputs(items)
        ego.update(label_dict={k:torch.from_numpy(np.stack([x['label_dict'][k] for x in items])) for k in items[0]['label_dict']},
                   sample_ids=[x['sample_id'] for x in items],gt_boxes=[x['gt_boxes'] for x in items],
                   gt_ids=[x['gt_ids'] for x in items],metadata=[x['metadata'] for x in items],
                   target_diagnostics=[x['target_diagnostics'] for x in items],
                   cameras=[x['camera'] for x in items],augmentations=[x['augmentation'] for x in items],
                   anchor_box=torch.from_numpy(self.anchor_box))
        out={'ego':ego}
        if self.training:
            targets=[t for x in batch for t in x['train_targets']]
            out['train_targets']={k:torch.from_numpy(np.stack([t[k] for t in targets])) for k in targets[0]}
        return out
    collate_batch_train=collate_batch
    collate_batch_test=collate_batch
