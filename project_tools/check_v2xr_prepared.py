"""Validate complete split routing and sample decoding without any RGB reads.
This is a data-loader diagnostic, not an experiment or fixed-GT evaluation.
"""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
ROOT=REPO.parent;DATA=ROOT/'datasets/v2xr'

def main():
 splits=json.loads((DATA/'split_manifest.json').read_text())['splits']
 config=load_yaml(str(REPO/'cscwd_configs/baseline_full_data_check.yaml'))
 report={'status':'running','scope':'all official split lengths; first sample of every nonempty sequence in three weathers; one collated sample per split/weather','checks':[],'rgb_reads':0,'fixed_gt_validated':False,'training_run':False}
 try:
  with patch('opencood.data_utils.datasets.opv2v.basedataset.load_camera_data',side_effect=AssertionError('RGB loader called')),patch('PIL.Image.open',side_effect=AssertionError('RGB pixels read')):
   for split in ('train','validate','test'):
    for weather,suffix in [('normal',None),('fog','_fog_0.060'),('snow','_snow_2.5_2.0')]:
     cfg=copy.deepcopy(config);cfg['validate_dir']=str(DATA/split);cfg['train_sim']=False;cfg['eval_sim']=suffix is not None
     if suffix:cfg['sim_weather']=suffix
     ds=build_dataset(cfg,visualize=False,train=False)
     expected=splits[split]['cooperative_samples'];assert len(ds)==expected,(split,len(ds),expected)
     starts=sorted(set([0]+ds.len_record[:-1]));starts=[i for i in starts if i<len(ds)]
     counts=[]
     for index in starts:
      raw,seq,timestamp=ds.retrieve_base_data(index)
      ego=next(v for v in raw.values() if v['ego'])
      for cav in raw.values():
       assert 'camera0' not in cav
       assert cav['lidar_np'].shape[1]==4 and cav['radar_np'].shape[1]==4
       assert np.isfinite(cav['lidar_np']).all() and np.isfinite(cav['radar_np']).all()
       before=dict(cav);before['camera0']=SimpleNamespace(size=before.pop('camera0_size'))
       np.testing.assert_array_equal(ds.mask_ego_fov_flag(cav,cav['lidar_np'],ego['params']),ds.mask_ego_fov_flag(before,cav['lidar_np'],ego['params']))
      counts.append({'index':index,'timestamp':timestamp,'agents':len(raw)})
     batch=ds.collate_batch_train([ds[0]])
     check={'split':split,'weather':weather,'dataset_samples':len(ds),'sampled_sequences':len(starts),'samples':counts,'batch_record_len':batch['ego']['record_len'].tolist(),'fov_metadata_equivalent':True}
     report['checks'].append(check);print('PASS',split,weather,len(ds),'sequences',len(starts),flush=True)
     del ds,batch
  report['status']='passed'
 except Exception as exc:
  report['status']='failed';report['error']=repr(exc);raise
 finally:
  (ROOT/'setup/v2xr_full_loader_check.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
