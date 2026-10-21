"""Read the complete sample sequence without RGB; optional real-batch GPU check.

Uses the upstream label path for diagnostics only. No AP is calculated and this
does not validate the proposed fixed-GT project evaluation protocol.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'cscwd_configs/baseline_sample.yaml')
    parser.add_argument('--output', type=Path, default=REPO.parent / 'setup/v2xr_sample_check.json')
    parser.add_argument('--gpu-step', action='store_true')
    args = parser.parse_args()
    hypes = load_yaml(str(args.config))
    assert hypes['load_rgb'] is False
    report = {'scope': 'one_train_sequence_diagnostic_only', 'domains': {},
              'fixed_gt_protocol_validated': False, 'official_splits_complete': False,
              'status': 'running'}
    try:
        with patch('opencood.data_utils.datasets.opv2v.basedataset.load_camera_data',
                   side_effect=AssertionError('RGB loader was called')), \
             patch('PIL.Image.open', side_effect=AssertionError('Image file was opened')):
            for domain, weather in (('normal', None), ('fog', '_fog_0.060'), ('snow', '_snow_2.5_2.0')):
                config = copy.deepcopy(hypes)
                config['train_sim'] = False
                config['eval_sim'] = weather is not None
                if weather:
                    config['sim_weather'] = weather
                dataset = build_dataset(config, visualize=False, train=False)
                raw, sequence, timestamp = dataset.retrieve_base_data(0)
                assert len(raw) == 3
                ego = next(v for v in raw.values() if v['ego'])
                for cav in raw.values():
                    assert 'camera0' not in cav
                    assert np.isfinite(cav['lidar_np']).all() and np.isfinite(cav['radar_np']).all()
                    expected_path = dict(cav)
                    size = expected_path.pop('camera0_size')
                    expected_path['camera0'] = SimpleNamespace(size=size)
                    # Same original geometry calculation; dimensions come from actual PNG IHDR.
                    np.testing.assert_array_equal(
                        dataset.mask_ego_fov_flag(cav, cav['lidar_np'], ego['params']),
                        dataset.mask_ego_fov_flag(expected_path, cav['lidar_np'], ego['params']))
                item = dataset[0]
                batch = dataset.collate_batch_train([item])
                assert batch['ego']['record_len'].item() >= 2
                report['domains'][domain] = {
                    'dataset_samples': len(dataset), 'raw_agents': list(raw),
                    'batch_record_len': batch['ego']['record_len'].tolist(),
                    'first_timestamp': timestamp, 'fov_metadata_equivalent': True,
                    'raw_point_counts': {k: {'lidar': len(v['lidar_np']), 'radar': len(v['radar_np'])}
                                         for k, v in raw.items()},
                    'voxel_counts': {m: int(batch['ego']['processed_' + m]['voxel_features'].shape[0])
                                     for m in ('lidar', 'radar')}, 'rgb_reads': 0}
                print('PASS real sequence', domain, report['domains'][domain]['batch_record_len'], flush=True)
                if args.gpu_step and domain == 'normal':
                    model = train_utils.create_model(config).cuda().train()
                    criterion = train_utils.create_loss(config).cuda()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                    device_batch = train_utils.to_device(batch['ego'], torch.device('cuda'))
                    optimizer.zero_grad()
                    output = model(device_batch)
                    loss = criterion(output, device_batch['label_dict'])
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    optimizer.step()
                    torch.cuda.synchronize()
                    report['real_baseline_gpu_step'] = {
                        'loss': loss.item(), 'output_shapes': {k: list(v.shape) for k, v in output.items()},
                        'peak_gpu_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        'legacy_labels_diagnostic_only': True}
                    del model, optimizer, criterion, device_batch, output, loss
                    torch.cuda.empty_cache()
            report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
