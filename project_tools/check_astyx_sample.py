"""Check real Astyx sample fields/calibration; this is not a dataset adapter."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--root', type=Path, default=root / 'datasets/astyx_sample/dataset_astyx_hires2019')
    parser.add_argument('--output', type=Path, default=root / 'setup/astyx_sample_check.json')
    args = parser.parse_args()
    lidar = np.loadtxt(args.root / 'lidar_vlp16/000000.txt', skiprows=1)
    radar = np.loadtxt(args.root / 'radar_6455/000000.txt', skiprows=2)
    assert lidar.shape[1] == 6 and radar.shape[1] == 5
    assert np.isfinite(lidar).all() and np.isfinite(radar).all()
    calib = json.loads((args.root / 'calibration/000000.json').read_text())
    sensors = {s['sensor_uid']: s['calib_data'] for s in calib['sensors']}
    transform = np.asarray(sensors['lidar_vlp16']['T_to_ref_COS'])
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(sensors['radar_6455']['T_to_ref_COS'], np.eye(4))
    xyz = lidar[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    reconstructed = (xyz - transform[:3, 3]) @ np.linalg.inv(transform[:3, :3]).T
    np.testing.assert_allclose(reconstructed, lidar[:, :3], atol=1e-10)
    gt = json.loads((args.root / 'groundtruth_obj3d/000000.json').read_text())
    cars = [o for o in gt['objects'] if o['classname'] == 'Car']
    for car in cars:
        assert np.isfinite(car['center3d']).all()
        assert (np.asarray(car['dimension3d']) > 0).all()
        np.testing.assert_allclose(np.linalg.norm(car['orientation_quat']), 1, atol=1e-6)
    lidar_input = np.column_stack((xyz, lidar[:, 3])).astype(np.float32)
    radar_input = radar[:, :4].astype(np.float32)  # V_r is column 3; Mag is excluded.
    out = {'sample_id': '000000', 'status': 'passed', 'scope': 'one_raw_sample_only',
           'lidar_shape': list(lidar.shape), 'radar_shape': list(radar.shape),
           'lidar_model_input_shape': list(lidar_input.shape),
           'radar_model_input_shape': list(radar_input.shape),
           'radar_fourth_feature': 'V_r', 'reference_frame': 'radar_6455',
           'car_annotations': len(cars), 'record_len_for_future_adapter': [1],
           'rgb_read': False, 'split_created': False, 'weather_created': False,
           'loader_adapter_implemented': False}
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
