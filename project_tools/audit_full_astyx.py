"""Audit the pinned Astyx non-RGB sparse checkout against Git object hashes."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/root/autodl-tmp/CSCWD')
REPO = ROOT / 'datasets/astyx_source'
DATA = REPO / 'dataset_astyx_hires2019'
REVISION = '2310e7f35a4f2b76de85463f0dd1b0b4e50c2f69'


def main():
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if actual != REVISION:
        raise RuntimeError('Unexpected Astyx source revision')
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', REVISION], cwd=REPO)
    objects = {}
    for item in tree.split(b'\0'):
        if not item:
            continue
        fields, name = item.split(b'\t', 1)
        objects[name.decode()] = fields.split()[2].decode()
    dataset = json.loads((DATA / 'dataset.json').read_text())
    records = []
    classes = Counter()
    point_counts = {'lidar': [], 'radar': []}
    empty_clouds = []
    for sample_id, sample in sorted(dataset['data'].items(), key=lambda kv: int(kv[0])):
        files = {}
        for sensor in ('lidar_vlp16', 'radar_6455', 'calibration', 'groundtruth_obj3d'):
            relative = sample[sensor]
            path = DATA / relative
            content = path.read_bytes()
            source_path = 'dataset_astyx_hires2019/' + relative
            expected = objects[source_path]
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if blob != expected:
                raise RuntimeError('Source object mismatch: ' + relative)
            files[sensor] = {'path': relative, 'bytes': len(content), 'git_blob_sha1': blob,
                             'sha256': hashlib.sha256(content).hexdigest()}
        for sensor, header_lines, columns, modality in (
                ('lidar_vlp16', 1, 6, 'lidar'), ('radar_6455', 2, 5, 'radar')):
            array = np.loadtxt(DATA / sample[sensor], skiprows=header_lines, ndmin=2)
            if not array.size:
                empty_clouds.append({'sample_id': sample_id, 'modality': modality})
            elif array.shape[1] != columns or not np.isfinite(array).all():
                raise RuntimeError('Invalid fields: ' + sample[sensor])
            point_counts[modality].append(array.shape[0] if array.size else 0)
        calibration = json.loads((DATA / sample['calibration']).read_text())
        sensors = {s['sensor_uid']: s['calib_data'] for s in calibration['sensors']}
        for sensor in ('lidar_vlp16', 'radar_6455'):
            matrix = np.asarray(sensors[sensor]['T_to_ref_COS'])
            assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
            np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        gt = json.loads((DATA / sample['groundtruth_obj3d']).read_text())
        car_count = 0
        for obj in gt['objects']:
            classes[obj['classname']] += 1
            if obj['classname'] == 'Car':
                car_count += 1
                assert np.isfinite(obj['center3d']).all()
                assert (np.asarray(obj['dimension3d']) > 0).all()
                np.testing.assert_allclose(np.linalg.norm(obj['orientation_quat']), 1, atol=1e-5)
        records.append({'sample_id': sample_id, 'files': files, 'car_count': car_count})
    if len(records) != 546:
        raise RuntimeError('Expected 546 Astyx samples')
    alias = ROOT / 'datasets/astyx/dataset_astyx_hires2019'
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.symlink_to(DATA, target_is_directory=True)
    for source in ('license.txt', 'readme.txt'):
        target = alias.parent / source
        if not target.exists():
            target.symlink_to(REPO / source)
    manifest = {'source': 'https://github.com/under-the-radar/radar_dataset_astyx',
                'revision': REVISION, 'root': str(DATA), 'sample_count': len(records),
                'samples': records, 'rgb_downloaded': False}
    (alias.parent / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    audit = {'completed_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
             'sample_count': len(records), 'verified_sensor_files': 4 * len(records),
             'source_revision': REVISION, 'all_git_blob_hashes_match': True,
             'all_point_fields_finite': True, 'radar_fourth_feature': 'V_r',
             'point_counts': {k: {'min': min(v), 'max': max(v), 'total': sum(v)} for k, v in point_counts.items()},
             'classes': dict(classes), 'empty_clouds': empty_clouds,
             'rgb_downloaded': False, 'split_created': False, 'synthetic_weather_created': False}
    (alias.parent / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
