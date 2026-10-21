"""CPU data-path regressions and tiny real TRAIN/normal-fog samples only.

Run with the prepared CSCWD environment (spconv/Open3D are intentional runtime
dependencies). No test-split weather is inspected or used for tuning here.
"""
import hashlib
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
from opencood.project.data_io import read_pcd
from opencood.project.data_geometry import augmentation_matrix, augment_boxes
from opencood.project.geometry import boxes_to_corners, transform_points
from opencood.project.anchor_targets import AnchorTargets


ROOT = Path(os.environ.get('CSCWD_DATA_ROOT', str(Path(__file__).resolve().parents[1] / 'datasets/v2xr')))
CONFIG = Path(__file__).resolve().parents[1] / 'cscwd_configs' / 'baseline_full_data_check.yaml'


class _DigestDataset(torch.utils.data.Dataset):
    """Return compact evidence, avoiding huge BEVs in multiprocessing queues."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return 2
    def __getitem__(self, index):
        result = self.dataset[index]
        sample = result['ego']
        digest = hashlib.sha256(sample['augmentation'].tobytes())
        for agent in sample['agents']:
            for modality in ('lidar', 'radar'):
                for key in ('voxel_features', 'voxel_coords', 'voxel_num_points'):
                    digest.update(agent[modality][key].tobytes())
        digest.update(json.dumps(sample['metadata'], sort_keys=True).encode())
        if result.get('train_targets') is not None:
            for target in result['train_targets']:
                for key,value in sorted(target.items()):digest.update(value.tobytes())
        return digest.hexdigest()


class DatasetGeometryTests(unittest.TestCase):
    def test_green_channel_cannot_enter_model_features(self):
        header = 'VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n'
        points = np.array([[1, 2, 3, 0], [1, 2, 3, 0]], np.float32)
        points[:, 3] = np.array([(17 << 16), (17 << 16) | (255 << 8)], dtype=np.uint32).view(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'measurement.pcd'
            path.write_bytes(header.encode() + points.tobytes())
            output = read_pcd(path)
        np.testing.assert_array_equal(output[0], output[1])
        self.assertEqual(output.shape[1], 4)
        self.assertAlmostEqual(output[0, 3], 17 / 255)

    def test_affine_augmentation_boxes_match_transformed_corners(self):
        boxes = np.array([[9, 4, 1, 4, 2, 1.5, .7], [3, -2, 0, 6, 3, 2, -1.2]])
        for seed in range(5):
            matrix = augmentation_matrix(np.random.default_rng(seed))
            transformed = transform_points(boxes_to_corners(boxes).reshape(-1, 3), matrix).reshape(2, 8, 3)
            predicted = boxes_to_corners(augment_boxes(boxes, matrix))
            for a, b in zip(transformed, predicted):
                a = a[np.lexsort(a.T[::-1])]
                b = b[np.lexsort(b.T[::-1])]
                np.testing.assert_allclose(a, b, atol=1e-10)

    def test_anchor_assignment_preserves_more_than_100_ground_truth(self):
        count = 125
        anchors = np.array([[i * 10, 0, 0, 2, 2, 4, 0] for i in range(count)], np.float32).reshape(1, count, 1, 7)
        gt = anchors.reshape(-1, 7)[:, [0, 1, 2, 5, 4, 3, 6]]
        labels = AnchorTargets(anchors)(gt)
        self.assertEqual(labels['pos_equal_one'].sum(), count)
        self.assertEqual(labels['neg_equal_one'].sum(), 0)
        np.testing.assert_array_equal(labels['targets'], 0)
        empty = AnchorTargets(anchors)(np.empty((0, 7)))
        self.assertEqual(empty['pos_equal_one'].sum(), 0)
        self.assertEqual(empty['neg_equal_one'].sum(), count)


@unittest.skipUnless((ROOT / 'eval_gt_manifest.json').exists(), 'prepared V2X-R manifest unavailable')
class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
        torch.set_num_threads(2)
        hypes = load_point_pillar_params(yaml.safe_load(CONFIG.read_text()))
        cls.dataset = ProjectV2XRDataset(hypes, root=ROOT, split='train', training=True,
            track='C', paired=False, augment=True, cache_bytes=16 << 20)

    def setUp(self):
        self.dataset.set_epoch(0)
        self.dataset.disabled = set()
        self.dataset.training = True
        self.dataset.track = 'C'
        self.dataset.supervision_source = 'none'
        from opencood.project.supervision import controlled_corruption_config
        self.dataset.controlled_corruption = controlled_corruption_config()

    def test_real_reader_matches_upstream_open3d(self):
        from opencood.utils.pcd_utils import pcd_to_np
        sample = self.dataset.manifest['samples'][self.dataset.sample_ids[0]]
        base = ROOT / sample['sequence'] / str(sample['ego_id']) / sample['timestamp']
        for suffix in ('.pcd', '_fog_0.060.pcd', '_radar.pcd'):
            with self.subTest(suffix=suffix):
                path = str(base) + suffix
                np.testing.assert_array_equal(read_pcd(path), pcd_to_np(path))

    def test_teacher_free_inputs_keep_supervision_separate(self):
        sample = self.dataset[0]
        current = sample['ego']['agents']
        self.assertNotIn('teacher_inputs', sample)
        self.assertNotIn('pairing_keys', sample)
        batch = self.dataset.collate_batch([sample])
        self.assertNotIn('train_targets', batch['ego'])
        self.assertNotIn('teacher_inputs', batch['ego'])
        self.assertNotIn('teacher_inputs', batch)
        self.assertEqual(batch['train_targets']['miss'].count_nonzero(), 0)
        self.assertEqual(batch['train_targets']['spur'].count_nonzero(), 0)
        self.assertEqual(batch['ego']['record_len'].sum(), len(current))

    def test_empty_modalities_preserve_agents_ids_and_ground_truth(self):
        self.dataset.disabled = {'lidar', 'radar'}
        sample = self.dataset[0]
        batch = self.dataset.collate_batch([sample])
        for modality in ('lidar', 'radar'):
            self.assertEqual(batch['ego']['processed_' + modality]['voxel_features'].shape[0], 0)
            self.assertEqual(batch['ego']['observation_maps'][modality + '_count'].count_nonzero(), 0)
        self.assertGreater(batch['ego']['record_len'].sum(), 0)
        manifest = self.dataset.manifest['samples'][self.dataset.sample_ids[0]]
        self.assertEqual(batch['ego']['gt_ids'][0], [g['gt_id'] for g in manifest['gt']])
        self.assertEqual(len(batch['ego']['gt_boxes'][0]), len(manifest['gt']))
        self.assertEqual(batch['train_targets']['miss_valid'].count_nonzero(), 0)

    def test_voxel_generator_buffers_are_owned_and_sensor_origin_is_per_agent(self):
        sample = self.dataset[0]['ego']
        before = sample['agents'][0]['lidar']['voxel_features'].copy()
        self.dataset._voxelize(np.array([[20, 0, 0, 1], [21, 1, 0, 1]], np.float32), np.random.default_rng(0))
        np.testing.assert_array_equal(sample['agents'][0]['lidar']['voxel_features'], before)
        xx, yy = self.dataset.grid
        denominator = np.linalg.norm(np.asarray(self.dataset.bounds[3:5]) - np.asarray(self.dataset.bounds[:2]))
        for agent, metadata in zip(sample['agents'], sample['metadata']):
            for i, modality in enumerate(('lidar', 'radar')):
                origin = metadata['sensor_origins'][i]
                expected = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2) / denominator
                np.testing.assert_allclose(agent['maps'][modality + '_distance'][0], expected, rtol=1e-7)
                occupied = agent['maps'][modality + '_occupancy'] > 0
                self.assertTrue((agent['maps'][modality + '_valid'][occupied] > 0).all())

    def test_persistent_workers_follow_shared_epoch_not_global_rng(self):
        self.dataset.track='W'
        self.dataset.supervision_source='provenance_v1'
        wrapped = _DigestDataset(self.dataset)
        direct = [wrapped[i] for i in range(2)]
        np.random.seed(12345)
        np.random.rand(1000)
        loader = torch.utils.data.DataLoader(wrapped, batch_size=None, num_workers=2,
                                            persistent_workers=True, prefetch_factor=1)
        try:
            self.assertEqual(list(loader), direct)
            self.dataset.set_epoch(1)
            updated = [wrapped[i] for i in range(2)]
            self.assertNotEqual(updated, direct)
            self.assertEqual(list(loader), updated)
        finally:
            loader._iterator._shutdown_workers()

    def test_provenance_has_both_positive_types_and_no_teacher_or_label_inputs(self):
        self.dataset.track='W';self.dataset.supervision_source='provenance_v1'
        self.dataset.controlled_corruption['probability']=1
        with patch('opencood.data_utils.datasets.project_v2xr_dataset.weather_for',return_value='normal'):
            sample=self.dataset[0]
        self.assertNotIn('teacher_inputs',sample)
        for agent,target,meta in zip(sample['ego']['agents'],sample['train_targets'],sample['ego']['metadata']):
            self.assertEqual(set(agent),{'lidar','radar','maps','valid'})
            self.assertTrue(meta['controlled_corruption'])
            self.assertTrue(meta['supervision']['source_ids_verified'])
            self.assertGreater(meta['supervision']['positive_miss_cells'],0)
            self.assertGreater(meta['supervision']['positive_spur_cells'],0)
            self.assertEqual(target['normal_mask'].sum(),0)
            self.assertEqual(meta['lidar_points'],int(agent['maps']['lidar_count'].sum()))
        batch=self.dataset.collate_batch([sample])
        self.assertNotIn('teacher_inputs',batch)
        self.assertNotIn('source_ids',batch['ego'])

    def test_natural_fog_is_unknown_and_evaluation_is_never_perturbed(self):
        self.dataset.track='W';self.dataset.supervision_source='provenance_v1'
        self.dataset.controlled_corruption['probability']=1
        with patch('opencood.data_utils.datasets.project_v2xr_dataset.weather_for',return_value='fog'):
            sample=self.dataset[0]
        for target,meta in zip(sample['train_targets'],sample['ego']['metadata']):
            self.assertFalse(meta['controlled_corruption'])
            self.assertTrue(all(value.sum()==0 for value in target.values()))
        # The evaluation code path is exercised on train frames only.
        self.dataset.training=False;self.dataset.weather='normal'
        sample=self.dataset[0]
        self.assertIsNone(sample['train_targets'])
        self.assertTrue(all(not meta['controlled_corruption'] for meta in sample['ego']['metadata']))

    def test_training_snow_is_rejected_before_any_data_read(self):
        from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
        with self.assertRaisesRegex(ValueError,'Snow'):
            ProjectV2XRDataset(self.dataset.hypes,root=ROOT,training=True,weather='snow',
                track='W',supervision_source='provenance_v1')


if __name__ == '__main__':
    unittest.main()
