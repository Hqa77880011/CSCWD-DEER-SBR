"""Supplemental protocol boundaries. All point clouds and scores here are fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from opencood.project.config import validate_config
from opencood.project.supplement import defaults
from opencood.project.supplement_data import (build_real_manifest, real_points, kradar_label,
    kradar_calibration, SupplementalDataset, perturb_points)
from opencood.project.supplement_evaluation import evaluate_supplement, sequence_bootstrap
from opencood.project.supplement_reporting import summarize
from opencood.models.common_modules.supported_residual import SupportedResidual
from project_tools.run_experiments import read_hypes
from scripts.supplement import recipe, expand_suite


def pcd(path, rows, fields=('x', 'y', 'z', 'intensity')):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = '\n'.join(['VERSION .7', 'FIELDS ' + ' '.join(fields),
        'SIZE ' + ' '.join(['4'] * len(fields)), 'TYPE ' + ' '.join(['F'] * len(fields)),
        'COUNT ' + ' '.join(['1'] * len(fields)), 'WIDTH ' + str(len(rows)),
        'HEIGHT 1', 'POINTS ' + str(len(rows)), 'DATA ascii'])
    path.write_text(header + '\n' + '\n'.join(' '.join(map(str, row)) for row in rows) + '\n')


def cooperative_fixture(root, frames=3):
    import yaml
    for split, seq in (('train', 'train_a'), ('train', 'dev_b'), ('validate', 'final_c')):
        for i in range(frames):
            for agent, origin in (('-1', 0), ('7', 10)):
                base = root / split / seq / agent / f'{i:05d}'
                base.parent.mkdir(parents=True, exist_ok=True)
                annotation = dict(lidar_pose=[origin, 0, 0, 0, 0, 0], vehicles={
                    '10': dict(location=[20, 0, 0], angle=[0, 0, 0], extent=[2, 1, 1])})
                base.with_suffix('.yaml').write_text(yaml.safe_dump(annotation))
                pcd(base.with_suffix('.pcd'), [[15 - origin, 2, 0, 128], [16 - origin, 3, 0, 64]])
                pcd(base.with_name(base.name + '_radar.pcd'), [[15 - origin, 2, 0, 128, 100]],
                    fields=('x', 'y', 'z', 'dopple', 'intensity'))


def kradar_fixture(root):
    labels, radar = root / 'labels', root / 'radar'
    train, test = [], []
    for seq in ('1', '2', '3'):
        (root / seq / 'info_calib').mkdir(parents=True)
        (root / seq / 'info_calib/calib_radar_lidar.txt').write_text('frame offset,x,y\n0,-2.54,0.3\n')
        (root / seq / 'description.txt').write_text('urban,day,normal')
        for number in (1, 2):
            filename = f'{number:05d}_{number+1:05d}.txt'
            (labels / seq).mkdir(parents=True, exist_ok=True)
            header = f'* idx(rdr,ldr64,camf,ldr128,camr)={number:05d}_{number+1:05d}_00003_00004_00005, timestamp={number}.0\n'
            (labels / seq / filename).write_text(header +
                '*, 0, Sedan, 12.54, -0.3, 0.3, 90, 2.1, 1.05, 1\n' +
                '*, 1, Bus or Truck, 32.54, 3.7, 0.8, 0, 5, 1.3, 1.5\n')
            pcd(root / seq / 'os2-64' / f'os2-64_{number+1:05d}.pcd', [[12.54, -.3, .3, 20], [14, 1, .3, 10]])
            (radar / seq).mkdir(parents=True, exist_ok=True)
            np.save(radar / seq / f'sprdr_{number:05d}.npy', np.array([[10, 0, 1, 5, 2], [12, 1, 1, 6, -3]], np.float32))
            if number == 1:
                train.append(seq + ',' + filename)
            elif seq == '1':
                test.append(seq + ',' + filename)
    (root / 'train.txt').write_text('\n'.join(train))
    (root / 'test.txt').write_text('\n'.join(test))
    return labels, radar


class SupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_budgets_native_recipes_and_unresolved_F(self):
        for epochs, milestones in ((20, [10, 15]), (30, [15, 23]), (60, [30, 45]), (90, [45, 68])):
            self.assertEqual(recipe(epochs=epochs)['training']['milestones'], milestones)
        self.assertEqual(len(expand_suite('internal_main')), 18)
        external = expand_suite('external_main')
        self.assertEqual(len(external), 12)
        cobevt = next(c for c in external if c['experiment']['method'] == 'cobevt')
        self.assertEqual(cobevt['training']['epochs'], 40)
        self.assertEqual(cobevt['training']['scheduler'], 'cosine_warmup')
        self.assertEqual(cobevt['training']['effective_batch_size'], 2)
        self.assertEqual(cobevt['training']['micro_batch_size'], 2)
        self.assertEqual(cobevt['training']['accumulation_steps'], 1)
        self.assertEqual(len(expand_suite('budgets')), 16)
        with self.assertRaisesRegex(ValueError, 'F is unresolved'):
            recipe(variant='F')
        invalid = defaults()
        invalid['experiment']['variant'] = 'NO_NULL'
        with self.assertRaisesRegex(ValueError, 'actual amplitude'):
            validate_config(invalid)
        self.assertFalse(recipe(variant='NO_NULL')['model']['null_donor'])

    def test_all_suites_expand_and_F_has_one_frozen_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = dict(schema='supplement_frozen_final_v1', variant='N2', source_split='validate', seeds=[17, 29, 43],
                test_used=False, one_structure_for_all_seeds_and_weather=True,
                normal_tolerance_pp=1., development_means=dict(AB=.2, N2=.3),
                normal_development_means=dict(S0=.2, AB=.2, N2=.2),
                evidence={v: {str(s): dict(score=.3 if v == 'N2' else .2,
                              scores=dict(normal=.2, fog=.4 if v == 'N2' else .2), checkpoint_sha256='0' * 64)
                              for s in (17, 29, 43)} for v in ('S0', 'AB', 'N2')})
            path = Path(directory) / 'fixture_frozen.json'
            path.write_text(json.dumps(frozen))
            matrix = json.loads((Path(__file__).resolve().parents[1] / 'cscwd_configs/supplement/matrix.json').read_text())
            for name in matrix['suites']:
                configs = expand_suite(name, final_model_file=path)
                self.assertTrue(configs)
                self.assertEqual(len({c['experiment']['run_id'] for c in configs}), len(configs))
                for config in configs:
                    if config['experiment']['variant'] == 'F' and config['experiment']['method'] == 'deer_sbr':
                        self.assertEqual(config['model']['type_prior_mode'], 'multiplicative')
            from scripts.supplement import SUITE_ALIASES
            for alias, canonical in SUITE_ALIASES.items():
                self.assertEqual(expand_suite(alias, final_model_file=path),
                                 expand_suite(canonical, final_model_file=path))

    def test_legacy_runtime_keeps_original_scientific_identity(self):
        import hashlib
        import zipfile
        import sys
        from unittest.mock import patch
        from scripts.legacy_runtime import execute_original
        root = Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(root / 'repro/archive/legacy_v4_runtime.zip') as archive:
            paths = sorted(n for n in archive.namelist() if n.endswith('.py') and (n.startswith('opencood/') or n.startswith('project_tools/') and n.count('/') == 1))
            digest = hashlib.sha256()
            for name in paths:
                digest.update(name.encode())
                digest.update(hashlib.sha256(archive.read(name)).digest())
            self.assertEqual(digest.hexdigest(), json.loads((root / 'repro/pretrained_release.json').read_text())['source_sha256'])
            for variant in ('S0', 'A', 'B', 'AB', 'SUP', 'N1', 'N2'):
                self.assertEqual(archive.read('repro/configs/' + variant + '.yaml'), (root / 'repro/configs' / (variant + '.yaml')).read_bytes())
        for script, arguments, expected_output in (
            ('reproduce.py', ['results'], root / 'outputs/reproduction'),
            ('pretrained.py', ['verify', '--weights-dir=/chosen/weights'], root / 'outputs/pretrained_evaluation'),
        ):
            with patch.object(sys, 'argv', [script, *arguments]), patch.object(sys, 'path', list(sys.path)), patch('scripts.legacy_runtime.runpy.run_path') as run:
                execute_original(script)
                run.assert_called_once()
                self.assertEqual(sys.argv[sys.argv.index('--data-root') + 1], str(root / 'datasets/v2xr'))
                self.assertEqual(sys.argv[sys.argv.index('--output-root') + 1], str(expected_output))
                if script == 'pretrained.py':
                    self.assertIn('--weights-dir=/chosen/weights', sys.argv)
                    self.assertNotIn('--weights-dir', sys.argv)

    def test_field_semantics_and_vehicle_ego_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cooperative_fixture(root)
            manifest = build_real_manifest('v2xradar', root, ['dev_b'], clean_sequences=['train_a'])
            sample = manifest['samples']['train/train_a/00000']
            self.assertEqual(sample['ego_id'], '7')
            self.assertEqual(sample['agent_ids'], ['7', '-1'])
            np.testing.assert_allclose(sample['gt'][0]['box'][:3], [10, 0, 0])
            self.assertEqual(manifest['sequence_partitions']['validate'], ['dev_b'])
            self.assertEqual(manifest['final_identity'], 'official_validate_reserved_as_final')
            v = real_points(root / 'train/train_a/7/00000_radar.pcd', 'v2xradar', 'radar', '7')
            r = real_points(root / 'train/train_a/-1/00000_radar.pcd', 'v2xradar', 'radar', '-1')
            self.assertEqual(float(v[0, 3]), 1.)
            self.assertAlmostEqual(float(r[0, 3]), .5)
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('v2xradar', data_root=root)
            h = read_hypes(c)
            dataset = SupplementalDataset(c, h, 'test')
            item = dataset[0]
            # Identical world measurements from two actual agents coincide
            # after the single projection into vehicle ego coordinates.
            np.testing.assert_array_equal(item['agents'][0]['maps']['radar_count'], item['agents'][1]['maps']['radar_count'])
            c['evaluation'].update(radar_keep=0., kind='artificial_corruption')
            degraded = SupplementalDataset(c, h, 'test')
            self.assertEqual(dataset.manifest['samples'][dataset.sample_ids[0]]['gt'], degraded.manifest['samples'][degraded.sample_ids[0]]['gt'])
            self.assertEqual(degraded[0]['agents'][0]['maps']['radar_count'].sum(), 0)

    def test_async_history_is_independent_of_common_sample_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cooperative_fixture(root, frames=4)
            manifest = build_real_manifest('v2xradar', root, ['dev_b'])
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            common = ['validate/final_c/00002', 'validate/final_c/00003']
            sample_path = root / 'common.json'
            sample_path.write_text(json.dumps(common))
            c = recipe('v2xradar', data_root=root)
            c['evaluation'].update(delay_frames=1, sample_ids=str(sample_path))
            dataset = SupplementalDataset(c, read_hypes(c), 'test')
            self.assertEqual(dataset.sample_ids, common)
            self.assertEqual(dataset.delayed[common[0]], 'validate/final_c/00001')
            item = dataset[0]
            self.assertEqual(item['metadata'][0]['source_timestamp'], '00002')
            self.assertEqual(item['metadata'][1]['source_timestamp'], '00001')

    def test_kradar_sync_calibration_class_and_one_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, radar = kradar_fixture(root)
            calib = kradar_calibration(root / '1/info_calib/calib_radar_lidar.txt')
            idx, stamp, objects = kradar_label(labels / '1/00001_00002.txt', calib)
            self.assertEqual(idx['rdr'], '00001')
            self.assertEqual(idx['ldr64'], '00002')
            np.testing.assert_allclose(objects[0]['box'], [10, 0, 1, 4.2, 2.1, 2, np.pi / 2], atol=1e-6)
            manifest = build_real_manifest('kradar', root, ['2'], radar_root=radar,
                label_root=labels, train_split=root / 'train.txt', test_split=root / 'test.txt')
            self.assertEqual(manifest['sequence_partitions']['validate'], ['2'])
            self.assertIn('official_sequence_overlap_disclosed', manifest['final_identity'])
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('kradar', data_root=root, radar_root=radar, label_root=labels, epochs=30)
            dataset = SupplementalDataset(c, read_hypes(c), 'train', training=True)
            batch = dataset.collate_batch([dataset[0]])
            self.assertEqual(batch['ego']['record_len'].tolist(), [1])
            self.assertEqual(set(batch['ego']['label_dict']['class_labels']), {'Sedan', 'Bus or Truck'})
            self.assertEqual(float(real_points(radar / '1/sprdr_00001.npy', 'kradar', 'radar')[0, 3]), 5.)
            self.assertGreater(int(batch['ego']['label_dict']['pos_equal_one'].sum()), 0)
            # A real-format fixture through the full-size grid and both heads,
            # not a parser-only check. No optimizer or AP is run on this data.
            from opencood.project.supplement_models import SupplementalModel, SupplementalLoss
            from opencood.project.training import select_model_inputs
            hypes = read_hypes(c)
            model = SupplementalModel(c, hypes).train()
            prediction = model(select_model_inputs(batch['ego']))
            criterion = SupplementalLoss({**hypes['loss']['args'], 'training_line': 'C', 'efficient_features': True}, 'deer_sbr')
            loss = criterion(prediction, batch['ego']['label_dict'], batch['train_targets'])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_mdd_real_refuses_unknown_natural_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cooperative_fixture(root)
            manifest = build_real_manifest('v2xradar', root, ['dev_b'])
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('v2xradar', 'mdd', 'S0', epochs=30, data_root=root)
            with self.assertRaisesRegex(ValueError, 'no verified clear'):
                SupplementalDataset(c, read_hypes(c), 'train', training=True)

    def test_source_keyed_corruption_and_empty_modalities(self):
        points = np.column_stack((np.linspace(1, 7, 100), np.ones((100, 2)), np.full(100, .3))).astype(np.float32)
        c = defaults()
        c['evaluation'].update(kind='artificial_corruption', drop_fraction=.4, inject_fraction=.1, radar_keep=0.)
        kwargs = dict(sample_id='test/scene/1', agent='1', ego='1', config=c, bounds=[0, 0, 0, 10, 10, 10], voxel=[1, 1, 10])
        a = perturb_points(points, points, [True, True], **kwargs)
        c2 = deepcopy(c)
        c2['experiment']['variant'] = 'SUP'
        b = perturb_points(points, points, [True, True], **{**kwargs, 'config': c2})
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(len(a[1]), 0)
        self.assertGreater(float(a[3]['miss'].max()), 0)
        self.assertGreater(float(a[3]['spur'].max()), 0)
        c['evaluation']['missing_modality'] = 'both'
        empty = perturb_points(points, points, [True, True], **kwargs)
        self.assertEqual((len(empty[0]), len(empty[1])), (0, 0))
        self.assertFalse(empty[2].any())

    def test_no_null_donor_empty_support_remains_finite_and_zero(self):
        from tests.test_innovation_modules import observations
        block = SupportedResidual(4, 4, 2, null_donor=False)
        self.assertIsNone(block.null_logit)
        obs = observations(batch=1)
        obs['radar_occupancy'].zero_()
        z = torch.randn(1, 4, 8, 8, requires_grad=True)
        output, aux = block(z, torch.randn_like(z), obs, torch.ones(1, 2))
        torch.testing.assert_close(output, z)
        self.assertEqual(aux['delta'].count_nonzero(), 0)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_v5_online_resume_and_native_selected_roles_are_separate(self):
        from opencood.project.training import EpochBatchSampler, TrainState, save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            config = recipe(method='attfuse', variant='S0', epochs=30)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 23])
            sampler = EpochBatchSampler(8, 4, 2, 17)
            resume = Path(directory) / 'latest.pt'
            save_checkpoint(resume, model, optimizer, scheduler, None, TrainState(), sampler, config)
            load_checkpoint(resume, model, optimizer, scheduler, None, sampler, config)
            for role in ('selected', 'budget_snapshot'):
                artifact = Path(directory) / (role + '.pt')
                save_checkpoint(artifact, model, optimizer, scheduler, None, TrainState(), sampler, config,
                    dict(checkpoint_role=role, bn_statistics_role='native_running_frozen'))
                with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'resume'):
                    load_checkpoint(artifact, model, optimizer, scheduler, None, sampler, config)

    def test_true3d_extension_does_not_relabel_bev(self):
        manifest = dict(dataset='v2xradar', classes=['Vehicle'], bounds=[-100, -100, -10, 100, 100, 10],
            final_identity='fixture_only', protocol_sha256='fixture', samples={
                's': dict(split='test', sequence='seq', gt=[dict(class_name='Vehicle', box=[10, 0, 0, 4, 2, 1, 0])])})
        records = [dict(sample_id='s', predictions=[dict(prediction_id='p', class_name='Vehicle', score=.9, box=[10, 0, 4, 4, 2, 1, 0])])]
        result = evaluate_supplement(manifest, records, 'test', defaults()['scoring'])
        self.assertEqual(result['metrics']['bev']['0.7']['ap'], 1.)
        self.assertEqual(result['metrics']['3d']['0.7']['ap'], 0.)

    def test_official_kradar_AP41_known_answer(self):
        samples, records = {}, []
        for i in range(41):
            gt = [dict(class_name=name, box=[10 + k * 20, 0, 1, 4 + k * 6, 2, 2, .3])
                  for k, name in enumerate(('Sedan', 'Bus or Truck'))]
            sid = str(i)
            samples[sid] = dict(split='test', sequence=str(i // 10), gt=gt)
            records.append(dict(sample_id=sid, predictions=[dict(g, prediction_id=str(k), score=.9) for k, g in enumerate(gt)]))
        manifest = dict(dataset='kradar', classes=['Sedan', 'Bus or Truck'], bounds=[0, -15, -2, 72, 15, 7.6],
            final_identity='fixture_only', protocol_sha256='fixture', samples=samples)
        result = evaluate_supplement(manifest, records, 'test', defaults()['scoring'])
        self.assertAlmostEqual(result['metrics']['3d']['0.7']['ap'], 1.)
        self.assertAlmostEqual(result['per_class']['Bus or Truck']['bev']['0.5']['ap'], 1.)

    def test_sequence_bootstrap_recomputes_detection_AP(self):
        manifest = dict(dataset='v2xradar', classes=['Vehicle'], bounds=[-100, -100, -10, 100, 100, 10],
            final_identity='fixture_only', protocol_sha256='fixture', samples={
                s: dict(split='test', sequence=s, gt=[dict(class_name='Vehicle', box=[10, 0, 0, 4, 2, 1, 0])]) for s in ('a', 'b')})
        empty = [dict(sample_id=s, predictions=[]) for s in ('a', 'b')]
        perfect = [dict(sample_id=s, predictions=[dict(prediction_id='p', class_name='Vehicle', score=.9, box=[10, 0, 0, 4, 2, 1, 0])]) for s in ('a', 'b')]
        result = sequence_bootstrap(manifest, empty, perfect, 'test', defaults()['scoring'], repetitions=3)
        self.assertEqual(result['paired_deltas_pp'], [100., 100., 100.])
        self.assertTrue(result['ap_recomputed_per_draw'])

    def test_paired_seed_summary_sample_std_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            for method, values in (('S0', [.1, .1, .1]), ('AB', [.2, .4, .3])):
                for seed, ap in zip((17, 29, 43), values):
                    m = dict(dataset='fixture', protocol='fixture', transfer='retrained', condition=dict(sample_ids=None),
                        weather='normal', split='test', method='deer_sbr', variant=method, seed=seed, recipe='e60', sample_ids=['a', 'b'],
                        protocol_sha256='fixture', gt_manifest_sha256='fixture',
                        checkpoint_sha256=f'fixture_{method}_{seed}', checkpoint_role='selected')
                    path = Path(directory) / f'{method}_{seed}.scores.json'
                    path.write_text(json.dumps(dict(schema='supplement_evaluation_v1', metadata=m,
                        result=dict(metrics={'3d': {'0.7': dict(ap=ap)}}))))
            rows = summarize([directory], 'deer_sbr/S0')['rows']
            ab = next(r for r in rows if r['method'] == 'deer_sbr/AB')
            self.assertAlmostEqual(ab['mean_ap'], .3)
            self.assertAlmostEqual(ab['sample_std_ap'], .1)
            self.assertAlmostEqual(ab['paired_mean_delta_pp'], 20.)
            self.assertTrue(ab['complete_three_seeds'])

    def test_weather_mean_std_is_computed_after_seed_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            for weather, values in (('fog', [.1, .3, .5]), ('snow', [.5, .3, .1])):
                for seed, ap in zip((17, 29, 43), values):
                    metadata = dict(dataset='v2xr', protocol='fixture', transfer='retrained', condition={},
                        weather=weather, split='test', method='deer_sbr', variant='AB', seed=seed, recipe='e60', sample_ids=['a'],
                        protocol_sha256='fixture', gt_manifest_sha256='fixture',
                        checkpoint_sha256=f'fixture_AB_{seed}', checkpoint_role='selected')
                    path = Path(directory) / f'{weather}_{seed}.scores.json'
                    path.write_text(json.dumps(dict(schema='supplement_evaluation_v1', metadata=metadata,
                        result=dict(metrics={'3d': {'0.7': dict(ap=ap)}}))))
            rows = summarize([directory], 'deer_sbr/S0')['rows']
            mean = next(r for r in rows if r['setting']['weather'] == 'fog_snow_mean')
            self.assertAlmostEqual(mean['mean_ap'], .3)
            self.assertAlmostEqual(mean['sample_std_ap'], 0.)


if __name__ == '__main__':
    unittest.main()
