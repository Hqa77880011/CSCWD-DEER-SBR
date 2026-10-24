"""CPU synthetic regressions for transfer, input and execution boundaries."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from opencood.project.config import config_hash, file_sha256
from opencood.project.supplement_data import build_real_manifest, SupplementalDataset
from opencood.project.supplement_reporting import load_selected
from project_tools import run_experiments as runner
from scripts.supplement import recipe
from tests.test_supplement import cooperative_fixture, kradar_fixture
from tests.test_supplement_performance import ToyDataset, ToyLoss, ToyModel


class SupplementBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def _source(self, root, value=1):
        model = ToyModel()
        with torch.no_grad():
            model.layers[-1].bias.fill_(value)
        path = root / f'source_{value}.pt'
        torch.save(dict(schema_version=5, training_protocol='supplement_v1',
                        model=model.state_dict(), metadata={'checkpoint_role': 'selected'}), path)
        return path

    def _fine_config(self, root, source):
        c = recipe(output_root=root, device='cpu')
        c['experiment'].update(transfer='fine_tuned', initialization_checkpoint=str(source),
                               initialization_checkpoint_sha256=file_sha256(source))
        return c

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_transfer_acceptance_source_binding_and_zero_shot(self):
        from opencood.project.supplement import initialization_metadata, shared_initialization_path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            for transfer in ('retrained', 'fine_tuned'):
                c = self._fine_config(root, source) if transfer == 'fine_tuned' else recipe(output_root=root)
                c['training']['batchnorm'].update(evaluation_statistics='running', refit_samples=0)
                model = torch.nn.Linear(1, 1)
                metadata = dict(checkpoint_role='selected', source_sha256='fixture',
                    bn_statistics_role='native_running_frozen', **initialization_metadata(c))
                saved = dict(schema_version=5, config=c, config_sha256=config_hash(c),
                             model=model.state_dict(), metadata=metadata)
                path = root / 'selected.pt'
                torch.save(saved, path)
                with patch.object(runner, 'source_tree_hash', return_value='fixture'), \
                     patch.object(runner, 'data_contract_metadata', return_value={}), \
                     patch.object(runner, 'build_model', return_value=model):
                    load_selected(c, path, torch.device('cpu'))
                    wrong = (self._fine_config(root, source) if transfer == 'retrained'
                             else recipe(output_root=root))
                    wrong['training'] = deepcopy(c['training'])
                    with self.assertRaisesRegex(ValueError, 'transfer'):
                        load_selected(wrong, path, torch.device('cpu'))
                    zero = deepcopy(c)
                    zero['experiment'].update(transfer='zero_shot', dataset_id='v2xradar',
                        initialization_checkpoint=None, initialization_checkpoint_sha256=None)
                    load_selected(zero, path, torch.device('cpu'))
                    if transfer == 'fine_tuned':
                        other = deepcopy(c)
                        other['experiment']['initialization_checkpoint_sha256'] = '0' * 64
                        with self.assertRaisesRegex(ValueError, 'initialization'):
                            load_selected(other, path, torch.device('cpu'))
                        source.rename(root / 'source_absent.pt')
                        # Inference uses the recorded digest, without the training source file.
                        load_selected(c, path, torch.device('cpu'))
                paths = set()
                for variant in ('S0', 'AB', 'N2'):
                    paired = deepcopy(c)
                    paired['experiment']['variant'] = variant
                    paths.add(shared_initialization_path(paired))
                self.assertEqual(len(paths), 1)

    def test_fine_source_change_missing_completion_and_resume_reject(self):
        from opencood.project.training import (EpochBatchSampler, TrainState,
                                               save_checkpoint, load_checkpoint)
        from scripts.supplement import completed_run_matches
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            c = self._fine_config(root, source)
            with patch.object(runner, 'data_contract_metadata', return_value={}), \
                 patch.object(runner, 'supervision_contract', return_value={}), \
                 patch.object(runner, 'source_tree_hash', return_value='fixture'):
                metadata = runner.artifact_metadata(c)
                self.assertEqual(metadata['initialization_checkpoint_sha256'], file_sha256(source))
                model = ToyModel()
                optimizer = torch.optim.Adam(model.parameters())
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10])
                sampler = EpochBatchSampler(32, 4, 2, 17)
                resume = root / 'online.pt'
                save_checkpoint(resume, model, optimizer, scheduler, None, TrainState(), sampler, c, metadata)
                load_checkpoint(resume, model, optimizer, scheduler, None, sampler, c, metadata)
                (root / 'report.rank0.json').write_text(json.dumps(dict(passed=True,
                    source_unchanged=True, status='checkpoint_selected')))
                (root / 'resolved_config.json').write_text(json.dumps(c))
                (root / 'run_metadata.json').write_text(json.dumps(metadata))
                selected = dict(schema_version=5, training_protocol='supplement_v1', config=c,
                    config_sha256=config_hash(c), metadata={**metadata, 'checkpoint_role': 'selected',
                    'bn_statistics_role': 'train_refit_cma_v1'})
                torch.save(selected, root / 'selected.pt')
                self.assertTrue(completed_run_matches(c, root))
                original = source.read_bytes()
                source.write_bytes(original + b'changed source')
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    runner.artifact_metadata(c)
                self.assertFalse(completed_run_matches(c, root))
                changed = deepcopy(c)
                changed['experiment']['initialization_checkpoint_sha256'] = file_sha256(source)
                current_metadata = runner.artifact_metadata(changed)
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    load_checkpoint(resume, model, optimizer, scheduler, None, sampler, changed, current_metadata)
                source.rename(root / 'missing.pt')
                with self.assertRaises(FileNotFoundError):
                    runner.artifact_metadata(c)
                self.assertFalse(completed_run_matches(c, root))

    def test_configuration_binds_source_and_shared_initialization_isolates_each_digest(self):
        from opencood.project.config import validate_config, load_config
        from opencood.project.supplement import shared_initialization_path
        from scripts.supplement import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [self._source(root, value) for value in (1, 2)]
            configurations = [recipe(output_root=root)] + [recipe(output_root=root,
                transfer='fine_tuned', initialize_from=path) for path in sources]
            self.assertEqual(len({shared_initialization_path(c) for c in configurations}), 3)
            self.assertEqual(len({c['experiment']['run_id'] for c in configurations}), 3)
            broken = recipe()
            broken['experiment']['transfer'] = 'fine_tuned'
            with self.assertRaisesRegex(ValueError, 'fine_tuned'):
                validate_config(broken)
            output = io.StringIO()
            with redirect_stdout(output):
                main(['configure', '--output-root', str(root), '--transfer', 'fine_tuned',
                      '--initialize-from', str(sources[0])])
            generated = load_config(Path(output.getvalue().strip()))
            self.assertEqual(generated['experiment']['initialization_checkpoint_sha256'], file_sha256(sources[0]))
            self.assertEqual(generated['experiment']['transfer'], 'fine_tuned')

    def test_real_inputs_participate_in_resume_and_completion(self):
        from opencood.project.training import EpochBatchSampler, TrainState, save_checkpoint
        from scripts.supplement import completed_run_matches
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cooperative_fixture(root, frames=1)
            manifest = build_real_manifest('v2xradar', root, ['dev_b'])
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('v2xradar', 'attfuse', 'S0', data_root=root, output_root=root, device='cpu')
            c['resources'].update(num_workers=0, cpu_threads=2)
            with patch.object(runner, 'source_tree_hash', return_value='fixture'):
                metadata = runner.artifact_metadata(c)
                (root / 'report.rank0.json').write_text(json.dumps(dict(passed=True,
                    source_unchanged=True, status='checkpoint_selected')))
                (root / 'resolved_config.json').write_text(json.dumps(c))
                (root / 'run_metadata.json').write_text(json.dumps(metadata))
                torch.save(dict(schema_version=5, training_protocol='supplement_v1', config=c,
                    config_sha256=config_hash(c), metadata={**metadata, 'checkpoint_role': 'selected',
                    'bn_statistics_role': 'native_running_frozen'}), root / 'selected.pt')
                self.assertTrue(completed_run_matches(c, root))
                model = ToyModel()
                optimizer = torch.optim.Adam(model.parameters())
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10])
                resume = root / 'online.pt'
                save_checkpoint(resume, model, optimizer, scheduler, None, TrainState(),
                                EpochBatchSampler(32, 4, 2, 17), c, metadata)
                path = root / next(iter(manifest['samples'].values()))['agents']['7']['lidar']
                path.write_bytes(path.read_bytes() + b'\n')
                self.assertFalse(completed_run_matches(c, root))
                args = Namespace(mode='benchmark', allow_formal_run=False, steps=1, warmup=0,
                                 resume=resume, initialize_from=None)
                with patch.object(runner, 'formal_blockers', return_value=[]), \
                     patch.object(runner, 'build_components', side_effect=lambda *_: (ToyDataset(), ToyModel(), ToyLoss())):
                    with self.assertRaisesRegex(ValueError, 'sensor source SHA256'):
                        runner.run(c, args)

    def test_real_manifest_missing_sensor_evidence_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cooperative_fixture(root, frames=1)
            manifest = build_real_manifest('v2xradar', root, ['dev_b'])
            for sample in manifest['samples'].values():
                for record in sample['agents'].values():
                    record.pop('sensor_sha256')
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('v2xradar', data_root=root)
            with self.assertRaisesRegex(ValueError, 'lacks sensor SHA256'):
                SupplementalDataset(c, runner.read_hypes(c), 'train', training=True)
            with self.assertRaisesRegex(ValueError, 'lacks sensor SHA256'):
                runner.data_contract_metadata(c)

    def test_pointnet_source_binary_and_runtime_evidence_are_separate(self):
        from opencood.project.l4dr import pointnet2_identity
        from opencood.project import config as config_module
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'synthetic_extension.bin'
            binary.write_bytes(b'not executable, identity fixture only')
            wrapper = SimpleNamespace(pointnet2=SimpleNamespace(__file__=str(binary)))
            with patch('importlib.import_module', return_value=wrapper):
                first = pointnet2_identity()
                binary.write_bytes(b'changed synthetic binary')
                second = pointnet2_identity()
                self.assertEqual(first['source_sha256'], second['source_sha256'])
                self.assertNotEqual(first['binary']['sha256'], second['binary']['sha256'])
                def changed_cuda(path):
                    return '0' * 64 if Path(path).suffix == '.cu' else file_sha256(path)
                with patch.object(config_module, 'file_sha256', side_effect=changed_cuda):
                    third = pointnet2_identity()
                self.assertNotEqual(second['source_sha256'], third['source_sha256'])
                self.assertEqual(second['binary'], third['binary'])
            with patch('importlib.import_module', side_effect=ImportError('CPU fixture')):
                missing = pointnet2_identity()
            self.assertEqual(missing['binary'], dict(status='unavailable', sha256=None))
            self.assertEqual(missing['operator_numerics'], 'not_verified_by_identity')
            self.assertIn('torch_cuda', missing['runtime'])

    def test_l4dr_evaluation_rejects_changed_extension_identity(self):
        from opencood.project.supplement import initialization_metadata
        from opencood.project import l4dr
        with tempfile.TemporaryDirectory() as directory:
            c = recipe(method='l4dr', variant='S0')
            model = torch.nn.Linear(1, 1)
            identity = l4dr.pointnet2_identity()
            saved = dict(schema_version=5, config=c, config_sha256=config_hash(c), model=model.state_dict(),
                metadata=dict(checkpoint_role='selected', source_sha256='fixture',
                    bn_statistics_role='native_running_frozen', l4dr_pointnet2=identity, **initialization_metadata(c)))
            path = Path(directory) / 'selected.pt'
            torch.save(saved, path)
            with patch.object(runner, 'source_tree_hash', return_value='fixture'), \
                 patch.object(runner, 'data_contract_metadata', return_value={}), \
                 patch.object(runner, 'build_model', return_value=model):
                load_selected(c, path, torch.device('cpu'))
                for key in ('binary', 'source_sha256', 'runtime'):
                    changed = deepcopy(identity)
                    changed[key] = 'synthetic changed evidence'
                    with patch.object(l4dr, 'pointnet2_identity', return_value=changed):
                        with self.assertRaisesRegex(ValueError, 'extension'):
                            load_selected(c, path, torch.device('cpu'))

    def test_export_uses_method_configuration_before_transfer(self):
        from project_tools.export_predictions import export_loader
        from tests.test_validation_fastpath import manifest_for
        c = recipe(method='l4dr', variant='S0')
        class Detector(torch.nn.Module):
            def forward(self, inputs, return_aux=False):
                self.keys_seen = set(inputs)
                return dict(psm=torch.zeros(1, 1, 1, 1), rm=torch.zeros(1, 7, 1, 1))
        model = Detector()
        dataset = SimpleNamespace(experiment_config=c, weather='normal',
            manifest=manifest_for({'a': []}), anchor_box=torch.tensor([[[[10., 0., 1., 2., 2., 4., 0.]]]]))
        ego = dict(sample_ids=['a'], metadata=[[dict(agent_id='1', weather='normal')]],
            processed_lidar={}, processed_radar={}, raw_radar=[torch.zeros(0, 4)],
            record_len=torch.ones(1, dtype=torch.long), observation_maps={'unused': torch.ones(1)},
            modality_valid=torch.ones(1, 2), pairwise_t_matrix=torch.eye(4))
        records = export_loader(model, [dict(ego=ego)], dataset, torch.device('cpu'), postprocess=False)
        self.assertEqual(model.keys_seen, {'processed_lidar', 'raw_radar', 'record_len'})
        self.assertEqual(records[0]['sample_id'], 'a')

    def test_external_dataset_skips_unused_maps_but_keeps_diagnostics_and_reference(self):
        from opencood.project.training import select_model_inputs, rng_state, restore_rng
        for dataset_id in ('v2xradar', 'kradar'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if dataset_id == 'kradar':
                    labels, radar = kradar_fixture(root)
                    options = dict(radar_root=radar, label_root=labels,
                                   train_split=root / 'train.txt', test_split=root / 'test.txt')
                    dev = ['2']
                else:
                    cooperative_fixture(root, frames=1)
                    options, dev = {}, ['dev_b']
                manifest = build_real_manifest(dataset_id, root, dev, **options)
                # All files are synthetic; mark only these known fixtures clear.
                for sample in manifest['samples'].values():
                    sample['verified_clear'] = True
                (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
                methods = ('attfuse', 'mdd', 'l4dr') if dataset_id == 'kradar' else ('attfuse', 'mdd', 'l4dr', 'cobevt', 'where2comm')
                for method in methods:
                    c = recipe(dataset_id, method, 'S0', data_root=root,
                               radar_root=options.get('radar_root'), label_root=options.get('label_root'))
                    for training in (False, True):
                        with self.subTest(dataset=dataset_id, method=method, training=training):
                            a = SupplementalDataset(c, runner.read_hypes(c), 'train', training=training, augment=training)
                            b = SupplementalDataset(c, runner.read_hypes(c), 'train', training=training, augment=training)
                            # Re-enable discarded producers as the reference path.
                            b.observation_maps_required = b.auxiliary_targets_required = True
                            for epoch in (0, 1, 0):
                                a.set_epoch(epoch)
                                b.set_epoch(epoch)
                                initial = rng_state()
                                optimized = a[0]
                                after_a = rng_state()
                                restore_rng(initial)
                                reference = b[0]
                                self.assert_nested_equal(after_a, rng_state())
                                for key in ('metadata', 'target_diagnostics', 'labels', 'refs'):
                                    self.assert_nested_equal(optimized[key], reference[key])
                                if training:
                                    self.assert_nested_equal(optimized['targets'], reference['targets'])
                                else:
                                    self.assertEqual(optimized['targets'], [{} for _ in optimized['agents']])
                                self.assertTrue(all(not x['maps'] for x in optimized['agents']))
                                self.assertTrue(all(x['maps'] for x in reference['agents']))
                                if method == 'l4dr':
                                    self.assertTrue(all('radar' not in x for x in optimized['agents']))
                                    for x in reference['agents']:
                                        x['radar'] = b.pre_processor.preprocess(x['raw_radar'])
                                batch_a, batch_b = a.collate_batch([optimized]), b.collate_batch([reference])
                                self.assert_nested_equal(select_model_inputs(batch_a['ego'], c),
                                                         select_model_inputs(batch_b['ego'], c))
                                if method == 'mdd' and training:
                                    self.assert_nested_equal(batch_a['ego']['reference_lidar'], batch_b['ego']['reference_lidar'])
                                if method == 'l4dr' and training:
                                    self.assert_nested_equal(batch_a['train_targets']['fad_boxes'], batch_b['train_targets']['fad_boxes'])
                            # The visualization consumer can explicitly request maps.
                            a.observation_maps_required = True
                            self.assertTrue(a[0]['agents'][0]['maps'])

    def test_method_input_projection_preserves_outputs_losses_gradients_bn_and_rng(self):
        from opencood.project.supplement_models import SupplementalModel, SupplementalLoss
        from opencood.project.training import select_model_inputs, select_loss_targets, rng_state, restore_rng
        from project_tools.smoke_supplement import synthetic_inputs, synthetic_labels

        def detached(value):
            if torch.is_tensor(value):
                return value.detach().clone()
            if isinstance(value, dict):
                return {k: detached(v) for k, v in value.items()}
            return value

        for method, variant in [(m, 'S0') for m in ('attfuse', 'mdd', 'l4dr', 'cobevt', 'where2comm')] + [('where2comm', 'AB')]:
            with self.subTest(method=method, variant=variant):
                c = recipe(method=method, variant=variant)
                h = runner.read_hypes(c)
                h['model']['args']['point_pillar_scatter']['grid_size'] = [32, 32, 1]
                model = SupplementalModel(c, h).train()
                inputs = synthetic_inputs(h, single_agent=True)
                if method == 'mdd':
                    inputs['processed_de_lidar'] = deepcopy(inputs['processed_lidar'])
                if method == 'l4dr':
                    raw = inputs['processed_radar']['voxel_features']
                    inputs['raw_radar'] = [raw[:, :2].reshape(-1, 4)]
                targets = {k: torch.zeros_like(inputs['observation_maps']['lidar_count'])
                           for k in ('miss', 'spur', 'deg_valid')}
                targets.update({k + '_valid': inputs['observation_maps']['lidar_occupancy'].clone() for k in ('miss', 'spur')})
                targets['normal_mask'] = torch.ones(1, 1, 1, 1)
                if method == 'l4dr':
                    targets['fad_boxes'] = [torch.tensor([[0., 0., 0., 4., 2., 2., 0.]])]
                initial_weights, initial_rng = deepcopy(model.state_dict()), rng_state()
                evidence = []
                for filtered in (False, True):
                    model.load_state_dict(initial_weights)
                    model.zero_grad(set_to_none=True)
                    restore_rng(initial_rng)
                    projected = select_model_inputs(inputs, c) if filtered else dict(inputs)
                    if method == 'mdd':
                        projected['processed_de_lidar'] = inputs['processed_de_lidar']
                    criterion = SupplementalLoss({**h['loss']['args'], 'training_line': c['experiment']['track'],
                        'lambda_a': c['loss']['type_weight'], 'lambda_correction': c['loss']['correction_weight'],
                        'lambda_id': c['loss']['identity_weight'], 'efficient_features': c['model']['efficient_features']}, method)
                    output = model(projected)
                    loss = criterion(output, synthetic_labels(output), select_loss_targets(c, targets))
                    loss.backward()
                    evidence.append((detached(output), detached(criterion.loss_dict), loss.detach().clone(),
                        {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None},
                        detached(dict(model.named_buffers())), rng_state()))
                self.assert_nested_equal(evidence[0], evidence[1])

    def test_retrained_label_cannot_load_fine_tuned_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self._fine_config(root, self._source(root))
            original['training']['batchnorm'].update(evaluation_statistics='running', refit_samples=0)
            model = torch.nn.Linear(1, 1)
            saved = dict(schema_version=5, config=original, config_sha256=config_hash(original),
                model=model.state_dict(), metadata=dict(checkpoint_role='selected', source_sha256='fixture',
                    training_transfer='fine_tuned', initialization_checkpoint_sha256=original['experiment']['initialization_checkpoint_sha256'],
                    bn_statistics_role='native_running_frozen'))
            path = root / 'selected.pt'
            torch.save(saved, path)
            current = deepcopy(original)
            current['experiment'].update(transfer='retrained', initialization_checkpoint=None,
                                         initialization_checkpoint_sha256=None)
            with patch.object(runner, 'source_tree_hash', return_value='fixture'), \
                 patch.object(runner, 'data_contract_metadata', return_value={}), \
                 patch.object(runner, 'build_model', return_value=model):
                with self.assertRaisesRegex(ValueError, 'transfer'):
                    load_selected(current, path, torch.device('cpu'))

    def test_retrained_and_fine_tuned_initializations_do_not_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            retrained = recipe(output_root=root, device='cpu')
            fine = self._fine_config(root, source)
            fine['experiment']['run_id'] += '_fine'
            for c, initialize in ((retrained, None), (fine, source)):
                c['resources'].update(num_workers=0, cpu_threads=2, raw_cache_gib_per_worker=0.)
                args = Namespace(mode='benchmark', allow_formal_run=False, steps=1, warmup=0,
                                 resume=None, initialize_from=initialize, timing_mode='throughput',
                                 diagnose_validation_samples=0)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(runner, 'build_components',
                        side_effect=lambda *_: (ToyDataset(), ToyModel(), ToyLoss())))
                    stack.enter_context(patch.object(runner, 'formal_blockers', return_value=[]))
                    stack.enter_context(patch.object(runner, 'artifact_metadata', return_value={'source_sha256': 'fixture'}))
                    stack.enter_context(patch.object(runner, 'amp_audit_contract', return_value={'source_sha256': 'fixture'}))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    runner.run(c, args)
            self.assertEqual(len(list((root / 'initializations').glob('*.pt'))), 2)

    def test_real_sensor_replacement_is_rejected_with_and_without_cache(self):
        for dataset_id in ('v2xradar', 'kradar'):
            with self.subTest(dataset=dataset_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if dataset_id == 'kradar':
                    labels, radar = kradar_fixture(root)
                    options = dict(radar_root=radar, label_root=labels,
                                   train_split=root / 'train.txt', test_split=root / 'test.txt')
                    dev = ['2']
                else:
                    cooperative_fixture(root, frames=1)
                    options, dev = {}, ['dev_b']
                manifest = build_real_manifest(dataset_id, root, dev, **options)
                (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
                c = recipe(dataset_id, data_root=root, radar_root=options.get('radar_root'),
                           label_root=options.get('label_root'))
                for budget in (0., .001):
                    c['resources']['raw_cache_gib_per_worker'] = budget
                    dataset = SupplementalDataset(c, runner.read_hypes(c), 'train', training=True)
                    record = {**manifest['samples'][dataset.sample_ids[0]]['agents'][
                        manifest['samples'][dataset.sample_ids[0]]['ego_id']], 'agent_id': manifest['samples'][dataset.sample_ids[0]]['ego_id']}
                    for modality in ('lidar', 'radar'):
                        base = options['radar_root'] if dataset_id == 'kradar' and modality == 'radar' else root
                        path = base / record[modality]
                        payload = path.read_bytes()
                        self.assert_nested_equal(dataset._points(record, modality), dataset._points(record, modality))
                        stat = path.stat()
                        if path.suffix == '.npy':
                            values = np.load(path)
                            values[:, 3] += 1
                            np.save(path, values)
                        else:
                            path.write_bytes(payload[:-2] + (b'2' if payload[-2:-1] == b'1' else b'1') + payload[-1:])
                        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                        self.assertEqual(path.stat().st_size, stat.st_size)
                        try:
                            with self.assertRaisesRegex(ValueError, 'SHA256'):
                                dataset._points(record, modality)
                            with self.assertRaisesRegex(ValueError, 'SHA256'):
                                runner.data_contract_metadata(c)
                        finally:
                            path.write_bytes(payload)


if __name__ == '__main__':
    unittest.main()
