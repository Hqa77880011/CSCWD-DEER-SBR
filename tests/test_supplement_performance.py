"""Synthetic regressions for execution-only supplement optimizations."""
from copy import deepcopy
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from opencood.project.config import validate_config
from scripts.supplement import recipe


class ToyDataset(torch.utils.data.Dataset):
    """Sample-keyed data for exercising the real runner without dataset files."""
    def __init__(self):
        self.epoch = 0
    def __len__(self):
        return 32
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __getitem__(self, index):
        return index, torch.tensor([index / 32., self.epoch / 20., .5])
    def collate_batch(self, rows):
        return {'ego': {'processed_lidar': torch.stack([r[1] for r in rows]),
                        'record_len': torch.ones(len(rows), dtype=torch.long),
                        'sample_ids': [str(r[0]) for r in rows], 'label_dict': {}}}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                          torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    def forward(self, inputs):
        return self.layers(inputs['processed_lidar'])


class ToyLoss(torch.nn.Module):
    def forward(self, prediction, labels, targets):
        loss = prediction.square().mean()
        self.loss_dict = {'total_loss': loss.detach(), 'aux': loss.detach() * .1}
        return loss


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for one, two in zip(a, b):
                self.assert_nested_equal(one, two)
        else:
            self.assertEqual(a, b)

    def test_gradient_check_names_nonfinite_without_mutation(self):
        from opencood.project.training import nonfinite_gradients
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        model(torch.ones(2, 3)).sum().backward()
        before = [p.grad.clone() for p in model.parameters()]
        self.assertEqual(nonfinite_gradients(model), [])
        for old, parameter in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, parameter.grad))
        model[0].weight.grad[0, 0] = float('nan')
        model[1].bias.grad[0] = float('inf')
        self.assertEqual(nonfinite_gradients(model), ['0.weight', '1.bias'])
        model.zero_grad(set_to_none=True)
        self.assertEqual(nonfinite_gradients(model), [])

    def test_loss_logging_matches_python_double_accumulation(self):
        from opencood.project.training import loss_scalars
        rows = [dict(loss=torch.tensor(.1234567, requires_grad=True),
                     conf=torch.tensor(.25), count=torch.tensor(123456789)),
                dict(loss=torch.tensor(.7654321, requires_grad=True),
                     conf=torch.tensor(.5), count=torch.tensor(123456790))]
        expected = {k: sum(float(row[k].detach()) for row in rows) / len(rows) for k in rows[0]}
        self.assertEqual(loss_scalars(rows), expected)
        self.assertTrue(all(row['loss'].grad is None for row in rows))

    def test_resource_validation_accounts_for_all_live_loader_caches(self):
        c = recipe()
        c['resources'].update(num_workers=8, raw_cache_gib_per_worker=1., ram_budget_gib=32)
        with self.assertRaisesRegex(ValueError, 'cache|RAM'):
            validate_config(c)
        for key, value in [('num_workers', -1), ('prefetch_factor', 0), ('cpu_threads', 0)]:
            c = recipe()
            c['resources'][key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_config(c)

    def test_candidates_only_change_execution_settings(self):
        from scripts.supplement import performance_configs
        c = recipe()
        for stage, count in [('workers', 2), ('prefetch', 2), ('cache', 2),
                             ('cpu', 2), ('activation', 2), ('checkpoint', 3), ('workers12', 1)]:
            rows = performance_configs(c, stage, 'outputs/performance-test')
            self.assertEqual(len(rows), count)
            for row in rows:
                a, b = deepcopy(c), deepcopy(row)
                for value in (a, b):
                    value.pop('resources')
                    value['experiment'].pop('run_id')
                    value['artifacts'].pop('output_root')
                    value['model'].pop('activation_checkpoint')
                    value['training'].pop('checkpoint_interval_updates')
                self.assertEqual(a, b)
                self.assertEqual(row['training']['precision'], 'fp32')
                self.assertEqual(row['training']['micro_batch_size'], 4)
                self.assertEqual(row['training']['accumulation_steps'], 2)

    def test_raw_cache_preserves_calibration_augmentation_sources_and_fixed_gt(self):
        from test_supplement import kradar_fixture
        from opencood.project.supplement_data import build_real_manifest, SupplementalDataset
        from project_tools.run_experiments import read_hypes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, radar = kradar_fixture(root)
            manifest = build_real_manifest('kradar', root, ['2'], radar_root=radar,
                label_root=labels, train_split=root / 'train.txt', test_split=root / 'test.txt')
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('kradar', epochs=30, data_root=root, radar_root=radar, label_root=labels)
            c['resources']['raw_cache_gib_per_worker'] = 0.
            uncached = SupplementalDataset(c, read_hypes(c), 'train', training=True, augment=True)
            c['resources']['raw_cache_gib_per_worker'] = .001
            cached = SupplementalDataset(c, read_hypes(c), 'train', training=True, augment=True)
            cached.cache.profile = True
            for epoch in (0, 1, 0):
                cached.set_epoch(epoch)
                uncached.set_epoch(epoch)
                for index in range(len(cached)):
                    a, b = cached[index], uncached[index]
                    timing = a.pop('data_timing')
                    self.assertGreaterEqual(timing['preprocessing_seconds'], 0)
                    self.assert_nested_equal(a, b)
            self.assertGreater(cached.cache.hits, 0)
            self.assertEqual(cached.cache.misses, 2 * len(cached))
            self.assertLessEqual(cached.cache.bytes, cached.cache.max_bytes)
            self.assertEqual(cached.manifest, uncached.manifest)
            record = {**manifest['samples'][cached.sample_ids[0]]['agents']['0'], 'agent_id': '0'}
            old = cached._points(record, 'radar')
            path = radar / record['radar']
            values = np.load(path)
            values[:, 3] += 1
            np.save(path, values)
            # A changed stat key must reload, never serve the old raw cloud.
            np.testing.assert_array_equal(cached._points(record, 'radar')[:, 3], old[:, 3] + 1)

    def _run_toy(self, directory, *, steps=4, resume=None, timing_mode='throughput', broken=None):
        from project_tools import run_experiments as runner
        c = recipe(output_root=directory, device='cpu')
        c['resources'].update(num_workers=0, cpu_threads=2, raw_cache_gib_per_worker=0.)
        c['training']['checkpoint_interval_updates'] = 2
        c = validate_config(c)
        def build(*_):
            model, criterion = ToyModel(), ToyLoss()
            if broken == 'gradient':
                model.layers[0].weight.register_hook(lambda value: value * float('nan'))
            if broken == 'loss':
                model.layers[-1].bias.data.fill_(float('nan'))
            return ToyDataset(), model, criterion
        args = Namespace(mode='benchmark', allow_formal_run=False, steps=steps, warmup=0,
                         resume=resume, timing_mode=timing_mode, diagnose_validation_samples=0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'build_components', side_effect=build))
            stack.enter_context(patch.object(runner, 'formal_blockers', return_value=[]))
            stack.enter_context(patch.object(runner, 'artifact_metadata', return_value={'source_sha256': 'synthetic'}))
            stack.enter_context(patch.object(runner, 'amp_audit_contract', return_value={'source_sha256': 'synthetic'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = runner.run(c, args)
        output = Path(report['output'])
        checkpoint = torch.load(output / 'latest.rank0.pt', weights_only=False)
        rows = [json.loads(row) for row in (output / 'steps.rank0.jsonl').read_text().splitlines()]
        events = [json.loads(row) for row in (output / 'telemetry.rank0.jsonl').read_text().splitlines()]
        return report, checkpoint, rows, events

    def test_runner_timing_modes_resume_and_checkpoint_dedup_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, full, rows, events = self._run_toy(root / 'full')
            _, synchronized, sync_rows, _ = self._run_toy(root / 'sync', timing_mode='synchronized')
            first, _, _, _ = self._run_toy(root / 'resume', steps=2)
            _, resumed, resumed_rows, _ = self._run_toy(root / 'resume', steps=2,
                resume=str(Path(first['output']) / 'latest.rank0.pt'))
            for key in ('model', 'optimizer', 'scheduler', 'train_state', 'sampler', 'rng'):
                self.assert_nested_equal(full[key], synchronized[key])
                self.assert_nested_equal(full[key], resumed[key])
            for expected, actual, restarted in zip(rows, sync_rows, resumed_rows):
                self.assertEqual(expected['loss'], actual['loss'])
                self.assertEqual(expected['loss'], restarted['loss'])
                self.assertEqual(expected['loss_terms'], actual['loss_terms'])
                self.assertGreaterEqual(expected['update_wall_seconds'], expected['data_seconds'])
            commits = [e for e in events if e['event'] == 'checkpoint_commit']
            self.assertEqual([e['optimizer_step'] for e in commits], [2, 4])
            self.assertEqual(full['metadata']['checkpoint_role'], 'resume')
            self.assertEqual(full['metadata']['bn_statistics_role'], 'online_running')
            self.assertTrue(report['passed'])
            self.assertFalse(report['formal_experiment_started'])

    def test_runner_rejects_nonfinite_loss_and_gradients_before_save(self):
        for broken in ('loss', 'gradient'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(FloatingPointError, 'nonfinite training'):
                    self._run_toy(directory, broken=broken)
                self.assertEqual(list(Path(directory).rglob('latest.rank0.pt')), [])

    def test_activation_checkpoint_keeps_forward_gradients_and_bn(self):
        from project_tools.run_experiments import read_hypes
        from project_tools.smoke_supplement import synthetic_inputs, synthetic_labels
        from opencood.project.supplement_models import SupplementalModel, SupplementalLoss
        c = recipe()
        h = read_hypes(c)
        h['model']['args']['point_pillar_scatter']['grid_size'] = [32, 32, 1]
        torch.manual_seed(17)
        checkpointed = SupplementalModel(c, h).train()
        plain_config = deepcopy(c)
        plain_config['model']['activation_checkpoint'] = False
        plain = SupplementalModel(plain_config, h).train()
        plain.load_state_dict(checkpointed.state_dict())
        self.assertTrue(checkpointed.native.activation_checkpoint)
        self.assertFalse(plain.native.activation_checkpoint)
        self.assertTrue(plain.native.efficient_features)
        inputs = synthetic_inputs(h)
        targets = {name: torch.zeros_like(inputs['observation_maps']['lidar_count']) for name in ('miss', 'spur')}
        targets.update({name + '_valid': inputs['observation_maps']['lidar_occupancy'].clone() for name in ('miss', 'spur')})
        targets['normal_mask'] = torch.ones(inputs['modality_valid'].shape[0], 1, 1, 1)
        evidence = []
        for model in (plain, checkpointed):
            criterion = SupplementalLoss({**h['loss']['args'], 'training_line': 'W', 'efficient_features': True}, 'deer_sbr')
            output = model(deepcopy(inputs))
            loss = criterion(output, synthetic_labels(output), targets)
            loss.backward()
            evidence.append((output['psm'].detach(), output['rm'].detach(), loss.detach(),
                             {n: p.grad for n, p in model.named_parameters() if p.grad is not None},
                             dict(model.named_buffers()), criterion.loss_dict))
        # Recomputed branches are stateless: the CPU FP32 fixture is exact.
        self.assert_nested_equal(evidence[0], evidence[1])

    def test_validation_reuses_only_inputs_and_still_predicts_every_epoch(self):
        from project_tools.run_experiments import validate_epoch
        class Dataset:
            sample_ids = ['validate/a', 'validate/b']
            manifest = {}
            def __len__(self):
                return len(self.sample_ids)
        model = torch.nn.Linear(1, 1)
        model.weight.data.fill_(.25)
        c, cache, events = recipe(), {}, []
        batch = {'ego': {'sample_ids': Dataset.sample_ids}}
        calls = []
        def export(model, loader, dataset, device, profile_callback=None, close_loader=True):
            calls.append((profile_callback, close_loader))
            return [{'score': float(model.weight.detach())} for _ in loader]
        def evaluate(manifest, records, split, scoring, sample_ids):
            return {'metrics': {'3d': {'0.7': {'ap': records[0]['score']}}}}
        with patch('project_tools.run_experiments.make_eval_loader', side_effect=lambda *a: (Dataset(), [batch])) as make, \
             patch('project_tools.export_predictions.export_loader', side_effect=export), \
             patch('opencood.project.supplement_evaluation.evaluate_supplement', side_effect=evaluate):
            scores, _ = validate_epoch(model, c, 'cpu', lambda event, **row: events.append((event, row)),
                                      loader_cache=cache, timing_mode='throughput')
            self.assertEqual(scores, {'normal': .25, 'fog': .25})
            model.weight.data.fill_(.75)
            scores, _ = validate_epoch(model, c, 'cpu', loader_cache=cache, timing_mode='throughput')
            self.assertEqual(scores, {'normal': .75, 'fog': .75})
        self.assertEqual(make.call_count, 2)
        self.assertEqual(calls, [(None, False)] * 4)

    def test_bn_input_reuse_refits_current_weights_and_restores_online_buffers(self):
        from test_bn_runner import Toy
        from project_tools.run_experiments import batchnorm_evaluation
        c, model, cache = recipe(), Toy().train(), {}
        batch = {'ego': {'processed_lidar': torch.arange(12).reshape(4, 3).float(),
                        'record_len': torch.ones(4), 'sample_ids': [f'train/{i}' for i in range(4)]}}
        with patch('project_tools.run_experiments.make_bn_refit_loader', return_value=(None, [batch] * 8, list(range(32)))) as make, \
             patch('project_tools.run_experiments.data_contract_metadata', return_value={}):
            for _ in range(2):
                before = deepcopy(model.state_dict())
                random_before = torch.get_rng_state().clone()
                with batchnorm_evaluation(model, c, 'cpu', loader_cache=cache) as report:
                    self.assertEqual(int(model.norm.num_batches_tracked), 8)
                    self.assertTrue(report['all_modules_eval_at_yield'])
                self.assert_nested_equal(before, model.state_dict())
                self.assertTrue(torch.equal(random_before, torch.get_rng_state()))
        self.assertEqual(make.call_count, 1)

    def test_loader_parameters_and_sample_order_are_effective(self):
        from project_tools.run_experiments import make_loader, shutdown_loader
        from opencood.project.training import EpochBatchSampler
        orders = []
        for workers, prefetch in ((0, 1), (2, 1), (2, 2)):
            c = recipe()
            c['resources'].update(num_workers=workers, prefetch_factor=prefetch)
            c = validate_config(c)
            sampler = EpochBatchSampler(32, 4, 2, 17)
            loader = make_loader(ToyDataset(), c, sampler=sampler)
            self.assertEqual(loader.num_workers, workers)
            self.assertTrue(loader.pin_memory)
            self.assertEqual(loader.persistent_workers, bool(workers))
            if workers:
                self.assertEqual(loader.prefetch_factor, prefetch)
            try:
                epochs = []
                for epoch in (0, 1):
                    sampler.set_epoch(epoch)
                    epochs.append([batch['ego']['sample_ids'] for batch in loader])
                orders.append(epochs)
            finally:
                shutdown_loader({'loader': loader})
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0], orders[2])


if __name__ == '__main__':
    unittest.main()
