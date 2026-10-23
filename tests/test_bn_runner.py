"""The runner's evaluation boundary restores the continuing training state."""
import unittest
from unittest.mock import patch
import json
import torch
from opencood.project.config import validate_config
from opencood.project.training import rng_state
from project_tools.run_experiments import (batchnorm_evaluation, batchnorm_snapshot,
                                           monitor_batchnorm, cuda_memory_snapshot, validate_epoch)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(3)
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, inputs, return_aux=False):
        return {'psm': self.dropout(self.norm(inputs['processed_lidar']))}


class RunnerBatchNormTests(unittest.TestCase):
    def test_registered_loader_boundary_restores_rng_and_training_buffers(self):
        model = Toy().train()
        config = validate_config({'version': 4})
        before = {name: value.clone() for name, value in model.state_dict().items()}
        random_before = torch.get_rng_state().clone()
        events = []
        def loader(_):
            torch.rand(5)  # independent loader construction must also be scoped
            batches = [{'ego': {'processed_lidar': torch.randn(4, 3), 'record_len': torch.ones(4),
                       'sample_ids': [f'train/{4*i+j}' for j in range(4)]}} for i in range(8)]
            return None, batches, list(range(32))
        with patch('project_tools.run_experiments.make_bn_refit_loader', loader), \
             patch('project_tools.run_experiments.data_contract_metadata', return_value={}):
            with batchnorm_evaluation(model, config, 'cpu', lambda event, **row: events.append((event, row))) as report:
                self.assertEqual(report['sample_count'], 32)
                self.assertEqual(report['micro_batches'], 8)
                self.assertEqual(report['dataset_split'], 'train')
                self.assertFalse(any(m.training for m in model.modules()))
                self.assertEqual(int(model.norm.num_batches_tracked), 8)
            self.assertTrue(report['original_buffers_restored_on_exit'])
            self.assertTrue(report['rng_restored_on_exit'])
        self.assertTrue(model.training)
        self.assertTrue(model.dropout.training)
        self.assertTrue(torch.equal(random_before, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertEqual([row['phase'] for event, row in events if event == 'batchnorm'],
                         ['before_refit_online', 'refitted_for_validation', 'after_refit_online_restored'])
        restored = next(row for event, row in events if event == 'refit_restore')
        self.assertTrue(restored['original_bn_hashes_restored'])
        self.assertTrue(restored['original_buffers_restored_on_exit'])
        refit = next(row for event, row in events if event == 'refit_complete')
        self.assertEqual(len(refit['input_sample_ids']), 32)
        json.dumps(events, allow_nan=False)

    def test_evaluation_exception_restores_outer_state(self):
        model = Toy().train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        config = validate_config({'version': 4})
        events = []
        batch = {'ego': {'processed_lidar': torch.ones(4, 3), 'record_len': torch.ones(4),
                         'sample_ids': [f'train/{j}' for j in range(4)]}}
        with patch('project_tools.run_experiments.make_bn_refit_loader', return_value=(None,[batch]*8,list(range(32)))), \
             patch('project_tools.run_experiments.data_contract_metadata', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                with batchnorm_evaluation(model, config, 'cpu', lambda event, **row: events.append((event, row))):
                    raise RuntimeError('evaluation failed')
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(next(row for event, row in events if event == 'refit_restore')['original_bn_hashes_restored'])

    def test_bn_monitor_preserves_state_and_records_bad_statistics_before_failing(self):
        model = Toy().train()
        model.norm.weight.grad = torch.ones_like(model.norm.weight)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        gradient = model.norm.weight.grad.clone()
        random_before = torch.get_rng_state().clone()
        summary = monitor_batchnorm(model, phase='checkpoint_resume')
        self.assertTrue(summary['all_finite'])
        self.assertEqual(summary['running_var_min'], 1.)
        self.assertEqual(summary['modules']['norm']['num_batches_tracked'], 0)
        self.assertTrue(torch.equal(random_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(gradient, model.norm.weight.grad))
        self.assertTrue(model.norm.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        with torch.no_grad():
            model.norm.running_var[0] = float('nan')
        events = []
        with self.assertRaisesRegex(FloatingPointError, 'norm'):
            monitor_batchnorm(model, lambda event, **row: events.append((event, row)), phase='checkpoint_resume')
        bad = events[0][1]['batchnorm']
        self.assertFalse(bad['all_finite'])
        self.assertEqual(bad['invalid_modules'], ['norm'])
        json.dumps(bad, allow_nan=False)
        with torch.no_grad():
            model.norm.running_var[0] = -.1
        self.assertEqual(batchnorm_snapshot(model)['modules']['norm']['negative_variance_count'], 1)

    def test_allocator_counters_do_not_convert_recovered_oom_to_failure(self):
        self.assertEqual(cuda_memory_snapshot('cpu')['allocator_ooms'], 0)
        with patch('torch.cuda.memory_stats', return_value={'num_ooms': 3, 'num_alloc_retries': 4}), \
             patch('torch.cuda.memory_allocated', return_value=10), \
             patch('torch.cuda.memory_reserved', return_value=20), \
             patch('torch.cuda.max_memory_allocated', return_value=30), \
             patch('torch.cuda.max_memory_reserved', return_value=40):
            memory = cuda_memory_snapshot('cuda:0')
        self.assertEqual(memory['allocator_ooms'], 3)
        self.assertEqual(memory['allocator_retries'], 4)
        self.assertEqual(memory['peak_reserved_bytes'], 40)

    def test_validation_progress_preserves_normal_fog_scope_and_closes_workers(self):
        class Dataset:
            manifest = {}
            def __len__(self):
                return 52
        class Iterator:
            def __init__(self):
                self.batches = iter([{'ego': {'sample_ids': [str(i)]}} for i in range(52)])
                self.closed = False
            def __iter__(self):
                return self
            def __next__(self):
                return next(self.batches)
            def _shutdown_workers(self):
                self.closed = True
        class Loader:
            _iterator = None  # also cover nonpersistent workers
            def __init__(self):
                self.iterator = Iterator()
            def __iter__(self):
                return self.iterator
        loaders = [Loader(), Loader()]
        events = []
        def export(model, batches, dataset, device, profile_callback=None):
            if profile_callback:
                profile_callback({"stage": "forward", "seconds": 1.25})
            return [batch['ego']['sample_ids'][0] for batch in batches]
        with patch('project_tools.run_experiments.make_eval_loader', side_effect=[(Dataset(), loader) for loader in loaders]) as make, \
             patch('project_tools.export_predictions.export_loader', side_effect=export), \
             patch('opencood.project.evaluation.evaluate_predictions', return_value={'metrics': {'3d': {'0.7': {'ap': .5}}}}):
            scores, _ = validate_epoch(Toy(), validate_config({'version': 4, 'experiment': {'track': 'W', 'variant': 'S0'}}), 'cpu',
                                       lambda event, **row: events.append((event, row)), timing_mode='synchronized')
        self.assertEqual(scores, {'normal': .5, 'fog': .5})
        self.assertEqual([call.args[1:] for call in make.call_args_list], [('validate', 'normal'), ('validate', 'fog')])
        self.assertTrue(all(loader.iterator.closed for loader in loaders))
        for weather in ('normal', 'fog'):
            progress = [row['completed_samples'] for event, row in events if event == 'validation_progress' and row['weather'] == weather]
            self.assertEqual(progress, [1, 25, 50, 52])
        self.assertEqual(len([event for event, row in events if event == 'validation_weather_complete']), 2)
        for event, row in events:
            if event == 'validation_weather_complete':
                self.assertEqual(row['stage_seconds']['forward'], 1.25)
                self.assertEqual(row['stage_calls']['forward'], 1)
                self.assertGreaterEqual(row['stage_seconds']['scoring'], 0)


if __name__ == '__main__':
    unittest.main()
