"""CPU checks of known answers and routing; no CUDA validation is claimed."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from project_tools import check_environment as checker


class MinimalEnvironmentTests(unittest.TestCase):
    def test_cpu_operator_fixture_has_independent_known_answers(self):
        case = checker._l4dr_operator_case('cpu')
        self.assertEqual(case['fps'].tolist(), [0, 3, 2, 1])
        self.assertEqual(case['ball'].tolist(), [[0, 1, 0, 0], [2, 2, 2, 2]])
        self.assertEqual(case['nn_index'].tolist(), [[0, 1, 2], [3, 2, 1]])
        self.assertEqual(case['nn_distance'].tolist(), [[.25, .75, 2.75], [1.75, 2.25, 4.25]])
        self.assertTrue(torch.isfinite(case['feature_gradient']).all())
        self.assertGreater(float(case['feature_gradient'].abs().sum()), 0.)

    def test_operator_comparison_rejects_wrong_indices_or_features(self):
        cpu = checker._l4dr_operator_case('cpu')
        for key in ('fps', 'ball', 'nn_index', 'nn_distance', 'interpolated', 'feature_gradient'):
            changed = deepcopy(cpu)
            changed[key].reshape(-1)[0] += 1
            with self.subTest(key=key), patch.object(checker, '_l4dr_operator_case', side_effect=[cpu, changed]):
                with self.assertRaises(AssertionError):
                    checker.check_l4dr_cuda()

    def test_minimal_routing_excludes_baseline_optimizer_and_import_sweep(self):
        # Mock dispatch only. This test does not execute any CUDA operator.
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_name', return_value='mock device'), \
             patch.object(checker, 'check_l4dr_cuda', return_value={'mock': True}) as l4dr, \
             patch.object(checker, 'check_spconv', return_value={'mock': True}) as spconv, \
             patch.object(checker, 'check_model') as baseline, \
             patch.object(checker, 'check_extensions') as extensions, \
             patch.object(checker.importlib, 'import_module') as imports, \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text') as write, \
             redirect_stdout(io.StringIO()):
            checker.main(['--checks', 'l4dr', 'spconv', '--output', 'mock-report.json'])
        l4dr.assert_called_once()
        spconv.assert_called_once()
        baseline.assert_not_called()
        extensions.assert_not_called()
        imports.assert_not_called()
        report = json.loads(write.call_args.args[0])
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['baseline_optimizer_check_requested'])
        self.assertFalse(report['dataset_read'])
        self.assertEqual(set(report['checks']), {'l4dr_cuda', 'spconv_gpu'})

    def test_missing_cuda_fails_and_writes_failure_report(self):
        with patch.object(torch.cuda, 'is_available', return_value=False), \
             patch.object(torch.cuda, 'get_device_name') as gpu_name, \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text') as write:
            with self.assertRaisesRegex(RuntimeError, 'CUDA GPU is required'):
                checker.main(['--checks', 'l4dr', 'spconv', '--output', 'mock-report.json'])
        gpu_name.assert_not_called()
        report = json.loads(write.call_args.args[0])
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['checks'], {})

    def test_default_keeps_original_full_checks(self):
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_name', return_value='mock device'), \
             patch.object(checker, 'check_l4dr_cuda') as l4dr, \
             patch.object(checker, 'check_spconv', return_value={}) as spconv, \
             patch.object(checker, 'check_model', return_value={}) as baseline, \
             patch.object(checker, 'check_extensions', return_value={}) as extensions, \
             patch.object(checker.importlib, 'import_module', return_value=SimpleNamespace(__version__='mock')) as imports, \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text') as write, \
             redirect_stdout(io.StringIO()):
            checker.main(['--output', 'mock-report.json'])
        l4dr.assert_not_called()
        spconv.assert_called_once()
        extensions.assert_called_once()
        baseline.assert_called_once()
        self.assertEqual(imports.call_count, 8)
        self.assertTrue(json.loads(write.call_args.args[0])['baseline_optimizer_check_requested'])

    def test_documented_data_commands_parse_without_model_or_loss_execution(self):
        from scripts import supplement
        from project_tools import run_experiments as runner
        class TwoSamples:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                return index

            def collate_batch(self, items):
                return dict(ego=dict(sample_ids=[str(items[0])], record_len=torch.tensor([1]),
                    metadata=[[dict(lidar_points=2, radar_points=2)]],
                    label_dict={'pos_equal_one': torch.ones(1)}))

        for dataset in ('v2xr', 'v2xradar', 'kradar'):
            extra = ['--radar-root', '/synthetic/radar', '--label-root', '/synthetic/labels'] if dataset == 'kradar' else []
            with self.subTest(dataset=dataset), patch.object(supplement, 'write_config', return_value=Path('mock.yaml')) as write:
                with redirect_stdout(io.StringIO()):
                    supplement.main(['configure', '--dataset', dataset, '--method', 'l4dr', '--variant', 'S0',
                        '--data-root', '/synthetic/data', '--manifest-dir', '/synthetic/meta',
                        '--device', 'cuda:0', '--output-root', '/synthetic/checks', *extra])
                config = write.call_args.args[0]
                output = io.StringIO()
                with patch.object(supplement, 'runtime_config', return_value=config), \
                     patch.object(runner, 'build_components') as build, redirect_stdout(output):
                    model, criterion = unittest.mock.Mock(), unittest.mock.Mock()
                    build.return_value = (TwoSamples(), model, criterion)
                    supplement.main(['check-data', '--config', 'mock.yaml', '--samples', '2'])
                model.assert_not_called()
                criterion.assert_not_called()
                report = json.loads(output.getvalue())
                self.assertFalse(report['training_started'])
                self.assertEqual(len(report['samples']), 2)


if __name__ == '__main__':
    unittest.main()
