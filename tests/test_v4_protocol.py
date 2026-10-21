"""Bind train-only BN refitting, losses and checkpoint roles to schema 4."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from opencood.project.config import (load_config, require_matching_training_contract,
                                    scientific_training_contract, validate_config)
from opencood.project.training import (EpochBatchSampler, TrainState, load_checkpoint,
                                      save_checkpoint)


class V4ProtocolTests(unittest.TestCase):
    def test_registered_20_epoch_budget_is_v4_only_and_comparison_bound(self):
        for epochs in (20, 30):
            with self.subTest(version=4, epochs=epochs):
                config = validate_config({'version': 4, 'training': {'epochs': epochs}})
                self.assertEqual(config['training']['epochs'], epochs)
        for version in (2, 3):
            self.assertEqual(validate_config({'version': version})['training']['epochs'], 30)
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'budget'):
                validate_config({'version': version, 'training': {'epochs': 20}})
        for epochs in (1, 19, 21, 29, 31, True, 20.0):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                validate_config({'version': 4, 'training': {'epochs': epochs}})
        with self.assertRaisesRegex(ValueError, 'training contract mismatch'):
            require_matching_training_contract(
                validate_config({'version': 4}),
                validate_config({'version': 4, 'training': {'epochs': 20}}), 'paired budget')

    def test_registered_statistics_policy_keeps_online_training(self):
        current = validate_config({'version': 4})
        self.assertEqual(current['experiment']['training_protocol'], 'teacher_free_v4')
        self.assertEqual(current['training']['normalization'], 'batchnorm')
        self.assertEqual(current['training']['precision'], 'fp32')
        self.assertEqual(current['training']['batchnorm'], {
            'eps': .001, 'momentum': .01, 'evaluation_statistics': 'train_refit_cma_v1',
            'refit_samples': 32, 'refit_sampling': 'uniform_midpoints_v1', 'refit_epoch': 0,
            'refit_augment': False, 'refit_micro_batch_size': 4})
        self.assertEqual(current['model']['amplitude_context'], 'rms')
        self.assertEqual(current['model']['type_prior_mode'], 'input_only')
        self.assertTrue(current['model']['efficient_features'])
        for version in (2, 3):
            old = validate_config({'version': version})
            self.assertEqual(old['training']['batchnorm']['evaluation_statistics'], 'running')
            self.assertEqual(old['training']['batchnorm']['refit_samples'], 0)
            self.assertEqual(old['training']['batchnorm']['momentum'], .01)

    def test_unregistered_statistics_or_precision_changes_are_rejected(self):
        for change in ({'momentum': .1}, {'eps': .01}, {'evaluation_statistics': 'running'},
                       {'refit_samples': 16}, {'refit_epoch': 1}, {'refit_augment': True},
                       {'refit_sampling': 'validation'}, {'refit_micro_batch_size': 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({'version': 4, 'training': {'batchnorm': change}})
        for change in ({'precision': 'amp_bf16'}, {'normalization': 'frozen_batchnorm'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({'version': 4, 'training': change})
        with self.assertRaises(ValueError):
            validate_config({'version': 4, 'experiment': {'training_protocol': 'teacher_free_v3'}})
        with self.assertRaises(ValueError):
            validate_config({'version': 3, 'training': {
                'batchnorm': {'evaluation_statistics': 'train_refit_cma_v1', 'refit_samples': 32}}})

    def test_loss_and_entire_statistics_policy_are_in_comparison_contract(self):
        current = validate_config({'version': 4})
        contract = scientific_training_contract(current)
        self.assertEqual(contract['loss'], current['loss'])
        self.assertEqual(contract['training']['batchnorm'], current['training']['batchnorm'])
        for group, field, value in (('loss', 'type_weight', .2),
                                    ('loss', 'correction_weight', .002),
                                    ('loss', 'identity_weight', .02)):
            changed = copy.deepcopy(current)
            changed[group][field] = value
            with self.assertRaisesRegex(ValueError, 'training contract mismatch'):
                require_matching_training_contract(current, changed, 'paired comparison')
        for field, value in (('momentum', .1), ('refit_samples', 64), ('refit_epoch', 1),
                             ('evaluation_statistics', 'running'), ('refit_micro_batch_size', 2)):
            changed = copy.deepcopy(current)
            changed['training']['batchnorm'][field] = value
            with self.assertRaisesRegex(ValueError, 'training contract mismatch'):
                require_matching_training_contract(current, changed, 'paired comparison')
        contract['training']['batchnorm']['momentum'] = .9
        self.assertEqual(current['training']['batchnorm']['momentum'], .01)

    def test_all_current_configs_share_registered_input_budget_loss_and_bn_policy(self):
        root = Path(__file__).resolve().parents[1] / 'cscwd_configs'
        paths = list(root.glob('project_v4*.yaml'))
        self.assertEqual(len(paths), 19)
        for path in paths:
            config = load_config(path)
            self.assertEqual(config['version'], 4)
            self.assertEqual(config['experiment']['training_protocol'], 'teacher_free_v4')
            self.assertEqual(config['training']['batchnorm'], validate_config({'version': 4})['training']['batchnorm'])
            self.assertIn('teacher_free_v4', config['artifacts']['output_root'])
        for track in ('C', 'W'):
            variants = ('S0', 'A', 'B', 'AB') if track == 'C' else (
                'S0', 'A', 'B', 'AB', 'SUP', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6')
            configs = [load_config(root / f'project_v4_{track}_{variant}.yaml') for variant in variants]
            for config in configs[1:]:
                require_matching_training_contract(configs[0], config, 'v4 matrix')
        for variant in ('N1', 'N2'):
            control = load_config(root / f'project_v4_W_{variant}.yaml')
            reference = load_config(root / 'project_v4_W_AB.yaml')
            self.assertEqual(control['training'], reference['training'])
            self.assertEqual(control['loss'], reference['loss'])
            self.assertEqual(control['data'], reference['data'])
            with self.assertRaisesRegex(ValueError, 'training contract mismatch'):
                require_matching_training_contract(reference, control, 'registered mechanism ablation')
        matrix = json.loads((root / 'experiment_matrix.json').read_text())
        self.assertEqual(matrix['schema_version'], 4)
        self.assertEqual(matrix['training_protocol'], 'teacher_free_v4')
        self.assertEqual(len(matrix['runs']), 78)
        self.assertFalse(matrix['formal_experiments_started'])
        self.assertTrue(all(row['status'] == 'pending' for row in matrix['runs']))
        self.assertTrue(all('tf3' not in row['run_id'] for row in matrix['runs']))
        self.assertNotIn('teacher_free_v3', json.dumps(matrix))

    def test_schema4_resume_requires_online_role_and_rejects_selected(self):
        config = validate_config({'version': 4})
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10])
        sampler = EpochBatchSampler(32, 4, 2, 17)
        state = TrainState()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, model, optimizer, scheduler, None, state, sampler, config)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload['schema_version'], 4)
            self.assertEqual(payload['metadata']['checkpoint_role'], 'resume')
            self.assertEqual(payload['metadata']['bn_statistics_role'], 'online_running')
            load_checkpoint(path, model, optimizer, scheduler, None, sampler, config)
            selected = {'checkpoint_role': 'selected', 'bn_statistics_role': 'train_refit_cma_v1'}
            save_checkpoint(path, model, optimizer, scheduler, None, state, sampler, config, selected)
            with self.assertRaisesRegex(ValueError, 'selected checkpoint.*refitted BN'):
                load_checkpoint(path, model, optimizer, scheduler, None, sampler, config)
            for metadata in ({'checkpoint_role': 'selected'},
                             {'checkpoint_role': 'selected', 'bn_statistics_role': 'online_running'},
                             {'checkpoint_role': 'resume', 'bn_statistics_role': 'train_refit_cma_v1'},
                             {'checkpoint_role': 'unknown'}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    save_checkpoint(path, model, optimizer, scheduler, None, state, sampler, config, metadata)
            for metadata in ({}, {'checkpoint_role': 'resume'},
                             {'checkpoint_role': 'resume', 'bn_statistics_role': 'train_refit_cma_v1'}):
                payload['metadata'] = metadata
                torch.save(payload, path)
                with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, 'online BN statistics role'):
                    load_checkpoint(path, model, optimizer, scheduler, None, sampler, config)

    def test_schema4_cannot_resume_old_protocol_or_reverse(self):
        current = validate_config({'version': 4})
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10])
        sampler = EpochBatchSampler(32, 4, 2, 17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            for version in (2, 3):
                old = validate_config({'version': version})
                save_checkpoint(path, model, optimizer, scheduler, None, TrainState(), sampler, old)
                with self.assertRaisesRegex(ValueError, 'configuration/schema mismatch'):
                    load_checkpoint(path, model, optimizer, scheduler, None, sampler, current)
                save_checkpoint(path, model, optimizer, scheduler, None, TrainState(), sampler, current)
                with self.assertRaisesRegex(ValueError, 'configuration/schema mismatch'):
                    load_checkpoint(path, model, optimizer, scheduler, None, sampler, old)


if __name__ == '__main__':
    unittest.main()
