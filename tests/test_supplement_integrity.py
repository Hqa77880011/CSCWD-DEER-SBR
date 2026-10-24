"""Regressions for experiment decisions and actual observation diagnostics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from opencood.project.supplement import resolve_final
from opencood.project.supplement_data import SupplementalDataset
from opencood.project.supplement_reporting import diagnose, summarize
from opencood.project.supplement_models import SupplementalModel
from project_tools.run_experiments import read_hypes
from scripts.supplement import recipe
from tests.test_supplement import cooperative_fixture


def frozen_fixture():
    """Synthetic development evidence; never a real model decision."""
    return dict(schema='supplement_frozen_final_v1', variant='N2',
        source_split='validate', seeds=[17, 29, 43], test_used=False,
        one_structure_for_all_seeds_and_weather=True, normal_tolerance_pp=1.,
        development_means=dict(AB=.4, N2=.5),
        normal_development_means=dict(S0=.5, AB=.5, N2=.5),
        evidence={v: {str(s): dict(score=score, scores=dict(normal=.5, fog=2 * score - .5),
                                  checkpoint_sha256=str(s % 10) * 64)
                      for s in (17, 29, 43)}
                  for v, score in (('S0', .45), ('AB', .4), ('N2', .5))})


def score_fixture(variant='AB', seed=17, weather='normal'):
    return dict(schema='supplement_evaluation_v1', metadata=dict(
        dataset='v2xr', protocol='fixture', protocol_sha256='protocol',
        gt_manifest_sha256='gt', transfer='retrained', condition={},
        weather=weather, split='test', method='deer_sbr', variant=variant,
        seed=seed, recipe='independent_e60', sample_ids=['a'],
        checkpoint_sha256=f'{variant}_{seed}', checkpoint_role='selected',
        checkpoint_epoch=50, selected_epoch=50, resolved_model=dict(variant=variant)),
        result=dict(metrics={'3d': {'0.7': dict(ap=.5)}}))


class SupplementIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_F_recomputes_registered_rule_and_normal_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic_decision.json'
            record = frozen_fixture()
            path.write_text(json.dumps(record))
            self.assertEqual(resolve_final(path), 'N2')
            for mutation in ('wrong_winner', 'wrong_score', 'normal_ineligible'):
                broken = deepcopy(record)
                if mutation == 'wrong_winner':
                    broken['variant'] = 'AB'
                elif mutation == 'wrong_score':
                    broken['evidence']['N2']['17']['score'] = .9
                else:
                    for row in broken['evidence']['N2'].values():
                        row.update(score=.5, scores=dict(normal=.2, fog=.8))
                    broken['normal_development_means']['N2'] = .2
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError, msg=mutation):
                    resolve_final(path)

    def test_weather_mean_rejects_mixed_selected_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            for weather in ('fog', 'snow'):
                row = score_fixture(weather=weather)
                row['metadata']['checkpoint_sha256'] = weather
                (Path(temporary) / (weather + '.scores.json')).write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                summarize([temporary], 'deer_sbr/S0')

    def test_summary_keeps_distinct_rho_settings_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            for rho in (.125, .25, .5):
                row = score_fixture()
                row['metadata']['resolved_model']['rho'] = rho
                (Path(temporary) / f'rho_{rho}.scores.json').write_text(json.dumps(row))
            rows = summarize([temporary], 'deer_sbr/S0')['rows']
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['resolved_model']['rho'] for r in rows}, {.125, .25, .5})
            self.assertTrue(all(r['n'] == 1 for r in rows))

    def test_pairing_rejects_changed_fixed_GT_or_seed_sample_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = score_fixture('S0'), score_fixture('AB')
            b['metadata']['gt_manifest_sha256'] = 'different GT'
            for name, row in (('a', a), ('b', b)):
                (root / (name + '.scores.json')).write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'GT|provenance'):
                summarize([root], 'deer_sbr/S0')
            (root / 'b.scores.json').write_text(json.dumps(a))
            b = score_fixture('S0', seed=29)
            b['metadata']['sample_ids'] = ['another scene']
            (root / 'b.scores.json').write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError, 'sample'):
                summarize([root], 'deer_sbr/S0')

    def test_native_and_independent_recipes_have_distinct_run_ids(self):
        a = recipe(method='attfuse', variant='S0', epochs=30, native=False)
        b = recipe(method='attfuse', variant='S0', epochs=30, native=True)
        self.assertNotEqual(a['experiment']['run_id'], b['experiment']['run_id'])

    def test_omitted_budget_uses_dataset_template(self):
        for dataset, epochs in [('v2xr', 60), ('v2xradar', 30), ('kradar', 30)]:
            self.assertEqual(recipe(dataset)['training']['epochs'], epochs)

    def test_configuration_rejects_disconnected_method_and_recipe_fields(self):
        from opencood.project.config import validate_config
        mutations = [
            ('data', 'base_hypes', 'cscwd_configs/baseline_full_data_check.yaml'),
            ('model', 'local_variant', 'AB'),
            ('training', 'adam_eps', 0.),
            ('training', 'recipe', 'independent_e90'),
            ('training', 'milestones', [10, 15]),
            ('loss', 'type_weight', -1.),
        ]
        for section, key, value in mutations:
            c = recipe(method='attfuse', variant='S0', epochs=30)
            c[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(c)
        c = recipe()
        c['training']['batchnorm']['eps'] = .01
        with self.assertRaisesRegex(ValueError, 'BN|BatchNorm|batchnorm'):
            validate_config(c)

    def test_external_configs_do_not_advertise_internal_execution_or_losses(self):
        from opencood.project.config import validate_config
        for method in ('attfuse', 'mdd', 'l4dr', 'cobevt', 'where2comm'):
            c = recipe(method=method, variant='S0', native=True)
            self.assertFalse(c['model']['activation_checkpoint'], method)
            self.assertFalse(c['model']['efficient_features'], method)
            self.assertEqual(set(c['loss'].values()), {0.})
            c['model']['activation_checkpoint'] = True
            with self.assertRaisesRegex(ValueError, 'local|internal|checkpoint'):
                validate_config(c)

    def test_skip_completed_checks_current_source_data_and_selected_role(self):
        from unittest.mock import patch
        from opencood.project.config import config_hash
        from scripts.supplement import completed_run_matches
        c = recipe()
        metadata = dict(source_sha256='source', gt_manifest_sha256='data')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'report.rank0.json').write_text(json.dumps(dict(
                passed=True, source_unchanged=True, status='checkpoint_selected')))
            (output / 'resolved_config.json').write_text(json.dumps(c))
            (output / 'run_metadata.json').write_text(json.dumps(metadata))
            selected = dict(schema_version=5, training_protocol='supplement_v1', config_sha256=config_hash(c), config=c,
                metadata={**metadata, 'checkpoint_role': 'selected', 'bn_statistics_role': 'train_refit_cma_v1'})
            torch.save(selected, output / 'selected.pt')
            with patch('project_tools.run_experiments.artifact_metadata', return_value=metadata):
                self.assertTrue(completed_run_matches(c, output))
            for changed in ({**metadata, 'source_sha256': 'new_source'},
                            {**metadata, 'gt_manifest_sha256': 'new_data'}):
                with patch('project_tools.run_experiments.artifact_metadata', return_value=changed):
                    self.assertFalse(completed_run_matches(c, output))
            for role in ('resume', 'budget_snapshot'):
                selected['metadata']['checkpoint_role'] = role
                torch.save(selected, output / 'selected.pt')
                with patch('project_tools.run_experiments.artifact_metadata', return_value=metadata):
                    self.assertFalse(completed_run_matches(c, output))

    def test_microbatch_and_selection_labels_cannot_bypass_registered_policy(self):
        from opencood.project.config import validate_config
        c = recipe()
        c['training'].update(micro_batch_size=8, accumulation_steps=1)
        with self.assertRaisesRegex(ValueError, 'microbatch4'):
            validate_config(c)
        c = recipe(method='mdd', variant='S0', native=True)
        c['training'].update(micro_batch_size=1, accumulation_steps=2)
        with self.assertRaisesRegex(ValueError, 'source single-device microbatch'):
            validate_config(c)
        c = recipe('kradar')
        c['selection']['class_name'] = 'Sedan'
        with self.assertRaisesRegex(ValueError, 'class_name'):
            validate_config(c)

    def test_mdd_runtime_betas_and_reference_objective_match_author_code(self):
        from opencood.loss.point_pillar_loss_mdd import PointPillarLossMDD
        from opencood.project.supplement_models import SupplementalLoss
        c = recipe(method='mdd', variant='S0', native=True)
        h = read_hypes(c)
        diffusion = h['model']['args']['mdd_block']['diffusion']
        self.assertEqual((diffusion['beta_start'], diffusion['beta_end']), (.005, .05))
        model = SupplementalModel(c, h)
        expected = torch.linspace(.005 ** .5, .05 ** .5, 3, dtype=torch.float64).square().float()
        torch.testing.assert_close(model.native.mdd.betas, expected, atol=0, rtol=0)
        loss_fn = SupplementalLoss(h['loss']['args'], 'mdd')
        native_loss = PointPillarLossMDD(h['loss']['args'])
        for epoch in (0, 19, 29):
            loss_fn.epoch = epoch
            reference = torch.ones(1, 2, 2, 2, requires_grad=True)
            prediction = torch.full_like(reference, 3., requires_grad=True)
            output = dict(psm=torch.zeros(1, 2, 2, 2, requires_grad=True),
                          rm=torch.zeros(1, 14, 2, 2, requires_grad=True),
                          gt_feature=reference, pred_feature=prediction)
            target = dict(pos_equal_one=torch.zeros(1, 2, 2, 2), neg_equal_one=torch.ones(1, 2, 2, 2),
                          targets=torch.zeros(1, 2, 2, 14))
            loss_fn(output, target).backward()
            native_prediction = prediction.detach().clone().requires_grad_()
            expected_loss = native_loss.diff_loss_func(native_prediction, reference.detach(),
                native_loss.sigmoid_weight(3, epoch + 1))
            expected_loss.backward()
            torch.testing.assert_close(loss_fn.loss_dict['mdd_reference_loss'], expected_loss, atol=0, rtol=0)
            self.assertIsNone(reference.grad)
            torch.testing.assert_close(prediction.grad, native_prediction.grad, atol=0, rtol=0)

    def test_Where2comm_common_weights_and_module_identity(self):
        from opencood.project.config import validate_config
        baseline = recipe(method='where2comm', variant='S0', native=True)
        local = recipe(method='where2comm', variant='AB', native=True)
        a = SupplementalModel(baseline, read_hypes(baseline))
        b = SupplementalModel(local, read_hypes(local))
        self.assertIsNone(a.local)
        self.assertIsNotNone(b.local)
        self.assertEqual(set(a.native.state_dict()), set(b.native.state_dict()))
        for key, value in a.native.state_dict().items():
            torch.testing.assert_close(value, b.native.state_dict()[key], atol=0, rtol=0)
        local['model']['local_variant'] = 'none'
        with self.assertRaisesRegex(ValueError, 'actual local module'):
            validate_config(local)

    def test_cobevt_window_padding_preserves_agent_and_spatial_masks(self):
        from opencood.project.supplement_models import WindowPaddedFusion
        class Capture(torch.nn.Module):
            def forward(self, features, mask):
                self.features, self.mask = features, mask
                return features[:, 0]
        native = Capture()
        fusion = WindowPaddedFusion(native, 4)
        features = torch.arange(1 * 3 * 2 * 5 * 6, dtype=torch.float32).reshape(1, 3, 2, 5, 6)
        mask = torch.zeros(1, 5, 6, 1, 3)
        mask[..., :2] = 1
        result = fusion(features, mask)
        self.assertTrue(torch.equal(result, features[:, 0]))
        self.assertEqual(native.features.shape[-2:], (8, 8))
        self.assertTrue(torch.equal(native.mask[:, :5, :6], mask))
        self.assertEqual(int(native.mask[:, 5:].sum() + native.mask[:, :, 6:].sum()), 0)
        self.assertEqual(int(native.mask[..., 2].sum()), 0)

    def test_agent_count_uses_communicating_agents_including_ego(self):
        from opencood.project.data_geometry import communicating_agents
        annotations = {str(i): dict(lidar_pose=[x, 0, 0, 0, 0, 0])
                       for i, x in enumerate((10, 20, 30, 80, 100))}
        self.assertEqual(communicating_agents(annotations, '0', 70), ['0', '1', '2', '3'])
        self.assertEqual(communicating_agents(annotations, '0', 0), ['0'])

    def test_archived_release_helpers_match_tested_portable_helpers(self):
        import ast
        import zipfile
        root = Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(root / 'repro/archive/legacy_v4_runtime.zip') as archive:
            archived = ast.parse(archive.read('scripts/pretrained.py').decode('utf8'))
        primary = ast.parse((root / 'scripts/pretrained.py').read_text(encoding='utf8'))
        for name in ('unpack_metadata', 'dataset_view'):
            a = next(n for n in archived.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in primary.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b))

    def test_registered_schedules_change_actual_optimizer_LR(self):
        from opencood.project.supplement_runner import scheduler_for
        from opencood.project.supplement import SCHEDULES
        for budget, milestones in SCHEDULES.items():
            c = recipe(epochs=budget)
            optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=c['training']['lr'])
            scheduler = scheduler_for(optimizer, c)
            used = []
            for _ in range(budget):
                used.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                scheduler.step()
            for milestone in milestones:
                self.assertAlmostEqual(used[milestone] / used[milestone - 1], .1)
        c = recipe(method='cobevt', variant='S0', native=True)
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=c['training']['lr'])
        scheduler = scheduler_for(optimizer, c)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], c['training']['warmup_lr'])
        for _ in range(c['training']['warmup_epochs']):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], c['training']['lr'])

    def test_freeze_final_uses_matched_complete_development_recipes(self):
        from opencood.project.config import config_hash
        from opencood.project.supplement_reporting import freeze_final
        with tempfile.TemporaryDirectory() as temporary:
            runs = []
            for variant, final_score in (('S0', .45), ('AB', .4), ('N2', .5)):
                for seed in (17, 29, 43):
                    c = recipe(variant=variant, seed=seed, epochs=20)
                    run = Path(temporary) / f'{variant}_{seed}'
                    run.mkdir()
                    (run / 'resolved_config.json').write_text(json.dumps(c))
                    (run / 'report.rank0.json').write_text(json.dumps(dict(passed=True, status='checkpoint_selected')))
                    (run / 'state.json').write_text(json.dumps(dict(selected_epoch=20)))
                    for epoch in range(1, 21):
                        ap = final_score - (20 - epoch) * .001
                        result = {weather: dict(split='validate', metrics={'3d': {'0.7': dict(ap=ap)}})
                                  for weather in ('normal', 'fog')}
                        (run / f'validation_epoch{epoch:03d}.json').write_text(json.dumps(result))
                    saved = dict(schema_version=5, config=c, config_sha256=config_hash(c),
                        metadata=dict(checkpoint_role='selected', source_sha256='1' * 64, gt_manifest_sha256='2' * 64),
                        train_state=dict(selected_epoch=20, best_score=final_score))
                    torch.save(saved, run / 'selected.pt')
                    runs.append(run)
            frozen = freeze_final(runs, 1.)
            self.assertEqual(frozen['variant'], 'N2')
            target = Path(temporary) / 'AB_29'
            c = json.loads((target / 'resolved_config.json').read_text())
            c['model']['rho'] = .5
            (target / 'resolved_config.json').write_text(json.dumps(c))
            saved = torch.load(target / 'selected.pt', weights_only=False)
            saved.update(config=c, config_sha256=config_hash(c))
            torch.save(saved, target / 'selected.pt')
            with self.assertRaisesRegex(ValueError, 'matched'):
                freeze_final(runs, 1.)

    def test_clear_development_has_normal_type_targets_and_real_donor_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cooperative_fixture(root)
            from opencood.project.supplement_data import build_real_manifest
            manifest = build_real_manifest('v2xradar', root, ['dev_b'],
                                           clean_sequences=['train_a', 'dev_b'])
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('v2xradar', data_root=root)
            h = read_hypes(c)
            dataset = SupplementalDataset(c, h, 'validate')
            targets = dataset[0]['targets'][0]
            self.assertGreater(targets['miss_valid'].sum(), 0)
            self.assertGreater(targets['spur_valid'].sum(), 0)
            self.assertEqual(targets['miss'].max(), 0)
            self.assertEqual(targets['spur'].max(), 0)
            # Restrict only the synthetic extent; real donor observations and
            # local module computations still pass through the actual model.
            h['model']['args']['point_pillar_scatter']['grid_size'] = [32, 32, 1]
            from project_tools.smoke_supplement import synthetic_inputs
            inputs = synthetic_inputs(h)
            model = SupplementalModel(c, h).eval()
            target = {k: torch.zeros_like(inputs['observation_maps']['lidar_count'])
                      for k in ('miss', 'spur')}
            target.update({k + '_valid': torch.ones_like(v) for k, v in list(target.items())})
            batch = dict(ego={**inputs, 'sample_ids': ['fixture'], 'metadata': [[]]},
                         train_targets=target)
            class OneScene:
                def __len__(self):
                    return 1

                def __getitem__(self, index):
                    return batch

                def collate_batch(self, items):
                    return items[0]
            result = diagnose(model, OneScene(), torch.device('cpu'), SimpleNamespace(samples=1), {})
            row = result['rows'][0]
            for key in ('donor_rms_normalized', 'donor_z_cosine', 'raw_abs_max', 'tanh_saturation_fraction'):
                self.assertIn(key, row)
            self.assertFalse(model.native.supported_residual.collect_diagnostics)


if __name__ == '__main__':
    unittest.main()
