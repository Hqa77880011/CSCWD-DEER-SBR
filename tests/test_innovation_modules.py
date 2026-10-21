"""Mechanism invariants; CPU-only and no dataset/test-weather dependence."""
import copy
import unittest
import torch
from opencood.models.common_modules.evidence_router import EvidenceRouter, observation_rms
from opencood.models.common_modules.supported_residual import SupportedResidual
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.point_pillar_deer_sbr import PointPillarDeerSbr, select_model_inputs
from opencood.loss.project_loss import ProjectLoss


def observations(batch=3, height=8, width=8):
    shape = (batch, 1, height, width)
    return {name: torch.ones(shape) for name in (
        'lidar_count', 'radar_count', 'lidar_occupancy', 'radar_occupancy',
        'lidar_valid', 'radar_valid', 'lidar_distance', 'radar_distance')}


def model_args(a=False, b=False):
    return {
        'pillar_vfe': {'use_norm': True, 'with_distance': False,
                       'use_absolute_xyz': True, 'num_filters': [4]},
        'voxel_size': [1, 1, 4], 'lidar_range': [0, 0, -2, 8, 8, 2],
        'point_pillar_scatter': {'grid_size': [8, 8, 1], 'num_features': 4},
        'base_bev_backbone': {'layer_nums': [0], 'layer_strides': [1],
                             'num_filters': [8], 'upsample_strides': [1],
                             'num_upsample_filter': [8]},
        'anchor_number': 2, 'anchor_num': 2, 'shared_init_seed': 17,
        'innovation': {'use_a': a, 'use_b': b, 'hidden_channels': 4,
                       'attention_channels': 2, 'support_radius_m': 2.},
    }


def model_inputs(empty=False):
    def sparse():
        count = 0 if empty else 3
        features = torch.zeros(count, 4, 4)
        if count:
            features[:, :2] = torch.tensor([1.5, 2.5, 0., 1.])
        coords = torch.tensor([[b, 0, 2, 1] for b in range(count)], dtype=torch.int64).reshape(-1, 4)
        return {'voxel_features': features, 'voxel_coords': coords,
                'voxel_num_points': torch.full((count,), 2, dtype=torch.int64)}
    obs = observations()
    if empty:
        for key in ('lidar_count', 'radar_count', 'lidar_occupancy', 'radar_occupancy'):
            obs[key].zero_()
    return {'processed_lidar': sparse(), 'processed_radar': sparse(),
            'record_len': torch.tensor([2, 1]), 'observation_maps': obs,
            'modality_valid': torch.ones(3, 2, dtype=torch.bool) if not empty else torch.zeros(3, 2, dtype=torch.bool)}


class InnovationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)

    def test_router_equal_initialization_and_frame_masks(self):
        router = EvidenceRouter(4, 4)
        lidar, radar = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        availability = torch.tensor([[1, 1], [0, 1], [0, 0]], dtype=torch.bool)
        left, right, aux = router(lidar, radar, observations(), availability)
        torch.testing.assert_close(aux['route_weights'][0], torch.full((2, 8, 8), .5))
        torch.testing.assert_close(left[0], lidar[0])
        torch.testing.assert_close(right[0], radar[0])
        self.assertEqual(left[1:].count_nonzero(), 0)
        self.assertEqual(right[2].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(aux['route_weights']).all())
        # Local emptiness does not mask a globally available sensor.
        obs = observations()
        obs['lidar_occupancy'].zero_()
        _, _, empty_cell = router(lidar, radar, obs, torch.ones(3, 2))
        torch.testing.assert_close(empty_cell['route_weights'], torch.full((3, 2, 8, 8), .5))

    def test_type_heads_are_detached_from_detection_and_b(self):
        router = EvidenceRouter(4, 4)
        repair = SupportedResidual(4, 4, 2)
        nn = torch.nn
        nn.init.normal_(router.route_head.weight)
        nn.init.normal_(repair.residual_net[-1].weight)
        lidar, radar = torch.randn(3, 4, 8, 8, requires_grad=True), torch.randn(3, 4, 8, 8)
        obs, available = observations(), torch.ones(3, 2)
        left, right, aux = router(lidar, radar, obs, available)
        z, _ = repair(left + right, radar, obs, available, aux['miss'], aux['spur'])
        z.square().mean().backward()
        self.assertIsNone(router.type_head.weight.grad)
        self.assertIsNone(router.type_head.bias.grad)
        self.assertGreater(router.route_head.weight.grad.abs().sum(), 0)
        self.assertGreater(router.input_project.weight.grad.abs().sum(), 0)
        self.assertGreater(lidar.grad.abs().sum(), 0)
        router.zero_grad(set_to_none=True)
        left, _, aux = router(lidar, radar, obs, available)
        criterion = ProjectLoss({'cls_weight': 1, 'reg': 2})
        aux.update(z_local=left, use_a=True, use_b=False)
        targets = {'miss': torch.ones_like(aux['miss']), 'spur': torch.ones_like(aux['spur']),
                   'miss_valid': torch.ones_like(aux['miss']), 'spur_valid': torch.ones_like(aux['spur'])}
        criterion.auxiliary_loss(aux, targets).backward()
        self.assertGreater(router.type_head.weight.grad.abs().sum(), 0)
        with torch.no_grad():
            router.type_head.bias.fill_(10)
        _, _, aux = router(lidar, radar, obs, available)
        self.assertTrue((aux['miss'] > .99).all() and (aux['spur'] > .99).all())

    def test_saturated_route_retains_both_valid_branches(self):
        router = EvidenceRouter(4, 4)
        with torch.no_grad():
            router.route_head.bias.copy_(torch.tensor([-100., 100.]))
        lidar, radar = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        left, right, aux = router(lidar, radar, observations(), torch.ones(3, 2))
        torch.testing.assert_close(left, .5 * lidar)
        torch.testing.assert_close(right, 1.5 * radar)
        self.assertTrue((aux['route_gains'] >= .5).all())
        self.assertTrue((aux['route_gains'] <= 1.5).all())

    def test_control_normalization_ignores_modality_amplitude(self):
        router = EvidenceRouter(4, 4)
        with torch.no_grad():
            torch.nn.init.normal_(router.route_head.weight)
        lidar, radar = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        _, _, before = router(lidar, radar, observations(), torch.ones(3, 2))
        _, _, after = router(lidar, 100 * radar, observations(), torch.ones(3, 2))
        torch.testing.assert_close(before['route_weights'], after['route_weights'])

    def test_sbr_support_null_bound_and_trainable_zero_init(self):
        repair = SupportedResidual(4, 4, 2)
        z, radar = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        obs, available = observations(), torch.ones(3, 2)
        out, aux = repair(z, radar, obs, available)
        torch.testing.assert_close(out, z, rtol=0, atol=0)
        out.sum().backward()
        self.assertGreater(repair.residual_net[-1].weight.grad.abs().sum(), 0)
        with torch.no_grad():
            repair.residual_net[-1].bias.fill_(100)
        out, aux = repair(z, radar, obs, available)
        self.assertGreater(aux['delta'].abs().sum(), 0)  # +B works without A
        self.assertTrue((aux['delta'].abs() <= .25 * aux['feature_scale']).all())
        obs['radar_occupancy'].zero_()
        out, aux = repair(z, radar, obs, available)
        torch.testing.assert_close(out, z, rtol=0, atol=0)
        torch.testing.assert_close(aux['null_weight'], torch.ones_like(aux['null_weight']))
        obs['radar_occupancy'].fill_(1)
        available[:, 1] = 0
        _, aux = repair(z, radar, obs, available)
        self.assertEqual(aux['delta'].count_nonzero(), 0)

    def test_donor_uses_only_local_measured_candidates(self):
        repair = SupportedResidual(1, 2, 1)
        for layer in (repair.query, repair.key):
            torch.nn.init.zeros_(layer.weight)
        z = torch.zeros(1, 1, 4, 4)
        radar = torch.full_like(z, 100.)  # Nonzero activations are NOT observations.
        radar[:, :, 0, 0] = 2
        occupied = torch.zeros_like(z)
        occupied[:, :, 0, 0] = 1
        donor, support, null = repair.local_donor(z, radar, occupied, torch.ones_like(z), torch.ones(1, 1, 1, 1))
        expected = torch.zeros_like(z)
        expected[:, :, :2, :2] = 1  # Dead Z uses current observed Radar RMS=2.
        expected[:, :, 2, 0] = 1
        expected[:, :, 0, 2] = 1
        torch.testing.assert_close(donor, expected)
        self.assertEqual(support.sum(), 6)
        self.assertEqual(null[0, 0, -1, -1], 1)

    def test_metric_support_scale_matching_and_density_corrected_null(self):
        repair = SupportedResidual(2, 2, 2)
        self.assertEqual(len(repair.offsets), 13)
        for layer in (repair.query, repair.key):
            torch.nn.init.zeros_(layer.weight)
        z = torch.ones(1, 2, 7, 7)
        radar = torch.ones_like(z) * 100
        occupied = torch.zeros(1, 1, 7, 7)
        occupied[:, :, 3, 3] = 1
        available = torch.ones(1, 1, 1, 1)
        donor, support, null = repair.local_donor(z, radar, occupied, torch.ones_like(occupied), available)
        self.assertEqual(support.sum(), 13)
        self.assertEqual(support[0, 0, 1, 1], 0)  # 1.13m diagonal is outside radius.
        self.assertEqual(support[0, 0, 1, 3], 1)  # 0.8m axial neighbour is supported.
        torch.testing.assert_close(donor[support.expand_as(donor).bool()],
                                   torch.full_like(donor[support.expand_as(donor).bool()], .5))
        occupied.fill_(1)
        dense, _, dense_null = repair.local_donor(z, radar, occupied, torch.ones_like(occupied), available)
        torch.testing.assert_close(dense_null, torch.full_like(dense_null, .5))
        torch.testing.assert_close(dense, torch.full_like(dense, .5))
        rescaled, _, _ = repair.local_donor(z, radar * 10, occupied, torch.ones_like(occupied), available)
        torch.testing.assert_close(rescaled, dense)

    def test_zero_initialized_residual_learns_and_opens_donor_gradients(self):
        repair = SupportedResidual(4, 4, 2)
        optimizer = torch.optim.SGD(repair.parameters(), lr=.2)
        z, radar = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        obs, available = observations(), torch.ones(3, 2)
        out, first = repair(z, radar, obs, available)
        self.assertEqual(first['delta'].count_nonzero(), 0)
        (out - (z + .1)).square().mean().backward()
        self.assertGreater(repair.residual_net[-1].weight.grad.abs().sum(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        out, second = repair(z, radar, obs, available)
        self.assertGreater(second['delta'].count_nonzero(), 0)
        (out - (z + .1)).square().mean().backward()
        self.assertGreater(repair.input_project.weight.grad.abs().sum(), 0)
        self.assertGreater(repair.key.weight.grad.abs().sum(), 0)
        self.assertGreater(repair.null_logit.weight.grad.abs().sum(), 0)

    def test_radar_only_dead_z_uses_real_radar_scale_and_learns(self):
        repair = SupportedResidual(4, 4, 2)
        z, radar = torch.zeros(2, 4, 7, 7), torch.full((2, 4, 7, 7), 5.)
        obs = observations(2, 7, 7)
        obs['lidar_occupancy'].zero_()
        available = torch.tensor([[0, 1], [0, 0]])
        optimizer = torch.optim.SGD(repair.parameters(), lr=.5)
        out, aux = repair(z, radar, obs, available)
        self.assertTrue(aux['scale_fallback'][0].item())
        self.assertFalse(aux['scale_fallback'][1].item())
        torch.testing.assert_close(aux['feature_scale'][0], torch.full_like(aux['feature_scale'][0], 5.))
        torch.testing.assert_close(aux['feature_scale'][1], torch.full_like(aux['feature_scale'][1], .001))
        (out[0] - .5).square().mean().backward()
        optimizer.step()
        _, learned = repair(z, radar, obs, available)
        self.assertGreater(learned['delta'][0].abs().mean(), .01)
        self.assertEqual(learned['delta'][1].count_nonzero(), 0)
        # A single live Z channel must keep the Z reference for every channel.
        z[:, :1] = 1
        _, live = repair(z, radar, obs, available)
        self.assertFalse(live['scale_fallback'][0].item())
        self.assertLess(live['feature_scale'][0, 1].item(), .1)

    def test_current_scale_is_detached_masked_bounded_and_stateless(self):
        z = torch.zeros(2, 4, 4, 4, requires_grad=True)
        occupied = torch.zeros(2, 1, 4, 4)
        occupied[0, :, 0, 0] = 1
        with torch.no_grad():
            z[0, :, 0, 0] = torch.tensor([2., 4., 0., 8.])
            z[0, :, 3, 3] = 1000.  # Unobserved features cannot inflate scales.
        scale = observation_rms(z, occupied)
        torch.testing.assert_close(scale[0, :2, 0, 0], torch.tensor([2., 4.]))
        self.assertGreater(scale[0, 2, 0, 0], 0)
        torch.testing.assert_close(scale[1], torch.full_like(scale[1], .001))
        self.assertFalse(scale.requires_grad)
        repair = SupportedResidual(4, 4, 2)
        self.assertFalse(any('scale' in key for key in repair.state_dict()))
        restored = SupportedResidual(4, 4, 2)
        restored.load_state_dict(copy.deepcopy(repair.state_dict()))
        for mode in (True, False):
            restored.train(mode)
            _, aux = restored(z, z, observations(2, 4, 4), torch.ones(2, 2))
            self.assertTrue(torch.isfinite(aux['feature_scale']).all())
        args = model_args(True, True)
        args['innovation']['require_calibrated_scale'] = False
        with self.assertRaises(ValueError):
            PointPillarDeerSbr(args)

    def test_four_variants_share_initial_parameters_and_output(self):
        models = [PointPillarDeerSbr(model_args(a, b)).eval() for a, b in [(False, False), (True, False), (False, True), (True, True)]]
        common = dict(models[0].named_parameters())
        inputs = model_inputs()
        with torch.no_grad():
            baseline = models[0](inputs)
            for model in models:
                params = dict(model.named_parameters())
                for key, value in common.items():
                    torch.testing.assert_close(params[key], value, rtol=0, atol=0)
                result = model(inputs)
                torch.testing.assert_close(result['psm'], baseline['psm'], rtol=0, atol=0)
                torch.testing.assert_close(result['rm'], baseline['rm'], rtol=0, atol=0)
                self.assertNotIn('aux', result)

    def test_stateless_activation_checkpoint_preserves_outputs_gradients_and_bn(self):
        args = model_args(True, True)
        args['innovation']['activation_checkpoint'] = False
        plain = PointPillarDeerSbr(args).train()
        # Exercise all branch gradients beyond exact zero initialization.
        with torch.no_grad():
            torch.nn.init.normal_(plain.evidence_router.route_head.weight, std=.1)
            torch.nn.init.normal_(plain.supported_residual.residual_net[-1].weight, std=.1)
        args['innovation']['activation_checkpoint'] = True
        checked = PointPillarDeerSbr(args).train()
        checked.load_state_dict(copy.deepcopy(plain.state_dict()))
        targets = {'normal_mask': torch.ones(3)}
        outputs = []
        for model in (plain, checked):
            out = model(model_inputs())
            outputs.append(out)
            criterion = ProjectLoss({'cls_weight': 1, 'reg': 2})
            (out['psm'].square().mean() + out['rm'].square().mean() +
             criterion.auxiliary_loss(out['aux'], targets)).backward()
        for key in ('psm', 'rm'):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
        checked_params = dict(checked.named_parameters())
        for name, parameter in plain.named_parameters():
            other = checked_params[name]
            if parameter.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        checked_buffers = dict(checked.named_buffers())
        for name, value in plain.named_buffers():
            torch.testing.assert_close(value, checked_buffers[name], rtol=0, atol=0)

    def test_single_pillar_and_empty_agents_are_retained(self):
        args = model_args()
        vfe = PillarVFE(args['pillar_vfe'], 4, args['voxel_size'], args['lidar_range'])
        scatter = PointPillarScatter(args['point_pillar_scatter'])
        batch = {'voxel_features': torch.ones(1, 1, 4), 'voxel_num_points': torch.ones(1, dtype=torch.int64),
                 'voxel_coords': torch.tensor([[0, 0, 1, 1]]), 'record_len': torch.tensor([2, 1])}
        result = scatter(vfe(batch))
        self.assertEqual(result['spatial_features'].shape, (3, 4, 8, 8))
        self.assertEqual(result['spatial_features'][1:].count_nonzero(), 0)
        model = PointPillarDeerSbr(model_args(True, True)).train()
        out = model(model_inputs(empty=True))
        self.assertEqual(out['psm'].shape[0], 2)
        self.assertEqual(out['aux']['delta'].count_nonzero(), 0)
        (out['psm'].mean() + out['rm'].mean()).backward()
        self.assertTrue(torch.isfinite(out['psm']).all())

    def test_shared_projection_identity_and_original_radar_bypass(self):
        model = PointPillarDeerSbr(model_args()).eval()
        inputs = model_inputs()
        lidar, radar = model._encode(inputs)
        local, aux = model.local_features(inputs)
        torch.testing.assert_close(local, torch.cat((lidar, radar), 1), rtol=0, atol=0)
        torch.testing.assert_close(aux['z0'], lidar, rtol=0, atol=0)
        # Diagnostic shared-stage extraction cannot run the collaborative backbone.
        hook = model.backbone.register_forward_pre_hook(lambda *x: (_ for _ in ()).throw(AssertionError('feature helper invoked backbone')))
        torch.testing.assert_close(model.extract_z0(inputs), lidar)
        hook.remove()
        with self.assertRaises(ValueError):
            model({**inputs, 'train_targets': {}})
        self.assertNotIn('train_targets', select_model_inputs({**inputs, 'train_targets': {}}))

    def test_ignored_anchors_and_empty_auxiliary_masks_have_zero_loss(self):
        criterion = ProjectLoss({'cls_weight': 1, 'reg': 2, 'training_line': 'weather'})
        output = {'psm': torch.randn(1, 2, 1, 1, requires_grad=True),
                  'rm': torch.randn(1, 14, 1, 1, requires_grad=True)}
        labels = {'pos_equal_one': torch.zeros(1, 1, 1, 2), 'neg_equal_one': torch.zeros(1, 1, 1, 2),
                  'targets': torch.full((1, 1, 1, 14), torch.nan)}
        loss = criterion(output, labels)
        self.assertEqual(loss, 0)
        loss.backward()
        self.assertEqual(output['psm'].grad.count_nonzero(), 0)
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        z = torch.zeros(1, 4, 2, 2, requires_grad=True)
        aux = {'use_a': True, 'use_b': True, 'miss_logits': logits, 'spur_logits': logits,
               'z_local': z, 'support': logits + 1, 'delta': z, 'feature_scale': torch.ones(1, 4, 1, 1)}
        target = {'miss': torch.full_like(logits, torch.nan), 'spur': torch.full_like(logits, torch.nan),
                  'miss_valid': torch.zeros_like(logits), 'spur_valid': torch.zeros_like(logits),
                  'deg_valid': torch.zeros_like(logits),
                  'normal_mask': torch.zeros(1)}
        loss = criterion.auxiliary_loss(aux, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss, 0)
        loss.backward()
        self.assertEqual(logits.grad.count_nonzero(), 0)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_old_teacher_target_is_rejected(self):
        criterion = ProjectLoss({'cls_weight': 1, 'reg': 2, 'training_line': 'clean'})
        z = torch.zeros(1, 4, 2, 2, requires_grad=True)
        aux = {'use_a': False, 'use_b': True, 'z_local': z, 'delta': z,
               'support': torch.ones(1, 1, 2, 2), 'feature_scale': torch.ones(1, 4, 1, 1)}
        with self.assertRaises(ValueError):
            criterion.auxiliary_loss(aux, {'clean_z0': torch.ones_like(z), 'deg_valid': aux['support']})

    def test_sup_and_mechanism_variants_forward_backward(self):
        inputs = model_inputs()
        for variant in ('SUP', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'):
            with self.subTest(variant=variant):
                args = model_args()
                args['innovation']['variant'] = variant
                model = PointPillarDeerSbr(args).train()
                output = model(inputs)
                aux = output['aux']
                torch.testing.assert_close(aux['route_weights'], torch.full_like(aux['route_weights'], .5))
                if variant == 'M1':
                    self.assertIs(aux['miss_logits'], aux['spur_logits'])
                if variant == 'M2':
                    self.assertTrue((aux['miss'] + aux['spur'] <= 1).all())
                    torch.testing.assert_close(aux['miss'], torch.full_like(aux['miss'], 1 / 3))
                (output['psm'].square().mean() + output['rm'].square().mean()).backward()
                self.assertIsNone(model.evidence_router.type_head.weight.grad)
                self.assertGreater(model.supported_residual.residual_net[-1].weight.grad.abs().sum(), 0)
        args['innovation']['variant'] = 'typo'
        with self.assertRaises(ValueError):
            PointPillarDeerSbr(args)

    def test_mechanisms_change_only_their_declared_controls(self):
        z, radar = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
        obs, available = observations(), torch.ones(3, 2)
        obs['radar_occupancy'].zero_()
        ab = SupportedResidual(4, 4, 2)
        m3 = SupportedResidual(4, 4, 2, hard_support=False)
        m4 = SupportedResidual(4, 4, 2, bounded=False)
        with torch.no_grad():
            ab.residual_net[-1].bias.fill_(10)
        m3.load_state_dict(ab.state_dict())
        m4.load_state_dict(ab.state_dict())
        _, base = ab(z, radar, obs, available)
        _, no_support = m3(z, radar, obs, available)
        self.assertEqual(base['delta'].count_nonzero(), 0)
        self.assertGreater(no_support['delta'].count_nonzero(), 0)
        self.assertEqual(no_support['support'].count_nonzero(), 0)
        torch.testing.assert_close(no_support['null_weight'], torch.ones_like(no_support['null_weight']))
        obs['radar_occupancy'].fill_(1)
        _, no_bound = m4(z, radar, obs, available)
        self.assertGreater(no_bound['delta'].abs().max(), .25)
        # M6 and SUP must never pass type tensors to the residual module.
        for variant in ('M6', 'SUP'):
            args = model_args()
            args['innovation']['variant'] = variant
            model = PointPillarDeerSbr(args).eval()
            def check_no_type(module, arguments):
                self.assertIsNone(arguments[4])
                self.assertIsNone(arguments[5])
            handle = model.supported_residual.register_forward_pre_hook(check_no_type)
            model(model_inputs())
            handle.remove()

    def test_m5_only_disables_identity_penalty(self):
        criterion = ProjectLoss({'cls_weight': 1, 'reg': 2, 'training_line': 'clean', 'lambda_correction': 0})
        z = torch.ones(1, 4, 2, 2, requires_grad=True)
        aux = {'use_a': False, 'use_b': True, 'z_local': z, 'delta': z,
               'support': torch.ones(1, 1, 2, 2), 'feature_scale': torch.ones(1, 4, 1, 1)}
        targets = {'normal_mask': torch.ones(1)}
        self.assertGreater(criterion.auxiliary_loss(aux, targets), 0)
        aux['disable_identity'] = True
        self.assertEqual(criterion.auxiliary_loss(aux, targets), 0)

    def test_fp16_zero_loss_does_not_sum_and_overflow_full_bev(self):
        criterion = ProjectLoss({'cls_weight': 1, 'reg': 2})
        z = torch.full((1, 4, 8, 8), 1000., dtype=torch.float16, requires_grad=True)
        self.assertTrue(torch.isinf(z.sum()))
        loss = criterion.auxiliary_loss({'z_local': z, 'use_a': False, 'use_b': False})
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_cpu_bfloat16_autocast_forward_backward_is_finite(self):
        model = PointPillarDeerSbr(model_args(True, True)).train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(model_inputs())
            loss = output['psm'].float().square().mean() + output['rm'].float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
