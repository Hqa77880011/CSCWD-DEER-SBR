"""Train-only mechanism contracts; no AP, weather truth, or GPU dependence."""
import copy
import unittest
import torch
from opencood.models.common_modules.evidence_router import EvidenceRouter
from opencood.models.common_modules.supported_residual import SupportedResidual


def observed(batch=2, height=7, width=7):
    return {name: torch.ones(batch, 1, height, width) for name in (
        'lidar_count', 'radar_count', 'lidar_occupancy', 'radar_occupancy',
        'lidar_valid', 'radar_valid', 'lidar_distance', 'radar_distance')}


class V3MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(913)
        self.lidar = torch.randn(2, 4, 7, 7)
        self.radar = torch.randn_like(self.lidar)
        self.obs = observed()
        self.available = torch.ones(2, 2, dtype=torch.bool)

    def test_amplitude_restores_scale_information_and_preserves_initial_identity(self):
        router = EvidenceRouter(4, 4, amplitude_context='rms')
        left, right, initial = router(self.lidar, self.radar, self.obs, self.available)
        torch.testing.assert_close(left, self.lidar, rtol=0, atol=0)
        torch.testing.assert_close(right, self.radar, rtol=0, atol=0)
        torch.nn.init.normal_(router.route_head.weight)
        _, _, before = router(self.lidar, self.radar, self.obs, self.available)
        _, _, after = router(self.lidar, self.radar * 10, self.obs, self.available)
        self.assertFalse(before['amplitude_descriptor'].requires_grad)
        self.assertFalse(torch.allclose(before['amplitude_descriptor'], after['amplitude_descriptor']))
        self.assertFalse(torch.allclose(before['route_weights'], after['route_weights']))
        self.assertEqual(sum(p.numel() for p in router.amplitude_project.parameters()), 32)

    def test_active_capacity_control_is_amplitude_invariant_and_trainable(self):
        router = EvidenceRouter(4, 4, amplitude_context='normalized_mean')
        rms = EvidenceRouter(4, 4, amplitude_context='rms')
        self.assertEqual(sum(p.numel() for p in router.parameters()), sum(p.numel() for p in rms.parameters()))
        torch.nn.init.normal_(router.route_head.weight)
        left, right, before = router(self.lidar, self.radar, self.obs, self.available)
        _, _, after = router(self.lidar, self.radar * 10, self.obs, self.available)
        torch.testing.assert_close(before['amplitude_descriptor'], after['amplitude_descriptor'])
        torch.testing.assert_close(before['route_weights'], after['route_weights'])
        (left.square().mean() + right.square().mean()).backward()
        self.assertGreater(router.amplitude_project.weight.grad.abs().sum(), 0)

    def test_zero_descriptor_matches_legacy_with_same_seeded_parameters(self):
        torch.manual_seed(10)
        legacy = EvidenceRouter(4, 4)
        torch.manual_seed(10)
        zero = EvidenceRouter(4, 4, amplitude_context='zero')
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(value, zero.state_dict()[name], rtol=0, atol=0)
        torch.nn.init.normal_(legacy.route_head.weight)
        zero.load_state_dict(legacy.state_dict(), strict=False)
        for first, second in zip(legacy(self.lidar, self.radar, self.obs, self.available)[:2],
                                 zero(self.lidar, self.radar, self.obs, self.available)[:2]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_type_head_variants_share_amplitude_initialization_and_preserve_caller_rng(self):
        weights = []
        for type_mode in ('independent', 'shared', 'exclusive'):
            torch.manual_seed(29)
            legacy = EvidenceRouter(4, 4, type_mode=type_mode)
            legacy_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(29)
            candidate = EvidenceRouter(4, 4, type_mode=type_mode, amplitude_context='rms')
            torch.testing.assert_close(torch.random.get_rng_state(), legacy_rng, rtol=0, atol=0)
            for name, value in legacy.state_dict().items():
                torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
            weights.append(candidate.amplitude_project.weight.detach().clone())
        for weight in weights[1:]:
            torch.testing.assert_close(weight, weights[0], rtol=0, atol=0)

    def test_route_decomposition_and_type_gradient_isolation(self):
        router = EvidenceRouter(4, 4, amplitude_context='rms')
        torch.nn.init.normal_(router.route_head.weight)
        left, right, aux = router(self.lidar, self.radar, self.obs, self.available)
        torch.testing.assert_close(aux['generic_route_logits'] + aux['type_route_bias'], aux['final_route_logits'])
        self.assertFalse(aux['generic_route_logits'].requires_grad)
        (left.square().mean() + right.square().mean()).backward()
        self.assertIsNone(router.type_head.weight.grad)
        self.assertGreater(router.amplitude_project.weight.grad.abs().sum(), 0)

    def test_input_only_removes_redundant_multiplier_but_keeps_type_input(self):
        repair = SupportedResidual(4, 4, 2, type_prior_mode='input_only')
        legacy = SupportedResidual(4, 4, 2)
        with torch.no_grad():
            repair.residual_net[-1].weight.normal_(std=.1)
            repair.residual_net[-1].bias.fill_(.1)
        legacy.load_state_dict(copy.deepcopy(repair.state_dict()))
        miss = torch.full((2, 1, 7, 7), .5, requires_grad=True)
        spur = torch.full_like(miss, .5, requires_grad=True)
        out, aux = repair(self.lidar, self.radar, self.obs, self.available, miss, spur)
        _, old = legacy(self.lidar, self.radar, self.obs, self.available, miss, spur)
        torch.testing.assert_close(aux['type_prior'], torch.ones_like(miss))
        torch.testing.assert_close(old['delta'], .625 * aux['delta'])
        out.square().mean().backward()
        self.assertIsNone(miss.grad)
        self.assertIsNone(spur.grad)
        _, different = repair(self.lidar, self.radar, self.obs, self.available,
                              torch.zeros_like(miss), torch.ones_like(spur))
        self.assertFalse(torch.allclose(aux['delta'], different['delta']))

    def test_input_only_support_bound_empty_and_diagnostics(self):
        repair = SupportedResidual(4, 4, 2, type_prior_mode='input_only')
        repair.collect_diagnostics = True
        with torch.no_grad():
            repair.residual_net[-1].bias.fill_(100)
        _, aux = repair(self.lidar, self.radar, self.obs, self.available)
        self.assertTrue((aux['delta'].abs() <= .25 * aux['feature_scale']).all())
        self.assertEqual(float(aux['tanh_saturation_fraction']), 1.)
        self.assertFalse(aux['donor_z_cosine'].requires_grad)
        self.obs['radar_occupancy'].zero_()
        out, empty = repair(self.lidar, self.radar, self.obs, self.available)
        torch.testing.assert_close(out, self.lidar, rtol=0, atol=0)
        self.assertEqual(empty['delta'].count_nonzero(), 0)
        self.assertEqual(empty['donor_rms_normalized'].count_nonzero(), 0)

    def test_mask_skip_requires_already_masked_inputs_and_preserves_gradients(self):
        self.available[1, 1] = False
        router = EvidenceRouter(4, 4, amplitude_context='rms')
        torch.nn.init.normal_(router.route_head.weight)
        masked = copy.deepcopy(router)
        lidar = self.lidar.clone().requires_grad_()
        radar = self.radar.clone().requires_grad_()
        left, right, _ = router(lidar, radar, self.obs, self.available)
        (left.square().sum() + right.square().sum()).backward()
        lidar2, radar2 = self.lidar.clone().requires_grad_(), self.radar.clone().requires_grad_()
        left2, right2, _ = masked(lidar2 * self.available[:, :1, None, None],
                                 radar2 * self.available[:, 1:, None, None], self.obs, self.available,
                                 inputs_masked=True)
        (left2.square().sum() + right2.square().sum()).backward()
        torch.testing.assert_close(left, left2, rtol=0, atol=0)
        torch.testing.assert_close(right, right2, rtol=0, atol=0)
        torch.testing.assert_close(lidar.grad, lidar2.grad, rtol=0, atol=0)
        torch.testing.assert_close(radar.grad, radar2.grad, rtol=0, atol=0)
        for one, two in zip(router.parameters(), masked.parameters()):
            if one.grad is not None:
                torch.testing.assert_close(one.grad, two.grad, rtol=0, atol=0)

    def test_normalized_donor_optimization_preserves_outputs_and_gradients(self):
        ordinary = SupportedResidual(4, 4, 2, type_prior_mode='input_only')
        with torch.no_grad():
            ordinary.residual_net[-1].weight.normal_(std=.1)
        efficient = copy.deepcopy(ordinary)
        efficient.efficient_features = True
        result = []
        for repair in (ordinary, efficient):
            z, radar = self.lidar.clone().requires_grad_(), self.radar.clone().requires_grad_()
            out, aux = repair(z, radar, self.obs, self.available)
            out.square().mean().backward()
            result.append((out, z.grad, radar.grad))
        for one, two in zip(*result):
            torch.testing.assert_close(one, two, rtol=2e-5, atol=1e-7)
        for one, two in zip(ordinary.parameters(), efficient.parameters()):
            torch.testing.assert_close(one.grad, two.grad, rtol=2e-5, atol=1e-7)

    def test_invalid_modes_fail_closed(self):
        with self.assertRaises(ValueError):
            EvidenceRouter(amplitude_context='weather_label')
        with self.assertRaises(ValueError):
            SupportedResidual(type_prior_mode='unbounded')


if __name__ == '__main__':
    unittest.main()
