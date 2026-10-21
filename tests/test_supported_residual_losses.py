"""Supported reduction preserves the dense loss's units, weights and gradients."""
import unittest
import torch
from opencood.loss.project_loss import masked_mean, supported_residual_losses


class SupportedResidualLossTests(unittest.TestCase):
    def test_weighted_values_and_gradients_match_dense(self):
        torch.manual_seed(7)
        for empty in (False, True):
            for scale_agents in (1, 3):
                source = torch.randn(3, 5, 4, 6)
                support = torch.rand(3, 1, 4, 6)
                support[support < .65] = 0
                if empty:
                    support.zero_()
                scale = torch.rand(scale_agents, 5, 1, 1) + .1
                normal = torch.tensor([0., .3, 1.])
                dense = source.clone().requires_grad_()
                normalized = dense / scale
                reference = (masked_mean(normalized.square(), support),
                             masked_mean(normalized.abs(), support * normal[:, None, None, None]))
                (reference[0] + .3 * reference[1]).backward()
                sparse = source.clone().requires_grad_()
                actual = supported_residual_losses(sparse, scale, support, normal)
                (actual[0] + .3 * actual[1]).backward()
                torch.testing.assert_close(torch.stack(actual), torch.stack(reference), rtol=2e-6, atol=1e-7)
                torch.testing.assert_close(sparse.grad, dense.grad, rtol=2e-6, atol=1e-7)

    def test_nan_outside_support_is_excluded_before_square(self):
        delta = torch.tensor([[[[float('nan'), 2.]]]], requires_grad=True)
        support = torch.tensor([[[[0., 1.]]]])
        correction, identity = supported_residual_losses(delta, torch.ones(1, 1, 1, 1), support, torch.ones(1))
        self.assertEqual(correction.item(), 4.)
        self.assertEqual(identity.item(), 2.)
        (correction + identity).backward()
        torch.testing.assert_close(delta.grad, torch.tensor([[[[0., 5.]]]]))

    def test_no_normal_and_empty_normal_are_differentiable_zero(self):
        for normal in (None, torch.zeros(2)):
            delta = torch.randn(2, 3, 4, 5, requires_grad=True)
            support = torch.ones(2, 1, 4, 5)
            correction, identity = supported_residual_losses(delta, torch.ones(2, 3, 1, 1), support, normal)
            self.assertEqual(identity.item(), 0.)
            identity.backward()
            self.assertEqual(delta.grad.count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
