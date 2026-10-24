"""Known answers and end-to-end CPU checks for the complete L4DR adaptation."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from opencood.project.l4dr import foreground_labels, voxelize_radar, BidirectionalMME
from opencood.project.supplement_data import build_real_manifest, SupplementalDataset
from opencood.project.supplement_models import SupplementalModel, SupplementalLoss
from opencood.project.training import select_model_inputs
from project_tools.run_experiments import read_hypes
from project_tools.smoke_supplement import smoke_one
from scripts.supplement import recipe
from tests.test_supplement import kradar_fixture


class L4DRSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_foreground_ignore_shell_and_no_GT_agent(self):
        xyz = torch.tensor([[0., 0., 0., 0.], [0., 1.1, 0., 0.], [0., 2., 0., 0.], [1., 0., 0., 0.]])
        gt = [torch.tensor([[0., 0., 0., 2., 2., 2., .3]]), torch.empty(0, 7)]
        self.assertEqual(foreground_labels(xyz, gt).tolist(), [1, -1, 0, 0])

    def test_post_FAD_voxel_caps_keep_points_and_detach_detector_input(self):
        hypes = dict(cav_lidar_range=[0., 0., 0., 4., 4., 4.], args=dict(voxel_size=[1., 1., 4.],
            max_points_per_voxel=2, max_voxel_train=2, max_voxel_test=4))
        points = torch.tensor([[1.1, 1.1, 1., 1.], [.1, .1, 1., 2.], [1.2, 1.2, 1., 3.],
                               [1.3, 1.3, 1., 4.], [2.1, 2.1, 1., 5.]])
        scores = torch.full((5, 1), .5, requires_grad=True)
        value = voxelize_radar(points, scores, 2, hypes, True)
        self.assertEqual(value['voxel_num_points'].tolist(), [2, 1])
        self.assertEqual(value['voxel_coords'].tolist(), [[2, 0, 1, 1], [2, 0, 0, 0]])
        self.assertEqual(value['voxel_features'][0, :, 3].tolist(), [1., 3.])
        # The author's MMESparseProcessor detaches before hard voxelization.
        # Detection must not become a second training objective for FAD scores.
        self.assertFalse(value['voxel_features'].requires_grad)
        self.assertIsNone(scores.grad)

    def test_bidirectional_MME_propagates_scalar_and_cross_offsets(self):
        args = read_hypes(recipe(method='l4dr', variant='S0'))['model']['args']
        encoder = BidirectionalMME(args).eval()
        coord = torch.tensor([[0, 0, 2, 3]])
        lidar = dict(voxel_features=torch.tensor([[[1.1, 1., 0., 7.]]]),
                     voxel_coords=coord, voxel_num_points=torch.ones(1, dtype=torch.long))
        radar = dict(voxel_features=torch.tensor([[[1., 1., 0., 5., .8]]]),
                     voxel_coords=coord, voxel_num_points=torch.ones(1, dtype=torch.long))
        captured = {}
        hooks = [layer.register_forward_pre_hook(lambda module, args, key=key: captured.update({key: args[0].detach()}))
                 for key, layer in (('lidar', encoder.lidar), ('radar', encoder.radar))]
        encoder(lidar, radar)
        for hook in hooks:
            hook.remove()
        torch.testing.assert_close(captured['lidar'][0, 0, 10:13], torch.tensor([.1, 0., 0.]))
        torch.testing.assert_close(captured['lidar'][0, 0, 13:], torch.tensor([5., .8]))
        self.assertEqual(captured['radar'][0, 0, 3], 7.)
        torch.testing.assert_close(captured['radar'][0, 0, 4:7], torch.tensor([-.1, 0., 0.]))

    def test_FAD_interpolation_uses_native_additive_epsilon(self):
        from unittest.mock import patch
        from opencood.project.l4dr import ForegroundDenoising
        torch.manual_seed(17)
        model = ForegroundDenoising().eval()
        points = torch.tensor([[0., 0., 0., .2], [.2, .3, .4, .7],
                               [.6, .7, .8, 1.], [1.2, 1.4, 1.6, .3]])
        captured = {}
        hooks = [model.fp[1].register_forward_hook(
            lambda module, args, output: captured.update(source=output.detach().clone())),
            model.fp[0].register_forward_pre_hook(
                lambda module, args: captured.update(interpolated=args[0].detach().clone()))]
        def near_distances(query, source, **kwargs):
            distances = query.new_ones((1, query.shape[1], source.shape[1]))
            distances[..., :3] = query.new_tensor([0., 2e-8, 4e-8])
            return distances
        with patch('opencood.project.l4dr.torch.cdist', side_effect=near_distances), torch.no_grad():
            model(points)
        for hook in hooks:
            hook.remove()
        # Native 1/(d+1e-8) gives the analytic ratio 15:5:3. A clamp
        # would give 4:2:1 and fails this small-distance boundary case.
        expected = (captured['source'][..., :3] * torch.tensor([15., 5., 3.]) / 23).sum(-1)
        actual = captured['interpolated'][:, :256]
        torch.testing.assert_close(actual, expected[..., None].expand_as(actual), rtol=2e-6, atol=1e-7)

    def test_kradar_format_forward_FAD_loss_backward_and_label_free_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, radar = kradar_fixture(root)
            manifest = build_real_manifest('kradar', root, ['2'], radar_root=radar, label_root=labels,
                train_split=root / 'train.txt', test_split=root / 'test.txt')
            (root / 'supplement_manifest.json').write_text(json.dumps(manifest))
            c = recipe('kradar', method='l4dr', variant='S0', data_root=root, radar_root=radar, label_root=labels)
            hypes = read_hypes(c)
            dataset = SupplementalDataset(c, hypes, 'train', training=True)
            batch = dataset.collate_batch([dataset[0]])
            inputs = select_model_inputs(batch['ego'])
            self.assertNotIn('fad_boxes', inputs)
            self.assertEqual(len(inputs['raw_radar']), 1)
            model = SupplementalModel(c, hypes).train()
            # MSGF must use the inter-modal branch, not each modality's own map.
            fused, gate, hooks = {}, {}, []
            for i in range(len(model.native.backbone.blocks)):
                hooks.append(model.native.backbone.blocks[i].register_forward_hook(
                    lambda module, args, out, i=i: fused.update({i: out.data_ptr()})))
                hooks.append(model.native.backbone.Conv_LG[i].register_forward_pre_hook(
                    lambda module, args, i=i: gate.update({i: args[0].data_ptr()})))
            output = model(inputs)
            self.assertEqual(fused, gate)
            for hook in hooks:
                hook.remove()
            loss_fn = SupplementalLoss({**hypes['loss']['args'], 'training_line': 'C'}, 'l4dr')
            detection = loss_fn.detection_loss(output, batch['ego']['label_dict'])
            detection.backward(retain_graph=True)
            self.assertTrue(all(p.grad is None for p in model.l4dr_frontend.fad.parameters()))
            self.assertTrue(any(p.grad is not None for p in model.l4dr_frontend.mme.parameters()))
            model.zero_grad(set_to_none=True)
            loss = loss_fn(output, batch['ego']['label_dict'], batch['train_targets'])
            self.assertGreater(float(loss_fn.loss_dict['fad_foreground_loss']), 0)
            loss.backward()
            self.assertGreater(float(model.l4dr_frontend.fad.head[-1].weight.grad.abs().sum()), 0)
            model.eval()
            with torch.no_grad():
                prediction = model(inputs)
            self.assertNotIn('fad_logits', prediction)
            self.assertEqual(set(prediction['class_outputs']), {'Sedan', 'Bus or Truck'})
            self.assertTrue(torch.isfinite(prediction['rm']).all())

    def test_empty_radar_keeps_one_real_agent(self):
        result = smoke_one(recipe(method='l4dr', variant='S0'), single_agent=True, empty_radar=True)
        self.assertTrue(result['finite'])

    def test_native_KRadar_recipe_retains_AdamW_and_35_epoch_budget(self):
        config = recipe('kradar', method='l4dr', variant='S0', native=True)
        self.assertEqual(config['training']['optimizer'], 'adamw')
        self.assertEqual(config['training']['epochs'], 35)
        self.assertEqual(config['training']['effective_batch_size'], 3)
        self.assertEqual(config['training']['adam_eps'], 1e-8)
        self.assertEqual(config['training']['scheduler'], 'cosine_update')


if __name__ == '__main__':
    unittest.main()
