"""GPU/extension and synthetic baseline checks; no dataset scores are produced."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch


def check_extensions():
    from opencood.utils.box_overlaps import bbox_overlaps
    from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils as iou
    from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_utils as roi

    # Identical, vertical separation, and partial vertical overlap.
    a = torch.tensor([[0, 0, 0, 2, 2, 2, 0]], dtype=torch.float32, device='cuda')
    b = a.repeat(3, 1)
    b[1, 2], b[2, 2] = 4, 1
    bev = iou.boxes_iou_bev(a, b)
    volume = iou.boxes_iou3d_gpu(a, b)
    torch.testing.assert_close(bev, torch.ones_like(bev))
    torch.testing.assert_close(volume, torch.tensor([[1, 0, 1 / 3]], device='cuda'))
    torch.testing.assert_close(iou.boxes_bev_iou_cpu(a.cpu(), b.cpu()), bev.cpu())
    rotated = a.clone()
    rotated[:, 3:5] = torch.tensor([4, 2], device='cuda')
    perpendicular = rotated.clone()
    perpendicular[:, 6] = np.pi / 2
    torch.testing.assert_close(iou.boxes_iou_bev(rotated, perpendicular),
                               torch.tensor([[1 / 3]], device='cuda'))
    boxes = a.repeat(3, 1)
    boxes[2, 0] = 10
    keep, _ = iou.nms_gpu(boxes, torch.tensor([0.9, 0.8, 0.7], device='cuda'), 0.15)
    assert keep.tolist() == [0, 2], keep
    points = torch.tensor([[[0, 0, 0], [5, 5, 5]]], dtype=torch.float32, device='cuda')
    assert roi.points_in_boxes_gpu(points, a[None]).tolist() == [[0, -1]]
    assert roi.points_in_boxes_cpu(points[0].cpu(), a.cpu()).tolist() == [[1, 0]]
    square = np.array([[0, 0, 2, 2]], dtype=np.float32)
    np.testing.assert_allclose(bbox_overlaps(square, square), [[1]])

    from opencood.pcdet_utils.pointnet2.pointnet2_batch import pointnet2_batch_cuda
    from opencood.pcdet_utils.pointnet2.pointnet2_stack import pointnet2_stack_cuda
    assert pointnet2_batch_cuda is not None and pointnet2_stack_cuda is not None
    return {'bev_iou': bev.tolist(), 'iou3d': volume.tolist(),
            'rotated_bev_iou': 1 / 3, 'nms_keep': keep.tolist(),
            'points_in_boxes': [0, -1], 'pointnet2_extensions': 'imported'}


def check_spconv():
    import spconv as spconv_package
    import spconv.pytorch as spconv
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor

    params = dict(core_method='SpVoxelPreprocessor', cav_lidar_range=[0., 0., 0., 4., 4., 4.],
        args=dict(voxel_size=[1., 1., 4.], max_points_per_voxel=2, max_voxel_train=4, max_voxel_test=4))
    preprocessor = SpVoxelPreprocessor(params, train=False)
    points = np.array([[.1, .1, .1, 1.], [.2, .2, .1, 2.], [1.1, .1, .1, 3.]], dtype=np.float32)
    voxels = {k: v.copy() for k, v in preprocessor.preprocess(points).items()}
    np.testing.assert_array_equal(voxels['voxel_coords'], [[0, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(voxels['voxel_num_points'], [2, 1])
    np.testing.assert_array_equal(voxels['voxel_features'][0], points[:2])
    np.testing.assert_array_equal(voxels['voxel_features'][1, 0], points[2])
    batch = preprocessor.collate_batch([voxels])
    np.testing.assert_array_equal(batch['voxel_coords'].numpy(), [[0, 0, 0, 0], [0, 0, 0, 1]])

    features = torch.randn(3, 4, device='cuda', requires_grad=True)
    indices = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2], [0, 2, 2, 2]],
                           dtype=torch.int32, device='cuda')
    sparse = spconv.SparseConvTensor(features, indices, [4, 4, 4], 1)
    layer = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False).cuda()
    result = layer(sparse).features
    result.square().mean().backward()
    assert result.shape == (3, 8) and torch.isfinite(result).all()
    assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    return {'version': getattr(spconv_package, '__version__', None), 'voxelization_known_answer': True,
            'voxel_collate': True, 'shape': list(result.shape), 'backward': True}


def _l4dr_operator_case(device):
    """Small unique-distance inputs for the operators used by L4DR FAD."""
    from opencood.project.l4dr import _farthest, _ball
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [7., 0., 0.]], device=device)
    queries = xyz.new_tensor([[.25, 0., 0.], [5.25, 0., 0.]])
    fps = _farthest(xyz, 4)
    ball = _ball(xyz, xyz[[0, 2]], 2., 4)
    if xyz.is_cuda:
        from opencood.pcdet_utils.pointnet2.pointnet2_batch.pointnet2_utils import three_nn
        distance, index = three_nn(queries[None].contiguous(), xyz[None].contiguous())
        distance, index = distance[0], index[0].long()
    else:
        distance, index = torch.cdist(queries[None], xyz[None],
            compute_mode='donot_use_mm_for_euclid_dist')[0].topk(3, largest=False, sorted=True)
    # Same gather/weight/reduction and feature autograd as FAD propagation;
    # L4DR does not use the extension's three_interpolate kernel.
    features = xyz.new_tensor([[1., 2.], [3., 5.], [7., 11.], [13., 17.]], requires_grad=True)
    weight = (distance + 1e-8).reciprocal()
    weight = weight / weight.sum(-1, keepdim=True)
    interpolated = (features[index] * weight[..., None]).sum(1)
    interpolated.square().sum().backward()
    return {k: v.detach().cpu() for k, v in dict(fps=fps, ball=ball, nn_index=index,
        nn_distance=distance, interpolated=interpolated, feature_gradient=features.grad).items()}


def check_l4dr_cuda():
    """Execute L4DR's actual batch CUDA operators; no optimizer or dataset."""
    from opencood.project.l4dr import pointnet2_identity
    cpu, cuda = _l4dr_operator_case('cpu'), _l4dr_operator_case('cuda')
    expected = dict(fps=torch.tensor([0, 3, 2, 1]),
        ball=torch.tensor([[0, 1, 0, 0], [2, 2, 2, 2]]),
        nn_index=torch.tensor([[0, 1, 2], [3, 2, 1]]),
        nn_distance=torch.tensor([[.25, .75, 2.75], [1.75, 2.25, 4.25]]))
    for key, answer in expected.items():
        torch.testing.assert_close(cpu[key], answer, rtol=0, atol=0)
        torch.testing.assert_close(cuda[key], answer, rtol=0, atol=0)
    for key in ('interpolated', 'feature_gradient'):
        # FP32 reciprocal/sum/backward use device-specific arithmetic order.
        torch.testing.assert_close(cuda[key], cpu[key], rtol=2e-6, atol=1e-7)
    identity = pointnet2_identity(include_path=True)
    assert identity['binary']['status'] == 'loaded'
    return dict(pointnet2_identity=identity, fps_indices=cpu['fps'].tolist(),
        ball_indices=cpu['ball'].tolist(), three_nn_indices=cpu['nn_index'].tolist(),
        three_nn_distances=cpu['nn_distance'].tolist(), feature_interpolation_backward=True,
        index_distance_tolerance=dict(rtol=0, atol=0), feature_tolerance=dict(rtol=2e-6, atol=1e-7),
        scope='four-point known answers and feature gradients; not full FAD/model or training equivalence')


def check_model(checkpoint_path):
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(REPO / 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml'))
    # Small spatial crop, unchanged upstream channels/layers; not an experiment config.
    bounds = [-6.4, -6.4, -3, 6.4, 6.4, 1]
    hypes['preprocess']['cav_lidar_range'] = bounds
    hypes['postprocess']['anchor_args']['cav_lidar_range'] = bounds
    hypes['model']['args']['lidar_range'] = bounds
    hypes = yaml_utils.load_point_pillar_params(hypes)
    preprocessor = SpVoxelPreprocessor(hypes['preprocess'], train=True)
    batch = {'record_len': torch.tensor([2, 1], device='cuda')}
    for modality in ('lidar', 'radar'):
        agents = []
        for _ in range(3):
            points = np.random.uniform([-6, -6, -2, 0], [6, 6, 0.8, 1], (256, 4)).astype(np.float32)
            # cumm returns reusable views; freeze each agent before the next call.
            agents.append({k: v.copy() for k, v in preprocessor.preprocess(points).items()})
        batch['processed_' + modality] = {
            k: v.cuda() for k, v in preprocessor.collate_batch(agents).items()}
    model = train_utils.create_model(hypes).cuda().train()
    criterion = train_utils.create_loss(hypes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)
    output = model(batch)
    n, anchors, height, width = output['psm'].shape
    positives = torch.zeros(n, height, width, anchors, device='cuda')
    positives[:, height // 2, width // 2, 0] = 1
    target = {'pos_equal_one': positives,
              'targets': torch.zeros(n, height, width, anchors * 7, device='cuda')}
    loss = criterion(output, target)
    assert torch.isfinite(loss), loss
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        expected = model(batch)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'epoch': 0,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
             'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
             'synthetic_only': True}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location='cuda')
    restored_model = train_utils.create_model(hypes).cuda().eval()
    restored_model.load_state_dict(restored['model'])
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.002)
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored_optimizer, [10, 15], gamma=0.1)
    restored_optimizer.load_state_dict(restored['optimizer'])
    restored_scheduler.load_state_dict(restored['scheduler'])
    # RNG state tensors must be on CPU for restoration.
    torch.set_rng_state(restored['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in restored['cuda_rng']])
    np.random.set_state(restored['numpy_rng'])
    random.setstate(restored['python_rng'])
    with torch.no_grad():
        actual = restored_model(batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert restored_optimizer.state and restored_scheduler.last_epoch == scheduler.last_epoch
    for original_state, restored_state in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(original_state[key].cpu(), restored_state[key].cpu(), rtol=0, atol=0)
    return {'source': 'synthetic_points_only', 'record_len': [2, 1],
            'crop_range': bounds, 'loss': loss.item(),
            'outputs': {k: list(v.shape) for k, v in output.items()},
            'forward_backward_optimizer': True, 'checkpoint_roundtrip': True,
            'checkpoint': str(checkpoint_path),
            'peak_gpu_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO.parent / 'setup/environment_check.json')
    parser.add_argument('--checks', nargs='+', choices=['extensions', 'spconv', 'l4dr', 'baseline'],
                        default=['extensions', 'spconv', 'baseline'],
                        help='use --checks l4dr spconv for minimal checks without optimizer/checkpoint work')
    args = parser.parse_args(argv)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable,
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              'requested_checks': args.checks, 'baseline_optimizer_check_requested': 'baseline' in args.checks,
              'dataset_read': False, 'checks': {}, 'status': 'running'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is required for these checks; CPU checks cannot validate CUDA operators')
        modules = ('spconv', 'open3d', 'pypcd.pypcd', 'numba', 'mmcv', 'timm',
                   'opencood.tools.train_utils', 'opencood.data_utils.datasets')
        for module in modules if 'baseline' in args.checks else ():
            imported = importlib.import_module(module)
            report['checks'][module] = getattr(imported, '__version__', 'imported')
            print('PASS import', module, flush=True)
        if 'extensions' in args.checks:
            report['checks']['cuda_extensions'] = check_extensions()
            print('PASS CUDA IoU / NMS / points-in-boxes', flush=True)
        if 'l4dr' in args.checks:
            report['checks']['l4dr_cuda'] = check_l4dr_cuda()
            print('PASS L4DR FPS / ball-query / three_nn and feature backward', flush=True)
        if 'spconv' in args.checks:
            report['checks']['spconv_gpu'] = check_spconv()
            print('PASS spconv voxelization/collate and GPU forward/backward', flush=True)
        if 'baseline' in args.checks:
            report['checks']['baseline'] = check_model(args.output.parent / 'synthetic_smoke_checkpoint.pth')
            print('PASS synthetic baseline forward/backward/optimizer/checkpoint', flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
