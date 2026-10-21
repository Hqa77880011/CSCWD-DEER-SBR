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
    import spconv.pytorch as spconv
    features = torch.randn(3, 4, device='cuda', requires_grad=True)
    indices = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2], [0, 2, 2, 2]],
                           dtype=torch.int32, device='cuda')
    sparse = spconv.SparseConvTensor(features, indices, [4, 4, 4], 1)
    layer = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False).cuda()
    result = layer(sparse).features
    result.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    return {'shape': list(result.shape), 'backward': True}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO.parent / 'setup/environment_check.json')
    args = parser.parse_args()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable,
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'checks': {}, 'status': 'running'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for module in ('spconv', 'open3d', 'pypcd.pypcd', 'numba', 'mmcv', 'timm',
                       'opencood.tools.train_utils', 'opencood.data_utils.datasets'):
            imported = importlib.import_module(module)
            report['checks'][module] = getattr(imported, '__version__', 'imported')
            print('PASS import', module, flush=True)
        report['checks']['cuda_extensions'] = check_extensions()
        print('PASS CUDA IoU / NMS / points-in-boxes', flush=True)
        report['checks']['spconv_gpu'] = check_spconv()
        print('PASS spconv GPU forward/backward', flush=True)
        report['checks']['baseline'] = check_model(args.output.parent / 'synthetic_smoke_checkpoint.pth')
        report['status'] = 'passed'
        print('PASS synthetic baseline forward/backward/optimizer/checkpoint', flush=True)
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
