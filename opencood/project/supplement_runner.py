"""Supplemental factories used by the existing training/export runner."""
from copy import deepcopy
import json
from pathlib import Path

import torch

from opencood.project.supplement_data import V2XRADAR_BOUNDS, KRADAR_BOUNDS


def configure_hypes(config, hypes):
    hypes = deepcopy(hypes)
    dataset = config['experiment']['dataset_id']
    bounds = (V2XRADAR_BOUNDS if dataset == 'v2xradar' else KRADAR_BOUNDS if dataset == 'kradar'
              else [-140.8, -40., -3., 140.8, 40., 1.])
    voxel = [.375, .375, 9.6] if dataset == 'kradar' else [.4, .4, bounds[5] - bounds[2]]
    hypes['preprocess']['cav_lidar_range'] = list(bounds)
    hypes['preprocess']['args']['voxel_size'] = voxel
    args = hypes['model']['args']
    args.update(lidar_range=list(bounds), voxel_size=voxel, proj_first=True)
    if 'where2comm_fusion' in args:
        args['where2comm_fusion']['voxel_size'] = voxel
    if config['experiment']['method'] == 'mdd':
        # The author constructor uses these constants, ignoring the different
        # values in its YAML. Reflect the unchanged runtime in resolved hypes.
        args['mdd_block']['diffusion'].update(beta_schedule='linear', beta_start=.005, beta_end=.05)
    hypes['postprocess']['anchor_args']['cav_lidar_range'] = list(bounds)
    hypes['postprocess']['anchor_args']['feature_stride'] = (4 if config['experiment']['method'] in ('cobevt', 'where2comm') else 2)
    return hypes


def build_dataset(config, hypes, split, *, training=False, augment=False, weather='normal'):
    if config['experiment']['dataset_id'] != 'v2xr':
        from opencood.project.supplement_data import SupplementalDataset
        return SupplementalDataset(config, hypes, split, training=training, augment=augment, weather='natural')
    from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
    d, e, r = (config[k] for k in ('data', 'experiment', 'resources'))
    return ProjectV2XRDataset(hypes, root=d['root'], split=split, training=training, track=e['track'],
        seed=e['seed'], weather=weather, paired=False, augment=augment,
        cache_bytes=int(r['raw_cache_gib_per_worker'] * 1024 ** 3),
        supervision_source=d['supervision_source'] if training else 'none',
        gt_manifest=str(Path(d['manifest_dir']) / 'eval_gt_manifest.json'),
        controlled_corruption=d['controlled_corruption'], experiment_config=config,
        reference=e['method'] == 'mdd' and training)


def blockers(config):
    d, e = config['data'], config['experiment']
    issues = []
    if not Path(d['root']).is_dir():
        issues.append('data root missing: ' + d['root'])
    path = Path(d['manifest_dir']) / ('eval_gt_manifest.json' if e['dataset_id'] == 'v2xr' else 'supplement_manifest.json')
    if not path.is_file():
        issues.append('prepared immutable GT/split manifest missing: ' + str(path))
    elif e['dataset_id'] != 'v2xr':
        manifest = json.loads(path.read_text(encoding='utf8'))
        if manifest['dataset'] != e['dataset_id']:
            issues.append('manifest dataset differs from configuration')
        for name in manifest['classes']:
            if not any(g['class_name'] == name for s in manifest['samples'].values() if s['split'] == 'validate' for g in s['gt']):
                issues.append('development holdout has no GT for required class: ' + name)
        if d['clean_only'] or (e['method'] == 'deer_sbr' and e['variant'] not in ('S0', 'B')):
            if not any(s['split'] == 'train' and s['verified_clear'] for s in manifest['samples'].values()):
                issues.append('no verified clear train sequences for trustworthy type/MDD reference supervision; prepare with reviewed --clean-sequences')
        if e['dataset_id'] == 'kradar':
            for key in ('radar_root', 'label_root'):
                if not d[key] or not Path(d[key]).is_dir():
                    issues.append('official K-Radar ' + key + ' missing')
    if config['evaluation']['intervention'] != 'none':
        issues.append('fixed-weight signal interventions cannot be trained')
    return issues


def select_device(config, local_rank=0):
    name = config['resources'].get('device', 'auto')
    if name == 'auto':
        name = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError(f'requested device {name} is unavailable; choose cpu explicitly for local functional checks')
    return device


def scheduler_for(optimizer, config, updates_per_epoch=None):
    t = config['training']
    if t['scheduler'] == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=t['milestones'], gamma=t['gamma'])
    if t['scheduler'] == 'cosine_update':
        if updates_per_epoch is None or updates_per_epoch <= 0:
            raise ValueError('native K-Radar cosine needs the actual complete updates per epoch')
        # Author cfg TYPE_TOTAL_ITER='every': T_max is one epoch's updates,
        # with scheduler.step() after each optimizer update, without a reset.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=updates_per_epoch, eta_min=t['min_lr'])
    import math
    def rate(epoch):
        if epoch < t['warmup_epochs']:
            return (t['warmup_lr'] + (t['lr'] - t['warmup_lr']) * epoch / max(1, t['warmup_epochs'])) / t['lr']
        fraction = (epoch - t['warmup_epochs']) / (t['epochs'] - t['warmup_epochs'])
        return (t['min_lr'] + .5 * (t['lr'] - t['min_lr']) * (1 + math.cos(math.pi * fraction))) / t['lr']
    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate)


def environment_report(native_v2xr=False, device='auto'):
    import importlib
    import platform
    names = ['numpy', 'yaml', 'torch', 'scipy', 'shapely', 'numba', 'einops', 'timm', 'pyquaternion']
    if native_v2xr:
        names += ['cumm.tensorview', 'spconv', 'open3d', 'pypcd.pypcd',
                  'opencood.pcdet_utils.pointnet2.pointnet2_batch.pointnet2_utils',
                  'opencood.data_utils.datasets.project_v2xr_dataset']
    rows = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError) as error:
            rows[name] = dict(status='missing_or_unloadable', reason=str(error))
        else:
            rows[name] = dict(status='imported', version=getattr(module, '__version__', None))
    cuda = torch.cuda.is_available()
    from opencood.project.l4dr import pointnet2_identity
    extension = pointnet2_identity(include_path=True)
    ready = all(r['status'] == 'imported' for r in rows.values()) and (not device.startswith('cuda') or cuda)
    return dict(schema='supplement_environment_v1', status='ready' if ready else 'blocked',
        python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        cuda_available=cuda, requested_device=device, modules=rows, data_read=False,
        l4dr_pointnet2=extension,
        weights_downloaded=False, compiled_extensions_executed=False,
        note='import readiness only; run the explicit synthetic smoke and server data check next')
