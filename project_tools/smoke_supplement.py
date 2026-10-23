#!/usr/bin/env python3
"""Dataset-free forward/backward checks; never report experimental AP."""
from copy import deepcopy
import gc

import torch

from opencood.project.supplement import INTERNAL, METHODS, NATIVE_HYPES, defaults, validate_supplement_config
from opencood.project.supplement_models import SupplementalModel, SupplementalLoss
from project_tools.run_experiments import read_hypes


def synthetic_inputs(hypes, *, single_agent=False, empty_radar=False):
    args = hypes['model']['args']
    height, width = int(args['point_pillar_scatter']['grid_size'][1]), int(args['point_pillar_scatter']['grid_size'][0])
    agents = 1 if single_agent else 3
    def sparse(empty=False):
        coords = [[a, 0, 2 + i, 3 + i] for a in range(agents) for i in range(2)]
        if empty:
            coords = []
        features = torch.zeros(len(coords), 4, 4)
        for i, (a, _, y, x) in enumerate(coords):
            b, v = args['lidar_range'], args['voxel_size']
            features[i, :2, :3] = torch.tensor([b[0] + (x + .5) * v[0], b[1] + (y + .5) * v[1], b[2] + .5 * v[2]])
            features[i, :2, 3] = .5
        return dict(voxel_features=features,
            voxel_coords=torch.tensor(coords, dtype=torch.long).reshape(-1, 4),
            voxel_num_points=torch.full((len(coords),), 2, dtype=torch.long))
    maps = {key: torch.zeros(agents, 1, height, width) for key in (
        'lidar_count', 'radar_count', 'lidar_occupancy', 'radar_occupancy',
        'lidar_valid', 'radar_valid', 'lidar_distance', 'radar_distance')}
    for key in maps:
        if key.endswith('valid'):
            maps[key].fill_(1)
        elif key.startswith('lidar') or not empty_radar:
            maps[key][:, :, 2:4, 3:5] = 1
    return dict(processed_lidar=sparse(), processed_radar=sparse(empty_radar),
        record_len=torch.tensor([1] if single_agent else [2, 1]),
        pairwise_t_matrix=torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1 if single_agent else 2, 5, 5, 1, 1),
        observation_maps=maps, modality_valid=torch.ones(agents, 2, dtype=torch.bool))


def synthetic_labels(output):
    def labels(pred):
        batch, anchors, height, width = pred['psm'].shape
        positive = torch.zeros(batch, height, width, anchors)
        positive[:, height // 2, width // 2, 0] = 1
        return dict(pos_equal_one=positive, neg_equal_one=1 - positive,
                    targets=torch.zeros(batch, height, width, anchors * 7))
    result = labels(output)
    if 'class_outputs' in output:
        result['class_labels'] = {name: labels(pred) for name, pred in output['class_outputs'].items()}
    return result


def smoke_one(config, *, single_agent=False, empty_radar=False):
    torch.set_num_threads(2)
    torch.manual_seed(config['experiment']['seed'])
    hypes = read_hypes(config)
    # Only the synthetic grid extent is reduced. Native layer counts, channel
    # widths, collaboration modules, diffusion steps and losses are retained.
    hypes['model']['args']['point_pillar_scatter']['grid_size'] = [32, 40 if config['experiment']['method'] == 'cobevt' else 32, 1]
    model = SupplementalModel(config, hypes).cpu().train()
    inputs = synthetic_inputs(hypes, single_agent=single_agent, empty_radar=empty_radar)
    if config['experiment']['method'] == 'l4dr':
        sparse = inputs['processed_radar']
        inputs['raw_radar'] = [sparse['voxel_features'][sparse['voxel_coords'][:, 0] == agent][:, :2].reshape(-1, 4)
                               for agent in range(int(inputs['record_len'].sum()))]
    if config['experiment']['method'] == 'mdd':
        inputs['processed_de_lidar'] = deepcopy(inputs['processed_lidar'])
    output = model(inputs)
    targets = {k: torch.zeros_like(inputs['observation_maps']['lidar_count']) for k in ('miss', 'spur', 'deg_valid')}
    for key in ('miss', 'spur'):
        targets[key + '_valid'] = inputs['observation_maps']['lidar_occupancy'].clone()
    targets['normal_mask'] = torch.ones(inputs['modality_valid'].shape[0], 1, 1, 1)
    if config['experiment']['method'] == 'l4dr':
        targets['fad_boxes'] = [torch.tensor([[0., 0., 0., 4., 2., 2., 0.]]) for _ in inputs['raw_radar']]
    criterion = SupplementalLoss({**hypes['loss']['args'], 'training_line': config['experiment']['track'],
                                 'efficient_features': True}, config['experiment']['method'])
    loss = criterion(output, synthetic_labels(output), targets)
    if not torch.isfinite(loss):
        raise AssertionError('nonfinite synthetic loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
        raise AssertionError('synthetic backward produced invalid or entirely absent gradients')
    model.eval()
    with torch.no_grad():
        prediction = model({k: v for k, v in inputs.items() if k != 'processed_de_lidar'}, return_aux=False)
    if prediction['psm'].shape[0] != len(inputs['record_len']) or not torch.isfinite(prediction['rm']).all():
        raise AssertionError('native inference contract failed')
    report = dict(method=config['experiment']['method'], variant=config['experiment']['variant'],
        dataset_head=config['experiment']['dataset_id'], single_agent=single_agent, empty_radar=empty_radar,
        forward=True, backward=True, inference=True, finite=True, synthetic_only=True,
        optimizer_updates=0, experimental_metrics_produced=False)
    del output, prediction, model, loss, gradients, criterion, inputs
    gc.collect()
    return report


def smoke_matrix(config=None):
    rows = []
    if config is not None:
        return [smoke_one(config, single_agent=config['experiment']['dataset_id'] == 'kradar')]
    for variant in INTERNAL:
        c = defaults()
        c['experiment']['variant'] = variant
        c['model'].update(amplitude_context='normalized_mean' if variant == 'N1' else 'rms',
                          type_prior_mode='multiplicative' if variant == 'N2' else 'input_only',
                          null_donor=variant != 'NO_NULL')
        row = smoke_one(validate_supplement_config(c))
        rows.append(row)
        print(f"PASS synthetic forward/backward/inference: deer_sbr/{variant}", flush=True)
    for method in METHODS[1:]:
        c = defaults()
        c['experiment'].update(method=method, variant='S0', comparison='complete_method')
        c['data']['base_hypes'] = NATIVE_HYPES[method]
        c['training']['batchnorm'].update(evaluation_statistics='running', refit_samples=0)
        rows.append(smoke_one(validate_supplement_config(c), single_agent=method == 'mdd', empty_radar=method != 'l4dr'))
        print(f"PASS synthetic native forward/backward/inference: {method}", flush=True)
    c['experiment'].update(method='where2comm', variant='AB', comparison='controlled_module')
    c['data']['base_hypes'] = NATIVE_HYPES['where2comm']
    c['model']['local_variant'] = 'AB'
    rows.append(smoke_one(validate_supplement_config(c)))
    c['experiment'].update(method='deer_sbr', dataset_id='kradar', protocol='kradar_revised_v2', track='C', comparison='local_single_agent')
    c['data']['base_hypes'] = 'cscwd_configs/baseline_full_data_check.yaml'
    c['model']['local_variant'] = 'none'
    c['selection'].update(domains=['natural'], class_name='two_class_macro')
    c['evaluation'].update(kind='natural', agents=1)
    rows.append(smoke_one(validate_supplement_config(c), single_agent=True))
    return rows
