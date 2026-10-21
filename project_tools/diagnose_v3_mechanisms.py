#!/usr/bin/env python3
"""Bounded train-only DEER/SBR mechanism audit. Never selects a model or AP."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from opencood.project.config import load_config, config_hash
from opencood.project.data_geometry import augment_boxes
from opencood.project.training import seed_everything, select_model_inputs, to_device
from opencood.loss.project_loss import masked_bce_logits, masked_mean, supported_residual_losses
from project_tools.run_experiments import build_components, source_tree_hash, data_contract_metadata


def selected_stats(values, mask=None):
    values = values.detach().float()
    if mask is not None:
        values = values.masked_select(mask.bool().expand_as(values))
    else:
        values = values.flatten()
    if not values.numel():
        return {'count': 0, 'mean': None, 'p95': None, 'max': None}
    return {'count': values.numel(), 'mean': float(values.mean()),
            'p95': float(torch.quantile(values, .95)), 'max': float(values.max())}


def cpu_gradient_relation(first, second, names):
    aa = bb = ab = 0.
    for name in names:
        left, right = first[name], second[name]
        if left is not None:
            aa += float(left.detach().double().square().sum())
        if right is not None:
            bb += float(right.detach().double().square().sum())
        if left is not None and right is not None:
            ab += float((left.detach().double() * right.detach().double()).sum())
    return {'detection_l2': math.sqrt(aa), 'auxiliary_l2': math.sqrt(bb),
            'auxiliary_over_detection': math.sqrt(bb / aa) if aa > 0 else None,
            'cosine': ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None,
            'nonzero_both': aa > 0 and bb > 0}


def independent_gradient_probe(model, criterion, batch, device):
    """One new graph per objective; preserve the actual microbatch and BN state.

    Ordinary backward can release saved tensors as it traverses the graph.
    Repeated retain_graph probes defeat that lifecycle and can OOM even when
    ordinary training is stable. No graph or gradient is shared across probes.
    """
    groups = {'router_shared': ('evidence_router.input_project', 'evidence_router.encoder', 'evidence_router.amplitude_project'),
              'router_type': ('evidence_router.type_head',), 'router_route': ('evidence_router.route_head',),
              'lidar_vfe': ('lidar_pillar_vfe',), 'radar_vfe': ('radar_pillar_vfe',),
              'residual': ('supported_residual',), 'shared_projection': ('phi_base',)}
    parameters = {name: parameter for name, parameter in model.named_parameters()
                  if any(name.startswith(prefix) for prefixes in groups.values() for prefix in prefixes)}
    group_names = {name: [key for key in parameters if any(key.startswith(prefix) for prefix in prefixes)]
                   for name, prefixes in groups.items()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    collect_diagnostics = model.supported_residual.collect_diagnostics
    gradients = {}
    values = {}
    model.supported_residual.collect_diagnostics = False
    inputs = to_device(select_model_inputs(batch['ego']), device)
    targets = to_device(batch['train_targets'], device)
    labels = to_device(batch['ego']['label_dict'], device)

    def restore_state():
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)

    try:
        for objective in ('detection', 'type', 'size', 'identity'):
            restore_state()
            if objective == 'detection':
                output = model(inputs, return_aux=False)
                loss = criterion.detection_loss(output, labels)
                del output
            else:
                # The local auxiliary objectives do not need the collaborative
                # backbone. Deleting its unused local output also frees phi_out.
                local, aux = model.local_features(inputs, include_z0=False)
                del local
                if objective == 'type':
                    loss = aux['z_local'][0, 0, 0, 0] * 0
                    for name in ('miss', 'spur'):
                        loss = loss + criterion.lambda_a * masked_bce_logits(
                            aux[name + '_logits'], targets[name], targets[name + '_valid'])
                else:
                    normal = targets.get('normal_mask') if not aux.get('disable_identity', False) else None
                    if criterion.efficient_features:
                        correction, identity = supported_residual_losses(
                            aux['delta'], aux['feature_scale'], aux['support'], normal)
                    else:
                        normalized = aux['delta'].float() / aux['feature_scale'].detach().float().clamp_min(1e-6)
                        correction = masked_mean(normalized.square(), aux['support'])
                        identity = normalized[0, 0, 0, 0] * 0
                        if normal is not None:
                            identity = masked_mean(normalized.abs(), aux['support'] * normal.reshape(-1, 1, 1, 1))
                    loss = criterion.lambda_correction * correction if objective == 'size' else criterion.lambda_id * identity
                    del correction, identity
                    if not criterion.efficient_features:
                        del normalized
                del aux
            values[objective] = float(loss.detach())
            tensors = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True, retain_graph=False)
            gradients[objective] = {name: None if value is None else value.detach().cpu()
                                    for name, value in zip(parameters, tensors)}
            del tensors, loss
        gradients['all_auxiliary'] = {}
        for name in parameters:
            terms = [gradients[objective][name] for objective in ('type', 'size', 'identity')
                     if gradients[objective][name] is not None]
            gradients['all_auxiliary'][name] = sum(terms[1:], terms[0].clone()) if terms else None
        isolated = all(gradients['detection'][name] is None or not gradients['detection'][name].count_nonzero()
                       for name in group_names['router_type'])
        if not isolated:
            raise AssertionError('detection gradient reaches supervised type_head')
        result = {'micro_batch_size': len(batch['ego']['sample_ids']), 'sample_ids': batch['ego']['sample_ids'],
                  'optimizer_updates': 0, 'new_graph_per_objective': True, 'retain_graph': False,
                  'bn_mode': 'unchanged training mode; buffers restored before each objective and on exit',
                  'weighted_objective_losses': values, 'type_task_gradient_isolated': isolated,
                  'detection_vs_weighted_auxiliary_gradients': {
                      objective: {name: cpu_gradient_relation(gradients['detection'], gradients[objective], names)
                                  for name, names in group_names.items()}
                      for objective in ('type', 'size', 'identity', 'all_auxiliary')}}
    finally:
        restore_state()
        model.supported_residual.collect_diagnostics = collect_diagnostics
    result['bn_buffers_restored'] = all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    return result


def target_regions(batch, dataset):
    """GT is used only AFTER model inputs are selected, for detached statistics."""
    gx, gy = dataset.grid
    unions, object_regions = [], []
    for boxes, augmentation, agents in zip(batch['ego']['gt_boxes'], batch['ego']['augmentations'],
                                           batch['ego']['record_len'].tolist()):
        boxes = augment_boxes(np.asarray(boxes), np.asarray(augmentation))
        union = np.zeros(gx.shape, dtype=bool)
        footprints = []
        for box in boxes:
            dx, dy = gx - box[0], gy - box[1]
            cs, sn = np.cos(box[6]), np.sin(box[6])
            xx, yy = dx * cs + dy * sn, -dx * sn + dy * cs
            footprint = (np.abs(xx) <= box[3] / 2) & (np.abs(yy) <= box[4] / 2)
            nearby = (np.abs(xx) <= box[3] / 2 + .8) & (np.abs(yy) <= box[4] / 2 + .8)
            union |= footprint
            footprints.append((footprint, nearby))
        for _ in range(agents):
            unions.append(union)
            object_regions.append(footprints)
    return torch.from_numpy(np.stack(unions)[:, None]), object_regions


def object_support_statistics(support, radar_occupancy, object_regions):
    support, occupied = support.detach().cpu().numpy()[:, 0] > 0, radar_occupancy.detach().cpu().numpy()[:, 0] > 0
    total = nearby = covered = supported_cells = target_cells = in_grid = in_grid_nearby = in_grid_covered = 0
    for agent, regions in enumerate(object_regions):
        for footprint, neighbourhood in regions:
            total += 1
            nearby += int(occupied[agent][neighbourhood].any())
            covered += int(support[agent][footprint].any())
            if footprint.any():
                in_grid += 1
                in_grid_nearby += int(occupied[agent][neighbourhood].any())
                in_grid_covered += int(support[agent][footprint].any())
            supported_cells += int(support[agent][footprint].sum())
            target_cells += int(footprint.sum())
    return {'agent_object_pairs': total,
            'in_grid_agent_object_pairs': in_grid, 'outside_grid_pairs': total - in_grid,
            'pairs_with_real_radar_in_box_plus_08m': nearby,
            'pairs_with_support_in_box': covered,
            'fraction_pairs_real_radar_nearby': nearby / max(total, 1),
            'fraction_pairs_support_in_box': covered / max(total, 1),
            'fraction_in_grid_pairs_real_radar_nearby': in_grid_nearby / max(in_grid, 1),
            'fraction_in_grid_pairs_support_in_box': in_grid_covered / max(in_grid, 1),
            'target_cell_support_fraction': supported_cells / max(target_cells, 1),
            'interpretation': 'Augmented fixed train GT only; all-pair fractions retain GT outside the BEV, in-grid fractions exclude empty footprints. Box-plus-0.8m is a rectangle; support-in-box uses actual circular SBR reach. Never used to construct support or inference inputs.'}


def type_and_sparsity_statistics(aux, targets, observations, range_scale):
    distance = observations['lidar_distance'] * range_scale
    normal = targets['normal_mask'].reshape(-1, 1, 1, 1) > 0
    sparse = normal & (observations['lidar_count'] > 0) & (observations['lidar_count'] <= 1)
    unobserved = normal & (observations['lidar_valid'] > 0) & (observations['lidar_count'] == 0)
    report = {'normal_sparse_observed_cells': int(sparse.sum()),
              'normal_sparse_lidar_gain': selected_stats(aux['route_gains'][:, :1], sparse),
              'normal_unobserved_lidar_gain': selected_stats(aux['route_gains'][:, :1], unobserved),
              'natural_sparse_labels': 'Only uncorrupted Normal, observed count=1, trusted valid type negatives.'}
    for name in ('miss', 'spur'):
        valid, target, prediction = targets[name + '_valid'] > 0, targets[name], aux[name].detach()
        trusted_sparse = sparse & valid & (target == 0)
        report[name] = {'valid_cells': int(valid.sum()),
                        'positive_cells': int((valid & (target > 0)).sum()),
                        'negative_cells': int((valid & (target == 0)).sum()),
                        'natural_sparse_negative_cells': int(trusted_sparse.sum()),
                        'natural_sparse_false_positive_fraction': float(((prediction > .5) & trusted_sparse).sum() / trusted_sparse.sum().clamp_min(1)),
                        'distance_bands_m': []}
        for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 140), (140, float('inf'))):
            band = (distance >= lo) & (distance < hi)
            known = valid & band
            report[name]['distance_bands_m'].append({'lo': lo, 'hi': hi if math.isfinite(hi) else None,
                'grid_cells': int(band.sum()), 'valid_cells': int(known.sum()),
                'positive_cells': int((known & (target > 0)).sum()), 'negative_cells': int((known & (target == 0)).sum())})
    return report


def routing_statistics(aux, router, available):
    initial = (1 / 3 if router.type_mode == 'exclusive' else .5) if router.type_control else 0.
    initial_type_difference = initial * (router.beta_spur + router.beta_miss)
    generic = aux['generic_route_logits'][:, 1:] - aux['generic_route_logits'][:, :1] + initial_type_difference
    typed = aux['type_route_bias'][:, 1:] - aux['type_route_bias'][:, :1] - initial_type_difference
    mask = (available[:, 0] & available[:, 1])[:, None, None, None]
    active = mask & (generic.abs() > 1e-7) & (typed.abs() > 1e-7)
    x, y = generic.masked_select(mask.expand_as(generic)), typed.masked_select(mask.expand_as(typed))
    cosine = float((x * y).sum() / (x.square().sum() * y.square().sum()).sqrt()) if x.numel() and x.square().sum() > 0 and y.square().sum() > 0 else None
    return {'generic_centered_abs': selected_stats(generic.abs(), mask),
            'type_centered_abs': selected_stats(typed.abs(), mask),
            'generic_type_cosine': cosine,
            'opposite_sign_fraction_active': float(((generic * typed < 0) & active).sum() / active.sum().clamp_min(1)),
            'active_cells': int(active.sum()),
            'lidar_gain': selected_stats(aux['route_gains'][:, :1], available[:, :1, None, None]),
            'radar_gain': selected_stats(aux['route_gains'][:, 1:], available[:, 1:, None, None]),
            'interpretation': 'Initialization constants removed; opposite contributions alone do not establish harmful duplicate control.'}


def residual_statistics(aux, target_mask, rho):
    support = aux['support'].detach() > 0
    normalized = aux['delta'].detach().abs() / aux['feature_scale']
    target_support = support & target_mask
    return {'support_fraction': float(support.float().mean()),
            'support_near_targets_fraction': float(target_support.sum() / target_mask.sum().clamp_min(1)),
            'null_supported': selected_stats(aux['null_weight'], support),
            'null_target_supported': selected_stats(aux['null_weight'], target_support),
            'type_prior_supported': selected_stats(aux['type_prior'], support),
            'normalized_abs_delta_supported': selected_stats(normalized, support),
            'normalized_abs_delta_target_supported': selected_stats(normalized, target_support),
            'near_boundary_fraction_supported': float(((normalized >= .99 * rho) & support).sum() / (support.sum().clamp_min(1) * normalized.shape[1])),
            'near_zero_fraction_supported': float(((normalized < 1e-6) & support).sum() / (support.sum().clamp_min(1) * normalized.shape[1])),
            'raw_abs_max': float(aux['raw_abs_max']),
            'tanh_saturation_fraction': float(aux['tanh_saturation_fraction']),
            'feature_scale_min': float(aux['feature_scale'].min()), 'feature_scale_max': float(aux['feature_scale'].max()),
            'scale_fallback_agents': int(aux['scale_fallback'].sum()),
            'donor_rms_normalized_supported': selected_stats(aux['donor_rms_normalized'], support),
            'donor_z_cosine_target_supported': selected_stats(aux['donor_z_cosine'], target_support),
            'donor_match_interpretation': 'Channel cosine is descriptive: separately trained encoders need not share semantic axes. Donor-zero counterfactual tests task dependence.'}


@torch.no_grad()
def contribution_counterfactuals(model, criterion, batch, device):
    model.eval()
    inputs = to_device(select_model_inputs(batch['ego']), device)
    labels = to_device(batch['ego']['label_dict'], device)
    results = {}
    original_a, original_b = model.use_a, model.use_b
    residual = model.supported_residual
    original_donor = residual.local_donor
    try:
        for name in ('baseline', 'A_disabled', 'B_disabled', 'donor_zero'):
            model.use_a, model.use_b = original_a, original_b
            residual.local_donor = original_donor
            if name == 'A_disabled':
                model.use_a = False
            if name == 'B_disabled':
                model.use_b = False
            if name == 'donor_zero':
                def zero_donor(*args, **kwargs):
                    donor, support, null = original_donor(*args, **kwargs)
                    return torch.zeros_like(donor), support, null
                residual.local_donor = zero_donor
            output = model(inputs, return_aux=True)
            results[name] = {'detection_loss': float(criterion.detection_loss(output, labels)),
                             'psm': output['psm'].detach().cpu(), 'rm': output['rm'].detach().cpu()}
            del output
        for name in ('A_disabled', 'B_disabled', 'donor_zero'):
            results[name]['detection_loss_change'] = results[name]['detection_loss'] - results['baseline']['detection_loss']
            for key in ('psm', 'rm'):
                results[name][key + '_max_abs_change'] = float((results[name][key] - results['baseline'][key]).abs().max())
        for row in results.values():
            del row['psm'], row['rm']
    finally:
        model.use_a, model.use_b = original_a, original_b
        residual.local_donor = original_donor
        model.train()
    return {'sample_ids': batch['ego']['sample_ids'], 'results': results,
            'interpretation': 'Fixed short-trained weights, eval BN, same current observations; changes establish contribution/sensitivity, never AP or that a donor is semantically correct.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 2 <= args.steps <= 8:
        parser.error('bounded audit allows 2..8 updates')
    config = load_config(args.config)
    t = config['training']
    if t['precision'] != 'fp32' or config['experiment']['variant'] != 'AB' or t['world_size'] != 1:
        parser.error('requires single GPU, FP32, AB configuration')
    if t['micro_batch_size'] * t['accumulation_steps'] != 8:
        parser.error('fixed diagnostic scope is exactly 8 train samples')
    torch.set_num_threads(config['resources']['cpu_threads'])
    seed_everything(config['experiment']['seed'], True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    dataset, model, criterion = build_components(config)
    model, criterion = model.to(device).train(), criterion.to(device)
    model.supported_residual.collect_diagnostics = True
    micro, accum = t['micro_batch_size'], t['accumulation_steps']
    batches = [dataset.collate_batch([dataset[i] for i in range(k * micro, (k + 1) * micro)]) for k in range(accum)]
    regions = [target_regions(batch, dataset) for batch in batches]
    range_scale = max(float(np.linalg.norm(np.asarray(dataset.bounds[3:5]) - np.asarray(dataset.bounds[:2]))), 1.)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], eps=t['adam_eps'], weight_decay=t['weight_decay'])
    report = {'training_protocol': config['experiment']['training_protocol'], 'formal_experiment_started': False,
              'diagnostic_only': True, 'split': 'train', 'epoch': 0, 'sample_index_range_inclusive': [0, 7],
              'precision': 'fp32', 'teacher_networks': 0, 'updates': args.steps,
              'config_sha256': config_hash(config), 'source_sha256': source_tree_hash(),
              'data_contract': data_contract_metadata(config),
              'sample_ids': [sid for batch in batches for sid in batch['ego']['sample_ids']],
              'model_changes': {'amplitude_context': model.evidence_router.amplitude_context,
                                'type_prior_mode': model.supported_residual.type_prior_mode},
              'steps': [], 'empty_modalities': [], 'gradient_probes': [],
              'interpretation': 'Fixed 8 train samples, 6 updates by default. GT is detached diagnostic metadata only. No validation, test, Snow, AP, or scheme selection.'}
    torch.cuda.reset_peak_memory_stats(device)
    before = independent_gradient_probe(model, criterion, batches[0], device)
    before['completed_optimizer_updates'] = 0
    report['gradient_probes'].append(before)
    print(json.dumps({'gradient_probe': 'before_training', 'passed': before['bn_buffers_restored']}), flush=True)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        row = {'step': step + 1, 'loss': 0., 'detection_loss': 0., 'microbatches': []}
        started = time.perf_counter()
        for index, batch in enumerate(batches):
            inputs = to_device(select_model_inputs(batch['ego']), device)
            labels = to_device(batch['ego']['label_dict'], device)
            targets = to_device(batch['train_targets'], device)
            output = model(inputs, return_aux=True)
            aux = output['aux']
            detection = criterion.detection_loss(output, labels)
            auxiliary = criterion.auxiliary_loss(aux, targets)
            loss = detection + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite diagnostic loss')
            detail = {}
            (loss / accum).backward()
            target_mask = regions[index][0].to(device)
            detail.update(
                routing=routing_statistics(aux, model.evidence_router, inputs['modality_valid']),
                residual=residual_statistics(aux, target_mask, config['model']['rho']),
                type_coverage_and_sparse_behavior=type_and_sparsity_statistics(aux, targets, inputs['observation_maps'], range_scale))
            if step == 0:
                detail['target_radar_support'] = object_support_statistics(aux['support'], inputs['observation_maps']['radar_occupancy'], regions[index][1])
            if (aux['delta'].detach() * (1 - aux['support'])).count_nonzero():
                raise AssertionError('residual outside real support')
            if ((aux['delta'].detach().abs() / aux['feature_scale']) > config['model']['rho'] + 1e-6).any():
                raise AssertionError('residual bound violated')
            row['loss'] += float(loss.detach()) / accum
            row['detection_loss'] += float(detection.detach()) / accum
            row['microbatches'].append(detail)
            del inputs, labels, targets, output, aux, detection, auxiliary, loss, target_mask
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError('nonfinite diagnostic parameter gradient')
        row['gradient_l2'] = {}
        for prefix in ('evidence_router.type_head', 'evidence_router.route_head', 'evidence_router.amplitude_project',
                       'supported_residual.residual_net.3', 'supported_residual.input_project',
                       'supported_residual.query', 'supported_residual.key', 'supported_residual.null_logit'):
            row['gradient_l2'][prefix] = math.sqrt(sum(float(parameter.grad.detach().double().square().sum())
                for name, parameter in model.named_parameters() if name.startswith(prefix) and parameter.grad is not None))
        optimizer.step()
        torch.cuda.synchronize(device)
        row['seconds'] = time.perf_counter() - started
        report['steps'].append(row)
        print(json.dumps({'step': row['step'], 'loss': row['loss'], 'seconds': row['seconds'], 'gradient_l2': row['gradient_l2']}), flush=True)
    after = independent_gradient_probe(model, criterion, batches[0], device)
    after['completed_optimizer_updates'] = args.steps
    report['gradient_probes'].append(after)
    print(json.dumps({'gradient_probe': 'after_training', 'passed': after['bn_buffers_restored']}), flush=True)
    report['fixed_weight_contributions'] = contribution_counterfactuals(model, criterion, batches[0], device)
    del batches, regions
    gc.collect()
    for disabled in ({'radar'}, {'lidar'}, {'lidar', 'radar'}):
        dataset.disabled = disabled
        batch = dataset.collate_batch([dataset[0]])
        optimizer.zero_grad(set_to_none=True)
        inputs = to_device(select_model_inputs(batch['ego']), device)
        output = model(inputs, return_aux=True)
        loss = criterion(output, to_device(batch['ego']['label_dict'], device), to_device(batch['train_targets'], device))
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        zero = bool(output['aux']['delta'].count_nonzero() == 0) if 'radar' in disabled else None
        report['empty_modalities'].append({'disabled': sorted(disabled), 'finite_forward_backward': finite,
            'zero_delta_without_radar': zero, 'gt_count': len(batch['ego']['gt_ids'][0]), 'record_len': batch['ego']['record_len'].tolist()})
        if not finite or zero is False:
            raise AssertionError('empty modality invariant failed')
        del batch, inputs, output, loss
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024 ** 3,
                  type_task_gradient_isolated=True,
                  fixed_batch_loss_decreased=report['steps'][-1]['loss'] < report['steps'][0]['loss'],
                  residual_nonzero=any(m['residual']['normalized_abs_delta_supported']['max'] > 0 for m in report['steps'][-1]['microbatches']),
                  passed=True)
    report['passed'] = report['fixed_batch_loss_decreased'] and report['residual_nonzero']
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key not in ('steps', 'data_contract', 'sample_ids')}), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
