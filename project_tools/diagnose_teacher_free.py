#!/usr/bin/env python3
"""Bounded fixed-train-batch learning/gradient/empty-modality checks; never AP."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from opencood.project.config import load_config, config_hash
from opencood.project.training import seed_everything, select_model_inputs, to_device
from project_tools.run_experiments import build_components, source_tree_hash, data_contract_metadata


def group_norm(model, prefix):
    return sum(float(p.grad.detach().double().square().sum()) for n, p in model.named_parameters()
               if n.startswith(prefix) and p.grad is not None) ** .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO/'cscwd_configs/project_v4_W_AB.yaml'))
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 2 <= args.steps <= 8:
        parser.error('bounded diagnostic allows 2..8 updates')
    config = load_config(args.config)
    if config['training']['precision'] != 'fp32' or config['experiment']['variant'] != 'AB':
        parser.error('diagnostic requires stable FP32 AB')
    if config['training']['world_size'] != 1:
        parser.error('single-GPU diagnostic only')
    torch.set_num_threads(config['resources']['cpu_threads'])
    seed_everything(config['experiment']['seed'], True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    dataset, model, criterion = build_components(config)
    model, criterion = model.to(device), criterion.to(device)
    model.train()
    t = config['training']
    micro, accum = t['micro_batch_size'], t['accumulation_steps']
    # Fixed eight train frames, no validation/test reads or parameter selection.
    batches = [dataset.collate_batch([dataset[i] for i in range(k*micro, (k+1)*micro)]) for k in range(accum)]
    optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], eps=t['adam_eps'], weight_decay=t['weight_decay'])
    report = {'training_protocol': config['experiment']['training_protocol'], 'formal_experiment_started': False,
              'diagnostic_only': True, 'split': 'train', 'precision': 'fp32', 'teacher_networks': 0,
              'config_sha256': config_hash(config), 'source_sha256': source_tree_hash(),
              'data_contract': data_contract_metadata(config), 'steps': [], 'empty_modalities': [],
              'sample_ids': [sid for b in batches for sid in b['ego']['sample_ids']],
              'interpretation': 'Fixed-batch trainability only; no AP, generalization, or production throughput claim.'}
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        row = {'step': step+1, 'loss': 0., 'detection_loss': 0., 'delta_abs_max_normalized': 0.,
               'support_fraction': 0., 'null_mean_supported': 0., 'type_positive_cells': {'miss': 0, 'spur': 0}}
        started = time.perf_counter()
        for batch in batches:
            inputs = to_device(select_model_inputs(batch['ego']), device)
            labels = to_device(batch['ego']['label_dict'], device)
            targets = to_device(batch['train_targets'], device)
            output = model(inputs, return_aux=True)
            detection = criterion.detection_loss(output, labels)
            if step == 0:
                grads = torch.autograd.grad(detection, tuple(model.evidence_router.type_head.parameters()),
                                            allow_unused=True, retain_graph=True)
                if any(g is not None and g.count_nonzero() for g in grads):
                    raise AssertionError('task gradient leaks into supervised type outputs')
                del grads
            auxiliary = criterion.auxiliary_loss(output['aux'], targets)
            loss = detection + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite fixed-batch loss')
            (loss/accum).backward()
            aux = output['aux']
            delta, support = aux['delta'].detach(), aux['support'].detach()
            normalized = delta.abs() / aux['feature_scale']
            if (delta * (1-support)).count_nonzero() or (normalized > config['model']['rho']+1e-6).any():
                raise AssertionError('SBR support/bound violated')
            row['loss'] += float(loss.detach())/accum
            row['detection_loss'] += float(detection.detach())/accum
            row['delta_abs_max_normalized'] = max(row['delta_abs_max_normalized'], float(normalized.max()))
            row['support_fraction'] += float(support.float().mean())/accum
            row['null_mean_supported'] += float((aux['null_weight'].detach()*support).sum()/support.sum().clamp_min(1))/accum
            for kind in ('miss','spur'):
                row['type_positive_cells'][kind] += int(((targets[kind] > 0) & (targets[kind+'_valid'] > 0)).sum())
            del inputs, labels, targets, output, aux, delta, support, normalized, loss, detection, auxiliary
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite fixed-batch gradients')
        row['gradient_l2'] = {group: group_norm(model, group) for group in (
            'evidence_router.type_head', 'evidence_router.route_head',
            'supported_residual.residual_net.3', 'supported_residual.input_project',
            'supported_residual.query', 'supported_residual.key', 'lidar_pillar_vfe', 'radar_pillar_vfe')}
        optimizer.step()
        torch.cuda.synchronize(device)
        row['seconds'] = time.perf_counter()-started
        report['steps'].append(row)
        print(json.dumps(row), flush=True)
    del batches
    gc.collect()
    for disabled in ({'radar'}, {'lidar'}, {'lidar','radar'}):
        dataset.disabled = disabled
        batch = dataset.collate_batch([dataset[0]])
        optimizer.zero_grad(set_to_none=True)
        inputs = to_device(select_model_inputs(batch['ego']), device)
        output = model(inputs, return_aux=True)
        loss = criterion(output, to_device(batch['ego']['label_dict'], device), to_device(batch['train_targets'], device))
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        strict_zero = bool(output['aux']['delta'].count_nonzero() == 0) if 'radar' in disabled else None
        row = {'disabled': sorted(disabled), 'finite_forward_backward': finite,
               'zero_delta_without_radar': strict_zero, 'gt_count': len(batch['ego']['gt_ids'][0]),
               'record_len': batch['ego']['record_len'].tolist()}
        if not finite or strict_zero is False:
            raise AssertionError('empty modality invariant failed')
        report['empty_modalities'].append(row)
        del batch, inputs, output, loss
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,
                  type_task_gradient_isolated=True,
                  fixed_batch_loss_decreased=report['steps'][-1]['loss'] < report['steps'][0]['loss'],
                  residual_nonzero=report['steps'][-1]['delta_abs_max_normalized'] > 0,
                  passed=True)
    if not report['residual_nonzero'] or not report['fixed_batch_loss_decreased']:
        report['passed'] = False
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k,v in report.items() if k not in ('steps','data_contract','sample_ids')}), flush=True)
    if not report['passed']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
