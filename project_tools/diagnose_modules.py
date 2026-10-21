#!/usr/bin/env python3
"""Bounded train-only numerical audit or worst-case memory diagnostic.

No selected checkpoint, teacher artifact, RMS artifact or AP result is produced.
Numerical audit: restore identical parameters AND BN buffers before FP32/AMP.
Stress: repeat a real maximal-agent sample using the teacher-free training graph.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from opencood.project.config import load_config, config_hash, file_sha256
from opencood.project.training import (select_model_inputs, to_device, seed_everything,
    tensor_state_hash, freeze_batchnorm, FP16_INITIAL_SCALE)
from project_tools.run_experiments import (read_hypes, build_model, data_contract_metadata,
    AMP_TOLERANCES, amp_audit_contract, validate_amp_audit)


VARIANTS = ('S0', 'A', 'B', 'AB', 'SUP', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'N1', 'N2')
THRESHOLDS = AMP_TOLERANCES


def _finite_number(value):
    return float(value) if math.isfinite(float(value)) else None


def vector_metrics(reference, candidate):
    """Accumulate on CPU in double without concatenating every gradient."""
    if set(reference) != set(candidate):
        return {'finite': False, 'keys_match': False, 'relative_l2': None, 'cosine': None}
    norm_a = norm_b = difference = dot = 0.0
    for key in sorted(reference):
        a, b = reference[key].detach().double(), candidate[key].detach().double()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return {'finite': False, 'keys_match': True, 'relative_l2': None, 'cosine': None}
        norm_a += float(a.square().sum())
        norm_b += float(b.square().sum())
        difference += float((a - b).square().sum())
        dot += float((a * b).sum())
    relative = math.sqrt(difference) / max(math.sqrt(norm_a), 1e-12)
    if norm_a == 0 and norm_b == 0:
        cosine = 1.0
    else:
        cosine = dot / max(math.sqrt(norm_a * norm_b), 1e-24)
    return {'finite': True, 'keys_match': True, 'relative_l2': relative,
            'cosine': max(-1., min(1., cosine)), 'reference_l2': math.sqrt(norm_a),
            'candidate_l2': math.sqrt(norm_b)}


def compare_snapshots(reference, candidate):
    loss_relative = abs(reference['loss'] - candidate['loss']) / max(abs(reference['loss']), 1e-12)
    outputs = {key: vector_metrics({key: reference['outputs'][key]}, {key: candidate['outputs'][key]})
               for key in reference['outputs']}
    gradients = vector_metrics(reference['gradients'], candidate['gradients'])
    groups = {}
    for prefix in sorted({key.split('.')[0] for key in reference['gradients']}):
        a = {key: value for key, value in reference['gradients'].items() if key.startswith(prefix + '.')}
        b = {key: value for key, value in candidate['gradients'].items() if key.startswith(prefix + '.')}
        groups[prefix] = vector_metrics(a, b)
    passed = (reference['finite'] and candidate['finite'] and math.isfinite(loss_relative)
              and loss_relative <= THRESHOLDS['loss_relative_error']
              and all(value['finite'] and value['relative_l2'] <= THRESHOLDS[key + '_relative_l2'] for key, value in outputs.items())
              and gradients['finite'] and gradients['keys_match']
              and gradients['cosine'] >= THRESHOLDS['gradient_cosine_min']
              and gradients['relative_l2'] <= THRESHOLDS['gradient_relative_l2'])
    return {'passed': bool(passed), 'loss_fp32': _finite_number(reference['loss']),
            'loss_amp': _finite_number(candidate['loss']), 'loss_relative_error': _finite_number(loss_relative),
            'target_precision': candidate.get('precision'),
            'amp_loss_scale': candidate.get('loss_scale', 1.0),
            'scaled_gradient_finite': candidate.get('scaled_gradient_finite', True),
            'finite': bool(reference['finite'] and candidate['finite'] and gradients['finite']),
            'psm_relative_l2': outputs['psm']['relative_l2'], 'rm_relative_l2': outputs['rm']['relative_l2'],
            'gradient_cosine': gradients['cosine'], 'gradient_relative_l2': gradients['relative_l2'],
            'max_abs_psm': _finite_number((reference['outputs']['psm'] - candidate['outputs']['psm']).abs().max()),
            'max_abs_rm': _finite_number((reference['outputs']['rm'] - candidate['outputs']['rm']).abs().max()),
            'outputs': outputs, 'gradients': gradients, 'gradient_groups_diagnostic': groups,
            'fp32_peak_allocated_gib': reference['peak_allocated_gib'],
            'amp_peak_allocated_gib': candidate['peak_allocated_gib'],
            'fp32_loss_terms': reference['loss_terms'], 'amp_loss_terms': candidate['loss_terms']}


def make_dataset(config, track='C', paired=False, configured_supervision=False):
    from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
    return ProjectV2XRDataset(read_hypes(config), root=config['data']['root'], split='train',
        training=True, track=track, seed=config['experiment']['seed'], paired=paired, augment=True,
        cache_bytes=0, supervision_source=config['data']['supervision_source'] if configured_supervision else 'none',
        supervision_audit=config['data']['supervision_audit'] if configured_supervision else None,
        gt_manifest=str(Path(config['data']['manifest_dir']) / 'eval_gt_manifest.json'),
        matching=config['data']['matching'], controlled_corruption=config['data'].get('controlled_corruption'))


def make_criterion(config, track):
    from opencood.loss.project_loss import ProjectLoss
    args = {**read_hypes(config)['loss']['args'], 'training_line': track,
            'lambda_a': config['loss']['type_weight'], 'lambda_correction': config['loss']['correction_weight'],
            'lambda_id': config['loss']['identity_weight'],
            'efficient_features': config['model']['efficient_features']}
    return ProjectLoss(args)


def snapshot(model, criterion, batches, device, precision, config):
    seed_everything(config['experiment']['seed'], config['training']['deterministic'])
    model.train()
    if config['training']['normalization'] == 'frozen_batchnorm':
        freeze_batchnorm(model)
    model.zero_grad(set_to_none=True)
    amp_dtype = torch.float16 if precision == 'amp_fp16' else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=precision == 'amp_fp16', init_scale=FP16_INITIAL_SCALE)
    # SGD is only a parameter container for the same GradScaler.unscale_ path;
    # no optimizer step is taken and no state/weights are updated by this audit.
    unscale_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    torch.cuda.reset_peak_memory_stats(device)
    loss_values, loss_terms = [], {}
    outputs = {'psm': [], 'rm': []}
    for batch in batches:
        inputs = to_device(select_model_inputs(batch['ego']), device)
        targets = to_device(batch.get('train_targets', {}), device)
        labels = to_device(batch['ego']['label_dict'], device)
        # torch.cuda.amp.autocast supports PyTorch 2.1 and newer versions.
        with torch.cuda.amp.autocast(enabled=precision != 'fp32', dtype=amp_dtype):
            prediction = model(inputs, return_aux=True)
            loss = criterion(prediction, labels, targets)
        scaler.scale(loss / len(batches)).backward()
        loss_values.append(float(loss.detach()))
        for key in outputs:
            outputs[key].append(prediction[key].detach().float().cpu())
        for key, value in criterion.loss_dict.items():
            loss_terms[key] = loss_terms.get(key, 0.) + float(value) / len(batches)
        del inputs, targets, labels, prediction, loss
    torch.cuda.synchronize(device)
    scaled_gradient_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                 for parameter in model.parameters())
    scaler.unscale_(unscale_optimizer)
    outputs = {key: torch.cat(value) for key, value in outputs.items()}
    gradients = {key: parameter.grad.detach().float().cpu().clone()
                 for key, parameter in model.named_parameters() if parameter.grad is not None}
    loss_mean = sum(loss_values) / len(loss_values)
    finite = (scaled_gradient_finite and math.isfinite(loss_mean) and all(bool(torch.isfinite(value).all()) for value in outputs.values())
              and all(bool(torch.isfinite(value).all()) for value in gradients.values()))
    result = {'loss': loss_mean, 'outputs': outputs, 'gradients': gradients, 'finite': finite,
              'precision': precision, 'loss_scale': scaler.get_scale(), 'scaled_gradient_finite': scaled_gradient_finite,
              'loss_terms': {key: _finite_number(value) for key, value in loss_terms.items()},
              'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3}
    model.zero_grad(set_to_none=True)
    return result


def batch_identity(batches):
    digest = hashlib.sha256()
    for batch in batches:
        ego = batch['ego']
        digest.update(json.dumps({'sample_ids': ego['sample_ids'], 'metadata': ego['metadata'],
            'augmentations': [matrix.tolist() for matrix in ego['augmentations']]}, sort_keys=True).encode())
        for modality in ('processed_lidar', 'processed_radar'):
            for key, value in sorted(ego[modality].items()):
                digest.update((modality + '/' + key).encode())
                digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state(path, variant):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'config' in payload:
        declared = payload['config']['experiment']['variant']
        if declared != variant:
            raise ValueError(f'{variant} checkpoint declares variant {declared}')
    state = payload.get('model', payload.get('model_state_dict', payload))
    return state


def audit(config, args, device, report):
    target_precision = config['training']['precision']
    report.update(kind='train_numerical_audit', schema='project_amp_consistency_v1',
                  precision=target_precision, reference_precision='fp32', variants=args.variants,
                  contract=amp_audit_contract(config),
                  thresholds=THRESHOLDS, cases=[], passed=False,
                  interpretation='Bounded forward/loss/gradient numerical consistency; not AP equivalence or a trained-model result.',
                  scope='Current train measurements and configured supervision; unverified labels stay masked. The graph uses current student features and separately supplied type labels.')
    checkpoints = {}
    for spec in args.checkpoint:
        variant, separator, path = spec.partition('=')
        if not separator or variant not in args.variants or not Path(path).is_file():
            raise ValueError('--checkpoint must be an audited VARIANT=/existing/checkpoint.pt')
        checkpoints[variant] = path
    micro = config['training']['micro_batch_size']
    accumulation = config['training']['accumulation_steps']
    if config['training']['world_size'] != 1:
        raise ValueError('single-device diagnostic cannot certify a DDP precision contract')
    if target_precision not in ('amp_fp16', 'amp_bf16'):
        raise ValueError('numerical audit config must declare target precision amp_fp16 or amp_bf16')
    tracks = args.tracks or [config['experiment']['track']]
    if config['experiment']['track'] not in tracks:
        raise ValueError('audit tracks must include the configured training line')
    for track in tracks:
        configured = track == config['experiment']['track']
        paired = configured and config['data']['paired_supervision']
        dataset = make_dataset(config, track=track, paired=paired, configured_supervision=configured)
        if args.samples * micro * accumulation > len(dataset):
            raise ValueError('requested more samples than train contains')
        criterion = make_criterion(config, track).to(device)
        for index in range(args.samples):
            batches = []
            for chunk in range(accumulation):
                start = (index * accumulation + chunk) * micro
                batches.append(dataset.collate_batch([dataset[i] for i in range(start, start + micro)]))
            identity = batch_identity(batches)
            for variant in args.variants:
                stages = [('initialization', None)]
                if variant in checkpoints:
                    stages.append(('diagnostic_checkpoint', checkpoints[variant]))
                for stage, path in stages:
                    model = build_model(config, mode='smoke', variant=variant)
                    if path:
                        model.load_state_dict(checkpoint_state(path, variant), strict=True)
                    initial = deepcopy(model.state_dict())
                    state_hash = tensor_state_hash(initial)
                    model.to(device)
                    reference = snapshot(model, criterion, batches, device, 'fp32', config)
                    model.load_state_dict(initial, strict=True)
                    candidate = snapshot(model, criterion, batches, device, target_precision, config)
                    row = compare_snapshots(reference, candidate)
                    row.update(variant=variant, stage=stage, track=track,
                               sample_ids=[sid for batch in batches for sid in batch['ego']['sample_ids']],
                               agents_per_micro=[batch['ego']['record_len'].tolist() for batch in batches],
                               weather=[[item['weather'] for scene in batch['ego']['metadata'] for item in scene] for batch in batches],
                               batch_identity_sha256=identity, micro_batch_size=micro, accumulation_steps=accumulation,
                               state_sha256=state_hash, checkpoint_sha256=file_sha256(path) if path else None,
                               scale_mode='current_observed_detached_rms' if model.use_b else None)
                    report['cases'].append(row)
                    print(json.dumps({'variant': variant, 'stage': stage, 'track': track,
                        'sample': index, 'passed': row['passed'], 'loss_relative_error': row['loss_relative_error'],
                        'gradient_cosine': row['gradients']['cosine'], 'gradient_relative_l2': row['gradients']['relative_l2']}), flush=True)
                    write_report(report, args.output)
                    del model, initial, reference, candidate
                    gc.collect()
                    torch.cuda.empty_cache()
        del dataset, criterion
        gc.collect()
    report['passed'] = bool(report['cases']) and all(row['passed'] for row in report['cases'])
    report['evidence'] = {'case_count': len(report['cases']), 'all_cases_passed': report['passed'],
                          'thresholds_fixed_before_execution': True, 'checkpoint_paths': checkpoints}
    end_contract = amp_audit_contract(config)
    if report['contract'] != end_contract:
        report['passed'] = False
        report['source_or_artifacts_changed_during_audit'] = True
    gate_error = validate_amp_audit(config, report, require_stability=False)
    report['formal_amp_blocker'] = validate_amp_audit(config, report)
    report['strict_gate_error'] = gate_error
    report['passed'] = report['passed'] and gate_error is None


def find_max_agent_sample(dataset, limit, index=None):
    maximum = min(dataset.max_agents, max(len(dataset.manifest['samples'][sid]['agent_ids'])
                                         for sid in dataset.sample_ids))
    if index is not None:
        sample = dataset[index]
        if len(sample['ego']['agents']) != maximum:
            raise ValueError(f'selected stress index does not retain the manifest maximum {maximum} actual agents')
        return index, sample, maximum
    examined = 0
    for candidate, sample_id in enumerate(dataset.sample_ids):
        if len(dataset.manifest['samples'][sample_id]['agent_ids']) < maximum:
            continue
        examined += 1
        if examined > limit:
            break
        sample = dataset[candidate]
        if len(sample['ego']['agents']) == maximum:
            return candidate, sample, maximum
    raise ValueError(f'no real {maximum}-agent sample survives communication filtering among {limit} candidate frames')


def stress(config, args, device, report):
    report.update(kind='max_agent_memory_stress', precision=args.precision,
                  variant='AB', track='W', passed=False, steps=[],
                  training_protocol=config['experiment']['training_protocol'], teacher_networks=0,
                  feature_scale='current observed detached RMS; no calibration artifact',
                  supervision='Configured train-only provenance; unverified Fog labels masked.',
                  interpretation='Repeated real maximal-agent sample for memory safety; no duplicated agents, AP result or production throughput estimate.')
    dataset = make_dataset(config, track='W', configured_supervision=True)
    index, sample, agents_per_scene = find_max_agent_sample(dataset, args.search_limit, args.sample_index)
    batch = dataset.collate_batch([sample] * args.micro_batch)
    if batch['ego']['record_len'].tolist() != [agents_per_scene] * args.micro_batch:
        raise ValueError('stress batch does not retain the maximal real agent count per scene')
    report.update(sample_index=index, sample_id=sample['ego']['sample_id'], micro_batch=args.micro_batch,
                  agents_per_scene=agents_per_scene, max_agents_in_train_manifest=agents_per_scene,
                  total_agents=agents_per_scene * args.micro_batch, repeated_same_real_frame=True,
                  synthetic_agents_added=0,
                  points_per_agent=[{key: agent[key] for key in ('agent_id', 'lidar_points', 'radar_points', 'weather')}
                                    for agent in sample['ego']['metadata']])
    del sample, dataset
    gc.collect()
    model = build_model(config, mode='smoke', variant='AB').to(device)
    criterion = make_criterion(config, 'W').to(device)
    t = config['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], eps=t['adam_eps'], weight_decay=t['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == 'amp_fp16', init_scale=FP16_INITIAL_SCALE)
    amp_dtype = torch.float16 if args.precision == 'amp_fp16' else torch.bfloat16
    inputs = to_device(select_model_inputs(batch['ego']), device)
    labels = to_device(batch['ego']['label_dict'], device)
    targets = to_device(batch['train_targets'], device)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        model.train()
        if t['normalization'] == 'frozen_batchnorm':
            freeze_batchnorm(model)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=args.precision != 'fp32', dtype=amp_dtype):
            output = model(inputs, return_aux=True)
            loss = criterion(output, labels, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite diagnostic stress loss')
        loss_scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError('nonfinite scaled diagnostic stress gradient')
        scaler.unscale_(optimizer)
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError('nonfinite unscaled diagnostic stress gradient')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        row = {'step': step + 1, 'seconds': time.perf_counter() - start, 'loss': float(loss.detach()),
               'loss_scale_before': loss_scale_before, 'loss_scale_after': scaler.get_scale(),
               'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
               'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 1024 ** 3}
        report['steps'].append(row)
        print(json.dumps(row), flush=True)
        write_report(report, args.output)
        del output, loss
    report['passed'] = len(report['steps']) == args.steps


def fp32_matrix(config, args, device, report):
    """Real forward/backward of registered controls; no optimizer updates/AMP."""
    from opencood.project.config import validate_config
    if config['version'] not in (3, 4) or config['training']['precision'] != 'fp32':
        raise ValueError('FP32 matrix checks require a registered teacher-free FP32 config')
    if args.samples != 1:
        raise ValueError('FP32 matrix check is bounded to one fixed effective batch')
    track = config['experiment']['track']
    if args.tracks and args.tracks != [track]:
        raise ValueError('one registered track per FP32 matrix check')
    dataset = make_dataset(config, track=track, paired=config['data']['paired_supervision'], configured_supervision=True)
    t = config['training']
    batches = []
    for chunk in range(t['accumulation_steps']):
        start = chunk * t['micro_batch_size']
        batches.append(dataset.collate_batch([dataset[i] for i in range(start, start+t['micro_batch_size'])]))
    report.update(schema=f"project_fp32_variant_paths_v{config['version']}", precision='fp32', optimizer_updates=0,
                  sample_index_range_inclusive=[0,7], sample_ids=[sid for b in batches for sid in b['ego']['sample_ids']],
                  batch_identity_sha256=batch_identity(batches), cases=[])
    shared_hashes = set()
    for variant in args.variants:
        current = deepcopy(config)
        current['experiment']['variant'] = variant
        current['model']['amplitude_context'] = 'normalized_mean' if variant == 'N1' else 'rms'
        current['model']['type_prior_mode'] = 'multiplicative' if variant == 'N2' else 'input_only'
        current = validate_config(current)
        model = build_model(current, mode='smoke').to(device)
        shared = {key:value.detach().cpu() for key,value in model.state_dict().items()
                  if not key.startswith(('evidence_router.', 'supported_residual.'))}
        shared_hash = tensor_state_hash(shared)
        shared_hashes.add(shared_hash)
        del shared
        criterion = make_criterion(current, track).to(device)
        measured = snapshot(model, criterion, batches, device, 'fp32', current)
        row = {'variant':variant,'finite':measured['finite'],'loss':measured['loss'],
               'loss_terms':measured['loss_terms'], 'peak_allocated_gib':measured['peak_allocated_gib'],
               'shared_initialization_sha256':shared_hash,
               'gradient_l2':{prefix:math.sqrt(sum(float(g.double().square().sum()) for name,g in measured['gradients'].items()
                                                  if name.startswith(prefix)))
                              for prefix in ('lidar_pillar_vfe','radar_pillar_vfe','evidence_router.type_head',
                                             'evidence_router.amplitude_project','supported_residual.residual_net.3')}}
        report['cases'].append(row)
        print(json.dumps(row), flush=True)
        del model, criterion, measured
        gc.collect()
    report['shared_initialization_equal'] = len(shared_hashes) == 1
    report['passed'] = all(row['finite'] for row in report['cases']) and report['shared_initialization_equal']
    report['interpretation'] = 'Fixed 8 train samples, one effective forward/backward per variant, no updates, AP or mixed precision.'


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO / 'cscwd_configs/project_v4.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--audit', action='store_true', help='default: FP32 versus configured FP16/BF16 numerical comparison')
    modes.add_argument('--fp32-matrix', action='store_true', help='registered FP32 variant forward/backward, zero optimizer updates')
    modes.add_argument('--stress', action='store_true', help='real maximal-agent repeated-batch teacher-free W memory stress')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['S0', 'AB'])
    parser.add_argument('--tracks', nargs='+', choices=['C', 'W'], default=None)
    parser.add_argument('--samples', type=int, default=1, help='first N effective train batches per track, maximum 8')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='VARIANT=PATH')
    parser.add_argument('--micro-batch', type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument('--steps', type=int, default=2, help='stress Adam updates, bounded 1..3')
    parser.add_argument('--sample-index', type=int)
    parser.add_argument('--search-limit', type=int, default=100)
    parser.add_argument('--precision', choices=['fp32', 'amp_fp16', 'amp_bf16'], default='fp32', help='stress precision (stable default FP32)')
    args = parser.parse_args()
    if not 1 <= args.samples <= 8 or not 1 <= args.steps <= 3 or not 1 <= args.search_limit <= 1000:
        parser.error('samples must be 1..8, steps 1..3, search-limit 1..1000')
    config = load_config(args.config)
    torch.set_num_threads(config['resources']['cpu_threads'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('this diagnostic requires CUDA; --help and pure metric helpers are CPU-safe')
    torch.cuda.set_device(device)
    target_precision = args.precision if args.stress else config['training']['precision']
    if target_precision == 'amp_bf16' and not torch.cuda.is_bf16_supported():
        parser.error('device does not support BF16')
    seed_everything(config['experiment']['seed'], config['training']['deterministic'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {'schema_version': 1, 'formal_experiment_started': False, 'diagnostic_only': True,
              'dataset_id': 'v2xr', 'split': 'train', 'seed': config['experiment']['seed'],
              'config_sha256': config_hash(config), 'data_contract': data_contract_metadata(config),
              'torch': torch.__version__, 'device': str(device), 'gpu': torch.cuda.get_device_name(device),
              'gpu_total_gib': torch.cuda.get_device_properties(device).total_memory / 1024 ** 3,
              'artifacts_created': 'JSON diagnostic report only; no model/teacher/RMS checkpoint',
              'source_sha256': file_sha256(__file__), 'passed': False}
    try:
        if args.fp32_matrix:
            fp32_matrix(config, args, device, report)
        elif args.stress:
            stress(config, args, device, report)
        else:
            audit(config, args, device, report)
    except Exception as error:
        report.update(passed=False, failure={'type': type(error).__name__, 'message': str(error)},
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024 ** 3)
        write_report(report, args.output)
        raise
    write_report(report, args.output)
    print(json.dumps({'passed': report['passed'], 'output': str(Path(args.output).resolve()),
                      'formal_experiment_started': False}), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
