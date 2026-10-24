"""Selected-checkpoint inference, mechanism/efficiency probes and paired reports."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from opencood.project.config import config_hash, file_sha256
from opencood.project.training import select_model_inputs, to_device, seed_everything
from opencood.project.supplement import resolve_final, final_decision
from opencood.project.supplement_evaluation import evaluate_supplement, sequence_bootstrap


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _model_identity(config):
    e = config['experiment']
    variant = resolve_final(e['final_model_file']) if e['variant'] == 'F' else e['variant']
    model = deepcopy(config['model'])
    if model['local_variant'] == 'F':
        model['local_variant'] = resolve_final(e['final_model_file'])
    return dict(method=e['method'], variant=variant, model=model)


def load_selected(config, checkpoint, device, *, budget_snapshot=False):
    from project_tools.run_experiments import build_model, source_tree_hash, data_contract_metadata
    from opencood.project.batchnorm import batchnorm_statistics_hashes
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    original = saved.get('config', {})
    metadata = saved.get('metadata', {})
    role = 'budget_snapshot' if budget_snapshot else 'selected'
    if saved.get('schema_version') != 5 or metadata.get('checkpoint_role') != role:
        raise ValueError('evaluation requires a supplemental selected checkpoint (or explicitly marked budget snapshot); online resume is forbidden')
    if saved.get('config_sha256') != config_hash(original) or metadata.get('source_sha256') != source_tree_hash():
        raise ValueError('selected checkpoint configuration/source identity changed')
    original_identity_config = deepcopy(original)
    original_identity_config['experiment']['variant'] = metadata.get('resolved_variant', original['experiment']['variant'])
    original_identity_config['model']['local_variant'] = metadata.get('resolved_local_variant', original['model']['local_variant'])
    if _model_identity(original_identity_config) != _model_identity(config):
        raise ValueError('selected architecture differs from the frozen experiment; inference interventions belong in evaluation.intervention')
    transfer = config['experiment']['transfer']
    from opencood.project.supplement import initialization_metadata
    trained_identity = initialization_metadata(original, verify_file=False)
    if trained_identity['training_transfer'] not in ('retrained', 'fine_tuned'):
        raise ValueError('selected checkpoint has no trained transfer identity')
    if any(metadata.get(key) != value for key, value in trained_identity.items()):
        raise ValueError('selected checkpoint transfer/initialization evidence is missing or inconsistent')
    if transfer in ('retrained', 'fine_tuned'):
        if initialization_metadata(config, verify_file=False) != trained_identity:
            raise ValueError('evaluation transfer/initialization differs from the actual trained checkpoint')
    # zero_shot retains the architecture-only cross-dataset adaptation above.
    # Source files are unnecessary for inference from self-contained weights.
    if config['experiment']['method'] == 'l4dr':
        from opencood.project.l4dr import pointnet2_identity
        if metadata.get('l4dr_pointnet2') != pointnet2_identity():
            raise ValueError('selected L4DR extension source/binary/runtime identity changed or is unavailable in old evidence')
    final_file = config['experiment']['final_model_file']
    if final_file and metadata.get('final_model_sha256') != file_sha256(final_file):
        # V2X-R's winning AB/N2 weights were trained before F was frozen. Reuse
        # them only when this exact selected file is bound in the dev decision.
        frozen = json.loads(Path(final_file).read_text(encoding='utf8'))
        winner = frozen['variant']
        evidence = frozen['evidence'][winner][str(config['experiment']['seed'])]
        if (metadata.get('final_model_sha256') is not None or budget_snapshot
                or original['experiment']['variant'] != winner
                or evidence['checkpoint_sha256'] != file_sha256(checkpoint)):
            raise ValueError('the frozen F decision changed or does not identify these selected candidate weights')
    if transfer in ('retrained', 'fine_tuned'):
        for key in ('training', 'loss', 'selection'):
            if original[key] != config[key]:
                raise ValueError(f'selected checkpoint {key} recipe mismatch')
        if original['experiment']['dataset_id'] != config['experiment']['dataset_id'] or original['experiment']['seed'] != config['experiment']['seed']:
            raise ValueError('retrained evaluation must retain the dataset and seed identity')
        for key, value in data_contract_metadata(config).items():
            if metadata.get(key) != value:
                raise ValueError('selected checkpoint immutable data mismatch: ' + key)
    elif transfer != 'zero_shot':
        raise ValueError('unsupported inference transfer identity')
    model = build_model(config, mode='export').to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    bn_role = metadata.get('bn_statistics_role')
    if bn_role == 'train_refit_cma_v1':
        report = metadata.get('batchnorm_refit', {})
        actual = batchnorm_statistics_hashes(model)
        expected = {key: row['statistics_sha256'] for key, row in report.get('bn', {}).items()}
        if (actual != expected or report.get('dataset_split') != 'train'
                or report.get('optimizer_updates') != 0 or report.get('all_modules_eval_at_yield') is not True):
            raise ValueError('selected train-only BN statistics do not match their fitting record')
    elif bn_role != 'native_running_frozen':
        raise ValueError('selected normalization role is missing')
    # No BN fitting takes place on any target or evaluation partition.
    return model, saved


def _evaluation_context(config, args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    args.weather = args.weather or ('normal' if config['experiment']['dataset_id'] == 'v2xr' else 'natural')
    if (config['experiment']['dataset_id'] == 'v2xr') == (args.weather == 'natural'):
        raise ValueError('V2X-R uses simulated normal/fog/snow; real datasets use natural and official weather groups')
    if args.split == 'test' and not args.allow_test_evaluation:
        raise ValueError('final evaluation requires --allow-test-evaluation after development decisions are frozen')
    if args.weather == 'snow' and args.split != 'test':
        raise ValueError('Snow cannot be used for development/model selection')
    from opencood.project.supplement_runner import select_device
    from project_tools.run_experiments import make_eval_loader
    device = select_device(config)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    seed_everything(config['experiment']['seed'], True)
    torch.set_num_threads(config['resources']['cpu_threads'])
    model, saved = load_selected(config, args.checkpoint, device, budget_snapshot=args.budget_snapshot)
    dataset, loader = make_eval_loader(config, args.split, args.weather)
    from opencood.project.supplement_models import set_intervention
    set_intervention(model, config['evaluation']['intervention'], config['evaluation']['perturbation_seed'])
    return device, model, saved, dataset, loader


def _metadata(config, args, saved, dataset):
    from project_tools.run_experiments import data_contract_metadata
    e = config['experiment']
    training_e = saved['config']['experiment']
    result = dict(schema='supplement_evaluation_metadata_v1', dataset=e['dataset_id'], protocol=e['protocol'],
        method=e['method'], variant=e['variant'], resolved_model=_model_identity(config), seed=e['seed'],
        transfer=e['transfer'], comparison=e['comparison'], condition=deepcopy(config['evaluation']),
        weather=dataset.weather, split=args.split, final_identity=dataset.manifest.get('final_identity', 'official_test'),
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_role=saved['metadata']['checkpoint_role'],
        checkpoint_epoch=saved['train_state']['epoch'], selected_epoch=saved['train_state']['selected_epoch'],
        selected_optimizer_updates=saved['train_state']['optimizer_steps'], recipe=saved['config']['training']['recipe'],
        training_recipe=deepcopy(saved['config']['training']),
        training_epochs=saved['config']['training']['epochs'], config_sha256=config_hash(config),
        effective_batch_size=saved['config']['training']['effective_batch_size'],
        micro_batch_size=saved['config']['training']['micro_batch_size'],
        checkpoint_sha256=file_sha256(args.checkpoint), source_sha256=saved['metadata']['source_sha256'],
        sample_ids=dataset.sample_ids, protocol_sha256=dataset.manifest['protocol_sha256'],
        single_or_multi_frame='one current frame per agent' if not config['evaluation']['delay_frames'] else 'delayed neighbor frame, current fixed GT',
        extra_supervision='paired normal feature reference' if training_e['method'] == 'mdd' and training_e['dataset_id'] == 'v2xr' else
            'MDD-CORR: verified clear originals as reference for controlled corruptions' if training_e['method'] == 'mdd' else
            'L4DR FAD: train-GT Radar foreground classification with 0.2m ignore shell; no inference labels' if training_e['method'] == 'l4dr' else
            'traceable type fractions on controlled inputs; no clean feature target' if training_e['method'] == 'deer_sbr' and training_e['variant'] not in ('S0', 'B') or training_e['method'] == 'where2comm' and saved['config']['model']['local_variant'] != 'none'
            else 'none; shared input corruption only',
        model_selection_split='validate', evaluation_bn_fitting=False)
    input_contract = data_contract_metadata(config)
    result['gt_manifest_sha256'] = input_contract['gt_manifest_sha256']
    if 'sensor_inputs_sha256' in input_contract:
        result['sensor_inputs_sha256'] = input_contract['sensor_inputs_sha256']
    result['input_fields'] = dataset.manifest.get('fields', dict(lidar='xyz + encoded red scalar', radar='xyz + encoded red scalar; not calibrated Doppler'))
    result['training_sample_count'] = saved.get('sampler', {}).get('size')
    result['training_dataset'] = training_e['dataset_id']
    result['initialization_checkpoint_sha256'] = saved['metadata'].get('initialization_checkpoint_sha256')
    result['training_transfer'] = saved['metadata']['training_transfer']
    if 'l4dr_pointnet2' in saved['metadata']:
        result['l4dr_pointnet2'] = deepcopy(saved['metadata']['l4dr_pointnet2'])
    n_train = result['training_sample_count']
    result['planned_optimizer_updates'] = (saved['config']['training']['epochs'] * (n_train // result['effective_batch_size'])
                                            if n_train is not None else None)
    names = dict(attfuse='AttFuse', mdd='AttFuse+MDD' if training_e['dataset_id'] == 'v2xr' else 'MDD-CORR',
                 l4dr='L4DR (FAD+MME+MSGF; four-channel adaptation)', cobevt='CoBEVT', where2comm='Where2comm')
    result['method_identity'] = names.get(e['method'], e['variant'])
    if e['dataset_id'] == 'kradar':
        result['method_identity'] += '-local (two-class adaptation)'
    if e['method'] == 'l4dr':
        result['l4dr_adaptation'] = dict(fad='PointNet++ MSG, per genuine agent, variable measured point count',
            mme='bidirectional attributes and cross-modal offsets; foreground score retained',
            msgf='inter-modal branch generates modality gates', point_loss_weight=2.,
            point_ignore_shell_m=.2, denoising_threshold=.1,
            source_revision='fcf2d7f0b7c632f9c23f75f720fe691be519d732',
            target_protocol='project fixed GT, classes and cooperative detector adaptation')
    if args.budget_snapshot:
        result.update(snapshot_kind=saved['metadata']['snapshot_kind'], snapshot_recipe=saved['metadata']['recipe'])
    return result


def measure(config, args):
    if args.action == 'bn-sensitivity' and config['experiment']['transfer'] == 'zero_shot':
        raise ValueError('zero_shot forbids fitting BN on the target dataset, including sensitivity probes')
    from project_tools.run_experiments import shutdown_loader
    from project_tools.export_predictions import export_loader, write_predictions
    device, model, saved, dataset, loader = _evaluation_context(config, args)
    metadata = _metadata(config, args, saved, dataset)
    try:
        if args.action == 'bn-sensitivity':
            if args.split != 'validate' or config['experiment']['method'] != 'deer_sbr':
                raise ValueError('BN sensitivity uses fixed internal weights and development scores, never final-test fitting')
            from project_tools.run_experiments import batchnorm_evaluation, validate_epoch
            reports = []
            for count in args.refit_samples:
                policy = config['training']['batchnorm']
                if count <= 0 or count % policy['refit_micro_batch_size']:
                    raise ValueError('refit samples must form complete train-only microbatches')
                c = deepcopy(config)
                c['training']['batchnorm'].update(evaluation_statistics='train_refit_cma_v1', refit_samples=count)
                with batchnorm_evaluation(model, c, device) as fitting:
                    scores, details = validate_epoch(model, c, device)
                reports.append(dict(refit_samples=count, development_scores=scores,
                                    details=details, train_only_fitting=fitting))
            return dict(schema='supplement_bn_sensitivity_v1', metadata=metadata, rows=reports,
                        optimizer_updates=0, selected_weights_changed=False, original_BN_restored=True)
        if args.action == 'efficiency':
            return efficiency(model, dataset, device, args, metadata)
        if args.action == 'diagnose':
            if args.split != 'validate':
                raise ValueError('mechanism diagnostics use independent development scenes; final test is not a mechanism tuning set')
            if config['experiment']['dataset_id'] == 'v2xr' and args.weather != 'normal':
                raise ValueError('type accuracy probes require Normal inputs with traceable corruption, not unknown Fog/Snow origins')
            from project_tools.run_experiments import make_eval_loader
            names = ['none', 'drop', 'inject', 'coexist'] if args.probe_condition == 'all' else [args.probe_condition]
            if not 0 <= args.drop_fraction <= 1 or not 0 <= args.inject_fraction <= 1:
                raise ValueError('diagnostic corruption fractions must be in [0,1]')
            reports = {}
            common = None
            if config['experiment']['dataset_id'] != 'v2xr':
                common = [sid for sid in dataset.sample_ids if dataset.manifest['samples'][sid]['verified_clear']]
                if not common:
                    raise ValueError('type accuracy probes need verified clear development scenes; unknown natural-weather origins cannot be labeled')
            for name in names:
                c = deepcopy(config)
                c['evaluation'].update(kind='artificial_corruption',
                    drop_fraction=args.drop_fraction if name in ('drop', 'coexist') else 0.,
                    inject_fraction=args.inject_fraction if name in ('inject', 'coexist') else 0.)
                probe, probe_loader = make_eval_loader(c, args.split, args.weather)
                if common is not None:
                    probe.sample_ids = [sid for sid in probe.sample_ids if sid in set(common)]
                try:
                    probe_metadata = {**deepcopy(metadata), 'condition': deepcopy(c['evaluation']),
                                      'sample_ids': probe.sample_ids, 'config_sha256': config_hash(c)}
                    reports[name] = diagnose(model, probe, device, args, probe_metadata)
                finally:
                    shutdown_loader({'loader': probe_loader})
            return dict(schema='supplement_four_condition_diagnostics_v1', conditions=reports,
                        common_scene_protocol=True, source_split='validate', formal_training_performed=False)
        output = Path(args.output)
        predictions = output.with_name(output.stem + '.predictions.jsonl')
        if output.exists() or predictions.exists():
            raise FileExistsError('evaluation output exists; choose a new output path')
        records = export_loader(model, loader, dataset, device)
        result = evaluate_supplement(dataset.manifest, records, args.split, config['scoring'], sample_ids=dataset.sample_ids)
        write_predictions(records, predictions, metadata)
        # Natural weather and distance groups use the same predictions and
        # fixed GT. They never refit BN or resample stochastic model outputs.
        groups = {}
        if config['experiment']['dataset_id'] != 'v2xr':
            for weather in sorted({dataset.manifest['samples'][sid]['natural_weather'] for sid in dataset.sample_ids} - {'unlabeled'}):
                ids = [sid for sid in dataset.sample_ids if dataset.manifest['samples'][sid]['natural_weather'] == weather]
                prefix = 'natural/' if config['evaluation']['kind'] == 'natural' else 'corruption_on_natural/'
                groups[prefix + weather] = evaluate_supplement(dataset.manifest,
                    [r for r in records if r['sample_id'] in set(ids)], args.split, config['scoring'], sample_ids=ids)
            if config['experiment']['dataset_id'] == 'v2xradar':
                for lower, upper in ((0, 30), (30, 50), (50, 100)):
                    manifest = deepcopy(dataset.manifest)
                    for sample in manifest['samples'].values():
                        sample['gt'] = [g for g in sample['gt'] if lower <= np.linalg.norm(g['box'][:2]) < upper]
                    selected_records = deepcopy(records)
                    for row in selected_records:
                        row['predictions'] = [p for p in row['predictions'] if lower <= np.linalg.norm(p['box'][:2]) < upper]
                    groups[f'distance/{lower}_{upper}m'] = evaluate_supplement(manifest, selected_records, args.split,
                        config['scoring'], sample_ids=dataset.sample_ids)
        return dict(schema='supplement_evaluation_v1', metadata=metadata, result=result,
                    groups=groups, predictions=str(predictions), experimental_result=True)
    finally:
        shutdown_loader({'loader': loader})


@torch.inference_mode()
def diagnose(model, dataset, device, args, metadata):
    """Collect actual local-module outputs while restoring diagnostic flags."""
    modules = [m for m in model.modules() if hasattr(m, 'collect_diagnostics')]
    previous = [m.collect_diagnostics for m in modules]
    try:
        for module in modules:
            module.collect_diagnostics = True
        return _diagnose(model, dataset, device, args, metadata)
    finally:
        for module, value in zip(modules, previous):
            module.collect_diagnostics = value


def _diagnose(model, dataset, device, args, metadata):
    from project_tools.diagnose_v3_mechanisms import selected_stats
    if args.samples <= 0:
        raise ValueError('diagnostic sample count must be positive')
    sums, counts, rows = Counter(), Counter(), []
    config = getattr(dataset, 'config', getattr(dataset, 'experiment_config', None))
    for i in range(min(args.samples, len(dataset))):
        batch = dataset.collate_batch([dataset[i]])
        output = model(to_device(select_model_inputs(batch['ego'], config), device), return_aux=True)
        aux = output.get('aux')
        if aux is None:
            raise ValueError('selected method has no DEER/SBR mechanism outputs')
        row = dict(sample_id=batch['ego']['sample_ids'][0], agents=batch['ego']['metadata'][0])
        for name in ('miss', 'spur'):
            pred = aux[name].detach().cpu()
            target = batch.get('train_targets', {})
            if name in target:
                valid = target[name + '_valid'] > 0
                sums[name + '_abs_error'] += float((pred - target[name]).abs()[valid].sum())
                sums[name + '_squared_error'] += float((pred - target[name]).square()[valid].sum())
                counts[name] += int(valid.sum())
            row[name] = selected_stats(aux[name])
        for name in ('route_gains', 'null_weight', 'support'):
            if name in aux:
                row[name] = selected_stats(aux[name])
        for name in ('donor_rms_normalized', 'donor_z_cosine', 'raw_abs_max', 'tanh_saturation_fraction'):
            if name in aux:
                row[name] = selected_stats(aux[name])
        normalized = aux['delta'].float() / aux['feature_scale'].float()
        support = aux['support'] > 0
        row['normalized_delta_inside'] = selected_stats(normalized.abs(), support)
        row['normalized_delta_outside'] = selected_stats(normalized.abs(), ~support)
        row['normalized_delta_max'] = float(normalized.abs().max())
        row['nonzero_residual_outside_support'] = int(((normalized != 0) & ~support).sum())
        rows.append(row)
    metrics = {name: dict(valid_cells=counts[name], mae=sums[name + '_abs_error'] / counts[name] if counts[name] else None,
        soft_label_brier=sums[name + '_squared_error'] / counts[name] if counts[name] else None) for name in ('miss', 'spur')}
    return dict(schema='supplement_mechanism_diagnostic_v1', metadata=metadata, type_metrics=metrics, rows=rows,
        target_semantics='continuous fractions from known source IDs; unknown natural-weather origins stay masked',
        diagnostic_only=True, selection_performed=False)


@torch.inference_mode()
def efficiency(model, dataset, device, args, metadata):
    from project_tools.export_predictions import reference_decode_predictions
    from opencood.project.supplement_evaluation import postprocess_real
    from opencood.project.evaluation import postprocess_predictions
    if args.samples < 1 or args.warmup < 0:
        raise ValueError('efficiency needs positive measured samples and nonnegative warmup')
    config = getattr(dataset, 'config', getattr(dataset, 'experiment_config', None))
    # Training/development can reuse parsed raw arrays. This separate latency
    # protocol explicitly includes point-file reads and parsing on every frame.
    if hasattr(dataset, 'cache'):
        dataset.cache.items.clear()
        dataset.cache.bytes = dataset.cache.max_bytes = 0
    for i in range(args.warmup):
        batch = dataset.collate_batch([dataset[i % len(dataset)]])
        model(to_device(select_model_inputs(batch['ego'], config), device), return_aux=False)
    _sync(device)
    times, local_times, peaks, agents = [], [], [], []
    for i in range(min(args.samples, len(dataset))):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        batch = dataset.collate_batch([dataset[i]])
        inputs = to_device(select_model_inputs(batch['ego'], config), device)
        output = model(inputs, return_aux=False)
        class_outputs = output.get('class_outputs', {getattr(dataset, 'classes', ['Vehicle'])[0]: output})
        records = []
        for name, prediction in class_outputs.items():
            anchor = dataset.anchors[name] if hasattr(dataset, 'anchors') else dataset.anchor_box
            row = reference_decode_predictions(prediction, anchor, batch['ego']['sample_ids'])[0]
            for p in row['predictions']:
                p.update(class_name=name, prediction_id=name + '/' + p['prediction_id'])
            records.extend(row['predictions'])
        sample = dataset.manifest['samples'][batch['ego']['sample_ids'][0]]
        if getattr(dataset, 'dataset_id', 'v2xr') == 'v2xr':
            postprocess_predictions(records, sample)
        else:
            postprocess_real(records, dataset.manifest, dataset.config['scoring'])
        _sync(device)
        times.append((time.perf_counter() - start) * 1000.)
        agents.append(int(inputs['record_len'][0]))
        if device.type == 'cuda':
            peaks.append(torch.cuda.max_memory_allocated(device))
        del output, class_outputs, prediction
        local = model.native if model.method == 'deer_sbr' else model.local
        if local is not None:
            native = model.native
            if model.method == 'deer_sbr':
                lidar, radar = native._encode(inputs)
            else:
                encoded = []
                for modality in ('lidar', 'radar'):
                    value = dict(inputs['processed_' + modality], record_len=inputs['record_len'], batch_size=int(inputs['record_len'].sum()))
                    encoded.append(native.scatter(getattr(native, modality + '_pillar_vfe')(value))['spatial_features'])
                lidar, radar = encoded
                del encoded, value
            _sync(device)
            start = time.perf_counter()
            local.fuse_encoded(lidar, radar, inputs, include_z0=False)
            _sync(device)
            local_times.append((time.perf_counter() - start) * 1000.)
            del lidar, radar
        # The separate local probe and prior sample must not inflate the
        # next end-to-end peak through still-live tensor references.
        del inputs, batch
    def stats(values):
        return dict(n=len(values), mean_ms=float(np.mean(values)), p50_ms=float(np.percentile(values, 50)),
                    p95_ms=float(np.percentile(values, 95))) if values else None
    return dict(schema='supplement_efficiency_v1', metadata=metadata,
        device=str(device), precision='fp32', batch_size=1, agent_histogram=dict(Counter(agents)),
        warmup_samples=args.warmup, measured_samples=len(times), parameters=sum(p.numel() for p in model.parameters()),
        end_to_end=stats(times), local_module=stats(local_times),
        peak_allocated_bytes=max(peaks) if peaks else None,
        scope=dict(preprocessing=True, point_file_io=True, host_to_device=True, decode=True, nms=True,
            raw_point_cache=False, filesystem_cache='OS-managed; no claim of cold disk IO',
            scoring=False, disk_output=False, serialized_data_pipeline=True, cuda_synchronized=device.type == 'cuda',
            local_module='phi_base + DEER + SBR + phi_out, excludes PFN/Scatter/cooperation; separate timing pass'))


def _validation_rows(run):
    run = Path(run)
    config = json.loads((run / 'resolved_config.json').read_text(encoding='utf8'))
    rows = []
    for path in sorted(run.glob('validation_epoch*.json')):
        value = json.loads(path.read_text(encoding='utf8'))
        domains = config['selection'].get('domains', ['normal', 'fog'] if config['experiment']['track'] == 'W' else ['normal'])
        if any(value[domain].get('split') != 'validate' for domain in domains):
            raise ValueError('development logs must identify split=validate; final-test results are forbidden here')
        scores = {domain: value[domain]['metrics']['3d']['0.7']['ap'] for domain in domains}
        if any(v is None or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError('invalid development AP in ' + str(path))
        rows.append(dict(epoch=int(path.stem.removeprefix('validation_epoch')), scores=scores,
                         score=float(np.mean(list(scores.values()))), source=str(path)))
    return config, rows


def convergence(run_dirs):
    results = []
    for run in run_dirs:
        config, rows = _validation_rows(run)
        if not rows:
            raise ValueError('no actual epoch validation logs: ' + str(run))
        step_path = Path(run) / 'steps.rank0.jsonl'
        steps = [json.loads(line) for line in step_path.read_text().splitlines() if line.strip()]
        by_epoch = defaultdict(list)
        for step in steps:
            by_epoch[step['epoch'] + 1].append(step)
        for row in rows:
            values = by_epoch[row['epoch']]
            row.update(loss=float(np.mean([s['loss'] for s in values])) if values else None,
                lr=sorted({s['lr'] for s in values}), optimizer_updates=len(values),
                loss_terms={k: float(np.mean([s['loss_terms'][k] for s in values if k in s['loss_terms']]))
                            for k in set().union(*(s.get('loss_terms', {}) for s in values))})
        old = [r['score'] for r in rows[:-10]]
        best = max(r['score'] for r in rows)
        improvement = (best - max(old)) * 100 if old else None
        recipe_name = config['training'].get('recipe', f"historical_v{config['version']}_e{config['training']['epochs']}")
        snapshots = [dict(epoch=epoch, recipe=recipe_name,
            kind='independent_recipe_endpoint' if epoch == config['training']['epochs'] else 'long_recipe_intermediate',
            exists=(Path(run) / f'budget_epoch{epoch:03d}.pt').is_file()) for epoch in config['training'].get('retain_epochs', [])]
        results.append(dict(run_dir=str(run), experiment=config['experiment'], recipe={**config['training'], 'recipe': recipe_name}, rows=rows,
            best_development_score=best, best_epoch=max(rows, key=lambda r: r['score'])['epoch'],
            best_improvement_after_last10_start_pp=improvement,
            plateau_candidate=improvement is not None and improvement < .3 and len(rows) > max(config['training']['milestones'], default=0),
            total_optimizer_updates=len(steps), budget_snapshots=snapshots))
    return dict(schema='supplement_convergence_v1', runs=results,
        interpretation='recipe endpoints and long-recipe intermediate states are distinct; plateau is a development heuristic')


def freeze_final(run_dirs, tolerance):
    records, contracts = {}, []
    for run in run_dirs:
        config, rows = _validation_rows(run)
        e = config['experiment']
        if config['version'] != 5 or e.get('method') != 'deer_sbr' or e['dataset_id'] != 'v2xr' or e['variant'] not in ('S0', 'AB', 'N2'):
            raise ValueError('freeze-final accepts only V2X-R S0/AB/N2 development runs')
        if config['selection']['domains'] != ['normal', 'fog']:
            raise ValueError('F uses the registered W Normal/Fog development criterion')
        report = json.loads((Path(run) / 'report.rank0.json').read_text())
        state = json.loads((Path(run) / 'state.json').read_text())
        if not report.get('passed') or report.get('status') != 'checkpoint_selected' or len(rows) != config['training']['epochs']:
            raise ValueError('freeze-final requires complete independent recipes: ' + str(run))
        if [r['epoch'] for r in rows] != list(range(1, config['training']['epochs'] + 1)):
            raise ValueError('development log epochs must cover the complete independent recipe')
        from opencood.project.training import TrainState, consider_checkpoint
        expected = TrainState()
        for row in rows:
            consider_checkpoint(expected, 'W', row['scores'], row['epoch'],
                                tolerance=config['selection']['tie_tolerance'])
        if state['selected_epoch'] != expected.selected_epoch:
            raise ValueError('selected epoch does not follow the registered development rule')
        selected = next(r for r in rows if r['epoch'] == state['selected_epoch'])
        if tolerance != config['selection']['normal_tolerance_pp']:
            raise ValueError('Normal tolerance must equal the value registered in the training configuration')
        checkpoint = Path(run) / 'selected.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (saved.get('config_sha256') != config_hash(config)
                or saved.get('schema_version') != 5 or saved.get('config') != config
                or saved.get('metadata', {}).get('checkpoint_role') != 'selected'
                or saved.get('train_state', {}).get('selected_epoch') != selected['epoch']
                or abs(saved['train_state']['best_score'] - selected['score']) > 1e-6):
            raise ValueError('selected checkpoint and development log disagree: ' + str(run))
        selected['checkpoint_sha256'] = file_sha256(checkpoint)
        saved_metadata = saved.get('metadata', {})
        if any(not saved_metadata.get(k) for k in ('source_sha256', 'gt_manifest_sha256')):
            raise ValueError('F requires the selected checkpoint source and fixed GT provenance')
        provenance = {k: v for k, v in saved_metadata.items()
                      if k.endswith('_sha256') and k not in ('supervision_audit_sha256',)}
        del saved
        key = (e['variant'], e['seed'])
        if key in records:
            raise ValueError('duplicate development seed/variant')
        records[key] = dict(**selected, run_dir=str(Path(run).resolve()))
        model_contract = {k: v for k, v in config['model'].items() if k != 'type_prior_mode'}
        contracts.append((config['training'], config['data'], config['selection'], config['loss'],
                          model_contract, config['scoring'], e['track'], e['transfer'], provenance))
    if set(records) != {(variant, seed) for variant in ('S0', 'AB', 'N2') for seed in (17, 29, 43)} or any(c != contracts[0] for c in contracts):
        raise ValueError('freeze-final needs matched S0/AB/N2 recipes for exactly seeds 17,29,43')
    evidence = {v: {str(s): records[v, s] for s in (17, 29, 43)} for v in ('S0', 'AB', 'N2')}
    chosen, scores, normal = final_decision(evidence, tolerance)
    return dict(schema='supplement_frozen_final_v1', variant=chosen, source_split='validate', seeds=[17, 29, 43],
        criterion='mean of per-seed selected Normal/Fog development AP3D07, Normal tolerance vs S0',
        normal_tolerance_pp=tolerance, development_means=scores, normal_development_means=normal,
        evidence=evidence,
        test_used=False, one_structure_for_all_seeds_and_weather=True)


def summarize(paths, baseline):
    files = []
    for path in paths:
        p = Path(path)
        files.extend([*p.rglob('*.scores.json'), *p.rglob('scores.json')] if p.is_dir() else [p])
    grouped, setting_contracts = defaultdict(dict), {}
    for path in sorted(set(files)):
        value = json.loads(path.read_text(encoding='utf8'))
        if value.get('schema') != 'supplement_evaluation_v1':
            if Path(path).is_file() and not Path(path).name.endswith('.scores.json'):
                continue
            raise ValueError('not an executed supplemental evaluation: ' + str(path))
        m, result = value['metadata'], value['result']
        if any(not m.get(k) for k in ('protocol_sha256', 'gt_manifest_sha256', 'checkpoint_sha256', 'sample_ids')):
            raise ValueError('statistical summary requires checkpoint, sample and fixed GT provenance')
        if m.get('checkpoint_role') not in ('selected', 'budget_snapshot'):
            raise ValueError('statistical summary requires selected weights or an explicit budget snapshot')
        setting_data = {k: deepcopy(m[k]) for k in ('dataset', 'protocol', 'transfer', 'condition', 'weather', 'split')}
        setting_data['condition'].pop('sample_ids', None)
        setting_data['checkpoint_role'] = m['checkpoint_role']
        if m['checkpoint_role'] == 'budget_snapshot':
            setting_data.update(snapshot_epoch=m['checkpoint_epoch'], snapshot_kind=m['snapshot_kind'])
        setting = json.dumps(setting_data, sort_keys=True)
        contract = (m['protocol_sha256'], m['gt_manifest_sha256'], m['sample_ids'])
        if setting in setting_contracts and contract != setting_contracts[setting]:
            raise ValueError('paired summary fixed GT/provenance or sample set differs across methods/seeds')
        if len(set(m['sample_ids'])) != len(m['sample_ids']):
            raise ValueError('summary sample list contains duplicates')
        setting_contracts[setting] = contract
        recipe_key = (m['recipe'], json.dumps(dict(model=m.get('resolved_model'),
                      training=m.get('training_recipe')), sort_keys=True))
        key = (setting, m['method'] + '/' + m['variant'], recipe_key)
        if m['seed'] in grouped[key]:
            raise ValueError('duplicate seed in a statistical group')
        if grouped[key] and any(r['metadata'].get('resolved_model') != m.get('resolved_model') for r in grouped[key].values()):
            raise ValueError('one statistical group must use one resolved structure across all seeds')
        grouped[key][m['seed']] = dict(ap=result['metrics']['3d']['0.7']['ap'], source=str(path), metadata=m)
    if not grouped:
        raise ValueError('no executed .scores.json files found')
    # The registered V2X-R weather mean is computed within each seed first.
    # Its standard deviation is not the average of Fog/Snow deviations.
    for (setting, method, recipe_name), fog in list(grouped.items()):
        spec = json.loads(setting)
        if spec['dataset'] != 'v2xr' or spec['weather'] != 'fog':
            continue
        snow_spec = {**spec, 'weather': 'snow'}
        snow = grouped.get((json.dumps(snow_spec, sort_keys=True), method, recipe_name))
        if snow is None:
            continue
        combined = {}
        for seed in sorted(set(fog) & set(snow)):
            for key in ('checkpoint_sha256', 'resolved_model', 'protocol_sha256', 'gt_manifest_sha256'):
                if fog[seed]['metadata'].get(key) != snow[seed]['metadata'].get(key):
                    raise ValueError('Fog/Snow weather mean requires the same checkpoint and fixed GT: ' + key)
            if fog[seed]['metadata']['sample_ids'] != snow[seed]['metadata']['sample_ids']:
                raise ValueError('Fog/Snow weather mean requires the same fixed scene set')
            if fog[seed]['ap'] is None or snow[seed]['ap'] is None:
                raise ValueError('weather AP is undefined; do not substitute zero')
            combined[seed] = dict(ap=(fog[seed]['ap'] + snow[seed]['ap']) / 2,
                source=[fog[seed]['source'], snow[seed]['source']],
                metadata={**fog[seed]['metadata'], 'weather': 'fog_snow_mean'})
        if combined:
            grouped[(json.dumps({**spec, 'weather': 'fog_snow_mean'}, sort_keys=True), method, recipe_name)] = combined
    rows = []
    for (setting, method, recipe_name), seeds in grouped.items():
        values = [r['ap'] for r in seeds.values()]
        if any(v is None for v in values):
            raise ValueError('AP is undefined for a required class; do not average it as zero')
        baseline_matches = [v for (s, name, _), v in grouped.items() if s == setting and name == baseline]
        if len(baseline_matches) > 1:
            raise ValueError('multiple baseline budgets in one summary; summarize matched recipes separately')
        differences = []
        if baseline_matches:
            for seed in sorted(set(seeds) & set(baseline_matches[0])):
                left, right = baseline_matches[0][seed], seeds[seed]
                if left['metadata']['sample_ids'] != right['metadata']['sample_ids']:
                    raise ValueError('paired methods evaluated different samples/perturbation arrangements')
                differences.append(dict(seed=seed, delta_pp=(right['ap'] - left['ap']) * 100))
        rows.append(dict(setting=json.loads(setting), method=method, recipe=recipe_name[0], seeds=sorted(seeds),
            resolved_model=next(iter(seeds.values()))['metadata'].get('resolved_model'),
            training_recipe=next(iter(seeds.values()))['metadata'].get('training_recipe'),
            method_identity=next(iter(seeds.values()))['metadata'].get('method_identity', method),
            training_information={key: {str(s): r['metadata'].get(key) for s, r in seeds.items()} for key in (
                'comparison', 'training_dataset', 'training_sample_count', 'training_epochs', 'effective_batch_size',
                'micro_batch_size', 'planned_optimizer_updates', 'selected_optimizer_updates', 'extra_supervision')},
            complete_three_seeds=set(seeds) == {17, 29, 43}, n=len(values), mean_ap=float(np.mean(values)),
            sample_std_ap=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            paired_vs=baseline, paired_deltas=differences,
            paired_mean_delta_pp=float(np.mean([d['delta_pp'] for d in differences])) if differences else None,
            sources=[r['source'] for r in seeds.values()]))
    return dict(schema='supplement_summary_v1', rows=rows, std_definition='sample standard deviation, ddof=1')


def bootstrap(config, args):
    from opencood.project.evaluation import read_prediction_archive
    from project_tools.run_experiments import make_eval_loader, shutdown_loader
    if args.repetitions <= 0:
        raise ValueError('bootstrap repetitions must be positive')
    from project_tools.run_experiments import data_contract_metadata
    ma, a = read_prediction_archive(args.predictions_a)
    mb, b = read_prediction_archive(args.predictions_b)
    for key in ('dataset', 'protocol', 'protocol_sha256', 'weather', 'condition', 'sample_ids', 'gt_manifest_sha256'):
        left, right = deepcopy(ma.get(key)), deepcopy(mb.get(key))
        if key == 'condition' and left is not None and right is not None:
            left.pop('sample_ids', None)
            right.pop('sample_ids', None)
        if left is None or left != right:
            raise ValueError('paired bootstrap provenance mismatch or missing field: ' + key)
    if ma['gt_manifest_sha256'] != data_contract_metadata(config)['gt_manifest_sha256']:
        raise ValueError('bootstrap predictions refer to a different immutable GT source')
    condition = deepcopy(config)
    condition['evaluation'] = deepcopy(ma['condition'])
    condition['evaluation']['sample_ids'] = None
    dataset, loader = make_eval_loader(condition, args.split, ma['weather'])
    try:
        if not set(ma['sample_ids']) <= set(dataset.sample_ids):
            raise ValueError('bootstrap sample set is not available under the frozen condition')
        manifest = deepcopy(dataset.manifest)
        manifest['samples'] = {sid: dataset.manifest['samples'][sid] for sid in ma['sample_ids']}
        return sequence_bootstrap(manifest, a, b, args.split, config['scoring'], repetitions=args.repetitions)
    finally:
        shutdown_loader({'loader': loader})


def robustness(config, args):
    args.weather = args.weather or ('normal' if config['experiment']['dataset_id'] == 'v2xr' else 'natural')
    conditions = [('baseline', {})]
    if args.suite == 'corruption':
        conditions += [(f'drop_{v}', dict(kind='artificial_corruption', drop_fraction=v)) for v in (.2, .4, .6)]
        conditions += [(f'inject_{v}', dict(kind='artificial_corruption', inject_fraction=v)) for v in (.01, .05, .1)]
        conditions += [('coexist', dict(kind='artificial_corruption', drop_fraction=.4, inject_fraction=.05))]
    elif args.suite == 'radar':
        conditions += [(f'radar_keep_{v}', dict(kind='artificial_corruption', radar_keep=v)) for v in (.75, .5, .25, 0.)]
        conditions += [(f'missing_{v}', dict(kind='artificial_corruption', missing_modality=v)) for v in ('lidar', 'radar', 'both')]
    elif args.suite == 'agents':
        conditions = [(f'agents_{v}', dict(agents=v, min_agents=5)) for v in (1, 2, 3, 5)]
    elif args.suite == 'heterogeneous':
        conditions += [(scope, dict(kind='artificial_corruption', agent_scope=scope, drop_fraction=.4, inject_fraction=.05))
                       for scope in ('ego', 'neighbors', 'all')]
    elif args.suite == 'pose':
        conditions += [(f'position_{v}m', dict(position_std_m=v)) for v in (.1, .3, 1.)]
        conditions += [(f'yaw_{v}deg', dict(yaw_std_deg=v)) for v in (.5, 1., 2.)]
    elif args.suite == 'async':
        conditions += [(f'delay_{v}frames', dict(delay_frames=v)) for v in (1, 2, 5)]
    elif args.suite == 'interventions':
        conditions += [(v, dict(intervention=v)) for v in ('type_zero', 'type_shuffle', 'amplitude_zero',
                                                         'amplitude_shuffle', 'donor_zero', 'donor_shuffle')]
    else:
        raise ValueError('unknown robustness suite')
    if config['experiment']['dataset_id'] == 'kradar' and args.suite in ('heterogeneous', 'pose', 'async'):
        raise ValueError('agent heterogeneity, cooperative pose and cooperative delay do not apply to single-agent K-Radar')
    from project_tools.run_experiments import make_eval_loader, shutdown_loader
    common_ids = None
    max_agents = None
    if args.execute:
        # Establish one intersection for every level, especially asynchronous
        # first-frame exclusions and agent-count comparisons.
        available = []
        for name, changes in conditions:
            c = deepcopy(config)
            c['evaluation'].update(changes)
            if args.suite == 'agents':
                if max_agents is None:
                    base_dataset, loader = make_eval_loader(config, args.split, args.weather)
                    max_agents = min(5, max(len(base_dataset.available_agent_ids(sid)) if hasattr(base_dataset, 'available_agent_ids')
                                            else len(base_dataset.manifest['samples'][sid]['agent_ids'])
                                            for sid in base_dataset.sample_ids))
                    shutdown_loader({'loader': loader})
                c['evaluation']['min_agents'] = max_agents
                if changes['agents'] > max_agents:
                    continue
            dataset, loader = make_eval_loader(c, args.split, args.weather)
            available.append(set(dataset.sample_ids))
            shutdown_loader({'loader': loader})
        common_ids = sorted(set.intersection(*available))
        if not common_ids:
            raise ValueError('robustness conditions have no common scenes')
        sample_path = Path(args.output).with_suffix('.common_samples.json')
        if sample_path.exists():
            raise FileExistsError(sample_path)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(common_ids), encoding='utf8')
    rows = []
    for name, changes in conditions:
        c = deepcopy(config)
        c['evaluation'].update(changes)
        if args.suite == 'agents' and max_agents is not None:
            if changes['agents'] > max_agents:
                rows.append(dict(condition=name, status='not_applicable', reason=f'only {max_agents} genuine agents available'))
                continue
            c['evaluation']['min_agents'] = max_agents
        if common_ids is not None:
            c['evaluation']['sample_ids'] = str(sample_path.resolve())
        target = Path(args.output).parent / Path(args.output).stem / name / 'result.scores.json'
        rows.append(dict(condition=name, config=c['evaluation'], output=str(target), executed=args.execute))
        if args.execute:
            options = deepcopy(args)
            options.action, options.output = 'evaluate', str(target)
            value = measure(c, options)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    return dict(schema='supplement_robustness_suite_v1', suite=args.suite, conditions=rows,
        perturbation_seed=config['evaluation']['perturbation_seed'], common_sample_count=len(common_ids) if common_ids else None,
        dry_run=not args.execute, training_performed=False)


def write_report_artifacts(result, path):
    """Write inspectable CSV and convergence curves only from executed logs."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if result['schema'] == 'supplement_summary_v1':
        for row in result['rows']:
            rows.append(dict(dataset=row['setting']['dataset'], weather=row['setting']['weather'],
                condition=json.dumps(row['setting']['condition'], sort_keys=True), method=row['method'],
                method_identity=row['method_identity'], training_information=json.dumps(row['training_information']),
                recipe=row['recipe'], n=row['n'], seeds=','.join(map(str, row['seeds'])),
                mean_percent=row['mean_ap'] * 100,
                sample_std_pp=row['sample_std_ap'] * 100 if row['sample_std_ap'] is not None else None,
                paired_mean_delta_pp=row['paired_mean_delta_pp'],
                paired_deltas=json.dumps(row['paired_deltas']), complete_three_seeds=row['complete_three_seeds']))
    elif result['schema'] == 'supplement_convergence_v1':
        for run in result['runs']:
            for row in run['rows']:
                rows.append(dict(run_id=run['experiment']['run_id'], seed=run['experiment']['seed'],
                    variant=run['experiment']['variant'], recipe=run['recipe']['recipe'], epoch=row['epoch'],
                    mean_loss=row['loss'], lr=json.dumps(row['lr']), optimizer_updates=row['optimizer_updates'],
                    development_ap_percent=row['score'] * 100, domain_scores=json.dumps(row['scores']),
                    loss_terms=json.dumps(row['loss_terms'])))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for run in result['runs']:
            label = f"{run['experiment']['variant']} / {run['recipe']['recipe']} / seed {run['experiment']['seed']}"
            axes[0].plot([r['epoch'] for r in run['rows']], [r['score'] * 100 for r in run['rows']], label=label)
            axes[1].plot([r['epoch'] for r in run['rows']], [r['loss'] for r in run['rows']], label=label)
        for axis in axes:
            axis.set_xlabel('Epoch')
            axis.grid(alpha=.2)
        axes[0].set_ylabel('Development AP3D@0.7 (%)')
        axes[1].set_ylabel('Training loss')
        axes[0].legend(fontsize=7)
        figure.tight_layout()
        image_path = path.with_suffix('.png')
        if image_path.exists():
            raise FileExistsError(image_path)
        figure.savefig(image_path, dpi=180)
        plt.close(figure)
        result['convergence_plot'] = str(image_path)
    if rows:
        csv_path = path.with_suffix('.csv')
        with csv_path.open('x', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result['table_csv'] = str(csv_path)


def visualize(config, args):
    """Fixed-scene BEV observation/GT/prediction overlays from executed archives."""
    from opencood.project.evaluation import read_prediction_archive
    from opencood.project.geometry import boxes_to_corners
    from project_tools.run_experiments import make_eval_loader, shutdown_loader
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metadata, records = read_prediction_archive(args.predictions)
    if metadata.get('schema') != 'supplement_evaluation_metadata_v1':
        raise ValueError('visualize requires an executed supplemental prediction archive')
    c = deepcopy(config)
    c['evaluation'] = deepcopy(metadata['condition'])
    c['evaluation']['sample_ids'] = None
    dataset, loader = make_eval_loader(c, metadata['split'], metadata['weather'])
    # Observation counts are an explicit consumer of visualization-only maps.
    dataset.observation_maps_required = True
    rows = {r['sample_id']: r for r in records}
    ids = args.sample_ids or sorted(rows)[:5]
    if not 1 <= len(ids) <= 5 or not set(ids) <= set(rows) & set(dataset.sample_ids):
        raise ValueError('choose 1..5 fixed sample IDs present in both data and predictions')
    outputs = []
    folder = Path(args.output).parent / Path(args.output).stem
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for index, sid in enumerate(ids):
            item = dataset[dataset.sample_ids.index(sid)]
            agents = item['agents'] if 'agents' in item else item['ego']['agents']
            count = sum(a['maps']['lidar_count'] + a['maps']['radar_count'] for a in agents)[0]
            bounds = dataset.bounds
            figure, axis = plt.subplots(figsize=(8, 6))
            axis.imshow(np.log1p(count), origin='lower', extent=[bounds[0], bounds[3], bounds[1], bounds[4]],
                        cmap='Greys', interpolation='nearest', aspect='equal')
            for label, source, color in (('Fixed GT', dataset.manifest['samples'][sid]['gt'], '#228866'),
                                          ('Prediction', rows[sid]['predictions'], '#4477AA')):
                for i, obj in enumerate(source):
                    corner = boxes_to_corners([obj['box']])[0, [0, 1, 2, 3, 0], :2]
                    axis.plot(corner[:, 0], corner[:, 1], color=color, linewidth=1.1, label=label if i == 0 else None)
            axis.set(xlabel='Ego x (m)', ylabel='Ego y (m)', title=sid)
            axis.legend(loc='upper right')
            figure.tight_layout()
            path = folder / f'sample_{index + 1:02d}.png'
            if path.exists():
                raise FileExistsError(path)
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(dict(sample_id=sid, image=str(path)))
    finally:
        shutdown_loader({'loader': loader})
    return dict(schema='supplement_fixed_scene_visualizations_v1', metadata=metadata, images=outputs,
                fixed_gt=True, scores_computed=False)
