#!/usr/bin/env python3
"""Supplemental experiment entrypoint, using the existing project runner."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from opencood.project.config import load_config, validate_config
from opencood.project.supplement import INTERNAL, METHODS, NATIVE_HYPES, SCHEDULES, resolve_final

SUITE_ALIASES = {'real_b': 'real_cooperative', 'real_b_l4dr_fallback': 'real_cooperative_l4dr',
                 'real_a': 'real_weather'}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'output exists: {path}; choose a new output path')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def recipe(dataset='v2xr', method='deer_sbr', variant='AB', seed=17,
           epochs=None, *, data_root=None, output_root=None, device='auto',
           final_model_file=None, native=False, overrides=None,
           radar_root=None, label_root=None, manifest_dir=None, execution_config=None):
    c = load_config(REPO / 'cscwd_configs/supplement' / (dataset + '.yaml'))
    e, t, m = (c[k] for k in ('experiment', 'training', 'model'))
    epochs = t['epochs'] if epochs is None else epochs
    e.update(method=method, variant=variant, seed=seed,
             comparison='controlled_module' if method == 'deer_sbr' else 'complete_method')
    if dataset == 'kradar':
        e['comparison'] = 'local_single_agent'
    if final_model_file:
        e['final_model_file'] = str(Path(final_model_file).resolve())
    actual = resolve_final(e['final_model_file']) if variant == 'F' and e['final_model_file'] else variant
    m.update(amplitude_context='normalized_mean' if actual == 'N1' else 'rms',
             type_prior_mode='multiplicative' if actual == 'N2' else 'input_only', null_donor=actual != 'NO_NULL')
    if method != 'deer_sbr':
        c['data']['base_hypes'] = NATIVE_HYPES[method]
        t['batchnorm'].update(evaluation_statistics='running', refit_samples=0)
        if method != 'where2comm' or variant == 'S0':
            # These switches/weights belong to DEER/SBR, not native detectors.
            m.update(activation_checkpoint=False, efficient_features=False)
            c['loss'] = {key: 0. for key in c['loss']}
        if method == 'mdd' and dataset != 'v2xr':
            c['data']['clean_only'] = True
        if method == 'where2comm' and variant != 'S0':
            m['local_variant'] = variant
        if native:
            source = yaml.safe_load((REPO / NATIVE_HYPES[method]).read_text(encoding='utf8'))
            epochs = int(source['train_params']['epoches'])
            native_batch = int(source['train_params']['batch_size'])
            t.update(effective_batch_size=native_batch, micro_batch_size=native_batch,
                     accumulation_steps=1, world_size=1)
            t['optimizer'] = source['optimizer']['core_method'].lower()
            t.update(lr=float(source['optimizer']['lr']), weight_decay=float(source['optimizer']['args']['weight_decay']),
                     adam_eps=float(source['optimizer']['args']['eps']))
            schedule = source['lr_scheduler']
            if schedule['core_method'] == 'multistep':
                t.update(scheduler='multistep', milestones=schedule['step_size'], gamma=float(schedule['gamma']))
            elif schedule['core_method'] == 'cosineannealwarm':
                t.update(scheduler='cosine_warmup', warmup_epochs=schedule['warmup_epoches'],
                         warmup_lr=float(schedule['warmup_lr']), min_lr=float(schedule['lr_min']), milestones=[])
            else:
                raise ValueError('unimplemented native schedule: ' + schedule['core_method'])
            if method == 'l4dr' and dataset == 'kradar':
                # Official cfg_PP_L4DR.yml at fcf2d7f: use its 35-epoch
                # AdamW/cosine candidate, retaining this project's fixed GT.
                epochs = 35
                t.update(optimizer='adamw', lr=.001, weight_decay=.01, adam_eps=1e-8,
                         effective_batch_size=3, micro_batch_size=3, accumulation_steps=1,
                         scheduler='cosine_update', warmup_epochs=0, min_lr=.0001, milestones=[])
    t.update(epochs=epochs, recipe=('native_' if native else 'independent_') + f'e{epochs}',
             retain_epochs=sorted({x for x in (20, 30, 60, epochs) if x <= epochs}))
    if not native:
        t['milestones'] = SCHEDULES[epochs]
    if data_root:
        c['data'].update(root=str(Path(data_root).resolve()), manifest_dir=str(Path(data_root).resolve()))
    if manifest_dir:
        c['data']['manifest_dir'] = str(Path(manifest_dir).resolve())
    for key, value in (('radar_root', radar_root), ('label_root', label_root)):
        if value:
            c['data'][key] = str(Path(value).resolve())
    if output_root:
        c['artifacts']['output_root'] = str(Path(output_root).resolve())
    c['resources']['device'] = device
    suffix = ''
    for section, values in (overrides or {}).items():
        c[section].update(values)
        for key, value in values.items():
            suffix += '_' + key + str(value).replace('.', 'p')
    if method == 'where2comm' and m['local_variant'] != 'none':
        e['comparison'] = 'controlled_module'
    e['run_id'] = f'supp_{dataset}_{method}_{variant}_e{epochs}_s{seed}' + ('_native' if native else '') + suffix
    if execution_config:
        execution = load_config(execution_config)
        if execution['version'] != 5:
            raise ValueError('execution settings require a schema 5 configuration')
        for key in ('num_workers', 'prefetch_factor', 'raw_cache_gib_per_worker', 'cpu_threads',
                    'ram_budget_gib', 'pin_memory', 'persistent_workers'):
            c['resources'][key] = execution['resources'][key]
        t['checkpoint_interval_updates'] = execution['training']['checkpoint_interval_updates']
        if method == 'deer_sbr' or m['local_variant'] != 'none':
            m['activation_checkpoint'] = execution['model']['activation_checkpoint']
    return validate_config(c)


def performance_configs(config, stage, output_root):
    """Generate one-factor candidates from the last measured winner; never launch."""
    if config['version'] != 5:
        raise ValueError('performance candidates require supplement schema 5')
    stages = json.loads((REPO / 'cscwd_configs/supplement/performance_candidates.json').read_text())
    spec = stages[stage]
    if stage == 'prefetch' and not config['resources']['num_workers']:
        raise ValueError('prefetch comparison requires num_workers > 0')
    has_branch = (config['experiment']['method'] == 'deer_sbr' and config['experiment']['variant'] != 'S0'
                  or config['experiment']['method'] == 'where2comm' and config['model']['local_variant'] != 'none')
    if stage == 'activation' and not has_branch:
        raise ValueError('activation checkpoint comparison requires an active internal local module (use AB)')
    rows = []
    for value in spec['values']:
        candidate = deepcopy(config)
        candidate[spec['section']][spec['key']] = value
        candidate['artifacts']['output_root'] = str(Path(output_root).resolve())
        candidate['experiment']['run_id'] += f'_perf_{stage}_{str(value).lower().replace(".", "p")}'
        rows.append(validate_config(candidate))
    return rows


def write_config(config):
    path = Path(config['artifacts']['output_root']) / 'configs' / (config['experiment']['run_id'] + '.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if yaml.safe_load(path.read_text(encoding='utf8')) != config:
            raise ValueError(f'existing configuration differs: {path}')
    else:
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
    return path


def expand_suite(name, epochs_override=None, seeds_override=None, **options):
    matrix = json.loads((REPO / 'cscwd_configs/supplement/matrix.json').read_text(encoding='utf8'))
    rows = []
    for block in matrix['suites'][SUITE_ALIASES.get(name, name)]:
        if epochs_override is not None and block.get('native'):
            raise ValueError('--epochs applies to controlled suites; complete native methods keep their registered recipe')
        for method, variant, seed, epochs in itertools.product(block['methods'], block['variants'], seeds_override or block['seeds'],
                                                               [epochs_override] if epochs_override else block['epochs']):
            c = recipe(block['dataset'], method, variant, seed, epochs,
                native=block.get('native', False), overrides=block.get('overrides'), **options)
            rows.append(c)
    if len({c['experiment']['run_id'] for c in rows}) != len(rows):
        raise ValueError('matrix expansion contains duplicate run identities')
    return rows


def plan_records(configs, *, write=False):
    records = []
    for config in configs:
        e = config['experiment']
        path = (write_config(config) if write else Path(config['artifacts']['output_root']) / 'configs' / (e['run_id'] + '.yaml'))
        output = Path(config['artifacts']['output_root']) / e['run_id'] / 'train'
        records.append(dict(run_id=e['run_id'], dataset=e['dataset_id'], method=e['method'], variant=e['variant'], seed=e['seed'],
            epochs=config['training']['epochs'], recipe=config['training']['recipe'],
            effective_batch_size=config['training']['effective_batch_size'], micro_batch_size=config['training']['micro_batch_size'],
            config=str(path), configuration_written=write,
            argv=['python', 'project_tools/run_experiments.py', '--config', str(path), '--mode', 'train', '--allow-formal-run'],
            main_outputs=[str(output / name) for name in ('latest.rank0.pt', 'selected.pt', 'steps.rank0.jsonl', 'validation_epoch*.json')],
            experiment_started=False))
    return records


def runtime_config(args):
    c = load_config(args.config)
    if c['version'] != 5:
        raise ValueError('supplement entrypoint requires version 5; use scripts/reproduce.py for the historical seven groups')
    if getattr(args, 'device', None):
        c['resources']['device'] = args.device
    return validate_config(c)


def completed_run_matches(config, output):
    """Skip only a completed recipe with current artifacts and selected weights."""
    import torch
    from opencood.project.config import config_hash
    from project_tools.run_experiments import artifact_metadata
    output = Path(output)
    required = ('report.rank0.json', 'resolved_config.json', 'run_metadata.json', 'selected.pt')
    if not all((output / name).is_file() for name in required):
        return False
    report = json.loads((output / required[0]).read_text(encoding='utf8'))
    if (report.get('passed') is not True or report.get('source_unchanged') is not True
            or report.get('status') != 'checkpoint_selected'
            or json.loads((output / required[1]).read_text(encoding='utf8')) != config):
        return False
    expected = artifact_metadata(config)
    recorded = json.loads((output / required[2]).read_text(encoding='utf8'))
    if any(recorded.get(key) != value for key, value in expected.items()):
        return False
    selected = torch.load(output / 'selected.pt', map_location='cpu', weights_only=False)
    bn_role = ('train_refit_cma_v1' if config['training']['batchnorm']['evaluation_statistics'] != 'running'
               else 'native_running_frozen')
    return (selected.get('schema_version') == 5 and selected.get('training_protocol') == 'supplement_v1'
            and selected.get('config_sha256') == config_hash(config) and selected.get('config') == config
            and selected.get('metadata', {}).get('checkpoint_role') == 'selected'
            and selected['metadata'].get('bn_statistics_role') == bn_role
            and all(selected['metadata'].get(key) == value for key, value in expected.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    environment = commands.add_parser('environment', help='inspect dependencies/devices without data, training or downloads')
    environment.add_argument('--native-v2xr', action='store_true')
    environment.add_argument('--device', default='auto')
    environment.add_argument('--output')
    prepared = commands.add_parser('prepare', help='index official files and freeze sequence/GT metadata; never download')
    prepared.add_argument('--dataset', choices=['v2xradar', 'kradar'], required=True)
    prepared.add_argument('--data-root', required=True)
    prepared.add_argument('--manifest-dir', required=True)
    holdout = prepared.add_mutually_exclusive_group(required=True)
    holdout.add_argument('--dev-sequences', nargs='+')
    holdout.add_argument('--dev-count', type=int)
    prepared.add_argument('--split-seed', type=int, default=17)
    prepared.add_argument('--clean-sequences', nargs='*', default=[])
    prepared.add_argument('--weather-map', help='reviewed JSON mapping sequence to official weather tag (optional for V2X-Radar-C)')
    for key in ('radar-root', 'label-root', 'train-split', 'test-split'):
        prepared.add_argument('--' + key)
    configure = commands.add_parser('configure', help='write one concrete standalone recipe')
    performance = commands.add_parser('performance-configs', help='write a small one-factor execution comparison; never train')
    performance.add_argument('--config', required=True, help='base recipe or the last measured winning configuration')
    performance.add_argument('--stage', required=True, choices=['workers', 'prefetch', 'cache', 'cpu', 'activation', 'checkpoint', 'workers12'])
    performance.add_argument('--output-root', required=True)
    plan = commands.add_parser('plan', help='expand a suite without starting experiments')
    launch = commands.add_parser('run-suite', help='dry-run by default; --execute runs sequentially')
    suites = json.loads((REPO / 'cscwd_configs/supplement/matrix.json').read_text(encoding='utf8'))['suites']
    for command in (configure, plan, launch):
        command.add_argument('--data-root')
        command.add_argument('--manifest-dir')
        command.add_argument('--radar-root')
        command.add_argument('--label-root')
        command.add_argument('--output-root', required=True)
        command.add_argument('--device', default='auto')
        command.add_argument('--final-model-file')
        command.add_argument('--execution-config', help='copy only validated worker/cache/thread/checkpoint execution settings')
    configure.add_argument('--dataset', choices=['v2xr', 'v2xradar', 'kradar'], default='v2xr')
    configure.add_argument('--method', choices=METHODS, default='deer_sbr')
    configure.add_argument('--variant', choices=(*INTERNAL, 'F'))
    configure.add_argument('--seed', type=int, choices=[17, 29, 43], default=17)
    configure.add_argument('--epochs', type=int, choices=list(SCHEDULES),
                           help='independent budget; default comes from the dataset template (60/30/30)')
    configure.add_argument('--native-recipe', action='store_true')
    for command in (plan, launch):
        command.add_argument('--suite', choices=[*suites, *SUITE_ALIASES], required=True)
        command.add_argument('--epochs', type=int, choices=list(SCHEDULES), help='override controlled recipe budget and its schedule')
        command.add_argument('--seeds', type=int, nargs='+', choices=[17, 29, 43])
    plan.add_argument('--write-configs', action='store_true')
    launch.add_argument('--execute', action='store_true')
    launch.add_argument('--skip-completed', action='store_true')
    for name in ('preflight', 'check-data', 'train', 'evaluate', 'diagnose', 'efficiency', 'robustness', 'bn-sensitivity'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--device')
        if name == 'train':
            command.add_argument('--resume')
            command.add_argument('--initialize-from')
            command.add_argument('--execute', action='store_true')
        if name == 'check-data':
            command.add_argument('--samples', type=int, default=2)
        if name in ('evaluate', 'diagnose', 'efficiency', 'robustness', 'bn-sensitivity'):
            command.add_argument('--checkpoint', required=True)
            command.add_argument('--output', required=True)
            command.add_argument('--split', choices=['validate', 'test'], default='validate')
            command.add_argument('--weather', choices=['normal', 'fog', 'snow', 'natural'])
            command.add_argument('--allow-test-evaluation', action='store_true')
            command.add_argument('--budget-snapshot', action='store_true')
        if name in ('diagnose', 'efficiency'):
            command.add_argument('--samples', type=int, default=100)
        if name == 'diagnose':
            command.add_argument('--probe-condition', choices=['none', 'drop', 'inject', 'coexist', 'all'], default='all')
            command.add_argument('--drop-fraction', type=float, default=.4)
            command.add_argument('--inject-fraction', type=float, default=.05)
        if name == 'efficiency':
            command.add_argument('--warmup', type=int, default=20)
        if name == 'bn-sensitivity':
            command.add_argument('--refit-samples', type=int, nargs='+', default=[32, 128])
        if name == 'robustness':
            command.add_argument('--suite', choices=['corruption', 'radar', 'agents', 'heterogeneous', 'pose', 'async', 'interventions'], required=True)
            command.add_argument('--execute', action='store_true')
    smoke = commands.add_parser('smoke', help='synthetic only; zero dataset reads, no optimizer update or AP')
    smoke.add_argument('--config')
    smoke.add_argument('--output')
    summary = commands.add_parser('summarize')
    summary.add_argument('--results', nargs='+', required=True, help='scores.json files or directories containing them')
    summary.add_argument('--output', required=True)
    summary.add_argument('--baseline', default='deer_sbr/S0')
    curves = commands.add_parser('convergence')
    curves.add_argument('--run-dirs', nargs='+', required=True)
    curves.add_argument('--output', required=True)
    frozen = commands.add_parser('freeze-final')
    frozen.add_argument('--run-dirs', nargs='+', required=True, help='S0, AB and N2 train directories for seeds 17/29/43')
    frozen.add_argument('--normal-tolerance-pp', type=float, default=1.)
    frozen.add_argument('--output', required=True)
    bootstrap = commands.add_parser('bootstrap')
    bootstrap.add_argument('--config', required=True)
    bootstrap.add_argument('--predictions-a', required=True)
    bootstrap.add_argument('--predictions-b', required=True)
    bootstrap.add_argument('--split', choices=['validate', 'test'], required=True)
    bootstrap.add_argument('--repetitions', type=int, default=1000)
    bootstrap.add_argument('--output', required=True)
    visuals = commands.add_parser('visualize', help='render 1..5 fixed scenes from an executed prediction archive')
    visuals.add_argument('--config', required=True)
    visuals.add_argument('--predictions', required=True)
    visuals.add_argument('--sample-ids', nargs='+')
    visuals.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.action == 'environment':
        from opencood.project.supplement_runner import environment_report
        result = environment_report(args.native_v2xr, args.device)
        if args.output:
            save_json(args.output, result)
        print(json.dumps(result, indent=2))
        if result['status'] != 'ready':
            raise SystemExit(1)
    elif args.action == 'prepare':
        from opencood.project.supplement_data import build_real_manifest
        weather = json.loads(Path(args.weather_map).read_text(encoding='utf8')) if args.weather_map else None
        manifest = build_real_manifest(args.dataset, args.data_root, args.dev_sequences or (),
            radar_root=args.radar_root, label_root=args.label_root, train_split=args.train_split,
            test_split=args.test_split, clean_sequences=args.clean_sequences, weather_by_sequence=weather,
            dev_count=args.dev_count, split_seed=args.split_seed)
        path = Path(args.manifest_dir) / 'supplement_manifest.json'
        save_json(path, manifest)
        print(json.dumps(dict(manifest=str(path), samples=len(manifest['samples']),
            partitions=manifest['sequence_partitions'], summary=manifest['summary'],
            final_identity=manifest['final_identity']), ensure_ascii=False, indent=2))
    elif args.action == 'performance-configs':
        configs = performance_configs(load_config(args.config), args.stage, args.output_root)
        rows = plan_records(configs, write=True)
        for row in rows:
            row['argv'] = ['python', 'project_tools/run_experiments.py', '--config', row['config'],
                           '--mode', 'benchmark', '--steps', '40', '--warmup', '10', '--timing-mode', 'throughput']
            output = Path(row['main_outputs'][0]).parent.parent / 'benchmark'
            row['main_outputs'] = [str(output / name) for name in
                ('report.rank0.json', 'steps.rank0.jsonl', 'telemetry.rank0.jsonl', 'latest.rank0.pt')]
        print(json.dumps(dict(stage=args.stage, tasks=rows,
                              experiment_started=False), ensure_ascii=False, indent=2))
    elif args.action in ('configure', 'plan', 'run-suite'):
        options = dict(data_root=args.data_root, output_root=args.output_root, device=args.device,
            final_model_file=args.final_model_file, radar_root=args.radar_root,
            label_root=args.label_root, manifest_dir=args.manifest_dir, execution_config=args.execution_config)
        if args.action == 'configure':
            print(write_config(recipe(args.dataset, args.method, args.variant or ('AB' if args.method == 'deer_sbr' else 'S0'), args.seed,
                args.epochs, native=args.native_recipe, **options)))
        else:
            configs = expand_suite(args.suite, epochs_override=args.epochs, seeds_override=args.seeds, **options)
            write = args.write_configs if args.action == 'plan' else args.execute
            rows = plan_records(configs, write=write)
            print(json.dumps(dict(tasks=rows, count=len(rows), formal_experiment_started=False), ensure_ascii=False, indent=2), flush=True)
            if args.action == 'run-suite' and args.execute:
                for row in rows:
                    if args.skip_completed:
                        output = Path(row['main_outputs'][0]).parent
                        if completed_run_matches(load_config(row['config']), output):
                            print('SKIP complete matching recipe: ' + row['run_id'], flush=True)
                            continue
                    subprocess.run([sys.executable, *row['argv'][1:]], cwd=REPO, check=True)
    elif args.action == 'smoke':
        from project_tools.smoke_supplement import smoke_matrix
        rows = smoke_matrix(load_config(args.config) if args.config else None)
        result = dict(checks=rows, synthetic_only=True, formal_experiment_started=False,
                      experimental_metrics_produced=False)
        if args.output:
            save_json(args.output, result)
        print(json.dumps(dict(passed=True, checks=len(rows), synthetic_only=True,
                              experimental_metrics_produced=False, output=args.output), indent=2))
    elif args.action in ('preflight', 'check-data', 'train'):
        config = runtime_config(args)
        from project_tools.run_experiments import preflight, build_components, run
        if args.action == 'preflight':
            print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        elif args.action == 'check-data':
            if not 1 <= args.samples <= 32:
                parser.error('check-data permits 1..32 train samples')
            dataset, _, _ = build_components(config)
            rows = []
            for i in range(min(args.samples, len(dataset))):
                batch = dataset.collate_batch([dataset[i]])
                ego = batch['ego']
                rows.append(dict(sample_id=ego['sample_ids'][0], agents=int(ego['record_len'][0]),
                    sensor_points=ego['metadata'][0], positive_anchors=int(ego['label_dict']['pos_equal_one'].sum())))
            print(json.dumps(dict(samples=rows, training_started=False), ensure_ascii=False, indent=2))
        elif not args.execute:
            print(json.dumps(dict(preflight=preflight(config), formal_experiment_started=False,
                message='pass --execute to train this concrete recipe'), ensure_ascii=False, indent=2))
        else:
            from argparse import Namespace
            run(config, Namespace(mode='train', allow_formal_run=True, resume=args.resume,
                initialize_from=args.initialize_from, steps=1, warmup=0))
    else:
        from opencood.project import supplement_reporting as reporting
        if args.action == 'summarize':
            result = reporting.summarize(args.results, args.baseline)
            reporting.write_report_artifacts(result, args.output)
        elif args.action == 'convergence':
            result = reporting.convergence(args.run_dirs)
            reporting.write_report_artifacts(result, args.output)
        elif args.action == 'freeze-final':
            result = reporting.freeze_final(args.run_dirs, args.normal_tolerance_pp)
        elif args.action == 'bootstrap':
            config = runtime_config(args)
            result = reporting.bootstrap(config, args)
        elif args.action == 'visualize':
            config = runtime_config(args)
            result = reporting.visualize(config, args)
        else:
            config = runtime_config(args)
            if args.action == 'robustness':
                result = reporting.robustness(config, args)
            else:
                result = reporting.measure(config, args)
        save_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
