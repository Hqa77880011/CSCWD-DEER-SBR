"""Registered supplemental recipes. Historical v2/v3/v4 contracts stay frozen."""
from copy import deepcopy
import json
import math
from pathlib import Path

from opencood.project.config import DEFAULTS, _merge

INTERNAL = ('S0', 'A', 'B', 'AB', 'SUP', 'N1', 'N2',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'NO_NULL')
METHODS = ('deer_sbr', 'attfuse', 'mdd', 'l4dr', 'cobevt', 'where2comm')
SCHEDULES = {20: [10, 15], 30: [15, 23], 60: [30, 45], 90: [45, 68]}
NATIVE_HYPES = {
    'attfuse': 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml',
    'mdd': 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion_with_MDD/V2XR_AttFuse.yaml',
    'l4dr': 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_L4DR.yaml',
    'cobevt': 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_Cobevt.yaml',
    'where2comm': 'opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_where2comm.yaml',
}


def defaults():
    c = deepcopy(DEFAULTS)
    c['version'] = 5
    c['experiment'].update(run_id='supp_v2xr_deer_sbr_AB_e60_s17',
                           training_protocol='supplement_v1', method='deer_sbr',
                           comparison='controlled_module', transfer='retrained',
                           final_model_file=None)
    c['data'].update(root='datasets/v2xr', manifest_dir='datasets/v2xr',
                     radar_root=None, label_root=None, supervision_source='provenance_v1',
                     clean_only=False)
    c['experiment']['track'] = 'W'
    c['training'].update(epochs=60, optimizer='adam', milestones=[30, 45], scheduler='multistep',
                         warmup_epochs=0, warmup_lr=0.0002, min_lr=0.00002,
                         retain_epochs=[20, 30, 60], recipe='independent_e60')
    c['training']['batchnorm'].update(evaluation_statistics='train_refit_cma_v1', refit_samples=32)
    c['model'].update(amplitude_context='rms', type_prior_mode='input_only',
                       efficient_features=True, null_donor=True, local_variant='none')
    c['resources']['device'] = 'auto'
    c['selection']['domains'] = ['normal', 'fog']
    c['selection']['class_name'] = 'Vehicle'
    c['selection']['normal_tolerance_pp'] = 1.
    c['artifacts']['output_root'] = 'outputs/supplement'
    c['evaluation'] = dict(kind='simulated_weather', corruption='none', drop_fraction=0.,
                           inject_fraction=0., max_offset_m=0.3, radar_keep=1.,
                           missing_modality='none', agent_scope='all', agents=5,
                           min_agents=1, position_std_m=0., yaw_std_deg=0.,
                           delay_frames=0, perturbation_seed=17029,
                           intervention='none', sample_ids=None, natural_weather=None)
    return c


def final_decision(evidence, tolerance):
    """Apply one development-only structure rule to matched three-seed evidence."""
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('F requires a finite nonnegative registered Normal tolerance')
    normal, scores = {}, {}
    for variant in ('S0', 'AB', 'N2'):
        rows = evidence.get(variant, {})
        if set(rows) != {'17', '29', '43'}:
            raise ValueError('F requires all three development seeds for every candidate and S0')
        for row in rows.values():
            values = row.get('scores', {})
            if (set(values) != {'normal', 'fog'}
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                           for v in [row.get('score'), *values.values()])):
                raise ValueError('F evidence must contain finite Normal/Fog development AP')
            if abs(row['score'] - (values['normal'] + values['fog']) / 2) > 1e-10:
                raise ValueError('F evidence score differs from its Normal/Fog development mean')
        normal[variant] = math.fsum(r['scores']['normal'] for r in rows.values()) / 3
        if variant != 'S0':
            scores[variant] = math.fsum(r['score'] for r in rows.values()) / 3
    eligible = [v for v in ('AB', 'N2') if (normal['S0'] - normal[v]) * 100 <= tolerance + 1e-10]
    if not eligible:
        raise ValueError('neither candidate meets the registered Normal tolerance; no final structure was frozen')
    return max(eligible, key=lambda v: scores[v]), scores, normal


def resolve_final(path):
    record = json.loads(Path(path).read_text(encoding='utf8'))
    if (record.get('schema') != 'supplement_frozen_final_v1'
            or record.get('variant') not in ('AB', 'N2')
            or record.get('source_split') != 'validate'
            or record.get('seeds') != [17, 29, 43]
            or record.get('test_used') is not False
            or record.get('one_structure_for_all_seeds_and_weather') is not True
            or any(set(record.get('evidence', {}).get(v, {})) != {'17', '29', '43'} for v in ('S0', 'AB', 'N2'))):
        raise ValueError('F requires one AB/N2 structure frozen using all three development seeds')
    chosen, scores, normal = final_decision(record['evidence'], record.get('normal_tolerance_pp'))
    if chosen != record['variant']:
        raise ValueError('F variant disagrees with the frozen development rule')
    for key, expected in (('development_means', scores), ('normal_development_means', normal)):
        actual = record.get(key, {})
        if (set(actual) != set(expected) or any(not isinstance(actual[k], (int, float))
                or not math.isfinite(actual[k]) or abs(actual[k] - v) > 1e-10 for k, v in expected.items())):
            raise ValueError('F recorded development means disagree with their evidence')
    for variant in ('S0', 'AB', 'N2'):
        for seed in ('17', '29', '43'):
            row = record['evidence'][variant][seed]
            if (not isinstance(row.get('score'), (int, float)) or not 0 <= row['score'] <= 1
                    or set(row.get('scores', {})) != {'normal', 'fog'}
                    or len(row.get('checkpoint_sha256', '')) != 64):
                raise ValueError('F decision is missing complete development/checkpoint evidence')
    return record['variant']


def validate_supplement_config(config):
    c = _merge(defaults(), config)
    e, d, t, m, v = (c[k] for k in ('experiment', 'data', 'training', 'model', 'evaluation'))
    if e['dataset_id'] not in ('v2xr', 'v2xradar', 'kradar') or e['method'] not in METHODS:
        raise ValueError('unregistered supplemental dataset/method')
    if e['dataset_id'] == 'kradar' and e['method'] in ('cobevt', 'where2comm'):
        raise ValueError('registered K-Radar comparisons are single-agent S0/F/MDD/L4DR/AttFuse')
    if e['dataset_id'] == 'kradar' and e['comparison'] != 'local_single_agent':
        raise ValueError('K-Radar comparisons must identify local_single_agent')
    protocols = dict(v2xr='project_fix_v1', v2xradar='v2xradar_official_v1', kradar='kradar_revised_v2')
    if e['protocol'] != protocols[e['dataset_id']] or e['training_protocol'] != 'supplement_v1':
        raise ValueError('supplemental dataset/evaluator identity mismatch')
    if e['seed'] not in (17, 29, 43) or e['track'] not in ('C', 'W'):
        raise ValueError('registered seeds are 17/29/43; tracks are C/W')
    if e['variant'] not in (*INTERNAL, 'F'):
        raise ValueError('unregistered supplemental variant')
    if e['variant'] == 'F':
        if not e['final_model_file']:
            raise ValueError('F is unresolved: freeze AB/N2 from development results first')
        resolve_final(e['final_model_file'])
    if not e['run_id'] or Path(e['run_id']).name != e['run_id'] or any(x in e['run_id'] for x in ('/', '\\', ':')):
        raise ValueError('run_id must be one directory component')
    if e['comparison'] not in ('controlled_module', 'complete_method', 'local_single_agent'):
        raise ValueError('invalid comparison identity')
    if e['transfer'] not in ('retrained', 'zero_shot', 'fine_tuned'):
        raise ValueError('invalid transfer identity')
    native_kradar_l4dr = e['method'] == 'l4dr' and e['dataset_id'] == 'kradar' and t['epochs'] == 35
    if t['epochs'] not in SCHEDULES and not (e['method'] in ('cobevt', 'where2comm') and t['epochs'] == 40) and not native_kradar_l4dr:
        raise ValueError('registered budgets: 20/30/60/90; native CoBEVT/Where2comm 40; native K-Radar L4DR 35')
    if t['optimizer'] not in ('adam', 'adamw'):
        raise ValueError('registered optimizers are Adam and AdamW')
    if t['effective_batch_size'] not in ((8,) if e['method'] == 'deer_sbr' else (2, 3, 4, 8) if native_kradar_l4dr else (2, 4, 8)):
        raise ValueError('controlled batch is 8; external native batch is 2/4 (K-Radar L4DR: 3), or common batch 8')
    if t['micro_batch_size'] * t['accumulation_steps'] * t['world_size'] != t['effective_batch_size']:
        raise ValueError('micro batch * accumulation * world size must equal effective_batch_size')
    if min(t[k] for k in ('micro_batch_size', 'accumulation_steps', 'world_size', 'checkpoint_interval_updates')) <= 0:
        raise ValueError('batch sizes and checkpoint interval must be positive')
    if t['precision'] != 'fp32' or t['normalization'] != 'batchnorm':
        raise ValueError('supplement_v1 uses FP32 and online BatchNorm')
    if t['scheduler'] not in ('multistep', 'cosine_warmup'):
        raise ValueError('unregistered learning rate schedule')
    if (t['milestones'] != sorted(set(t['milestones']))
            or any(not 0 < x < t['epochs'] for x in t['milestones'])
            or not 0 <= t['warmup_epochs'] < t['epochs']):
        raise ValueError('invalid learning rate schedule bounds')
    if any(not 0 < x <= t['epochs'] for x in t['retain_epochs']):
        raise ValueError('retained epochs must be within the recipe')
    if t['lr'] <= 0 or t['weight_decay'] < 0 or not 0 < t['gamma'] <= 1:
        raise ValueError('invalid optimizer parameters')
    bn = t['batchnorm']
    if bn['evaluation_statistics'] not in ('running', 'train_refit_cma_v1'):
        raise ValueError('BN statistics may use train only or native running state')
    if bn['evaluation_statistics'] == 'train_refit_cma_v1' and (bn['refit_samples'] <= 0 or bn['refit_samples'] % bn['refit_micro_batch_size']):
        raise ValueError('BN train subset must contain complete micro batches')
    if e['method'] != 'deer_sbr' and bn['evaluation_statistics'] != 'running':
        raise ValueError('complete external methods retain native normalization')
    if e['method'] not in ('deer_sbr', 'where2comm') and e['variant'] != 'S0':
        raise ValueError('external complete methods do not accept internal variants')
    if e['method'] == 'where2comm' and m['local_variant'] not in ('none', 'AB', 'N2', 'F'):
        raise ValueError('Where2comm local module must be none/AB/N2/frozen F')
    if e['method'] == 'where2comm' and ((e['variant'] == 'S0' and m['local_variant'] != 'none')
            or (e['variant'] != 'S0' and m['local_variant'] != e['variant'])):
        raise ValueError('Where2comm variant identity must match the actual local module')
    if e['method'] == 'deer_sbr' or (e['method'] == 'where2comm' and m['local_variant'] != 'none'):
        actual = resolve_final(e['final_model_file']) if e['variant'] == 'F' else e['variant']
        if (m['amplitude_context'] != ('normalized_mean' if actual == 'N1' else 'rms')
                or m['type_prior_mode'] != ('multiplicative' if actual == 'N2' else 'input_only')
                or m['null_donor'] != (actual != 'NO_NULL')):
            raise ValueError('internal variant must match its actual amplitude/type-prior/null-donor mechanism')
    if m['local_variant'] == 'F' and not e['final_model_file']:
        raise ValueError('Where2comm+F requires a frozen final-model record')
    if e['dataset_id'] != 'v2xr' and (c['selection']['domains'] != ['natural'] or e['track'] != 'C'):
        raise ValueError('real-data selection is natural development data only')
    if e['dataset_id'] == 'v2xr' and c['selection']['domains'] != (['normal', 'fog'] if e['track'] == 'W' else ['normal']):
        raise ValueError('V2X-R development domains must exclude Snow')
    if c['selection']['metric'] != 'ap3d' or c['selection']['iou'] != .7 or c['selection']['tie_tolerance'] != 1e-6:
        raise ValueError('supplement_v1 model selection is frozen to development AP3D07 with tolerance 1e-6')
    if c['selection']['normal_tolerance_pp'] < 0:
        raise ValueError('Normal tolerance must be fixed nonnegative percentage points')
    if c['scoring'] != DEFAULTS['scoring']:
        raise ValueError('supplement_v1 postprocessing/IoU thresholds are registered; do not tune them on final results')
    if e['method'] == 'mdd' and e['dataset_id'] != 'v2xr' and not d['clean_only']:
        raise ValueError('MDD-CORR real training requires verified clear-only references')
    if d['supervision_source'] not in ('none', 'provenance_v1'):
        raise ValueError('supplement uses only traceable controlled type targets')
    if d['paired_supervision']:
        raise ValueError('MDD reference inputs are explicit and separate from DEER supervision')
    if v['kind'] not in ('simulated_weather', 'natural', 'artificial_corruption'):
        raise ValueError('weather and corruption result identities must be explicit')
    if not 0 <= v['drop_fraction'] <= 1 or not 0 <= v['inject_fraction'] <= 1 or not 0 <= v['radar_keep'] <= 1:
        raise ValueError('corruption fractions must be in [0,1]')
    if v['agent_scope'] not in ('ego', 'neighbors', 'all') or not 1 <= v['agents'] <= 5 or not 1 <= v['min_agents'] <= 5:
        raise ValueError('invalid agent intervention')
    if v['missing_modality'] not in ('none', 'lidar', 'radar', 'both'):
        raise ValueError('invalid modality missingness')
    if v['intervention'] not in ('none', 'type_zero', 'type_shuffle', 'amplitude_zero', 'amplitude_shuffle', 'donor_shuffle', 'donor_zero'):
        raise ValueError('unknown fixed-weight intervention')
    if v['corruption'] != 'none':
        raise ValueError('corruption uses explicit fractions and kind; no unnamed corruption implementation exists')
    if v['kind'] != 'artificial_corruption' and (v['drop_fraction'] or v['inject_fraction'] or v['radar_keep'] != 1 or v['missing_modality'] != 'none'):
        raise ValueError('artificial perturbations must be named artificial_corruption, separately from natural/physical weather')
    if min(v['position_std_m'], v['yaw_std_deg'], v['delay_frames']) < 0:
        raise ValueError('pose/delay perturbations must be nonnegative')
    if e['dataset_id'] == 'kradar' and (v['agents'] != 1 or v['delay_frames'] or v['position_std_m'] or v['yaw_std_deg']):
        raise ValueError('K-Radar uses one agent, without cooperative pose/delay perturbations')
    if m['rho'] <= 0 or m['support_radius_m'] <= 0 or not 0 <= m['route_strength'] < 1:
        raise ValueError('invalid residual/routing parameters')
    if (min(m['hidden_channels'], m['attention_channels']) <= 0 or not 0 <= m['g_min'] <= 1
            or not 0 < m['scale_floor'] < m['scale_ceiling'] or m['scale_mode'] != 'current_observation_rms'):
        raise ValueError('invalid local module size or amplitude scale bounds')
    from opencood.project.supervision import controlled_corruption_config
    controlled_corruption_config(d['controlled_corruption'])
    # Reject nonfinite values at the configuration boundary.
    json.dumps(c, allow_nan=False)
    return c
