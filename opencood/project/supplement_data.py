"""Published-format readers and immutable GT for supplemental real data.

V2X-Radar-C: official OPV2V export; positive-ID vehicle ego, negative-ID RSU.
K-Radar: label v2.0, OS2-64 PCD and official sprdr_*.npy, one Radar-frame agent.
No download, guessed calibration, or implicit replacement cloud is performed.
"""
from copy import deepcopy
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import re
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.project.data_io import read_annotation
from opencood.project.data_geometry import (augmentation_matrix, augment_boxes, bev_grid,
    count_map)
from opencood.project.geometry import (pose_to_world, transform_points,
    boxes_to_corners, validate_boxes)
from opencood.project.supervision import controlled_corruption, provenance_targets, normal_targets
from opencood.project.weather import augmentation_seed

V2XRADAR_BOUNDS = [-102.4, -102.4, -8.5, 102.4, 102.4, 3.5]
KRADAR_BOUNDS = [0., -15., -2., 72., 15., 7.6]
KRADAR_CLASSES = ('Sedan', 'Bus or Truck')
SOURCE_REVISIONS = {'v2xradar': '1624751e31260c68d544e60599079d2104ee8a73',
                    'kradar': 'f90c28993d1757ce0236632b4d912c92a14976b2'}


def read_pcd_fields(path):
    """Read actual named PCD fields, including organized/binary/compressed PCD."""
    path = Path(path)
    with path.open('rb') as stream:
        header = {}
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f'truncated PCD header: {path}')
            line = raw.decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, _, values = line.partition(' ')
            header[key] = values.split()
            if key == 'DATA':
                break
        payload = stream.read()
    fields = header['FIELDS']
    sizes = list(map(int, header['SIZE']))
    counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
    if not len(fields) == len(sizes) == len(counts) == len(header['TYPE']):
        raise ValueError(f'inconsistent PCD schema: {path}')
    kind = {'F': 'f', 'I': 'i', 'U': 'u'}
    types = [np.dtype('<' + kind[t] + str(s)) for t, s in zip(header['TYPE'], sizes)]
    n = int(header['POINTS'][0])
    dtype = np.dtype([(name, typ, (count,)) if count != 1 else (name, typ)
                      for name, typ, count in zip(fields, types, counts)])
    mode = header['DATA'][0]
    if mode == 'ascii':
        values = np.loadtxt(io.BytesIO(payload), ndmin=2) if n else np.empty((0, sum(counts)))
        if values.shape != (n, sum(counts)):
            raise ValueError(f'PCD point count/schema mismatch: {path}')
        result, offset = {}, 0
        for name, count, typ in zip(fields, counts, types):
            value = values[:, offset:offset + count].astype(typ)
            result[name] = value[:, 0] if count == 1 else value
            offset += count
    elif mode == 'binary':
        if len(payload) != n * dtype.itemsize:
            raise ValueError(f'PCD binary byte count mismatch: {path}')
        values = np.frombuffer(payload, dtype=dtype, count=n)
        result = {name: values[name].copy() for name in fields}
    elif mode == 'binary_compressed':
        import lzf
        compressed, uncompressed = struct.unpack('<II', payload[:8])
        if compressed != len(payload) - 8 or uncompressed != n * dtype.itemsize:
            raise ValueError(f'PCD compressed byte count mismatch: {path}')
        data = lzf.decompress(payload[8:], uncompressed)
        if data is None:
            raise ValueError(f'invalid LZF PCD payload: {path}')
        result, offset = {}, 0
        for name, typ, count in zip(fields, types, counts):
            size = n * typ.itemsize * count
            value = np.frombuffer(data[offset:offset + size], dtype=typ).reshape(n, count)
            result[name] = value[:, 0].copy() if count == 1 else value.copy()
            offset += size
    else:
        raise ValueError(f'unsupported PCD DATA={mode}: {path}')
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f'nonfinite PCD field: {path}')
    return result


def real_points(path, dataset, modality, agent_id='0'):
    """Return xyz plus one documented scalar, never a fabricated Doppler field."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'missing {dataset} {modality}: {path}; prepare official files first (no automatic download)')
    if dataset == 'kradar' and modality == 'radar':
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[1] < 5 or not np.isfinite(values).all():
            raise ValueError('official K-Radar sprdr requires [x,y,z,power,Doppler,...], at least five columns')
        # sprdr is the already generated official sparse product; do not apply
        # a second 1e13 normalization or interpret power as velocity.
        return np.ascontiguousarray(values[:, :4], dtype=np.float32)
    fields = read_pcd_fields(path)
    for name in ('x', 'y', 'z'):
        if name not in fields:
            raise ValueError(f'missing PCD coordinate {name}: {path}')
    xyz = np.column_stack([fields[k] for k in ('x', 'y', 'z')])
    if dataset == 'v2xradar' and modality == 'radar':
        if not {'dopple', 'intensity'} <= fields.keys():
            raise ValueError('V2X-Radar published radar schema requires dopple and intensity; aliases are not guessed')
        # Official read_radar, revision 1624751. The native four-channel models
        # use column 3; the fifth channel is recorded but not silently added.
        scalar = (np.clip(fields['dopple'] / 256., 0, 1) if int(agent_id) < 0
                  else np.clip((fields['dopple'] + 20.) / 35., 0, 1))
    elif 'intensity' in fields:
        scalar = fields['intensity'] / (256. if dataset == 'v2xradar' else 1.)
    elif dataset == 'v2xradar' and 'rgb' in fields:
        scalar = fields['rgb']  # official read_lidar uses the published raw field
    else:
        raise ValueError(f'official intensity field missing: {path}')
    result = np.ascontiguousarray(np.column_stack((xyz, scalar)), dtype=np.float32)
    if dataset == 'kradar':
        result = result[(np.abs(result[:, 0]) > .01) | (np.abs(result[:, 1]) > .01)]
    return result


def kradar_calibration(path, z_offset=.7):
    lines = Path(path).read_text(encoding='utf8').splitlines()
    values = [float(x.strip()) for x in lines[1].split(',')]
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError('K-Radar calibration second line must be frame_delta,x,y')
    return np.array([values[1], values[2], z_offset], dtype=np.float64)


def kradar_label(path, calibration, version='v2_0'):
    """Official v2.0 labels: degrees, half extents, LiDAR->Radar translation."""
    if version != 'v2_0':
        raise ValueError('this registered K-Radar experiment uses revised label v2_0 only')
    text = Path(path).read_text(encoding='utf8').strip()
    # A published v2_0 formatting issue occasionally joins the first object
    # directly to the header; preserve it as the official reader does.
    lines = [line.strip() for line in text.replace('*', '\n*').splitlines() if line.strip()]
    header = lines[0]
    match = re.search(r'idx\(rdr,ldr64,camf,ldr128,camr\)\s*=\s*([0-9_]+)', header)
    stamp = re.search(r'timestamp\s*=\s*([0-9.]+)', header)
    if not match or not stamp or len(match[1].split('_')) != 5:
        raise ValueError(f'K-Radar v2_0 synchronization header not recognized: {path}')
    indices = dict(zip(('rdr', 'ldr64', 'camf', 'ldr128', 'camr'), match[1].split('_')))
    objects = []
    for line in lines[1:]:
        parts = [x.strip() for x in line.split(',')]
        if len(parts) != 10 or parts[0] != '*':
            raise ValueError(f'K-Radar v2_0 object schema mismatch: {path}: {line}')
        if parts[2] not in KRADAR_CLASSES:
            continue
        values = np.asarray(list(map(float, parts[3:])))
        box = np.r_[values[:3] + calibration, 2 * values[4:7], np.deg2rad(values[3])]
        validate_boxes([box])
        objects.append(dict(gt_id=parts[1], class_name=parts[2], box=box.tolist()))
    return indices, stamp[1], objects


def inside_region(boxes, bounds, dataset):
    boxes = validate_boxes(boxes)
    bounds = np.asarray(bounds)
    if dataset == 'v2xradar':
        # Official mask_boxes_outside_range_numpy uses >=1 legal corner.
        corners = boxes_to_corners(boxes)
        return ((corners >= bounds[:3]) & (corners <= bounds[3:])).all(2).any(1)
    valid = ((boxes[:, :3] >= bounds[:3]) & (boxes[:, :3] <= bounds[3:])).all(1)
    return valid & (np.abs(np.rad2deg(np.arctan2(boxes[:, 1], boxes[:, 0]))) <= 53.)


def build_real_manifest(dataset, root, dev_sequences=(), *, radar_root=None,
                        label_root=None, train_split=None, test_split=None,
                        clean_sequences=(), weather_by_sequence=None, dev_count=None, split_seed=17):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'official data root missing: {root}; no files will be downloaded')
    dev = set(map(str, dev_sequences))
    def choose_development(train_sequences):
        if dev:
            return dev
        if dev_count is None or not 1 <= dev_count < len(train_sequences):
            raise ValueError('explicit sequence holdout or a valid development sequence count is required')
        rng = np.random.default_rng(split_seed)
        return set(rng.choice(sorted(train_sequences), dev_count, replace=False).tolist())
    samples, bounds = {}, V2XRADAR_BOUNDS if dataset == 'v2xradar' else KRADAR_BOUNDS
    if dataset == 'v2xradar':
        train_seqs = {p.name for p in (root / 'train').iterdir() if p.is_dir()}
        dev = choose_development(train_seqs)
        if not dev < train_seqs:
            raise ValueError('development sequences must be a proper subset of official train')
        if not set(clean_sequences) <= train_seqs:
            raise ValueError('reviewed clear sequences must belong to official train')
        final_seqs = {p.name for p in (root / 'validate').iterdir() if p.is_dir()}
        if train_seqs & final_seqs:
            raise ValueError('official train/final validation sequences overlap')
        for native_split, use_split in (('train', 'train'), ('validate', 'test')):
            for sequence in sorted((root / native_split).iterdir()):
                if not sequence.is_dir():
                    continue
                agents = sorted([p.name for p in sequence.iterdir() if p.is_dir() and p.name.lstrip('-').isdigit()], key=int)
                vehicles = [a for a in agents if int(a) >= 0]
                if not vehicles or not any(int(a) < 0 for a in agents):
                    raise ValueError(f'V2X-Radar-C requires vehicle ego and real roadside agent: {sequence}')
                ego = vehicles[0]
                for annotation in sorted((sequence / ego).glob('*.yaml')):
                    if 'additional' in annotation.name:
                        continue
                    timestamp = annotation.stem
                    ordered = [ego] + [a for a in agents if a != ego]
                    annotations, records = {}, {}
                    for agent in ordered:
                        base = sequence / agent / timestamp
                        paths = dict(annotation=base.with_suffix('.yaml'), lidar=base.with_suffix('.pcd'),
                                     radar=base.with_name(timestamp + '_radar.pcd'))
                        missing = [str(p) for p in paths.values() if not p.is_file()]
                        if missing:
                            raise FileNotFoundError('incomplete synchronized cooperative frame: ' + ', '.join(missing))
                        annotations[agent] = read_annotation(paths['annotation'])
                        records[agent] = {k: p.relative_to(root).as_posix() for k, p in paths.items()}
                        records[agent]['pose'] = list(map(float, annotations[agent]['lidar_pose']))
                    gt = {}
                    from opencood.utils import box_utils
                    for agent in ordered:
                        objects = deepcopy(annotations[agent]['vehicles'])
                        for obj in objects.values():
                            obj.setdefault('center', [0, 0, 0])
                        projected = {}
                        # Same official create_bbx -> x1_to_x2 -> corner_to_center
                        # -> any-corner ROI sequence, including roll/pitch.
                        box_utils.project_world_objects(objects, projected, annotations[ego]['lidar_pose'],
                            bounds, order='lwh', dataset='opv2v')
                        for key, box in projected.items():
                            box = np.asarray(box).reshape(7)
                            validate_boxes([box])
                            gt.setdefault(str(key), dict(gt_id=str(key), class_name='Vehicle', box=box.tolist()))
                    split = 'validate' if native_split == 'train' and sequence.name in dev else use_split
                    sid = f'{native_split}/{sequence.name}/{timestamp}'
                    samples[sid] = dict(split=split, official_split=native_split, sequence=sequence.name,
                        timestamp=timestamp, ego_id=ego, agent_ids=ordered, agents=records, gt=list(gt.values()),
                        natural_weather=(weather_by_sequence or {}).get(sequence.name, 'unlabeled'),
                        verified_clear=sequence.name in clean_sequences)
        final_identity = 'official_validate_reserved_as_final'
        fields = dict(lidar='official intensity/256 (rgb fallback verbatim)',
            radar='official normalized dopple: RSU clip(raw/256), vehicle clip((raw+20)/35)',
            omitted='official fifth scalar intensity normalization; no calibrated velocity claim',
            units='xyz metres; raw dopple/intensity units not established by public reader; normalized scalar dimensionless')
    elif dataset == 'kradar':
        if not all((radar_root, label_root, train_split, test_split)):
            raise ValueError('K-Radar requires official sparse radar, revised v2_0 labels and official train/test lists')
        radar_root, label_root = Path(radar_root).resolve(), Path(label_root).resolve()
        def split_entries(path):
            return [tuple(x.strip() for x in line.split(',')) for line in Path(path).read_text().splitlines() if line.strip()]
        train, test = split_entries(train_split), split_entries(test_split)
        if set(train) & set(test):
            raise ValueError('K-Radar official frame lists overlap')
        train_seqs = {s for s, _ in train}
        dev = choose_development(train_seqs)
        if not dev < train_seqs:
            raise ValueError('development sequences must be a proper subset of official training sequences')
        # The official K-Radar lists include all 58 sequences in BOTH splits.
        # Hold out development sequences from official train, while preserving
        # every official final-test frame and recording this benchmark overlap.
        final_seqs = {s for s, _ in test}
        dropped_train = 0
        for official_split, entries in (('train', train), ('test', test)):
            for seq, filename in entries:
                base = root / seq
                calibration_path = base / 'info_calib/calib_radar_lidar.txt'
                calibration = kradar_calibration(calibration_path)
                label = label_root / seq / filename
                indices, timestamp, gt = kradar_label(label, calibration)
                gt = [g for g, ok in zip(gt, inside_region([x['box'] for x in gt], bounds, dataset)) if ok]
                lidar = base / 'os2-64' / f"os2-64_{indices['ldr64']}.pcd"
                radar = radar_root / seq / f"sprdr_{indices['rdr']}.npy"
                for path in (lidar, radar, label):
                    if not path.is_file():
                        raise FileNotFoundError(f'official K-Radar input missing: {path}')
                description = [x.strip() for x in (base / 'description.txt').read_text().split(',')]
                if len(description) != 3:
                    raise ValueError('K-Radar description requires road_type,capture_time,climate')
                climate = description[2]
                split = 'test' if official_split == 'test' else 'validate' if seq in dev else 'train'
                sid = seq + '/' + filename.removesuffix('.txt')
                samples[sid] = dict(split=split, official_split=official_split, sequence=seq, timestamp=timestamp,
                    ego_id='0', agent_ids=['0'], gt=gt, natural_weather=climate,
                    verified_clear=(climate.lower() == 'normal' or seq in clean_sequences),
                    agents={'0': dict(lidar=lidar.relative_to(root).as_posix(),
                        radar=radar.relative_to(radar_root).as_posix(), annotation=label.relative_to(label_root).as_posix(),
                        calibration=calibration.tolist(), pose=[0.] * 6)})
        final_identity = 'official_test_frames_unchanged_official_sequence_overlap_disclosed'
        fields = dict(lidar='OS2-64 intensity, native sensor units; xyz translated to Radar',
                      radar='official sprdr [x,y,z,power,Doppler,...]; power is column 3, already generated product',
                      units='xyz metres; yaw degrees converted to rad; Doppler m/s omitted from four-channel models',
                      density='same official sparse extraction for all methods; no tensor extraction performed')
    else:
        raise ValueError('only v2xradar and kradar use this preparation path')
    partitions = {s: sorted({v['sequence'] for v in samples.values() if v['split'] == s}) for s in ('train', 'validate', 'test')}
    if any(not seqs for seqs in partitions.values()):
        raise ValueError('sequence holdout leaves an empty train/development/final partition')
    pairs = (('train', 'validate'), ('train', 'test'), ('validate', 'test')) if dataset == 'v2xradar' else (('train', 'validate'),)
    if any(set(partitions[a]) & set(partitions[b]) for a, b in pairs):
        raise ValueError('sequence leakage between partitions')
    for sid, sample in samples.items():
        sample['gt_sha256'] = hashlib.sha256(json.dumps(sample['gt'], sort_keys=True).encode()).hexdigest()
    result = dict(schema='supplement_real_manifest_v1', dataset=dataset, bounds=bounds,
        classes=['Vehicle'] if dataset == 'v2xradar' else list(KRADAR_CLASSES), samples=samples,
        sequence_partitions=partitions, final_identity=final_identity, fields=fields,
        source_revision=SOURCE_REVISIONS[dataset], label_version='opv2v_vehicle' if dataset == 'v2xradar' else 'v2_0',
        roi_rule='any_corner_xyz' if dataset == 'v2xradar' else 'center_xyz_azimuth53',
        official_train_frames_purged=dropped_train if dataset == 'kradar' else 0)
    result['development_holdout'] = dict(sequences=sorted(dev), split_seed=split_seed,
        source='explicit_sequence_ids' if dev_sequences else 'seeded_train_sequence_sample', dev_count=len(dev))
    result['summary'] = {}
    for split in ('train', 'validate', 'test'):
        part = [s for s in samples.values() if s['split'] == split]
        result['summary'][split] = dict(num_samples=len(part), num_gt=sum(len(s['gt']) for s in part),
            class_gt_counts={name: sum(g['class_name'] == name for s in part for g in s['gt']) for name in result['classes']},
            verified_clear_frames=sum(s['verified_clear'] for s in part))
    result['protocol_sha256'] = hashlib.sha256(json.dumps({k: result[k] for k in ('dataset', 'bounds', 'classes', 'source_revision', 'label_version', 'roi_rule', 'final_identity')}, sort_keys=True).encode()).hexdigest()
    return result


def perturb_points(lidar, radar, valid, *, sample_id, agent, ego, config, bounds, voxel, legal_mask=None):
    """A method-independent sample/agent-keyed perturbation schedule."""
    v = config['evaluation']
    valid = np.asarray(valid, dtype=bool).copy()
    seed = augmentation_seed(sample_id + '/' + str(agent), 0, v['perturbation_seed'], stream='supplement_robustness')
    rng = np.random.default_rng(seed)
    trace = dict(seed=seed, reference_lidar_points=len(lidar), reference_radar_points=len(radar))
    scope = v['agent_scope'] == 'all' or (v['agent_scope'] == 'ego') == (str(agent) == str(ego))
    target = None
    if scope:
        if v['kind'] == 'artificial_corruption' and (v['drop_fraction'] or v['inject_fraction']):
            reference = lidar.copy()
            def legal(points):
                b = np.asarray(bounds)
                result = ((points[:, :3] >= b[:3]) & (points[:, :3] < b[3:])).all(1)
                return result if legal_mask is None else result & legal_mask(points)
            lidar, origins, _ = controlled_corruption(lidar, rng, dict(probability=1., drop_fraction=v['drop_fraction'],
                inject_fraction=v['inject_fraction'], max_offset_m=v['max_offset_m']), legal)
            target = provenance_targets(reference, lidar, origins, bounds, voxel)
            trace.update(target.pop('stats'))
        if v['radar_keep'] < 1:
            ids = rng.permutation(len(radar))[:int(round(len(radar) * v['radar_keep']))]
            radar = radar[ids]
        for index, name in enumerate(('lidar', 'radar')):
            if v['missing_modality'] in (name, 'both'):
                valid[index] = False
        if not valid[0]:
            lidar = lidar[:0]
        if not valid[1]:
            radar = radar[:0]
        if not valid[0] and target is not None:
            target = {key: np.zeros_like(value) for key, value in target.items()}
    trace.update(scope_applied=scope, observed_lidar_points=len(lidar), observed_radar_points=len(radar))
    return lidar, radar, valid, target, trace


def perturbed_transform(transform, config, sample_id, agent, ego):
    if str(agent) == str(ego):
        return transform
    v = config['evaluation']
    rng = np.random.default_rng(augmentation_seed(sample_id + '/' + str(agent), 0, v['perturbation_seed'], stream='supplement_pose'))
    out = np.eye(4)
    out[:2, 3] = rng.normal(0, v['position_std_m'], 2)
    yaw = np.deg2rad(rng.normal(0, v['yaw_std_deg']))
    c, s = np.cos(yaw), np.sin(yaw)
    out[:2, :2] = [[c, -s], [s, c]]
    return out @ transform


class NumpyVoxelizer:
    """Explicit CPU hard voxelization, preserving first-seen caps and empty agents."""
    def __init__(self, hypes, training):
        p = hypes['preprocess']
        self.bounds = np.asarray(p['cav_lidar_range'])
        self.voxel = np.asarray(p['args']['voxel_size'])
        self.max_points = p['args']['max_points_per_voxel']
        self.max_voxels = p['args']['max_voxel_train' if training else 'max_voxel_test']

    def preprocess(self, points):
        ids = np.floor((points[:, :3] - self.bounds[:3]) / self.voxel).astype(np.int64)
        legal = ((points[:, :3] >= self.bounds[:3]) & (points[:, :3] < self.bounds[3:])).all(1)
        points, ids = points[legal], ids[legal]
        lookup, coords, groups = {}, [], []
        for point, coord in zip(points, ids):
            key = tuple(coord[::-1])
            if key not in lookup:
                if len(groups) >= self.max_voxels:
                    continue
                lookup[key] = len(groups)
                groups.append([])
                coords.append(key)
            group = groups[lookup[key]]
            if len(group) < self.max_points:
                group.append(point)
        features = np.zeros((len(groups), self.max_points, 4), np.float32)
        counts = np.asarray([len(x) for x in groups], dtype=np.int32)
        for i, group in enumerate(groups):
            features[i, :len(group)] = group
        return dict(voxel_features=features, voxel_coords=np.asarray(coords, np.int32).reshape(-1, 3), voxel_num_points=counts)

    def collate_batch(self, entries):
        coords = [np.pad(x['voxel_coords'], ((0, 0), (1, 0)), constant_values=i) for i, x in enumerate(entries)]
        return {k: torch.from_numpy(np.concatenate(coords if k == 'voxel_coords' else [x[k] for x in entries]))
                for k in ('voxel_features', 'voxel_coords', 'voxel_num_points')}


def anchors_for(hypes, class_name='Vehicle'):
    a = hypes['postprocess']['anchor_args']
    b = hypes['preprocess']['cav_lidar_range']
    stride = a.get('feature_stride', 2)
    x = np.linspace(b[0] + a['vw'], b[3] - a['vw'], a['W'] // stride)
    y = np.linspace(b[1] + a['vh'], b[4] - a['vh'], a['H'] // stride)
    xx, yy = np.meshgrid(x, y)
    result = np.zeros((*xx.shape, 2, 7), np.float32)
    result[..., 0] = xx[..., None]
    result[..., 1] = yy[..., None]
    result[..., 2] = -1.
    result[..., 3:6] = [a['h'], a['w'], a['l']]
    if class_name in KRADAR_CLASSES:
        result[..., 2] = 1.
        result[..., 3:6] = [2., 2.1, 4.2] if class_name == 'Sedan' else [3., 2.6, 10.]
    result[..., 1, 6] = np.pi / 2
    return result


class SupplementalDataset(Dataset):
    """Real-data dataset sharing the project's tensor, target and sampler contracts."""
    def __init__(self, config, hypes, split, *, training=False, augment=False, weather='natural'):
        self.config, self.hypes = deepcopy(config), deepcopy(hypes)
        d = config['data']
        self.root = Path(d['root'])
        path = Path(d['manifest_dir']) / 'supplement_manifest.json'
        if not path.is_file():
            raise FileNotFoundError(f'prepared real-data manifest missing: {path}; run scripts/supplement.py prepare first')
        self.manifest = json.loads(path.read_text(encoding='utf8'))
        self.dataset_id = config['experiment']['dataset_id']
        if self.manifest['dataset'] != self.dataset_id:
            raise ValueError('manifest dataset identity mismatch')
        self.training, self.split, self.augment = training, split, augment
        if training and split != 'train':
            raise ValueError('training and BN fitting require the train partition')
        self.weather = weather
        self.classes = self.manifest['classes']
        self.bounds = hypes['preprocess']['cav_lidar_range']
        if list(self.bounds) != self.manifest['bounds']:
            raise ValueError('model ROI and immutable real-data manifest disagree')
        self.voxel = hypes['preprocess']['args']['voxel_size']
        self.grid = bev_grid(self.bounds, self.voxel)
        self.epoch_value = mp.Value('q', 0)
        self.max_agents = config['evaluation']['agents']
        self.seed = config['experiment']['seed']
        self.pre_processor = NumpyVoxelizer(hypes, training)
        self.anchors = {name: anchors_for(hypes, name) for name in self.classes}
        self.anchor_box = self.anchors[self.classes[0]]
        from opencood.project.anchor_targets import AnchorTargets
        ta = hypes['postprocess']['target_args']
        self.assigners = {name: AnchorTargets(anchor, ta['pos_threshold'], ta['neg_threshold']) for name, anchor in self.anchors.items()}
        self.sample_ids = sorted(sid for sid, sample in self.manifest['samples'].items() if sample['split'] == split)
        history_ids = list(self.sample_ids)
        if training and d['clean_only']:
            self.sample_ids = [sid for sid in self.sample_ids if self.manifest['samples'][sid]['verified_clear']]
            if not self.sample_ids:
                raise ValueError('MDD-CORR has no verified clear train frames; provide reviewed clear sequences; never use natural rain as clean target')
        v = config['evaluation']
        if not training:
            self.sample_ids = [sid for sid in self.sample_ids if len(self.manifest['samples'][sid]['agent_ids']) >= v['min_agents']]
            if v['natural_weather']:
                self.sample_ids = [sid for sid in self.sample_ids if self.manifest['samples'][sid]['natural_weather'] == v['natural_weather']]
            if v['sample_ids']:
                selected = json.loads(Path(v['sample_ids']).read_text())
                unknown = set(selected) - set(self.sample_ids)
                if unknown:
                    raise ValueError(f'common sample list contains unavailable frames: {sorted(unknown)[:3]}')
                self.sample_ids = sorted(selected)
        self.delayed = {}
        if not training and v['delay_frames']:
            for seq in self.manifest['sequence_partitions'][split]:
                ids = sorted(sid for sid in history_ids if self.manifest['samples'][sid]['sequence'] == seq)
                for i in range(v['delay_frames'], len(ids)):
                    self.delayed[ids[i]] = ids[i - v['delay_frames']]
            self.sample_ids = [sid for sid in self.sample_ids if sid in self.delayed and set(self.manifest['samples'][sid]['agent_ids']) <= set(self.manifest['samples'][self.delayed[sid]]['agent_ids'])]
        if not self.sample_ids:
            raise ValueError(f'no {split} samples after explicit sequence/condition filtering')

    def __len__(self):
        return len(self.sample_ids)

    def set_epoch(self, epoch):
        self.epoch_value.value = int(epoch)

    def _points(self, record, modality):
        d = self.config['data']
        base = (Path(d['radar_root']) if self.dataset_id == 'kradar' and modality == 'radar' else self.root)
        return real_points(base / record[modality], self.dataset_id, modality, record['agent_id'])

    def __getitem__(self, index):
        sid = self.sample_ids[index]
        sample = self.manifest['samples'][sid]
        epoch = self.epoch_value.value
        rng = np.random.default_rng(augmentation_seed(sid, epoch, self.seed))
        aug = augmentation_matrix(rng, self.training and self.augment)
        current, targets, refs, metadata = [], [], [], []
        ego = sample['ego_id']
        world_to_ego = np.linalg.inv(pose_to_world(sample['agents'][ego]['pose']))
        agents = ([ego] + [a for a in sample['agent_ids'] if a != ego])[:self.max_agents]
        for agent in agents:
            source_sample = self.manifest['samples'][self.delayed[sid]] if sid in self.delayed and agent != ego else sample
            record = {**source_sample['agents'][agent], 'agent_id': agent}
            lidar, radar = (self._points(record, m) for m in ('lidar', 'radar'))
            transform = world_to_ego @ pose_to_world(record['pose'])
            if not self.training:
                transform = perturbed_transform(transform, self.config, sid, agent, ego)
            if self.dataset_id == 'kradar':
                lidar[:, :3] += np.asarray(record['calibration'], np.float32)
            for points in (lidar, radar):
                points[:, :3] = transform_points(points[:, :3], aug @ transform)
            def crop(points):
                b = np.asarray(self.bounds)
                keep = ((points[:, :3] >= b[:3]) & (points[:, :3] < b[3:])).all(1)
                return np.ascontiguousarray(points[keep])
            lidar, radar = crop(lidar), crop(radar)
            reference = lidar.copy()
            target = normal_targets(lidar, self.bounds, self.voxel)
            if not sample['verified_clear']:
                target = {k: np.zeros_like(v) for k, v in target.items()}
            applied = False
            valid = np.ones(2, dtype=bool)
            if self.training and sample['verified_clear']:
                corr_rng = np.random.default_rng(augmentation_seed(sid + '/' + agent, epoch, self.seed, stream='controlled_corruption'))
                lidar, origins, applied = controlled_corruption(lidar, corr_rng, self.config['data']['controlled_corruption'],
                    legal_mask=lambda p: ((p[:, :3] >= np.asarray(self.bounds[:3])) & (p[:, :3] < np.asarray(self.bounds[3:]))).all(1))
                if self.config['data']['supervision_source'] == 'provenance_v1':
                    target = provenance_targets(reference, lidar, origins, self.bounds, self.voxel)
                    target.pop('stats')
            trace = None
            if not self.training:
                lidar, radar, valid, explicit, trace = perturb_points(lidar, radar, valid, sample_id=sid,
                    agent=agent, ego=ego, config=self.config, bounds=self.bounds, voxel=self.voxel)
                if explicit is not None and sample['verified_clear']:
                    target = explicit
            target['normal_mask'] = np.full((1, 1, 1), float(sample['verified_clear'] and not applied), np.float32)
            origin = (aug @ transform)[:3, 3]
            lc, rc = (count_map(p, self.bounds, self.voxel, self.grid[0].shape) for p in (lidar, radar))
            # No camera FOV is borrowed from V2X-R. Valid geometry is the
            # dataset's own ROI; Radar donors always require measured occupancy.
            available = np.ones_like(lc)
            distance = np.sqrt((self.grid[0] - origin[0]) ** 2 + (self.grid[1] - origin[1]) ** 2)[None]
            distance = (distance / np.linalg.norm(np.asarray(self.bounds[3:5]) - np.asarray(self.bounds[:2]))).astype(np.float32)
            maps = dict(lidar_count=lc, radar_count=rc, lidar_occupancy=(lc > 0).astype(np.float32),
                radar_occupancy=(rc > 0).astype(np.float32), lidar_valid=available * valid[0],
                radar_valid=available * valid[1], lidar_distance=distance, radar_distance=distance.copy())
            if self.training:
                prng = np.random.default_rng(augmentation_seed(sid + '/' + agent, epoch, self.seed, stream='point_order'))
                lidar, radar = lidar[prng.permutation(len(lidar))], radar[prng.permutation(len(radar))]
            current.append(dict(lidar=self.pre_processor.preprocess(lidar), radar=self.pre_processor.preprocess(radar), maps=maps, valid=valid,
                                raw_radar=radar if self.config['experiment']['method'] == 'l4dr' else None))
            if self.config['experiment']['method'] == 'mdd' and self.training:
                if not sample['verified_clear']:
                    raise ValueError('unverified natural-weather clean reference reached MDD')
                refs.append(self.pre_processor.preprocess(reference))
            targets.append(target)
            metadata.append(dict(agent_id=agent, weather=sample['natural_weather'], lidar_points=len(lidar),
                radar_points=len(radar), source_timestamp=source_sample['timestamp'],
                source_id=self.delayed[sid] if sid in self.delayed and agent != ego else sid,
                agent_to_ego=transform.tolist(), perturbation_trace=trace,
                modality_valid=valid.tolist(), controlled_corruption=applied))
        labels, diagnostics = {}, []
        for name, assigner in self.assigners.items():
            boxes = np.asarray([g['box'] for g in sample['gt'] if g['class_name'] == name]).reshape(-1, 7)
            boxes = augment_boxes(boxes, aug) if self.training else boxes
            labels[name] = assigner(boxes)
            diagnostics.append(deepcopy(assigner.last_diagnostics))
        fad_boxes = augment_boxes(np.asarray([g['box'] for g in sample['gt']]).reshape(-1, 7), aug)
        return dict(agents=current, sample_id=sid, labels=labels, targets=targets, refs=refs, fad_boxes=fad_boxes,
                    metadata=metadata, target_diagnostics=diagnostics[0])

    def collate_batch(self, batch):
        agents = [a for item in batch for a in item['agents']]
        labels = {name: {k: torch.from_numpy(np.stack([item['labels'][name][k] for item in batch]))
                        for k in batch[0]['labels'][name]} for name in self.classes}
        ego = dict(processed_lidar=self.pre_processor.collate_batch([a['lidar'] for a in agents]),
            processed_radar=self.pre_processor.collate_batch([a['radar'] for a in agents]),
            record_len=torch.tensor([len(item['agents']) for item in batch]), batch_size=len(agents),
            observation_maps={k: torch.from_numpy(np.stack([a['maps'][k] for a in agents])) for k in agents[0]['maps']},
            modality_valid=torch.from_numpy(np.stack([a['valid'] for a in agents])),
            pairwise_t_matrix=torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(len(batch), self.max_agents, self.max_agents, 1, 1),
            sample_ids=[item['sample_id'] for item in batch], metadata=[item['metadata'] for item in batch],
            target_diagnostics=[item['target_diagnostics'] for item in batch],
            label_dict={**labels[self.classes[0]], 'class_labels': labels})
        if batch[0]['refs']:
            ego['reference_lidar'] = self.pre_processor.collate_batch([r for item in batch for r in item['refs']])
        targets = [t for item in batch for t in item['targets']]
        result = dict(ego=ego, train_targets={k: torch.from_numpy(np.stack([t[k] for t in targets])) for k in targets[0]})
        if self.config['experiment']['method'] == 'l4dr':
            ego['raw_radar'] = [torch.from_numpy(a['raw_radar']) for a in agents]
            if self.training:
                result['train_targets']['fad_boxes'] = [torch.from_numpy(item['fad_boxes']).float()
                    for item in batch for _ in item['agents']]
        return result

    collate_batch_train = collate_batch
    collate_batch_test = collate_batch
