"""Dataset-specific evaluators; no frame-AP averaging or borrowed V2X-R ROI."""
from copy import deepcopy

import numpy as np

from opencood.project.geometry import pairwise_iou_both, rotated_nms, validate_boxes
from opencood.project.supplement_data import inside_region


def postprocess_real(predictions, manifest, scoring):
    selected = []
    for name in manifest['classes']:
        rows = [p for p in predictions if p['class_name'] == name]
        boxes = validate_boxes([p['box'] for p in rows])
        scores = np.asarray([p['score'] for p in rows], dtype=np.float64)
        ids = [p['prediction_id'] for p in rows]
        if len(set(ids)) != len(ids) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('invalid confidence/duplicate prediction IDs')
        valid = (scores >= scoring['score_floor']) & inside_region(boxes, manifest['bounds'], manifest['dataset'])
        indices = np.flatnonzero(valid)
        keep = rotated_nms(boxes[valid], scores[valid], [ids[i] for i in indices],
                           scoring['nms_bev_iou'], scoring['max_detections'])
        selected.extend(rows[indices[i]] for i in keep)
    if any(p['class_name'] not in manifest['classes'] for p in predictions):
        raise ValueError('predicted class is outside this dataset protocol')
    return selected


def _v2xradar_scores(samples, records, thresholds):
    from opencood.project.official.v2xradar_eval import calculate_ap
    metrics = {g: {} for g in ('3d', 'bev')}
    num_gt = sum(len(s['gt']) for s in samples.values())
    for geometry in metrics:
        for threshold in thresholds:
            stat = {threshold: dict(fp=[], tp=[], score=[], gt=0)}
            row = stat[threshold]
            for sid, sample in sorted(samples.items()):
                predictions = records[sid]['predictions']
                scores = np.asarray([p['score'] for p in predictions])
                order = np.argsort(-scores)  # exactly the official sorting rule
                gt = [g['box'] for g in sample['gt']]
                matrices = pairwise_iou_both([p['box'] for p in predictions], gt)
                available = list(range(len(gt)))
                for i in order:
                    overlaps = matrices[geometry][i, available]
                    positive = bool(len(overlaps) and overlaps.max() >= threshold)
                    row['tp'].append(int(positive))
                    row['fp'].append(int(not positive))
                    row['score'].append(float(scores[i]))
                    if positive:
                        available.pop(int(overlaps.argmax()))
                row['gt'] += len(gt)
            ap, recall, precision = calculate_ap(stat, threshold) if num_gt else (None, [], [])
            metrics[geometry][str(float(threshold))] = dict(ap=ap, num_gt=num_gt,
                num_predictions=len(row['score']), tp=sum(row['tp']), fp=sum(row['fp']),
                precision=precision, recall=recall)
    return dict(metrics=metrics, num_gt=num_gt,
        evaluator_version='v2xradar_1624751_official_voc2010_bev_and_true3d_extension',
        ap_definition='official calculate_ap VOC2010; official BEV matching; true 3D IoU extension separately named')


def _kradar_anno(rows, prediction=False):
    names = {'Sedan': 'sed', 'Bus or Truck': 'bus'}
    boxes = validate_boxes([p['box'] for p in rows])
    if not prediction:
        # Official util_pipeline.dict_datum_to_kitti rounds GT centre,
        # dimensions and radian yaw to two decimals before evaluation.
        boxes = np.round(boxes, 2)
    n = len(boxes)
    # Official util_pipeline.dict_datum_to_kitti: (xcam,ycam,zcam)=(y,z,x),
    # dimensions after kitti_common conversion are (l,h,w); yaw is unchanged.
    return dict(name=np.asarray([names[p['class_name']] for p in rows]),
        bbox=np.tile(np.array([50., 50., 150., 150.]), (n, 1)),
        location=boxes[:, [1, 2, 0]], dimensions=boxes[:, [3, 5, 4]],
        rotation_y=boxes[:, 6], alpha=np.zeros(n),
        truncated=np.full(n, -1.) if prediction else np.zeros(n),
        occluded=np.full(n, -1, dtype=np.int64) if prediction else np.zeros(n, dtype=np.int64),
        score=np.asarray([p['score'] for p in rows]) if prediction else np.zeros(n))


def _kradar_scores(samples, records):
    from opencood.project.official.kradar_eval_revised import get_official_eval_result_revised
    gt = [_kradar_anno(samples[sid]['gt']) for sid in sorted(samples)]
    pred = [_kradar_anno(records[sid]['predictions'], True) for sid in sorted(samples)]
    per_class = {}
    for name, index in (('Sedan', 0), ('Bus or Truck', 1)):
        count = sum(g['class_name'] == name for s in samples.values() for g in s['gt'])
        if count:
            result, _ = get_official_eval_result_revised(gt, pred, index, difficultys=[0],
                z_axis=1, z_center=.5, is_return_with_dict=True)
            metrics = {geometry: {str(float(iou)): dict(ap=float(ap) / 100., num_gt=count)
                for iou, ap in zip(result['iou'], result[geometry])} for geometry in ('3d', 'bev')}
        else:
            metrics = {g: {str(t): dict(ap=None, num_gt=0) for t in (.3, .5, .7)} for g in ('3d', 'bev')}
        per_class[name] = metrics
    metrics = {g: {} for g in ('3d', 'bev')}
    for g in metrics:
        for threshold in (.3, .5, .7):
            values = [c[g][str(threshold)]['ap'] for c in per_class.values()]
            metrics[g][str(threshold)] = dict(ap=float(np.mean(values)) if all(v is not None for v in values) else None)
    return dict(metrics=metrics, per_class=per_class,
        num_gt=sum(len(s['gt']) for s in samples.values()),
        evaluator_version='kradar_f90c289_official_revised_AP41_center_half_CPU_IoU',
        ap_definition='official revised 41 recall samples; Sedan and Bus or Truck macro mean')


def evaluate_supplement(manifest, records, split, scoring, *, sample_ids=None):
    dataset = manifest.get('dataset', 'v2xr')
    if dataset == 'v2xr':
        from opencood.project.evaluation import evaluate_predictions
        if sample_ids is None:
            return evaluate_predictions(manifest, records, split)
        # Subset membership is a frozen condition, shared by every method.
        subset = deepcopy(manifest)
        subset['samples'] = {sid: manifest['samples'][sid] for sid in sample_ids}
        subset['summary'][split]['num_samples'] = len(sample_ids)
        subset['summary'][split]['num_gt'] = sum(len(s['gt']) for s in subset['samples'].values())
        return evaluate_predictions(subset, records, split)
    selected = {sid: s for sid, s in manifest['samples'].items() if s['split'] == split and (sample_ids is None or sid in sample_ids)}
    by_id = {r['sample_id']: r for r in records}
    if not selected or len(by_id) != len(records) or set(by_id) != set(selected):
        raise ValueError('evaluator requires each selected fixed-GT frame exactly once, including empty predictions')
    for sid, row in by_id.items():
        for key, expected in (('dataset', dataset), ('split', split), ('protocol_sha256', manifest['protocol_sha256']),
                              ('gt_sha256', selected[sid].get('gt_sha256'))):
            if key in row and row[key] != expected:
                raise ValueError('prediction provenance differs from fixed real-data source: ' + key)
    processed = {sid: {**by_id[sid], 'predictions': postprocess_real(by_id[sid]['predictions'], manifest, scoring)} for sid in selected}
    result = _v2xradar_scores(selected, processed, scoring['iou_thresholds']) if dataset == 'v2xradar' else _kradar_scores(selected, processed)
    result.update(dataset=dataset, split=split, num_samples=len(selected), protocol_sha256=manifest['protocol_sha256'],
        final_identity=manifest['final_identity'], class_names=manifest['classes'],
        sample_ids=sorted(selected), selection_primary='Vehicle AP3D07' if dataset == 'v2xradar' else 'two_class_macro_AP3D07')
    return result


def sequence_bootstrap(manifest, records_a, records_b, split, scoring, *, repetitions=1000, seed=17029):
    """Paired sequence draws, then re-run the dataset evaluator on duplicated scenes."""
    samples = {sid: s for sid, s in manifest['samples'].items() if s['split'] == split}
    if set(samples) != {r['sample_id'] for r in records_a} or set(samples) != {r['sample_id'] for r in records_b}:
        raise ValueError('paired bootstrap requires the same complete sample set for both methods')
    sequences = sorted({s['sequence'] for s in samples.values()})
    lookup_a, lookup_b = ({r['sample_id']: r for r in records} for records in (records_a, records_b))
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        remapped = deepcopy(manifest)
        remapped['samples'] = {}
        draws = [[], []]
        for copy_index, sequence in enumerate(rng.choice(sequences, len(sequences), replace=True)):
            for sid, sample in samples.items():
                if sample['sequence'] != sequence:
                    continue
                new_id = f'{copy_index}/{sid}'
                remapped['samples'][new_id] = deepcopy(sample)
                for target, lookup in zip(draws, (lookup_a, lookup_b)):
                    record = deepcopy(lookup[sid])
                    record['sample_id'] = new_id
                    target.append(record)
        if manifest.get('dataset', 'v2xr') == 'v2xr':
            remapped['summary'][split].update(num_samples=len(remapped['samples']),
                num_gt=sum(len(s['gt']) for s in remapped['samples'].values()))
        ap = [evaluate_supplement(remapped, records, split, scoring)['metrics']['3d']['0.7']['ap'] for records in draws]
        if any(x is None for x in ap):
            raise ValueError('bootstrap draw has no GT for a required class; CI is undefined for this subset')
        values.append((ap[1] - ap[0]) * 100.)
    return dict(unit='sequence', repetitions=repetitions, seed=seed, quantity='B_minus_A_AP3D07_pp',
        interval_percentile95=np.percentile(values, [2.5, 97.5]).tolist(), paired_deltas_pp=values,
        ap_recomputed_per_draw=True)
