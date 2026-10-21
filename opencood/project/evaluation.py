"""project_fix_v1: complete frames, true 3D/BEV, stable global VOC2010 AP."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .fixed_gt import canonical_hash, default_protocol, gt_boxes
from .geometry import admissibility, pairwise_iou, pairwise_iou_both, rotated_nms, validate_boxes

EVALUATOR_VERSION = "project_fix_v1_global_voc2010_unmatched_v1"


def postprocess_hash(protocol=None):
    protocol = default_protocol() if protocol is None else protocol
    return canonical_hash({key: protocol[key] for key in (
        "score_floor", "nms_bev_iou", "max_detections", "nms_version", "nms_suppression",
        "postprocess_order", "roi_xy", "roi_rule", "fov_rule", "geometry_version", "tie_break")})


def validate_record_metadata(record, sample, protocol_sha256, require=False):
    """Verify supplied provenance; strict archive replay requires every field."""
    expected = {"dataset": "v2xr", "protocol_sha256": protocol_sha256}
    for key in ("split", "sequence", "timestamp", "ego_id", "gt_sha256"):
        if key in sample:
            expected[key] = sample[key]
    for key, value in expected.items():
        if require and key not in record:
            raise ValueError(f"Missing record provenance {key}: {record['sample_id']}")
        if key in record and record[key] != value:
            raise ValueError(f"Prediction {key} disagrees with fixed source: {record['sample_id']}")
    weather = record.get("evaluation_weather")
    if (require or weather is not None) and weather not in ("normal", "fog", "snow"):
        raise ValueError("Prediction evaluation_weather must be normal/fog/snow")
    if weather == "snow" and sample["split"] != "test":
        raise ValueError("Snow is final test only, never validation/model selection")
    if "agent_ids" in sample:
        source = list(map(str, sample["agent_ids"]))
        if "source_agent_ids" in record and list(map(str, record["source_agent_ids"])) != source:
            raise ValueError("Prediction GT source_agent_ids disagree with manifest")
        for key in ("agent_ids", "evaluated_agent_ids"):
            if require and key == "agent_ids" and key not in record:
                raise ValueError("Archive must record actual participating agent_ids")
            if key in record:
                agents = list(map(str, record[key]))
                if len(set(agents)) != len(agents) or not set(agents).issubset(source) or str(sample["ego_id"]) not in agents:
                    raise ValueError("Invalid participating agent_ids for the fixed source sample")
        if "agent_ids" in record and "evaluated_agent_ids" in record and record["agent_ids"] != record["evaluated_agent_ids"]:
            raise ValueError("agent_ids and evaluated_agent_ids disagree")
        if "agent_weather" in record:
            weather_by_agent = record["agent_weather"]
            if set(weather_by_agent) != set(record.get("agent_ids", [])) or any(w != weather for w in weather_by_agent.values()):
                raise ValueError("Actual agent weather disagrees with homogeneous evaluation weather")


def validate_archive_metadata(metadata, records, manifest, split, gt_manifest_sha256,
                              predictions_file_sha256=None):
    """Strict saved-result provenance gate; in-memory unit tests need no archive."""
    required = ("protocol", "protocol_sha256", "gt_manifest_sha256", "dataset", "split",
                "evaluation_weather", "source_kind", "checkpoint_sha256", "config_sha256",
                "checkpoint_epoch", "postprocessed", "sample_count")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Prediction archive is missing provenance fields: {missing}")
    expected = {"protocol": "project_fix_v1", "protocol_sha256": manifest["protocol_sha256"],
                "gt_manifest_sha256": gt_manifest_sha256, "dataset": "v2xr", "split": split,
                "sample_count": manifest["summary"][split]["num_samples"]}
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f"Prediction archive {key} does not match frozen evaluation source")
    if metadata["source_kind"] not in ("retrained", "official_checkpoint_reevaluated"):
        raise ValueError("Prediction replay requires an executed checkpoint source kind")
    for key in ("checkpoint_sha256", "config_sha256", "gt_manifest_sha256", "protocol_sha256"):
        value = metadata[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid SHA256 field: {key}")
    if not isinstance(metadata["checkpoint_epoch"], int) or isinstance(metadata["checkpoint_epoch"], bool) or metadata["checkpoint_epoch"] < 0:
        raise ValueError("Checkpoint epoch must be a nonnegative integer")
    if type(metadata["postprocessed"]) is not bool:
        raise ValueError("postprocessed must be an explicit boolean")
    if metadata["evaluation_weather"] not in ("normal", "fog", "snow"):
        raise ValueError("Unknown archive evaluation weather")
    if metadata["evaluation_weather"] == "snow" and split != "test":
        raise ValueError("Snow is final test only")
    if metadata.get("postprocess_sha256", postprocess_hash()) != postprocess_hash():
        raise ValueError("Archive postprocessing settings hash differs from frozen protocol")
    if predictions_file_sha256 and metadata.get("predictions_file_sha256", predictions_file_sha256) != predictions_file_sha256:
        raise ValueError("Saved prediction bytes no longer match the archive SHA256")
    if len(records) != metadata["sample_count"]:
        raise ValueError("Prediction archive sample_count disagrees with saved records")
    for record in records:
        sid = record["sample_id"]
        if sid not in manifest["samples"] or manifest["samples"][sid]["split"] != split:
            raise ValueError("Archive contains a sample outside the specified split")
        validate_record_metadata(record, manifest["samples"][sid], manifest["protocol_sha256"], require=True)
        if record["evaluation_weather"] != metadata["evaluation_weather"]:
            raise ValueError("Prediction archive mixes evaluation weather/source records")
    return True


def voc2010_ap(recall, precision):
    """All-points integral of the monotonically decreasing precision envelope."""
    recall, precision = np.asarray(recall, dtype=float), np.asarray(precision, dtype=float)
    if recall.shape != precision.shape or recall.ndim != 1:
        raise ValueError("precision/recall must be aligned 1D arrays")
    if not np.isfinite(recall).all() or not np.isfinite(precision).all() or np.any(np.diff(recall) < 0):
        raise ValueError("Invalid precision/recall curve")
    mrec, mpre = np.r_[0., recall, 1.], np.r_[0., precision, 0.]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])+1
    return float(np.sum((mrec[changes]-mrec[changes-1])*mpre[changes]))


def validate_prediction(prediction):
    if "prediction_id" not in prediction or not str(prediction["prediction_id"]):
        raise ValueError("Every prediction requires a stable prediction_id")
    if prediction.get("class_name") != "Vehicle":
        raise ValueError("project_fix_v1 V2X-R predictions must use class_name='Vehicle'")
    box = validate_boxes([prediction["box"]])[0]
    score = float(prediction["score"])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Prediction confidence must be finite in [0,1]")
    return {"prediction_id": str(prediction["prediction_id"]), "class_name": "Vehicle", "score": score, "box": box.tolist()}


def reference_postprocess_predictions(predictions, sample, protocol=None):
    """Pre-optimization path retained for exact CPU regression comparisons."""
    from .geometry import reference_rotated_nms
    protocol = default_protocol() if protocol is None else protocol
    predictions = [validate_prediction(p) for p in predictions]
    ids = [p["prediction_id"] for p in predictions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction_id within sample")
    counts = {"input": len(predictions), "below_score_floor": 0, "outside_roi_fov": 0, "nms_or_detection_cap": 0}
    high_score = [p for p in predictions if p["score"] >= protocol["score_floor"]]
    counts["below_score_floor"] = len(predictions)-len(high_score)
    if high_score:
        valid, _, _ = admissibility([p["box"] for p in high_score], sample.get("camera"), protocol["roi_xy"])
        admitted = [p for p, ok in zip(high_score, valid) if ok]
    else:
        admitted = []
    counts["outside_roi_fov"] = len(high_score)-len(admitted)
    if admitted:
        keep = reference_rotated_nms([p["box"] for p in admitted], [p["score"] for p in admitted],
                                     [p["prediction_id"] for p in admitted], protocol["nms_bev_iou"], protocol["max_detections"])
        selected = [admitted[i] for i in keep]
    else:
        selected = []
    counts["nms_or_detection_cap"] = len(admitted)-len(selected)
    counts["output"] = len(selected)
    return selected, counts


def _prediction_arrays(predictions):
    """Validate every candidate, using one matrix check instead of N tiny ones."""
    ids, boxes, scores = [], [], []
    for prediction in predictions:
        if "prediction_id" not in prediction or not str(prediction["prediction_id"]):
            raise ValueError("Every prediction requires a stable prediction_id")
        if prediction.get("class_name") != "Vehicle":
            raise ValueError("project_fix_v1 V2X-R predictions must use class_name='Vehicle'")
        ids.append(str(prediction["prediction_id"]))
        boxes.append(prediction["box"])
        scores.append(float(prediction["score"]))
    return _validate_prediction_arrays(boxes, scores, ids)


def _validate_prediction_arrays(boxes, scores, prediction_ids):
    boxes = validate_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Prediction confidence must be finite in [0,1]")
    ids = [str(pid) for pid in prediction_ids]
    if len(ids) != len(boxes) or any(not pid for pid in ids):
        raise ValueError("Every prediction requires a stable prediction_id")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction_id within sample")
    return boxes, scores, ids


def _predictions_from_arrays(boxes, scores, ids, indices):
    return [{"prediction_id": ids[int(i)], "class_name": "Vehicle",
             "score": float(scores[i]), "box": boxes[i].tolist()} for i in indices]


def _postprocess_arrays(boxes, scores, ids, sample, protocol):
    counts = {"input": len(boxes), "below_score_floor": 0, "outside_roi_fov": 0, "nms_or_detection_cap": 0}
    high_score = np.flatnonzero(scores >= protocol["score_floor"])
    counts["below_score_floor"] = len(boxes) - len(high_score)
    if len(high_score):
        valid, _, _ = admissibility(boxes[high_score], sample.get("camera"), protocol["roi_xy"])
        admitted = high_score[valid]
    else:
        admitted = high_score
    counts["outside_roi_fov"] = len(high_score) - len(admitted)
    if len(admitted):
        keep = rotated_nms(boxes[admitted], scores[admitted], [ids[i] for i in admitted],
                           protocol["nms_bev_iou"], protocol["max_detections"])
        selected = admitted[keep]
    else:
        selected = admitted
    counts["nms_or_detection_cap"] = len(admitted) - len(selected)
    counts["output"] = len(selected)
    return _predictions_from_arrays(boxes, scores, ids, selected), counts


def postprocess_prediction_arrays(boxes, scores, prediction_ids, sample, protocol=None):
    """Same frozen contract for decoded arrays; materialize only selected boxes.

    All input boxes/scores/IDs are checked before any score/ROI filtering, just
    like the dictionary API. No candidate limit or extra threshold is applied.
    """
    protocol = default_protocol() if protocol is None else protocol
    boxes, scores, ids = _validate_prediction_arrays(boxes, scores, prediction_ids)
    return _postprocess_arrays(boxes, scores, ids, sample, protocol)


def postprocess_predictions(predictions, sample, protocol=None):
    """Frozen filter order; batch validation preserves all exclusion counts."""
    protocol = default_protocol() if protocol is None else protocol
    boxes, scores, ids = _prediction_arrays(predictions)
    return _postprocess_arrays(boxes, scores, ids, sample, protocol)


def reference_score_records(gt_by_sample, predictions_by_sample, iou_thresholds=(0.3, 0.5, 0.7)):
    """Score already postprocessed predictions; useful for geometry unit tests.

    Keys in both mappings must match exactly, including negative and empty
    frames. ``gt_by_sample`` contains (N,7) boxes. A match searches all currently
    unmatched GT; it never reuses the assignment at another IoU threshold.
    """
    if set(gt_by_sample) != set(predictions_by_sample):
        raise ValueError("GT and prediction sample IDs differ; explicit empty frames required")
    gt = {str(sid): validate_boxes(boxes) for sid, boxes in gt_by_sample.items()}
    predictions = {}
    flat = []
    for sid in sorted(gt):
        predictions[sid] = [validate_prediction(p) for p in predictions_by_sample[sid]]
        if len({p["prediction_id"] for p in predictions[sid]}) != len(predictions[sid]):
            raise ValueError(f"Duplicate prediction_id in {sid}")
        for i, p in enumerate(predictions[sid]):
            flat.append((sid, i, p))
    flat.sort(key=lambda row: (-row[2]["score"], row[0], row[2]["prediction_id"]))
    num_gt = sum(len(x) for x in gt.values())
    metrics = {}
    for geometry in ("3d", "bev"):
        matrices = {sid: pairwise_iou([p["box"] for p in predictions[sid]], gt[sid], geometry) for sid in gt}
        metrics[geometry] = {}
        for threshold in iou_thresholds:
            if not 0 < threshold <= 1:
                raise ValueError("IoU thresholds must be in (0,1]")
            matched = {sid: np.zeros(len(g), dtype=bool) for sid, g in gt.items()}
            tp = np.zeros(len(flat), dtype=np.int64)
            for k, (sid, row, prediction) in enumerate(flat):
                remaining = np.flatnonzero(~matched[sid])
                if len(remaining):
                    ious = matrices[sid][row, remaining]
                    best = int(np.argmax(ious))
                    if ious[best] >= threshold:
                        matched[sid][remaining[best]] = True
                        tp[k] = 1
            fp = 1-tp
            precision = np.cumsum(tp)/np.maximum(1, np.arange(len(flat))+1)
            recall = np.cumsum(tp)/num_gt if num_gt else np.zeros(len(flat))
            metrics[geometry][str(float(threshold))] = {
                "status": "completed" if num_gt else "not_applicable",
                "ap": voc2010_ap(recall, precision) if num_gt else None,
                "reason": None if num_gt else "No admissible Vehicle GT in the complete evaluation set",
                "num_gt": num_gt, "num_predictions": len(flat), "tp": int(tp.sum()), "fp": int(fp.sum()),
                "max_recall": float(recall[-1]) if len(recall) else 0.,
                "precision": precision.tolist(), "recall": recall.tolist(),
                "tp_flags": tp.tolist(), "fp_flags": fp.tolist(),
            }
    return {"metrics": metrics, "num_samples": len(gt), "num_gt": num_gt,
            "num_predictions": len(flat), "num_empty_prediction_frames": sum(not x for x in predictions.values()),
            "global_prediction_order": [{"sample_id": sid, "prediction_id": p["prediction_id"], "score": p["score"]} for sid, _, p in flat]}

def _match_frame_thresholds_python(ious, order, thresholds):
    """Identical unmatched argmax, retaining the first GT on exact ties."""
    result = np.zeros((len(thresholds), len(order)), dtype=np.int64)
    for threshold_index in range(len(thresholds)):
        matched = np.zeros(ious.shape[1], dtype=np.bool_)
        for index in range(len(order)):
            row, best, best_iou = order[index], -1, -np.inf
            for gt_index in range(ious.shape[1]):
                if not matched[gt_index]:
                    value = ious[row, gt_index]
                    # np.argmax selects the first NaN if one occurs. Geometry
                    # normally supplies finite [0,1] IoUs; preserve that edge too.
                    if best < 0 or value > best_iou or (np.isnan(value) and not np.isnan(best_iou)):
                        best, best_iou = gt_index, value
            if best >= 0 and best_iou >= thresholds[threshold_index]:
                matched[best] = True
                result[threshold_index, index] = 1
    return result


try:
    from numba import njit as _njit
except ImportError:
    _MATCH_JIT_AVAILABLE = False
    _match_frame_thresholds = _match_frame_thresholds_python
else:
    _MATCH_JIT_AVAILABLE = True
    _match_frame_thresholds = _njit(cache=True, fastmath=False)(_match_frame_thresholds_python)


def score_records(gt_by_sample, predictions_by_sample, iou_thresholds=(0.3, 0.5, 0.7), *, implementation="optimized"):
    """Score already postprocessed predictions; useful for geometry unit tests.

    Keys in both mappings must match exactly, including negative and empty
    frames. ``gt_by_sample`` contains (N,7) boxes. A match searches all currently
    unmatched GT; it never reuses the assignment at another IoU threshold.
    """
    if implementation == "reference":
        return reference_score_records(gt_by_sample, predictions_by_sample, iou_thresholds)
    if implementation != "optimized":
        raise ValueError("Unknown scoring implementation")
    if not _MATCH_JIT_AVAILABLE:
        return reference_score_records(gt_by_sample, predictions_by_sample, iou_thresholds)
    thresholds = np.asarray(tuple(iou_thresholds), dtype=np.float64)
    if thresholds.ndim != 1 or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("IoU thresholds must be in (0,1]")
    if set(gt_by_sample) != set(predictions_by_sample):
        raise ValueError("GT and prediction sample IDs differ; explicit empty frames required")
    gt = {str(sid): validate_boxes(boxes) for sid, boxes in gt_by_sample.items()}
    predictions = {}
    flat = []
    for sid in sorted(gt):
        boxes, scores, ids = _prediction_arrays(predictions_by_sample[sid])
        predictions[sid] = _predictions_from_arrays(boxes, scores, ids, range(len(boxes)))
        for i, p in enumerate(predictions[sid]):
            flat.append((sid, i, p))
    flat.sort(key=lambda row: (-row[2]["score"], row[0], row[2]["prediction_id"]))
    frame_order = {sid: [] for sid in gt}
    global_indices = {sid: [] for sid in gt}
    for index, (sid, row, _) in enumerate(flat):
        frame_order[sid].append(row)
        global_indices[sid].append(index)
    frame_order = {sid: np.asarray(rows, dtype=np.int64) for sid, rows in frame_order.items()}
    global_indices = {sid: np.asarray(rows, dtype=np.int64) for sid, rows in global_indices.items()}
    num_gt = sum(len(x) for x in gt.values())
    matrices_both = {sid: pairwise_iou_both([p["box"] for p in predictions[sid]], gt[sid]) for sid in gt}
    metrics = {}
    for geometry in ("3d", "bev"):
        matrices = {sid: matrices_both[sid][geometry] for sid in gt}
        metrics[geometry] = {}
        tp_by_threshold = np.zeros((len(thresholds), len(flat)), dtype=np.int64)
        for sid in gt:
            tp_by_threshold[:, global_indices[sid]] = _match_frame_thresholds(matrices[sid], frame_order[sid], thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            tp = tp_by_threshold[threshold_index]
            fp = 1-tp
            precision = np.cumsum(tp)/np.maximum(1, np.arange(len(flat))+1)
            recall = np.cumsum(tp)/num_gt if num_gt else np.zeros(len(flat))
            metrics[geometry][str(float(threshold))] = {
                "status": "completed" if num_gt else "not_applicable",
                "ap": voc2010_ap(recall, precision) if num_gt else None,
                "reason": None if num_gt else "No admissible Vehicle GT in the complete evaluation set",
                "num_gt": num_gt, "num_predictions": len(flat), "tp": int(tp.sum()), "fp": int(fp.sum()),
                "max_recall": float(recall[-1]) if len(recall) else 0.,
                "precision": precision.tolist(), "recall": recall.tolist(),
                "tp_flags": tp.tolist(), "fp_flags": fp.tolist(),
            }
    return {"metrics": metrics, "num_samples": len(gt), "num_gt": num_gt,
            "num_predictions": len(flat), "num_empty_prediction_frames": sum(not x for x in predictions.values()),
            "global_prediction_order": [{"sample_id": sid, "prediction_id": p["prediction_id"], "score": p["score"]} for sid, _, p in flat]}


def evaluate_predictions(manifest, records, split, protocol=None, *, implementation="optimized"):
    """Validate complete split coverage and run the frozen common postprocessor.

    records: iterable of {sample_id, predictions: [{prediction_id,class_name,
    score,box:[x,y,z,l,w,h,yaw]}]}. Missing frames, duplicates, unknown classes,
    invalid numbers and mismatched protocol hashes are errors, never skips.
    """
    if implementation not in ("optimized", "reference"):
        raise ValueError("Unknown evaluation implementation")
    postprocessor = postprocess_predictions if implementation == "optimized" else reference_postprocess_predictions
    protocol = default_protocol() if protocol is None else dict(protocol)
    recorded_hash = protocol.get("protocol_sha256")
    semantic = {k: v for k,v in protocol.items() if k != "protocol_sha256"}
    if recorded_hash != canonical_hash(semantic) or manifest.get("protocol_sha256") != recorded_hash:
        raise ValueError("Evaluation protocol is not the frozen GT protocol")
    if protocol != default_protocol():
        raise ValueError("This evaluator only supports the exact frozen project_fix_v1 settings")
    selected = {sid: sample for sid, sample in manifest["samples"].items() if sample["split"] == split}
    if not selected or split not in manifest["summary"]:
        raise ValueError(f"Unknown/empty split: {split}")
    by_sample = {}
    recorded_weathers = set()
    for record in records:
        sid = record["sample_id"]
        if sid in by_sample:
            raise ValueError(f"Duplicate prediction record: {sid}")
        if sid not in selected:
            raise ValueError(f"Unexpected prediction sample for {split}: {sid}")
        if not isinstance(record.get("predictions"), list):
            raise ValueError(f"Predictions must be a list, including [] for empty sample {sid}")
        validate_record_metadata(record, selected[sid], recorded_hash)
        recorded_weathers.add(record.get("evaluation_weather"))
        by_sample[sid] = record
    missing = set(selected)-set(by_sample)
    if missing:
        raise ValueError(f"Missing {len(missing)} prediction frames, e.g. {sorted(missing)[:3]}")
    if len(recorded_weathers) > 1:
        raise ValueError("Evaluation records mix weather or partially omit weather provenance")
    predictions, filtering = {}, {}
    for sid, sample in selected.items():
        predictions[sid], filtering[sid] = postprocessor(by_sample[sid]["predictions"], sample, protocol)
    result = score_records({sid: gt_boxes(sample) for sid, sample in selected.items()}, predictions, protocol["iou_thresholds"],
                           implementation=implementation)
    summary = manifest["summary"][split]
    if result["num_samples"] != summary["num_samples"] or result["num_gt"] != summary["num_gt"]:
        raise ValueError("Manifest GT counts disagree with the evaluation set")
    result.update({"protocol": "project_fix_v1", "protocol_sha256": recorded_hash,
                   "evaluator_version": EVALUATOR_VERSION, "split": split, "dataset": "v2xr",
                   "class_name": "Vehicle", "ap_definition": protocol["ap_definition"],
                   "postprocess_counts": {k: sum(c[k] for c in filtering.values()) for k in next(iter(filtering.values()))},
                   "postprocess_sha256": postprocess_hash(protocol),
                   "sample_ids_sha256": canonical_hash(sorted(selected)),
                   "fixed_gt_sha256": summary.get("gt_sha256"),
                   "evaluation_weather": next(iter(recorded_weathers)),
                   "prediction_records_sha256": canonical_hash([by_sample[sid] for sid in sorted(by_sample)]),
                   "postprocessed_predictions_sha256": canonical_hash([
                       {"sample_id": sid, "predictions": predictions[sid]} for sid in sorted(predictions)])})
    return result


def read_prediction_archive(path):
    """Read JSON {metadata,records}, a JSON list, or newline-delimited records."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        metadata_path = path.with_suffix(path.suffix+".meta.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        return metadata, records
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {}, data
    return data.get("metadata", {}), data["records"]
