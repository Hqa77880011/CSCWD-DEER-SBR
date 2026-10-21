"""Immutable V2X-R GT built from every original agent's annotations.

The manifest includes all cooperative samples, even zero-GT samples. Weather,
point counts, communication radius, training caps and model options are absent
from GT acceptance. Original labels remain auditable by path, SHA256 and ID.
"""
from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .geometry import (GEOMETRY_VERSION, admissibility, camera_from_annotation,
                       object_to_ego_box, validate_boxes)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AnnotationLoader(yaml.CSafeLoader):
    """Allow only data constructors for the NumPy scalar tags in source YAML."""


def _dtype(loader, node):
    data = loader.construct_mapping(node, deep=True)
    dtype = np.dtype(data["args"][0])
    if dtype.kind not in "fiu" or dtype.itemsize > 8:
        raise ValueError("Unsupported source NumPy dtype")
    return dtype.newbyteorder(data["state"][1])


def _scalar(loader, node):
    dtype, raw = loader.construct_sequence(node, deep=True)
    if not isinstance(dtype, np.dtype) or not isinstance(raw, bytes) or len(raw) != dtype.itemsize:
        raise ValueError("Invalid source NumPy scalar")
    return np.frombuffer(raw, dtype=dtype, count=1)[0].item()


AnnotationLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l,n: tuple(l.construct_sequence(n, deep=True)))
AnnotationLoader.add_constructor("tag:yaml.org,2002:python/object/apply:numpy.dtype", _dtype)
AnnotationLoader.add_constructor("tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar", _scalar)
AnnotationLoader.add_constructor("tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar", _scalar)


def load_annotation(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Source annotation SHA256 mismatch: {path}")
    value = yaml.load(raw, Loader=AnnotationLoader)
    if not isinstance(value, dict) or not isinstance(value.get("vehicles"), dict):
        raise ValueError(f"Invalid annotation structure: {path}")
    return value


def default_protocol():
    data = {
        "schema": "project_protocol_v1", "protocol": "project_fix_v1", "dataset": "v2xr",
        "class_name": "Vehicle", "class_mapping": {"vehicle.*": "Vehicle"},
        "box_format": "x_y_z_length_width_height_yaw", "z_reference": "center",
        "coordinates": "ego_lidar_meters", "yaw": "radians_about_positive_z",
        "roi_xy": [0., -40., 140., 40.], "roi_rule": "center_half_open_xy_no_z_filter",
        "fov_rule": "ego_camera0_positive_depth_center_half_open_image",
        "geometry_version": GEOMETRY_VERSION,
        "source_center_rule": "vehicle_local_center_rotated_by_full_CARLA_pose",
        "source_box_rule": "transform_eight_corners_then_enclose_in_projected_heading_upright_yaw",
        "ego_rule": "lexicographic_agent_sort_negative_ids_last_first_nonnegative",
        "deduplication": "original_object_id_union_all_agents_ego_first_then_agent_order",
        "duplicate_tolerance": {"center_m": 0.001, "dimension_m": 0.001, "yaw_rad": 0.0001},
        "duplicate_conflicts": "fail_build_above_tolerance_record_all_raw_disagreements",
        "weather_invariant": ["normal", "fog", "snow"], "point_count_filter": False,
        "gt_truncation": False, "score_floor": 0.001, "nms_bev_iou": 0.15,
        "max_detections": 500, "nms_version": "shapely_rotated_bev_cpu_v1",
        "nms_suppression": "strictly_greater_than_iou_threshold",
        "postprocess_order": ["validate", "score_floor", "center_roi_fov", "stable_rotated_bev_nms", "max_detections"],
        "iou_thresholds": [0.3, 0.5, 0.7], "iou_geometries": ["3d", "bev"],
        "ap_definition": "voc2010_all_points_global_sort", "global_score_sort": True,
        "matching": "maximum_iou_among_unmatched_gt_same_sample_class_greater_equal_threshold",
        "tie_break": ["sample_id", "prediction_id"], "invalid_output": "raise",
        "missing_prediction_frame": "raise_require_explicit_empty_predictions",
        "zero_gt_ap": "not_applicable", "difficulty": "ALL-admissible",
    }
    data["protocol_sha256"] = canonical_hash(data)
    return data


def ordered_agents(ids):
    ids = list(map(str, ids))
    return sorted((x for x in ids if int(x) >= 0)) + sorted((x for x in ids if int(x) < 0))


def gt_boxes(sample):
    return validate_boxes([obj["box"] for obj in sample["gt"]])


def build_sample(root, frames, image_sizes, protocol):
    sample_ids = {f["sample_id"] for f in frames}
    if len(sample_ids) != 1:
        raise ValueError("Mixed samples passed to GT builder")
    by_agent = {str(f["agent"]): f for f in frames}
    if len(by_agent) != len(frames):
        raise ValueError("Duplicate sensor frame in manifest")
    agents = ordered_agents(by_agent)
    if not agents or int(agents[0]) < 0:
        raise ValueError("No vehicle available as ego")
    source_annotations, annotations = {}, {}
    for agent in agents:
        entry = by_agent[agent]["modalities"]["annotation"]
        annotations[agent] = load_annotation(Path(root)/entry["path"], entry["sha256"])
        source_annotations[agent] = {"path": entry["path"], "sha256": entry["sha256"]}
    first = by_agent[agents[0]]
    camera_path = str(Path(first["modalities"]["annotation"]["path"]).with_suffix("")) + "_camera0.png"
    if camera_path not in image_sizes:
        raise ValueError(f"Missing verified camera dimensions: {camera_path}")
    ego = annotations[agents[0]]
    camera = camera_from_annotation(ego, image_sizes[camera_path])
    objects, duplicates, raw_types = {}, [], set()
    for agent in agents:
        for original_id, obj in annotations[agent]["vehicles"].items():
            oid = str(original_id)
            raw_type = str(obj.get("bp_id", ""))
            raw_types.add(raw_type)
            if not raw_type.startswith("vehicle."):
                raise ValueError(f"Unmapped original class {raw_type!r} in {first['sample_id']}/{oid}")
            box, _ = object_to_ego_box(obj, ego["lidar_pose"])
            # Preserve the full original label by immutable source reference;
            # this compact entry does not copy a million raw YAML dictionaries.
            entry = {"original_id": oid, "original_type": raw_type, "box": box.tolist(),
                     "source_agent_ids": [agent], "selected_source_agent_id": agent,
                     "original_annotation_sha256": canonical_hash(obj)}
            if oid in objects:
                chosen = objects[oid]
                chosen["source_agent_ids"].append(agent)
                old = np.asarray(chosen["box"])
                delta_yaw = abs(np.arctan2(np.sin(box[6]-old[6]), np.cos(box[6]-old[6])))
                delta_center = float(np.max(np.abs(box[:3]-old[:3])))
                delta_dimensions = float(np.max(np.abs(box[3:6]-old[3:6])))
                tol = protocol["duplicate_tolerance"]
                conflict = raw_type != chosen["original_type"] or delta_center > tol["center_m"] or delta_dimensions > tol["dimension_m"] or delta_yaw > tol["yaw_rad"]
                if entry["original_annotation_sha256"] != chosen["original_annotation_sha256"]:
                    duplicates.append({"original_id": oid, "selected_agent": chosen["selected_source_agent_id"], "other_agent": agent,
                                       "other_annotation_sha256": entry["original_annotation_sha256"], "center_max_abs_m": delta_center,
                                       "dimension_max_abs_m": delta_dimensions, "yaw_abs_rad": float(delta_yaw), "conflict": bool(conflict)})
                if conflict:
                    raise ValueError(f"Conflicting duplicate object {first['sample_id']}/{oid}: {duplicates[-1]}")
            else:
                objects[oid] = entry
    original, accepted, excluded = [], [], []
    for oid in sorted(objects):
        entry = objects[oid]
        valid, roi_mask, fov_mask = admissibility([entry["box"]], camera, protocol["roi_xy"])
        reasons = ([] if roi_mask[0] else ["center_outside_xy_roi"]) + ([] if fov_mask[0] else ["center_outside_camera0_fov"])
        entry["excluded_reasons"] = reasons
        original.append(entry)
        if valid[0]:
            accepted.append({"gt_id": first["sample_id"]+"/"+oid, "original_id": oid, "class_name": "Vehicle", "box": entry["box"]})
        else:
            excluded.append({"original_id": oid, "reasons": reasons})
    sample = {"sample_id": first["sample_id"], "split": first["split"], "sequence": first["sequence"],
              "timestamp": first["timestamp"], "ego_id": agents[0], "agent_ids": agents,
              "ego_lidar_pose": ego["lidar_pose"], "camera": camera,
              "source_annotations": source_annotations, "original_gt": original, "gt": accepted,
              "excluded": excluded, "duplicate_disagreements": duplicates,
              "num_original_gt": len(original), "num_gt": len(accepted),
              "num_source_objects": sum(len(x["vehicles"]) for x in annotations.values())}
    sample["gt_sha256"] = canonical_hash(accepted)
    return sample, sorted(raw_types)


def _build_job(args):
    return build_sample(*args)


def build_manifest(data_root, workers=8, progress=None):
    root = Path(data_root)
    data = json.loads((root/"data_manifest.json").read_text())
    splits = json.loads((root/"split_manifest.json").read_text())
    images = json.loads((root/"image_sizes.json").read_text())["images"]
    protocol = default_protocol()
    groups = collections.defaultdict(list)
    for frame in data["frames"]:
        groups[frame["sample_id"]].append(frame)
    expected = sum(s["cooperative_samples"] for s in splits["splits"].values())
    if len(groups) != expected:
        raise ValueError(f"Split sample count mismatch {len(groups)} != {expected}")
    jobs = []
    for sid in sorted(groups):
        frames = groups[sid]
        ego = ordered_agents(f["agent"] for f in frames)[0]
        frame = next(f for f in frames if str(f["agent"]) == ego)
        image_key = str(Path(frame["modalities"]["annotation"]["path"]).with_suffix(""))+"_camera0.png"
        jobs.append((str(root), frames, {image_key: images[image_key]}, protocol))
    samples, raw_types = {}, set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for n, (sample, types) in enumerate(executor.map(_build_job, jobs, chunksize=8), 1):
            samples[sample["sample_id"]] = sample
            raw_types.update(types)
            if progress and (n % 500 == 0 or n == expected):
                progress(n, expected)
    summary = {}
    for split, spec in splits["splits"].items():
        chosen = [s for s in samples.values() if s["split"] == split]
        if len(chosen) != spec["cooperative_samples"]:
            raise ValueError(f"Split sample count changed: {split}")
        summary[split] = {"num_samples": len(chosen), "num_gt": sum(s["num_gt"] for s in chosen),
                          "max_gt_per_sample": max((s["num_gt"] for s in chosen), default=0),
                          "num_zero_gt_samples": sum(s["num_gt"] == 0 for s in chosen),
                          "num_original_gt": sum(s["num_original_gt"] for s in chosen),
                          "num_source_objects": sum(s["num_source_objects"] for s in chosen),
                          "duplicate_raw_disagreements": sum(len(s["duplicate_disagreements"]) for s in chosen),
                          "sample_ids_sha256": canonical_hash([s["sample_id"] for s in chosen]),
                          "gt_sha256": canonical_hash({s["sample_id"]: s["gt_sha256"] for s in chosen})}
    manifest = {"schema": "v2xr_fixed_gt_v1", "protocol": "project_fix_v1", "protocol_sha256": protocol["protocol_sha256"],
                "dataset": "v2xr", "dataset_revision": data["source_revision"],
                "source_manifests": {p: file_sha256(root/p) for p in ("data_manifest.json", "split_manifest.json", "image_sizes.json")},
                "geometry_version": GEOMETRY_VERSION, "box_format": protocol["box_format"],
                "class_mapping": {t: "Vehicle" for t in sorted(raw_types)},
                "weather_invariant": ["normal", "fog", "snow"], "gt_truncation": False,
                "summary": summary, "samples": samples,
                "empty_source_sequences": splits.get("empty_upstream_sequences", []),
                "audit": {"all_source_annotations_sha256_verified": True, "conflicting_duplicates": 0,
                          "camera_pose_extrinsic_crosscheck": "passed", "point_clouds_read": False}}
    return manifest, protocol


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def load_manifest(path, validate=True):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "v2xr_fixed_gt_v1" or data.get("protocol") != "project_fix_v1":
        raise ValueError("Unsupported GT manifest schema/protocol")
    if validate:
        protocol_path = Path(path).with_name("protocol_manifest.json")
        protocol = json.loads(protocol_path.read_text())
        expected_hash = protocol.pop("protocol_sha256")
        if canonical_hash(protocol) != expected_hash or data["protocol_sha256"] != expected_hash:
            raise ValueError("GT/protocol hash mismatch")
        for sid, sample in data["samples"].items():
            if sid != sample["sample_id"] or sample["num_gt"] != len(sample["gt"]):
                raise ValueError(f"Corrupt sample metadata: {sid}")
            if sample["gt_sha256"] != canonical_hash(sample["gt"]):
                raise ValueError(f"GT hash mismatch: {sid}")
            gt_boxes(sample)
        for split, summary in data["summary"].items():
            selected = [s for s in data["samples"].values() if s["split"] == split]
            if len(selected) != summary["num_samples"] or sum(s["num_gt"] for s in selected) != summary["num_gt"]:
                raise ValueError(f"GT split counts disagree: {split}")
    return data
