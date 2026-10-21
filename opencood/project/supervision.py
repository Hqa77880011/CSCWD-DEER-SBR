"""Conservative pairing targets, independent from model inputs and PCD colors.

Unmatched points are UNKNOWN by default. Classifying them as deletions/new
returns requires an external audited coordinate/provenance assumption.
"""
import hashlib
import numpy as np
from scipy.spatial import cKDTree


CONTROLLED_CORRUPTION = {"probability": 0.5, "drop_fraction": 0.2,
                        "inject_fraction": 0.05, "max_offset_m": 0.3}


def controlled_corruption_config(config=None):
    """Explicit augmentation contract; these are not fitted weather parameters."""
    out = dict(CONTROLLED_CORRUPTION)
    if config is not None:
        if set(config) != set(out):
            raise ValueError("controlled corruption requires the four registered parameters")
        out.update({key: float(value) for key, value in config.items()})
    if not all(np.isfinite(value) for value in out.values()):
        raise ValueError("nonfinite controlled corruption parameter")
    if any(not 0 <= out[key] <= 1 for key in ("probability", "drop_fraction", "inject_fraction")):
        raise ValueError("controlled corruption probabilities/fractions must be in [0,1]")
    if out["max_offset_m"] <= 0:
        raise ValueError("controlled corruption offset must be positive")
    return out


def controlled_corruption(points, rng, config=None, legal_mask=None):
    """Delete exact source IDs and add explicitly tagged nearby ghost returns.

    Input is the accepted normal training cloud in augmented ego coordinates.
    Ghosts copy an existing measured fourth field and receive a bounded spatial
    offset; this is a controlled augmentation, not physical Fog/Snow simulation.
    The optional geometry predicate applies to ghosts before accepting them.
    No source IDs or flags are returned as model input channels.
    """
    config = controlled_corruption_config(config)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    source_ids = np.arange(len(points), dtype=np.int64)
    applied = bool(rng.random() < config["probability"] and len(points))
    if not applied:
        return points.copy(), source_ids, False
    keep = rng.random(len(points)) >= config["drop_fraction"]
    count = int(np.ceil(len(points) * config["inject_fraction"]))
    ghosts = points[rng.integers(0, len(points), size=count)].copy()
    if count:
        direction = rng.normal(size=(count, 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        radius = rng.uniform(0.25, 1.0, size=(count, 1)) * config["max_offset_m"]
        ghosts[:, :3] += (direction * radius).astype(np.float32)
        if legal_mask is not None:
            accepted = np.asarray(legal_mask(ghosts), dtype=bool)
            if accepted.shape != (len(ghosts),):
                raise ValueError("ghost geometry predicate must return one flag per point")
            ghosts = ghosts[accepted]
    observed = np.concatenate((points[keep], ghosts), axis=0)
    source_ids = np.concatenate((source_ids[keep], np.full(len(ghosts), -1, np.int64)))
    return np.ascontiguousarray(observed), source_ids, True


def provenance_targets(normal_points, observed_points, source_ids, point_cloud_range, voxel_size):
    """Exact fractions for the defined source-ID augmentation, never density deltas.

    Unchanged source IDs must retain their exact coordinates AND fourth field;
    -1 denotes a new injected return. Unknown natural-weather origins are never
    accepted by this function. Simultaneous deletion/injection remains visible.
    """
    normal_points = np.asarray(normal_points, dtype=np.float32).reshape(-1, 4)
    observed_points = np.asarray(observed_points, dtype=np.float32).reshape(-1, 4)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if source_ids.shape != (len(observed_points),) or np.any(source_ids < -1) or np.any(source_ids >= len(normal_points)):
        raise ValueError("invalid controlled source IDs")
    known = source_ids >= 0
    retained_ids = source_ids[known]
    if len(np.unique(retained_ids)) != len(retained_ids):
        raise ValueError("a retained source ID cannot appear twice")
    if not np.array_equal(observed_points[known], normal_points[retained_ids]):
        raise ValueError("retained source ID does not preserve the original measurement")
    clean, clean_ids, shape = _grid(normal_points, point_cloud_range, voxel_size)
    observed, observed_ids, _ = _grid(observed_points, point_cloud_range, voxel_size)
    if len(clean) != len(normal_points) or len(observed) != len(observed_points):
        raise ValueError("source-ID targets require geometry-filtered finite points")
    retained = np.zeros(len(normal_points), dtype=bool)
    retained[retained_ids] = True
    nc = _aggregate(clean_ids, np.ones(len(clean)), shape)
    no = _aggregate(observed_ids, np.ones(len(observed)), shape)
    miss = _aggregate(clean_ids, ~retained, shape) / np.maximum(nc, 1)
    spur = _aggregate(observed_ids, ~known, shape) / np.maximum(no, 1)
    vc, vo = nc > 0, no > 0
    result = {"miss": miss, "spur": spur, "miss_valid": vc, "spur_valid": vo,
              "deg_valid": (vc & (miss > 0)) | (vo & (spur > 0))}
    result = {key: value[None].astype(np.float32) for key, value in result.items()}
    result["stats"] = {"source": "provenance_v1", "normal_points": len(clean),
        "observed_points": len(observed), "removed_points": int((~retained).sum()),
        "injected_points": int((~known).sum()), "miss_valid_cells": int(vc.sum()),
        "spur_valid_cells": int(vo.sum()), "positive_miss_cells": int((vc & (miss > 0)).sum()),
        "positive_spur_cells": int((vo & (spur > 0)).sum()),
        "joint_positive_cells": int((vc & vo & (miss > 0) & (spur > 0)).sum()),
        "source_ids_verified": True, "unmatched_semantics_verified": False,
        "source_id_sha256": hashlib.sha256(source_ids.astype('<i8').tobytes()).hexdigest()}
    return result


def _grid(points, point_cloud_range, voxel_size):
    points = np.asarray(points, dtype=np.float64)[:, :3]
    bounds, voxel = np.asarray(point_cloud_range), np.asarray(voxel_size)
    width, height = np.rint((bounds[3:5] - bounds[:2]) / voxel[:2]).astype(int)
    valid = np.isfinite(points).all(1) & (points >= bounds[:3]).all(1) & (points < bounds[3:]).all(1)
    xy = np.floor((points[valid, :2] - bounds[:2]) / voxel[:2]).astype(int)
    return points[valid], xy[:, 1] * width + xy[:, 0], (height, width)


def _aggregate(ids, flags, shape):
    return np.bincount(ids, weights=np.asarray(flags, dtype=float), minlength=int(np.prod(shape))).reshape(shape)


def normal_targets(points, point_cloud_range, voxel_size):
    _, ids, shape = _grid(points, point_cloud_range, voxel_size)
    occupied = _aggregate(ids, np.ones(len(ids)), shape) > 0
    zeros = np.zeros((1, *shape), dtype=np.float32)
    return {"miss": zeros.copy(), "spur": zeros.copy(),
            "miss_valid": occupied[None].astype(np.float32), "spur_valid": occupied[None].astype(np.float32),
            "deg_valid": zeros.copy()}


def mutual_nn_targets(clean_xyz, observed_xyz, point_cloud_range, voxel_size,
                      max_distance=0.002, ambiguity_margin=0.001,
                      unmatched_min_distance=0.05, allow_unmatched=False):
    if not 0 < max_distance < unmatched_min_distance or ambiguity_margin < 0:
        raise ValueError("pairing distances must satisfy 0 < match < unmatched, ambiguity >= 0")
    clean, clean_ids, shape = _grid(clean_xyz, point_cloud_range, voxel_size)
    observed, observed_ids, _ = _grid(observed_xyz, point_cloud_range, voxel_size)
    # Query in 3D, not BEV: vertically distinct returns must not be paired.
    def nearest(a, b):
        if len(b) == 0:
            return np.full((len(a), 2), np.inf), np.full((len(a), 2), -1, dtype=np.int64)
        return cKDTree(b).query(a, k=2, workers=1)
    dc, ic = nearest(clean, observed)
    do, io = nearest(observed, clean)
    cm = np.zeros(len(clean), dtype=bool)
    om = np.zeros(len(observed), dtype=bool)
    if len(clean) and len(observed):
        candidates = np.flatnonzero((dc[:, 0] <= max_distance) & ((dc[:, 1] - dc[:, 0]) > ambiguity_margin))
        partners = ic[candidates, 0]
        good = ((io[partners, 0] == candidates) & (do[partners, 0] <= max_distance)
                & ((do[partners, 1] - do[partners, 0]) > ambiguity_margin))
        cm[candidates[good]] = True
        om[partners[good]] = True
    removed = (dc[:, 0] >= unmatched_min_distance) & bool(allow_unmatched)
    added = (do[:, 0] >= unmatched_min_distance) & bool(allow_unmatched)
    nc = _aggregate(clean_ids, np.ones(len(clean)), shape)
    no = _aggregate(observed_ids, np.ones(len(observed)), shape)
    vc = (nc > 0) & (_aggregate(clean_ids, cm | removed, shape) == nc)
    vo = (no > 0) & (_aggregate(observed_ids, om | added, shape) == no)
    miss = _aggregate(clean_ids, removed, shape) / np.maximum(nc, 1)
    spur = _aggregate(observed_ids, added, shape) / np.maximum(no, 1)
    result = {"miss": miss, "spur": spur, "miss_valid": vc, "spur_valid": vo,
              "deg_valid": (vc & (miss > 0)) | (vo & (spur > 0))}
    result = {key: value[None].astype(np.float32) for key, value in result.items()}
    result["stats"] = {"clean_points": len(clean), "observed_points": len(observed), "matched_points": int(cm.sum()),
                       "miss_valid_cells": int(vc.sum()), "spur_valid_cells": int(vo.sum()),
                       "positive_miss_cells": int((vc & (miss > 0)).sum()),
                       "positive_spur_cells": int((vo & (spur > 0)).sum()),
                       "unmatched_semantics_verified": bool(allow_unmatched)}
    return result
