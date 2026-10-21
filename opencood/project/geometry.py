"""Float64 reference geometry for upright boxes in (x,y,z,l,w,h,yaw).

All lengths are metres, z is the centre, yaw is radians about +z. Source
CARLA roll/pitch are retained while transforming corners, then explicitly
approximated with an upright enclosure. No point cloud is consulted here.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Polygon
try:
    # Load libtorch before the optional LLVM runtime. In this environment,
    # importing Numba first can break a later CUDA random_device initialization.
    import torch as _torch_runtime
    from numba import njit
except ImportError:
    njit = None

GEOMETRY_VERSION = "carla_corners_upright_enclosure_v1"
OPTICAL_FROM_CARLA = np.array([[0., 1., 0., 0.], [0., 0., -1., 0.],
                              [1., 0., 0., 0.], [0., 0., 0., 1.]])


def validate_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[None]
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Boxes must have shape (N,7): x,y,z,l,w,h,yaw")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("Boxes must be finite with strictly positive l,w,h")
    return boxes


def pose_to_world(pose):
    """CARLA [x,y,z,roll,yaw,pitch], with angles in degrees."""
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError("Invalid CARLA pose")
    r, y, t = np.deg2rad(p[3:])
    cr, sr, cy, sy, cp, sp = math.cos(r), math.sin(r), math.cos(y), math.sin(y), math.cos(t), math.sin(t)
    out = np.eye(4)
    out[:3, 3] = p[:3]
    out[:3, :3] = [[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                   [sy*cp, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                   [sp, -cp*sr, cp*cr]]
    return out


def transform_points(points, matrix):
    xyz, matrix = np.asarray(points, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Expected points (N,3) and transform (4,4)")
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def boxes_to_corners(boxes):
    boxes = validate_boxes(boxes)
    signs = np.array([[1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,-1],
                      [1,-1,1], [1,1,1], [-1,1,1], [-1,-1,1]], dtype=np.float64)
    corners = signs[None] * boxes[:, None, 3:6] / 2
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    x, y = corners[:, :, 0].copy(), corners[:, :, 1].copy()
    corners[:, :, 0] = c[:, None]*x - s[:, None]*y
    corners[:, :, 1] = s[:, None]*x + c[:, None]*y
    return corners + boxes[:, None, :3]


def object_to_ego_box(obj, ego_pose):
    """Transform all eight local corners; enclose them in an upright yaw box.

    ``center`` is the CARLA bounding-box offset in the vehicle's local frame,
    not a world translation. The enclosure records the roll/pitch approximation
    without dropping a corner or using the upstream average-angle wrap bug.
    """
    location, angle, center, extent = (np.asarray(obj[k], dtype=np.float64)
                                        for k in ("location", "angle", "center", "extent"))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (location, angle, center, extent)):
        raise ValueError("Invalid source object fields")
    if np.any(extent <= 0):
        raise ValueError("Non-positive source object extent")
    transform = np.linalg.solve(pose_to_world(ego_pose), pose_to_world(np.r_[location, angle]))
    local = boxes_to_corners([np.r_[center, 2*extent, 0.]])[0]
    corners = transform_points(local, transform)
    heading = transform[:2, 0]
    if np.linalg.norm(heading) < 1e-8:
        raise ValueError("Vertical source heading has no well-defined yaw")
    yaw = math.atan2(heading[1], heading[0])
    c, s = math.cos(yaw), math.sin(yaw)
    basis = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
    aligned = corners @ basis.T
    lo, hi = aligned.min(0), aligned.max(0)
    box = np.r_[((lo+hi)/2) @ basis, hi-lo, yaw]
    validate_boxes(box)
    return box, corners


def camera_from_annotation(annotation, image_size):
    """Ego LiDAR -> optical camera0, cross-checked against source extrinsic."""
    cam = annotation["camera0"]
    lidar_to_carla = np.linalg.solve(pose_to_world(cam["cords"]), pose_to_world(annotation["lidar_pose"]))
    source_extrinsic = np.asarray(cam["extrinsic"], dtype=np.float64)
    if source_extrinsic.shape != (4,4) or not np.allclose(source_extrinsic, lidar_to_carla, atol=2e-4, rtol=0):
        raise ValueError("camera0 extrinsic disagrees with camera/lidar world poses")
    intrinsic = np.asarray(cam["intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3,3) or not np.isfinite(intrinsic).all() or intrinsic[0,0] <= 0 or intrinsic[1,1] <= 0:
        raise ValueError("Invalid camera intrinsic")
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return {"intrinsic": intrinsic.tolist(), "lidar_to_camera": (OPTICAL_FROM_CARLA @ lidar_to_carla).tolist(),
            "image_width": width, "image_height": height}


def admissibility(boxes, camera=None, roi=(0., -40., 140., 40.)):
    """Half-open centre ROI and positive-depth centre FOV; no z/point filter."""
    boxes = validate_boxes(boxes)
    x0, y0, x1, y1 = roi
    roi_mask = (boxes[:,0] >= x0) & (boxes[:,0] < x1) & (boxes[:,1] >= y0) & (boxes[:,1] < y1)
    fov_mask = np.ones(len(boxes), dtype=bool)
    if camera is not None and len(boxes):
        fov_mask = points_in_fov(boxes[:, :3], camera)
    return roi_mask & fov_mask, roi_mask, fov_mask


def points_in_fov(points, camera):
    """Boolean geometry mask for points already expressed in ego LiDAR axes."""
    xyz = transform_points(np.asarray(points)[:, :3], camera["lidar_to_camera"])
    uvw = xyz @ np.asarray(camera["intrinsic"]).T
    uv = np.full((len(xyz), 2), np.inf)
    positive = xyz[:,2] > 0
    np.divide(uvw[:,:2], uvw[:,2,None], out=uv, where=positive[:,None])
    return positive & (uv[:,0] >= 0) & (uv[:,0] < camera["image_width"]) & (uv[:,1] >= 0) & (uv[:,1] < camera["image_height"])


def pairwise_iou(boxes_a, boxes_b, geometry="3d"):
    """Exact rotated polygon intersection; volume uses z interval overlap."""
    a, b = validate_boxes(boxes_a), validate_boxes(boxes_b)
    if geometry not in ("3d", "bev"):
        raise ValueError("geometry must be '3d' or 'bev'")
    result = np.zeros((len(a), len(b)), dtype=np.float64)
    if not len(a) or not len(b):
        return result
    ca, cb = boxes_to_corners(a)[:,:4,:2], boxes_to_corners(b)[:,:4,:2]
    pa, pb = [Polygon(x) for x in ca], [Polygon(x) for x in cb]
    area_a, area_b = a[:,3]*a[:,4], b[:,3]*b[:,4]
    amin, amax, bmin, bmax = ca.min(1), ca.max(1), cb.min(1), cb.max(1)
    for i in range(len(a)):
        overlap = np.all(np.minimum(amax[i], bmax) > np.maximum(amin[i], bmin), axis=1)
        if geometry == "3d":
            dz = np.maximum(0., np.minimum(a[i,2]+a[i,5]/2, b[:,2]+b[:,5]/2) - np.maximum(a[i,2]-a[i,5]/2, b[:,2]-b[:,5]/2))
            overlap &= dz > 0
        for j in np.flatnonzero(overlap):
            intersection = pa[i].intersection(pb[j]).area
            if geometry == "3d":
                intersection *= dz[j]
                union = area_a[i]*a[i,5]+area_b[j]*b[j,5]-intersection
            else:
                union = area_a[i]+area_b[j]-intersection
            result[i,j] = np.clip(intersection/union, 0., 1.)
    return result


def pairwise_iou_both(boxes_a, boxes_b):
    """Return exact BEV and 3D IoUs while sharing each GEOS intersection.

    The arithmetic order matches ``pairwise_iou`` for both geometries. An
    upright 3D intersection uses the same BEV polygon area times its positive
    z overlap, so corners, polygons and clipping need only be evaluated once.
    """
    a, b = validate_boxes(boxes_a), validate_boxes(boxes_b)
    results = {key: np.zeros((len(a), len(b)), dtype=np.float64) for key in ("3d", "bev")}
    if not len(a) or not len(b):
        return results
    ca, cb = boxes_to_corners(a)[:, :4, :2], boxes_to_corners(b)[:, :4, :2]
    pa, pb = [Polygon(x) for x in ca], [Polygon(x) for x in cb]
    area_a, area_b = a[:, 3]*a[:, 4], b[:, 3]*b[:, 4]
    amin, amax, bmin, bmax = ca.min(1), ca.max(1), cb.min(1), cb.max(1)
    for i in range(len(a)):
        overlap = np.all(np.minimum(amax[i], bmax) > np.maximum(amin[i], bmin), axis=1)
        dz = np.maximum(0., np.minimum(a[i, 2]+a[i, 5]/2, b[:, 2]+b[:, 5]/2)
                        - np.maximum(a[i, 2]-a[i, 5]/2, b[:, 2]-b[:, 5]/2))
        for j in np.flatnonzero(overlap):
            intersection = pa[i].intersection(pb[j]).area
            union = area_a[i]+area_b[j]-intersection
            results["bev"][i, j] = np.clip(intersection/union, 0., 1.)
            if dz[j] > 0:
                intersection *= dz[j]
                union = area_a[i]*a[i, 5]+area_b[j]*b[j, 5]-intersection
                results["3d"][i, j] = np.clip(intersection/union, 0., 1.)
    return results


def reference_rotated_nms(boxes, scores, prediction_ids=None, threshold=0.15, max_detections=500):
    """Pre-optimization v4 NMS, retained as an independent regression oracle.

    Lower-score boxes cannot suppress a previously selected box, so scan the
    stable score order until the 500th survivor. The prefix is never capped:
    if its candidates are suppressed, the scan continues. Only boxes whose
    AABB overlap could exceed the threshold need an exact polygon intersection.
    Polygons are cached on demand, avoiding construction for tens of thousands
    of low-score candidates that cannot affect the requested output.
    """
    boxes = validate_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError("Invalid NMS scores")
    if not 0 <= threshold <= 1 or max_detections < 1:
        raise ValueError("Invalid NMS settings")
    ids = [str(i) for i in range(len(boxes))] if prediction_ids is None else list(map(str, prediction_ids))
    if len(ids) != len(boxes) or len(set(ids)) != len(ids):
        raise ValueError("prediction_ids must be unique and complete")
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], ids[i]))
    corners = boxes_to_corners(boxes)[:,:4,:2]
    polygons = {}
    minima, maxima = corners.min(1), corners.max(1)
    areas = boxes[:,3]*boxes[:,4]
    keep = []
    for i in order:
        suppressed = False
        if keep:
            kept = np.asarray(keep)
            overlap_lengths = np.maximum(0., np.minimum(maxima[i], maxima[kept])-np.maximum(minima[i], minima[kept]))
            # min(box areas, AABB intersection) is an upper bound on the exact
            # intersection. Only use it to avoid impossible suppressions.
            upper_area = np.minimum(np.prod(overlap_lengths, axis=1), np.minimum(areas[i], areas[kept]))
            upper_iou = upper_area/(areas[i]+areas[kept]-upper_area)
            possible = np.flatnonzero((upper_area > 0) & (upper_iou >= threshold-1e-12))
            # Suppression is existential; checking the most likely overlaps
            # first reduces expensive clipping without changing greedy order.
            for index in possible[np.argsort(-upper_iou[possible], kind="stable")]:
                j = int(kept[index])
                if i not in polygons:
                    polygons[i] = Polygon(corners[i])
                if j not in polygons:
                    polygons[j] = Polygon(corners[j])
                intersection = polygons[i].intersection(polygons[j]).area
                if intersection/(areas[i]+areas[j]-intersection) > threshold:
                    suppressed = True
                    break
        if not suppressed:
            keep.append(i)
            if len(keep) == max_detections:
                break
    return np.asarray(keep, dtype=np.int64)


def _nms_overlap_candidates(i, kept, count, minima, maxima, areas, threshold):
    """Float64 AABB upper bounds in the reference's kept-box order.

    Compiling this small loop avoids allocating and reducing several NumPy
    arrays for every detector candidate. Fast math is deliberately disabled;
    these are the same operations and conservative bound as the reference.
    Polygon clipping and the final strict comparison remain in Shapely.
    """
    possible = np.empty(count, dtype=np.int64)
    upper_ious = np.empty(count, dtype=np.float64)
    found = 0
    for k in range(count):
        j = kept[k]
        dx = max(0., min(maxima[i, 0], maxima[j, 0])-max(minima[i, 0], minima[j, 0]))
        dy = max(0., min(maxima[i, 1], maxima[j, 1])-max(minima[i, 1], minima[j, 1]))
        upper_area = min(dx*dy, min(areas[i], areas[j]))
        upper_iou = upper_area/(areas[i]+areas[j]-upper_area)
        if upper_area > 0 and upper_iou >= threshold-1e-12:
            possible[found] = j
            upper_ious[found] = upper_iou
            found += 1
    return possible[:found], upper_ious[:found]


_nms_overlap_candidates_compiled = (njit(cache=True, fastmath=False, error_model="numpy")(_nms_overlap_candidates)
                                    if njit is not None else None)


def rotated_nms(boxes, scores, prediction_ids=None, threshold=0.15, max_detections=500):
    """Exact reference NMS with a compiled, conservative AABB prefilter.

    The score/ID order, lazy unbounded prefix, GEOS intersection operands and
    strict IoU threshold match ``reference_rotated_nms``. Numba is optional:
    environments without it execute the preserved NumPy reference directly.
    """
    if _nms_overlap_candidates_compiled is None:
        return reference_rotated_nms(boxes, scores, prediction_ids, threshold, max_detections)
    boxes = validate_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(boxes),) or not np.isfinite(scores).all():
        raise ValueError("Invalid NMS scores")
    if not 0 <= threshold <= 1 or max_detections < 1:
        raise ValueError("Invalid NMS settings")
    ids = [str(i) for i in range(len(boxes))] if prediction_ids is None else list(map(str, prediction_ids))
    if len(ids) != len(boxes) or len(set(ids)) != len(ids):
        raise ValueError("prediction_ids must be unique and complete")
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], ids[i]))
    corners = boxes_to_corners(boxes)[:, :4, :2]
    minima, maxima = corners.min(1), corners.max(1)
    areas = boxes[:, 3]*boxes[:, 4]
    # Extremely large but finite input dimensions can overflow derived areas
    # or corners. Preserve the reference's NumPy handling in that edge case.
    if not (np.isfinite(minima).all() and np.isfinite(maxima).all() and np.isfinite(areas).all()):
        return reference_rotated_nms(boxes, scores, ids, threshold, max_detections)
    polygons = {}
    keep = []
    kept = np.empty(len(boxes), dtype=np.int64)
    for i in order:
        suppressed = False
        if keep:
            possible, upper_iou = _nms_overlap_candidates_compiled(
                i, kept, len(keep), minima, maxima, areas, threshold)
            for index in np.argsort(-upper_iou, kind="stable"):
                j = int(possible[index])
                if i not in polygons:
                    polygons[i] = Polygon(corners[i])
                if j not in polygons:
                    polygons[j] = Polygon(corners[j])
                intersection = polygons[i].intersection(polygons[j]).area
                if intersection/(areas[i]+areas[j]-intersection) > threshold:
                    suppressed = True
                    break
        if not suppressed:
            kept[len(keep)] = i
            keep.append(i)
            if len(keep) == max_detections:
                break
    return np.asarray(keep, dtype=np.int64)
