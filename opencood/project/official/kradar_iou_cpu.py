"""Exact CPU rectangle intersection for the official K-Radar AP accumulator.

Same [x,y,length,width,yaw] and criterion=-1/0/1/2 contract as nms_gpu.py.
This replaces only the IoU execution backend, not matching or interpolation.
"""
import numpy as np
from shapely.geometry import Polygon


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    del device_id
    def polygons(rows):
        out = []
        for x, y, length, width, yaw in np.asarray(rows).reshape(-1, 5):
            corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * [length / 2, width / 2]
            # Official K-Radar nms_gpu uses clockwise positive angles.
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, s], [-s, c]])
            out.append(Polygon(corners @ rotation.T + [x, y]))
        return out
    left, right = polygons(boxes), polygons(query_boxes)
    result = np.zeros((len(left), len(right)), dtype=np.float64)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            area = a.intersection(b).area
            denominator = (a.area + b.area - area if criterion == -1 else
                           a.area if criterion == 0 else b.area if criterion == 1 else 1.)
            result[i, j] = area / denominator if denominator > 0 else 0.
    return result
