"""Exact CPU NMS regression cases; no dataset or held-out split is read."""
import math
import unittest
from unittest import mock

import numpy as np

from opencood.project import geometry


BOX = np.array([10., 0., 0., 2., 2., 2., 0.])


class ExactNmsOptimizationTests(unittest.TestCase):
    def assert_same(self, boxes, scores, ids=None, threshold=.15, cap=500):
        expected = geometry.reference_rotated_nms(boxes, scores, ids, threshold, cap)
        actual = geometry.rotated_nms(boxes, scores, ids, threshold, cap)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        return actual

    def test_empty_single_and_lexical_tie_order(self):
        self.assertEqual(self.assert_same([], []).tolist(), [])
        self.assertEqual(self.assert_same([BOX], [.3]).tolist(), [0])
        boxes = np.tile(BOX, (4, 1))
        boxes[3, 0] += 20
        self.assertEqual(self.assert_same(boxes, [.8]*4, ["z", "a10", "a2", "a0"]).tolist(), [3, 1])

    def test_contact_and_strict_iou_threshold_neighbors(self):
        boxes = np.tile(BOX, (2, 1))
        boxes[1, 0] += 2
        self.assertEqual(self.assert_same(boxes, [1., .9], threshold=0.).tolist(), [0, 1])
        boxes[1] = boxes[0]
        self.assertEqual(self.assert_same(boxes, [1., .9], threshold=1.).tolist(), [0, 1])
        boxes[1, 0] += 2*(1-.15)/(1+.15)
        exact_iou = geometry.pairwise_iou(boxes[:1], boxes[1:], "bev")[0, 0]
        self.assertEqual(self.assert_same(boxes, [1., .9], threshold=exact_iou).tolist(), [0, 1])
        self.assertEqual(self.assert_same(boxes, [1., .9], threshold=np.nextafter(exact_iou, 0.)).tolist(), [0])
        self.assertEqual(self.assert_same(boxes, [1., .9], threshold=np.nextafter(exact_iou, 1.)).tolist(), [0, 1])

    def test_suppressed_box_cannot_suppress_later_survivor(self):
        boxes = np.tile(BOX, (3, 1))
        boxes[:, 0] += np.array([0., 1., 2.])
        self.assertEqual(self.assert_same(boxes, [1., .9, .8]).tolist(), [0, 2])

    def test_dense_rotations_match_independent_pairwise_reference(self):
        rng = np.random.default_rng(1701)
        boxes = np.column_stack([rng.uniform(-8, 8, (180, 2)), rng.uniform(-3, 3, 180),
                                 rng.uniform(.2, 8, (180, 3)), rng.uniform(-math.pi, math.pi, 180)])
        scores = rng.integers(0, 6, len(boxes))/6
        ids = [f"anchor_{i:07d}" for i in rng.permutation(len(boxes))]
        iou = geometry.pairwise_iou(boxes, boxes, "bev")
        for threshold in (0., .15, .5, 1.):
            for cap in (1, 17, 500):
                expected = []
                for i in sorted(range(len(boxes)), key=lambda i: (-scores[i], ids[i])):
                    if not expected or np.all(iou[i, expected] <= threshold):
                        expected.append(i)
                        if len(expected) == cap:
                            break
                with self.subTest(threshold=threshold, cap=cap):
                    self.assertEqual(self.assert_same(boxes, scores, ids, threshold, cap).tolist(), expected)

    def test_large_suppressed_prefix_retains_late_survivors(self):
        boxes = np.tile(BOX, (8195, 1))
        boxes[-3:, 0] += np.array([20., 40., 60.])
        scores = np.linspace(.9, .1, len(boxes))
        self.assertEqual(self.assert_same(boxes, scores, cap=4).tolist(), [0, 8192, 8193, 8194])

    def test_compiled_bounds_equal_numpy_reference_bits(self):
        if geometry._nms_overlap_candidates_compiled is None:
            self.skipTest("Numba is optional")
        rng = np.random.default_rng(17)
        boxes = np.column_stack([rng.uniform(-25, 25, (256, 3)),
                                 rng.uniform(.01, 12, (256, 3)), rng.uniform(-math.pi, math.pi, 256)])
        corners = geometry.boxes_to_corners(boxes)[:, :4, :2]
        minima, maxima = corners.min(1), corners.max(1)
        areas = boxes[:, 3]*boxes[:, 4]
        kept = rng.permutation(len(boxes)).astype(np.int64)
        for i in range(0, len(boxes), 7):
            lengths = np.maximum(0., np.minimum(maxima[i], maxima[kept])-np.maximum(minima[i], minima[kept]))
            upper_area = np.minimum(np.prod(lengths, axis=1), np.minimum(areas[i], areas[kept]))
            upper_iou = upper_area/(areas[i]+areas[kept]-upper_area)
            selected = (upper_area > 0) & (upper_iou >= .15-1e-12)
            possible, actual = geometry._nms_overlap_candidates_compiled(i, kept, len(kept), minima, maxima, areas, .15)
            np.testing.assert_array_equal(possible, kept[selected])
            np.testing.assert_array_equal(actual.view(np.uint64), upper_iou[selected].view(np.uint64))

    def test_optional_dependency_fallback_and_input_errors(self):
        with mock.patch.object(geometry, "_nms_overlap_candidates_compiled", None):
            self.assertEqual(self.assert_same([BOX, BOX], [.9, .8]).tolist(), [0])
        for fn in (geometry.reference_rotated_nms, geometry.rotated_nms):
            for args in (([BOX], [float("nan")]), ([BOX], []), ([BOX, BOX], [.8, .8], ["same", "same"])):
                with self.subTest(function=fn.__name__, args=args), self.assertRaises(ValueError):
                    fn(*args)

    def test_shared_iou_matches_both_reference_matrices_exactly(self):
        rng = np.random.default_rng(117)
        a = np.column_stack([rng.uniform(-8, 8, (75, 3)), rng.uniform(.1, 8, (75, 3)),
                             rng.uniform(-math.pi, math.pi, 75)])
        # Include identical, z-disjoint and arbitrary rotated pairs.
        b = np.concatenate([a[:15], a[15:30].copy(), a[30:45].copy(), a[45:60].copy()])
        b[15:30, 0] += 2*b[15:30, 3]
        b[30:45, 2] += 20
        analytic_b = np.tile(BOX, (4, 1))
        analytic_b[1, 0] += BOX[3]  # BEV edges only touch.
        analytic_b[2, 2] += BOX[5]  # BEV identical, z intervals only touch.
        analytic_b[3, 2] += 20     # BEV identical, disjoint z intervals.
        for left, right in ((a, b), ([BOX], analytic_b), (a, []), ([], b), ([], [])):
            both = geometry.pairwise_iou_both(left, right)
            for key in ("3d", "bev"):
                expected = geometry.pairwise_iou(left, right, key)
                np.testing.assert_array_equal(both[key].view(np.uint64), expected.view(np.uint64))


if __name__ == "__main__":
    unittest.main()
