"""Exact-output acceptance checks for CPU validation acceleration."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from opencood.project.evaluation import (
    evaluate_predictions, postprocess_prediction_arrays, postprocess_predictions,
    reference_postprocess_predictions, reference_score_records, score_records,
    _match_frame_thresholds, _match_frame_thresholds_python,
)
from opencood.project.fixed_gt import canonical_hash, default_protocol
from opencood.project.geometry import validate_boxes
from project_tools.export_predictions import (
    decode_prediction_arrays, decode_predictions, export_loader,
    reference_decode_predictions,
)


def prediction(box, score, pid):
    return {"prediction_id": pid, "class_name": "Vehicle", "score": score, "box": list(box)}


def manifest_for(samples):
    result = {"protocol_sha256": default_protocol()["protocol_sha256"], "samples": {},
              "summary": {"validate": {"num_samples": len(samples), "num_gt": sum(map(len, samples.values()))}}}
    for sid, boxes in samples.items():
        sample = {"sample_id": sid, "split": "validate", "sequence": "scene", "timestamp": sid,
                  "ego_id": "1", "agent_ids": ["1", "2"], "gt": [
                      {"gt_id": f"{sid}/{i}", "class_name": "Vehicle", "box": list(box)}
                      for i, box in enumerate(boxes)]}
        sample["gt_sha256"] = canonical_hash(sample["gt"])
        result["samples"][sid] = sample
    return result


class ValidationFastpathTests(unittest.TestCase):
    def test_matching_exact_threshold_ties_unmatched_nan_and_empty(self):
        matrix = np.array([[.5, .5], [.9, .4], [.2, .7], [.7, .7], [np.nan, 1.]])
        order = np.arange(len(matrix), dtype=np.int64)
        thresholds = np.array([.5, .7, np.nextafter(.7, 1.)])
        expected = np.zeros((len(thresholds), len(order)), dtype=np.int64)
        for t, threshold in enumerate(thresholds):
            matched = np.zeros(matrix.shape[1], dtype=bool)
            for k, row in enumerate(order):
                remaining = np.flatnonzero(~matched)
                if len(remaining):
                    values = matrix[row, remaining]
                    best = int(np.argmax(values))
                    if values[best] >= threshold:
                        matched[remaining[best]] = True
                        expected[t, k] = 1
        self.assertEqual(expected[0].tolist(), [1, 0, 1, 0, 0])
        self.assertEqual(expected[1].tolist(), [0, 1, 1, 0, 0])
        for matcher in (_match_frame_thresholds_python, _match_frame_thresholds):
            np.testing.assert_array_equal(matcher(matrix, order, thresholds), expected)
            np.testing.assert_array_equal(matcher(matrix[-1:], np.array([0]), thresholds), np.zeros((3, 1)))
            np.testing.assert_array_equal(matcher(np.empty((3, 0)), np.arange(3), thresholds), np.zeros((3, 3)))
            self.assertEqual(matcher(np.empty((0, 2)), np.empty(0, dtype=np.int64), thresholds).shape, (3, 0))

    def test_framewise_scoring_matches_independent_global_algorithm(self):
        rng = np.random.default_rng(23)
        gt, records = {"negative": [], "empty": [[10, 0, 0, 4, 2, 2, 0]]}, {"negative": [], "empty": []}
        for sid in ("b", "a", "c"):
            boxes = np.column_stack([rng.uniform(5, 15, (120, 2)), np.zeros(120),
                                     rng.uniform(1, 5, (120, 3)), rng.uniform(-2, 2, 120)])
            gt[sid] = boxes[:7]
            records[sid] = [prediction(box, float(rng.integers(0, 5)) / 5, f"p{i:03}")
                            for i, box in enumerate(boxes)]
        before = reference_score_records(gt, records)
        self.assertEqual(score_records(gt, records), before)
        self.assertEqual(score_records(gt, records, implementation="reference"), before)

    def test_arrays_and_dict_path_exact_counts_order_values_and_full_metrics(self):
        rng = np.random.default_rng(17)
        boxes = np.column_stack([rng.uniform(-20, 160, (900, 1)), rng.uniform(-50, 50, (900, 1)),
                                 rng.uniform(-1, 2, (900, 1)), rng.uniform(.5, 6, (900, 3)),
                                 rng.uniform(-np.pi, np.pi, (900, 1))])
        scores = rng.integers(0, 10, len(boxes)) / 10
        ids = [f"anchor_{i:07d}" for i in range(len(boxes))]
        # Include exact floor, ROI boundaries, same-score duplicate geometry,
        # and below-floor candidates whose exclusion counts must be retained.
        boxes[:6] = [[0, 0, 1, 4, 2, 2, 0], [140, 0, 1, 4, 2, 2, 0],
                     [10, -40, 1, 4, 2, 2, 0], [10, 40, 1, 4, 2, 2, 0],
                     [10, 0, 1, 4, 2, 2, 0], [10, 0, 1, 4, 2, 2, 0]]
        scores[:6] = [.001, .9, .9, .9, .8, .8]
        records = [prediction(b, float(s), pid) for b, s, pid in zip(boxes, scores, ids)]
        reference = reference_postprocess_predictions(records, {})
        self.assertEqual(postprocess_predictions(records, {}), reference)
        self.assertEqual(postprocess_prediction_arrays(boxes, scores, ids, {}), reference)
        self.assertEqual(canonical_hash(postprocess_predictions(records, {})), canonical_hash(reference))
        manifest = manifest_for({"a": [boxes[4]], "b": []})
        complete = [{"sample_id": "a", "predictions": records}, {"sample_id": "b", "predictions": []}]
        before = evaluate_predictions(manifest, complete, "validate", implementation="reference")
        after = evaluate_predictions(manifest, complete, "validate")
        self.assertEqual(after, before)  # Includes AP/PR, flags, orders, counts, every hash.

    def test_fov_and_empty_arrays_match_reference(self):
        camera = {"intrinsic": np.eye(3).tolist(), "lidar_to_camera": np.eye(4).tolist(),
                  "image_width": 2, "image_height": 2}
        boxes = [[0, 0, 1, 1, 1, 1, 0], [2, 0, 1, 1, 1, 1, 0],
                 [0, 2, 1, 1, 1, 1, 0], [1.9, 1.9, 1, 1, 1, 1, 0],
                 [0, 0, 0, 1, 1, 1, 0], [0, 0, -1, 1, 1, 1, 0]]
        records = [prediction(b, .9, str(i)) for i, b in enumerate(boxes)]
        self.assertEqual(postprocess_predictions(records, {"camera": camera}),
                         reference_postprocess_predictions(records, {"camera": camera}))
        self.assertEqual(postprocess_prediction_arrays([], [], [], {}),
                         reference_postprocess_predictions([], {}))

    def test_batch_validation_eliminates_per_candidate_numpy_checks(self):
        records = [prediction([-1, 0, 0, 4, 2, 2, 0], .5, str(i)) for i in range(128)]
        with mock.patch("opencood.project.evaluation.validate_boxes", wraps=validate_boxes) as check:
            reference_postprocess_predictions(records, {})
            self.assertEqual(check.call_count, len(records))
        with mock.patch("opencood.project.evaluation.validate_boxes", wraps=validate_boxes) as check:
            postprocess_predictions(records, {})
            self.assertEqual(check.call_count, 1)

    def test_all_candidates_validated_before_filtering_and_markers_do_not_bypass(self):
        baseline = prediction([-1, 0, 0, 4, 2, 2, 0], 0., "low_outside")
        for field, value in (("score", float("nan")), ("score", 1.1), ("class_name", "Car"),
                             ("box", [-1, 0, 0, -1, 2, 2, 0]),
                             ("box", [-1, 0, 0, float("inf"), 2, 2, 0]), ("prediction_id", "")):
            bad = {**baseline, field: value}
            for postprocessor in (reference_postprocess_predictions, postprocess_predictions):
                with self.assertRaises(ValueError):
                    postprocessor([bad], {})
        for postprocessor in (reference_postprocess_predictions, postprocess_predictions):
            with self.assertRaises(ValueError):
                postprocessor([baseline, baseline], {})
        manifest = manifest_for({"a": []})
        marked = {"sample_id": "a", "predictions": [{**baseline, "class_name": "Car"}],
                  "postprocessed": True, "postprocess_sha256": "trusted", "filter_counts": {"output": 1}}
        with self.assertRaises(ValueError):
            evaluate_predictions(manifest, [marked], "validate")
        with self.assertRaises(ValueError):
            postprocess_prediction_arrays([[0, 0, 0, -1, 2, 2, 0]], [0.], ["low"], {})

    def test_decode_matches_reference_exactly_and_rejects_nonfinite_below_floor(self):
        torch.manual_seed(17)
        anchors = torch.randn(2, 3, 2, 7)
        anchors[..., 3:6] = anchors[..., 3:6].abs() + 1
        output = {"rm": torch.randn(2, 14, 2, 3) / 5, "psm": torch.randn(2, 2, 2, 3)}
        output["psm"][1].fill_(-100)
        before = reference_decode_predictions(output, anchors, ["a", "b"])
        after = decode_predictions(output, anchors, ["a", "b"])
        self.assertEqual(after, before)
        self.assertEqual(json.dumps(after), json.dumps(before))
        arrays = decode_prediction_arrays(output, anchors, ["a", "b"])
        for record, array in zip(before, arrays):
            self.assertEqual(postprocess_prediction_arrays(array["boxes"], array["scores"], array["prediction_ids"], {}),
                             reference_postprocess_predictions(record["predictions"], {}))
        output["rm"][1, 0, 0, 0] = float("nan")
        for decoder in (reference_decode_predictions, decode_predictions, decode_prediction_arrays):
            with self.assertRaises(ValueError):
                decoder(output, anchors, ["a", "b"])

    def test_export_full_record_bytes_profile_and_mode_restore(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = torch.nn.Identity()

            def forward(self, inputs, return_aux=False):
                return {"rm": torch.zeros(2, 7, 1, 2),
                        "psm": torch.tensor([[[[0., 1.]]], [[[-100., -100.]]]])}

        manifest = manifest_for({"a": [[10, 0, 1, 4, 2, 2, 0]], "b": []})
        dataset = SimpleNamespace(manifest=manifest, weather="normal", anchor_box=torch.tensor(
            [[[[10., 0., 1., 2., 2., 4., 0.]], [[10., 0., 1., 2., 2., 4., 0.]]]]))
        batch = {"ego": {"sample_ids": ["a", "b"], "metadata": [
            [{"agent_id": "1", "weather": "normal"}], [{"agent_id": "1", "weather": "normal"}]],
            "processed_lidar": {}}}
        model = Detector()
        model.train()
        model.child.eval()
        with mock.patch("project_tools.export_predictions.select_model_inputs", return_value={}):
            reference = export_loader(model, [copy.deepcopy(batch)], dataset, torch.device("cpu"), implementation="reference")
            events = []
            optimized = export_loader(model, [copy.deepcopy(batch)], dataset, torch.device("cpu"), profile_callback=events.append)
            raw_before = export_loader(model, [copy.deepcopy(batch)], dataset, torch.device("cpu"), postprocess=False, implementation="reference")
            raw_after = export_loader(model, [copy.deepcopy(batch)], dataset, torch.device("cpu"), postprocess=False)
        self.assertEqual(optimized, reference)
        self.assertEqual(json.dumps(optimized), json.dumps(reference))
        self.assertEqual(raw_after, raw_before)
        self.assertEqual(evaluate_predictions(manifest, optimized, "validate"),
                         evaluate_predictions(manifest, reference, "validate", implementation="reference"))
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertEqual([e["stage"] for e in events], ["loader_wait", "h2d", "forward", "decode", "postprocess", "postprocess"])
        self.assertTrue(all(e["seconds"] >= 0 for e in events))


if __name__ == "__main__":
    unittest.main()
