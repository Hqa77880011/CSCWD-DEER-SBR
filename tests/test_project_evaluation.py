"""Independent analytical acceptance cases; these are not dataset scores."""
import copy
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from opencood.project.evaluation import (evaluate_predictions, postprocess_predictions,
                                         read_prediction_archive, score_records,
                                         validate_archive_metadata, voc2010_ap)
from opencood.project.fixed_gt import canonical_hash, default_protocol
from opencood.project.geometry import (admissibility, pairwise_iou, points_in_fov,
                                      rotated_nms, validate_boxes)


BOX = [10., 0., 0., 2., 2., 2., 0.]


def prediction(box=BOX, score=0.9, pid="p"):
    return {"prediction_id": pid, "class_name": "Vehicle", "score": score, "box": list(box)}


def simple_manifest(samples):
    return {"protocol_sha256": default_protocol()["protocol_sha256"],
            "samples": {sid: {"sample_id": sid, "split": "validate", "gt": [
                {"gt_id": f"{sid}/{i}", "original_id": str(i), "class_name": "Vehicle", "box": b}
                for i,b in enumerate(boxes)]} for sid, boxes in samples.items()},
            "summary": {"validate": {"num_samples": len(samples), "num_gt": sum(map(len, samples.values()))}}}


class ProjectEvaluationTests(unittest.TestCase):
    def test_eval_z_001(self):
        displaced = list(BOX)
        displaced[2] = 3.
        self.assertAlmostEqual(pairwise_iou([BOX], [displaced], "bev")[0,0], 1.)
        self.assertAlmostEqual(pairwise_iou([BOX], [displaced], "3d")[0,0], 0.)
        result = score_records({"a": [BOX]}, {"a": [prediction(displaced)]})
        self.assertEqual(result["metrics"]["3d"]["0.3"]["ap"], 0.)
        self.assertEqual(result["metrics"]["bev"]["0.7"]["ap"], 1.)

    def test_eval_empty_002(self):
        result = score_records({"a": [BOX], "b": [BOX]}, {"a": [prediction()], "b": []})
        self.assertEqual(result["num_gt"], 2)
        self.assertEqual(result["num_empty_prediction_frames"], 1)
        for geometry in ("3d", "bev"):
            self.assertEqual(result["metrics"][geometry]["0.7"]["max_recall"], .5)
            self.assertEqual(result["metrics"][geometry]["0.7"]["ap"], .5)

    def test_eval_order_003(self):
        gt = {"a": [], "b": [BOX]}
        predictions = {"a": [prediction(score=.9)], "b": [prediction(score=.8)]}
        result = score_records(gt, predictions)
        reversed_result = score_records(dict(reversed(list(gt.items()))), dict(reversed(list(predictions.items()))))
        self.assertEqual(result, reversed_result)
        self.assertEqual(result["metrics"]["3d"]["0.7"]["ap"], .5)

    def test_same_score_stable_ids(self):
        gt = {"a": [], "b": [BOX]}
        p = {"a": [prediction(score=.5)], "b": [prediction(score=.5)]}
        self.assertEqual(score_records(gt,p)["metrics"]["3d"]["0.7"]["ap"], .5)
        far = list(BOX)
        far[0] = 50.
        p = {"b": [prediction(pid="z"), prediction(far, pid="a")]}
        r = score_records({"b": [BOX]}, p)
        self.assertEqual(r["global_prediction_order"][0]["prediction_id"], "a")
        self.assertEqual(r["metrics"]["3d"]["0.7"]["ap"], .5)

    def test_duplicates_and_each_threshold_new_matching(self):
        p = {"a": [prediction(pid="1"), prediction(score=.8,pid="2")]}
        r = score_records({"a": [BOX]}, p)
        for g in ("3d", "bev"):
            for threshold in ("0.3", "0.5", "0.7"):
                self.assertEqual(r["metrics"][g][threshold]["tp_flags"], [1,0])
                self.assertEqual(r["metrics"][g][threshold]["fp_flags"], [0,1])

    def test_searches_unmatched_gt_not_matched_best(self):
        second = list(BOX)
        second[0] += .5
        r = score_records({"a": [BOX,second]}, {"a": [prediction(pid="1"),prediction(score=.8,pid="2")]})
        self.assertEqual(r["metrics"]["3d"]["0.5"]["tp"], 2)
        self.assertEqual(r["metrics"]["3d"]["0.7"]["tp"], 1)

    def test_rotated_square_exact_and_partial_height(self):
        rotated = list(BOX)
        rotated[6] = math.pi/4
        self.assertAlmostEqual(pairwise_iou([BOX],[rotated],"bev")[0,0], 1/math.sqrt(2), places=12)
        shifted = list(BOX)
        shifted[2] = 1.
        self.assertAlmostEqual(pairwise_iou([BOX],[shifted],"3d")[0,0], 1/3)
        thin_a, thin_b = list(BOX), list(BOX)
        thin_a[3:5], thin_b[3:5], thin_b[6] = [10.,1.], [10.,1.], math.pi/2
        self.assertAlmostEqual(pairwise_iou([thin_a],[thin_b],"bev")[0,0], 1/19)

    def test_touching_boxes_have_zero_iou(self):
        touching = list(BOX)
        touching[0] += 2
        self.assertEqual(pairwise_iou([BOX],[touching],"3d")[0,0], 0.)

    def test_no_gt_is_not_applicable_and_fp_retained(self):
        r = score_records({"a": []}, {"a": [prediction()]})
        self.assertIsNone(r["metrics"]["3d"]["0.7"]["ap"])
        self.assertEqual(r["metrics"]["3d"]["0.7"]["status"], "not_applicable")
        self.assertEqual(r["metrics"]["3d"]["0.7"]["fp"], 1)

    def test_all_predictions_empty_scores_zero_with_gt(self):
        r = score_records({"a": [BOX]}, {"a": []})
        self.assertEqual(r["metrics"]["3d"]["0.7"]["ap"], 0.)

    def test_no_gt_truncation_at_100(self):
        boxes = [list(BOX) for _ in range(137)]
        r = score_records({"a": boxes}, {"a": []})
        self.assertEqual(r["num_gt"], 137)

    def test_center_roi_half_open_no_corner_or_z_filter(self):
        boxes = [list(BOX) for _ in range(6)]
        for box,x,y,z in zip(boxes, [0,140,10,10,139.9,10], [0,0,-40,40,39.9,0], [0,0,0,0,0,200]):
            box[:3] = [x,y,z]
        valid,_,_ = admissibility(boxes)
        np.testing.assert_array_equal(valid,[True,False,True,False,True,True])

    def test_fov_boundaries_positive_depth(self):
        camera = {"intrinsic": np.eye(3).tolist(), "lidar_to_camera": np.eye(4).tolist(), "image_width": 2,"image_height":2}
        points = [[0,0,1],[2,0,1],[0,2,1],[1.999,1.999,1],[0,0,0],[0,0,-1],[-.001,0,1]]
        np.testing.assert_array_equal(points_in_fov(points,camera),[True,False,False,True,False,False,False])

    def test_rotated_nms_score_floor_cap_and_idempotence(self):
        p = [prediction(score=.8,pid="z"),prediction(score=.8,pid="a"),prediction(score=.0009,pid="low")]
        out, counts = postprocess_predictions(p,{})
        self.assertEqual([x["prediction_id"] for x in out],["a"])
        self.assertEqual(counts["below_score_floor"],1)
        self.assertEqual(postprocess_predictions(out,{})[0],out)
        self.assertEqual(rotated_nms([BOX,BOX],[.8,.8],["z","a"]).tolist(),[1])
        distant = [list(BOX) for _ in range(4)]
        for i,b in enumerate(distant): b[0] += i*10
        self.assertEqual(rotated_nms(distant,[.1,.9,.5,.7],max_detections=2).tolist(),[1,3])

    def test_invalid_predictions_and_missing_duplicate_frames_raise(self):
        manifest = simple_manifest({"a":[BOX],"b":[]})
        with self.assertRaises(ValueError): evaluate_predictions(manifest,[{"sample_id":"a","predictions":[]}],"validate")
        records=[{"sample_id":"a","predictions":[]},{"sample_id":"b","predictions":[]}]
        with self.assertRaises(ValueError): evaluate_predictions(manifest,records+[records[0]],"validate")
        for field,value in [("score",float("nan")),("score",1.1),("class_name","Car"),("box",[10,0,0,-1,2,2,0])]:
            p = prediction()
            p[field] = value
            with self.assertRaises(ValueError): postprocess_predictions([p],{})
        with self.assertRaises(ValueError): postprocess_predictions([prediction(),prediction()],{})
        bad = copy.deepcopy(default_protocol())
        bad["score_floor"] = .2
        with self.assertRaises(ValueError): evaluate_predictions(manifest,records,"validate",bad)

    def test_public_api_complete_records_and_fixed_counts(self):
        manifest = simple_manifest({"a":[BOX],"b":[BOX]})
        r = evaluate_predictions(manifest,[{"sample_id":"a","predictions":[prediction()]},{"sample_id":"b","predictions":[]}],"validate")
        self.assertEqual(r["metrics"]["3d"]["0.7"]["ap"],.5)
        self.assertEqual(r["num_gt"],2)

    def test_voc2010_envelope(self):
        self.assertAlmostEqual(voc2010_ap([0,.5,.5,1],[0,.5,1/3,.5]),.5)
        with self.assertRaises(ValueError): voc2010_ap([.5,.1],[1,1])

    def test_optimized_nms_matches_independent_dense_pairwise_reference(self):
        rng=np.random.default_rng(29)
        boxes=np.column_stack([rng.uniform(-15,15,(150,2)),np.zeros(150),
                               rng.uniform(.8,5,(150,3)),rng.uniform(-math.pi,math.pi,150)])
        scores=rng.integers(0,8,150)/8
        ids=[f"id_{i:06d}" for i in range(150)]
        matrix=pairwise_iou(boxes,boxes,"bev")
        expected=[]
        for index in sorted(range(len(boxes)),key=lambda i:(-scores[i],ids[i])):
            if not expected or np.all(matrix[index,expected] <= .15):
                expected.append(index)
                if len(expected)==50:break
        self.assertEqual(rotated_nms(boxes,scores,ids,max_detections=50).tolist(),expected)

    def test_nms_adaptive_prefix_does_not_truncate_late_survivors(self):
        boxes=np.tile(BOX,(3002,1))
        boxes[3000:,0]=[30,50]
        scores=np.linspace(.9,.1,len(boxes))
        self.assertEqual(rotated_nms(boxes,scores,max_detections=3).tolist(),[0,3000,3001])

    def test_70k_nms_avoids_eager_polygon_construction(self):
        # A latency regression guard without hardware-specific wallclock limits:
        # 70k candidates and a 500 cap must not construct 70k Shapely polygons.
        boxes=np.tile(BOX,(70000,1))
        boxes[:,0]=np.arange(len(boxes))*5
        scores=np.linspace(.9,.1,len(boxes))
        from shapely.geometry import Polygon
        with mock.patch("opencood.project.geometry.Polygon",wraps=Polygon) as polygon:
            keep=rotated_nms(boxes,scores,max_detections=500)
        self.assertEqual(keep.tolist(),list(range(500)))
        self.assertLessEqual(polygon.call_count,500)

    def test_archive_weather_source_hash_and_sidecar_validation(self):
        manifest=simple_manifest({"a":[BOX],"b":[]})
        records=[]
        for sid,sample in manifest["samples"].items():
            sample.update(sequence="scene",timestamp=sid,ego_id="1",agent_ids=["1","2"],gt_sha256=canonical_hash(sample["gt"]))
            records.append({"sample_id":sid,"predictions":[],"split":"validate","dataset":"v2xr",
                            "sequence":"scene","timestamp":sid,"ego_id":"1","agent_ids":["1"],
                            "source_agent_ids":["1","2"],"agent_weather":{"1":"normal"},
                            "evaluation_weather":"normal","gt_sha256":sample["gt_sha256"],
                            "protocol_sha256":manifest["protocol_sha256"]})
        metadata={"protocol":"project_fix_v1","protocol_sha256":manifest["protocol_sha256"],
                  "gt_manifest_sha256":"a"*64,"dataset":"v2xr","split":"validate","evaluation_weather":"normal",
                  "source_kind":"retrained","checkpoint_sha256":"b"*64,"config_sha256":"c"*64,
                  "checkpoint_epoch":3,"postprocessed":True,"sample_count":2,"predictions_file_sha256":"d"*64}
        self.assertTrue(validate_archive_metadata(metadata,records,manifest,"validate","a"*64,"d"*64))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"predictions.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records)+"\n")
            path.with_suffix(".jsonl.meta.json").write_text(json.dumps(metadata))
            self.assertEqual(read_prediction_archive(path),(metadata,records))
        with self.assertRaises(ValueError): validate_archive_metadata(metadata,records,manifest,"validate","a"*64,"e"*64)
        wrong=copy.deepcopy(records)
        wrong[1]["evaluation_weather"]="fog"
        wrong[1]["agent_weather"]={"1":"fog"}
        with self.assertRaises(ValueError): evaluate_predictions(manifest,wrong,"validate")
        with self.assertRaises(ValueError): validate_archive_metadata(metadata,wrong,manifest,"validate","a"*64)
        for field,value in [("gt_sha256","0"*64),("agent_ids",["3"]),("source_agent_ids",["1"]),("sequence","other")]:
            wrong=copy.deepcopy(records)
            wrong[0][field]=value
            with self.assertRaises(ValueError): evaluate_predictions(manifest,wrong,"validate")
        absent=dict(metadata)
        del absent["checkpoint_sha256"]
        with self.assertRaises(ValueError): validate_archive_metadata(absent,records,manifest,"validate","a"*64)


if __name__ == "__main__":
    unittest.main()
