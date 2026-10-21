"""Source transformations, safe YAML, and complete prepared-data integration."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from opencood.project.fixed_gt import (AnnotationLoader, build_sample, canonical_hash,
                                      default_protocol, file_sha256, gt_boxes,
                                      load_manifest, ordered_agents)
from opencood.project.geometry import (admissibility, camera_from_annotation, object_to_ego_box,
                                      pose_to_world, transform_points)


def obj(location=(10,0,0), center=(0,0,1), angle=(0,0,0)):
    return {"location": list(location),"center":list(center),"angle":list(angle),
            "extent":[2,1,1],"bp_id":"vehicle.test.car"}


def annotation(objects):
    return {"lidar_pose":[0,0,0,0,0,0],"camera0":{"cords":[0,0,0,0,0,0],
            "extrinsic":np.eye(4).tolist(),"intrinsic":[[400,0,400],[0,400,300],[0,0,1]]},"vehicles":objects}


class FixedGTTests(unittest.TestCase):
    def test_agent_order_matches_upstream_lexicographic_ego(self):
        self.assertEqual(ordered_agents(["20","-1","100"]),["100","20","-1"])

    def test_rotates_vehicle_local_bbox_center(self):
        box,corners = object_to_ego_box(obj(location=(0,0,0),center=(2,0,1),angle=(0,90,0)),[0]*6)
        np.testing.assert_allclose(box[:3],[0,2,1],atol=1e-12)
        np.testing.assert_allclose(box[3:6],[4,2,2],atol=1e-12)
        self.assertAlmostEqual(box[6],math.pi/2)
        np.testing.assert_allclose(corners.mean(0),box[:3],atol=1e-12)

    def test_world_to_ego_transform_direction(self):
        box,_ = object_to_ego_box(obj(location=(10,10,0),center=(0,0,0),angle=(0,90,0)),[10,0,0,0,90,0])
        np.testing.assert_allclose(box[:3],[10,0,0],atol=1e-12)
        self.assertAlmostEqual(box[6],0.)
        transform=pose_to_world([10,0,0,0,90,0])
        np.testing.assert_allclose(transform_points([[10,0,0]],transform),[[10,10,0]],atol=1e-12)

    def test_roll_pitch_upright_encloses_all_eight_corners(self):
        box,corners = object_to_ego_box(obj(angle=(23,179,17)),[1,2,3,-5,12,7])
        c,s=np.cos(box[6]),np.sin(box[6])
        local=(corners-box[:3]) @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self.assertTrue(np.all(np.abs(local) <= box[3:6]/2+1e-10))
        np.testing.assert_allclose(local.max(0)-local.min(0),box[3:6],atol=1e-10)

    def test_yaw_near_wrap_is_not_averaged_to_zero(self):
        for yaw in [179.999,-179.999,180]:
            box,_=object_to_ego_box(obj(angle=(0,yaw,0)),[0]*6)
            self.assertAlmostEqual(abs(box[6]),abs(math.radians(yaw)))

    def test_camera_axis_conversion_and_bad_calibration_rejected(self):
        a=annotation({})
        camera=camera_from_annotation(a,[800,600])
        np.testing.assert_allclose(transform_points([[10,2,3]],camera["lidar_to_camera"]),[[2,-3,10]])
        a["camera0"]["extrinsic"][0][3]=1
        with self.assertRaises(ValueError): camera_from_annotation(a,[800,600])

    def test_safe_numpy_scalar_and_reject_arbitrary_python(self):
        encoded=yaml.dump({"value":np.float64(1.25)})
        self.assertEqual(yaml.load(encoded,Loader=AnnotationLoader)["value"],1.25)
        with self.assertRaises(yaml.constructor.ConstructorError):
            yaml.load("!!python/object/apply:os.system ['false']",Loader=AnnotationLoader)

    def _sample(self, root, second_object=None):
        frames=[]
        for agent,objects in [("1",{101:obj()}),("2",{101:obj() if second_object is None else second_object,102:obj(location=(150,0,0))})]:
            path=Path(root)/f"validate/scene/{agent}/000001.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(annotation(objects)))
            frames.append({"sample_id":"validate/scene/000001","split":"validate","sequence":"validate/scene", "timestamp":"000001","agent":agent,
                           "modalities":{"annotation":{"path":str(path.relative_to(root)),"sha256":file_sha256(path)}}})
        return build_sample(root,frames,{"validate/scene/1/000001_camera0.png":[800,600]},default_protocol())[0]

    def test_duplicate_union_original_ids_exclusions_and_no_point_reads(self):
        with tempfile.TemporaryDirectory() as root:
            sample=self._sample(root)
        self.assertEqual(sample["num_source_objects"],3)
        self.assertEqual(sample["num_original_gt"],2)
        self.assertEqual(sample["num_gt"],1)
        self.assertEqual(sample["gt"][0]["original_id"],"101")
        self.assertEqual(sample["original_gt"][0]["source_agent_ids"],["1","2"])
        self.assertEqual(sample["excluded"][0]["reasons"],["center_outside_xy_roi"])
        self.assertEqual(sample["gt_sha256"],canonical_hash(sample["gt"]))

    def test_conflicting_original_id_is_error_not_silent_pick(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,"Conflicting duplicate"):
                self._sample(root,obj(location=(11,0,0)))


DATA_ROOT = Path(os.environ.get("CSCWD_DATA_ROOT",str(Path(__file__).resolve().parents[1]/"datasets/v2xr")))


@unittest.skipUnless((DATA_ROOT/"eval_gt_manifest.json").exists(),"Prepared full dataset is not available")
class FullPreparedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=load_manifest(DATA_ROOT/"eval_gt_manifest.json")

    def test_all_official_samples_and_sensor_frames_covered(self):
        expected={"train":8083,"validate":829,"test":3166}
        for split,count in expected.items():
            self.assertEqual(self.manifest["summary"][split]["num_samples"],count)
        self.assertEqual(len(self.manifest["samples"]),12078)
        self.assertEqual(sum(len(s["source_annotations"]) for s in self.manifest["samples"].values()),37727)
        self.assertEqual(self.manifest["audit"]["conflicting_duplicates"],0)

    def test_every_accepted_gt_has_source_and_satisfies_same_geometry(self):
        for sid,sample in self.manifest["samples"].items():
            original={o["original_id"]:o for o in sample["original_gt"]}
            self.assertEqual(len(original),sample["num_original_gt"])
            self.assertEqual(len(sample["gt"])+len(sample["excluded"]),len(original))
            accepted_ids=[g["original_id"] for g in sample["gt"]]
            self.assertEqual(len(accepted_ids),len(set(accepted_ids)))
            self.assertTrue(all(not original[oid]["excluded_reasons"] for oid in accepted_ids))
            self.assertTrue(admissibility(gt_boxes(sample),sample["camera"])[0].all(),sid)

    def test_weather_invariance_no_gt_cap_and_empty_samples_retained(self):
        self.assertEqual(self.manifest["weather_invariant"],["normal","fog","snow"])
        self.assertFalse(self.manifest["gt_truncation"])
        self.assertFalse(self.manifest["audit"]["point_clouds_read"])
        self.assertTrue(any(s["num_gt"]==0 for s in self.manifest["samples"].values()))
        for sample in self.manifest["samples"].values():
            self.assertNotIn("weather",sample)
            self.assertEqual(sample["num_gt"],len(sample["gt"]))


if __name__ == "__main__":
    unittest.main()
