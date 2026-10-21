"""Exact controlled provenance labels; no weather color/count heuristics."""
import unittest
import numpy as np
from opencood.project.supervision import (controlled_corruption,controlled_corruption_config,
    provenance_targets,normal_targets)
from opencood.project.weather import augmentation_seed,weather_for


class SourceSupervisionTests(unittest.TestCase):
    bounds=[0,0,-2,4,4,2]
    voxel=[1,1,4]

    def test_same_cell_deletion_and_injection_survive_equal_point_counts(self):
        normal=np.array([[1.1,1.1,0,.2],[1.2,1.2,0,.4]],np.float32)
        observed=np.array([[1.1,1.1,0,.2],[1.3,1.3,0,.4]],np.float32)
        target=provenance_targets(normal,observed,np.array([0,-1]),self.bounds,self.voxel)
        self.assertEqual(target['miss'][0,1,1],.5)
        self.assertEqual(target['spur'][0,1,1],.5)
        self.assertEqual(target['stats']['joint_positive_cells'],1)
        self.assertEqual(target['stats']['removed_points'],1)
        self.assertEqual(target['stats']['injected_points'],1)
        self.assertFalse(target['stats']['unmatched_semantics_verified'])

    def test_false_source_ids_are_rejected(self):
        normal=np.array([[1.1,1.1,0,.2],[1.2,1.2,0,.4]],np.float32)
        for observed,ids in ((normal[[0,0]],[0,0]),(normal[[1]],[0]),(normal[[0]],[-2])):
            with self.subTest(ids=ids),self.assertRaises(ValueError):
                provenance_targets(normal,observed,ids,self.bounds,self.voxel)

    def test_probability_and_injected_geometry_filter_are_exact(self):
        normal=np.tile(np.array([[1.5,1.5,0,.2]],np.float32),(40,1))
        config=controlled_corruption_config();config['probability']=0
        observed,ids,applied=controlled_corruption(normal,np.random.default_rng(1),config)
        self.assertFalse(applied)
        np.testing.assert_array_equal(observed,normal)
        np.testing.assert_array_equal(ids,np.arange(len(normal)))
        config.update(probability=1,drop_fraction=1,inject_fraction=.5)
        observed,ids,applied=controlled_corruption(normal,np.random.default_rng(1),config,
            legal_mask=lambda points:np.zeros(len(points),bool))
        self.assertTrue(applied)
        self.assertEqual(len(observed),0)
        target=provenance_targets(normal,observed,ids,self.bounds,self.voxel)
        self.assertEqual(target['miss'][0,1,1],1)
        self.assertEqual(target['spur_valid'].sum(),0)

    def test_corruption_is_epoch_keyed_without_global_rng_dependency(self):
        normal=np.tile(np.array([[1.5,1.5,0,.2]],np.float32),(100,1))
        config=controlled_corruption_config();config['probability']=1
        def corrupt(epoch):
            seed=augmentation_seed('sample/agent',epoch,17,stream='controlled_corruption')
            return controlled_corruption(normal,np.random.default_rng(seed),config)
        a=corrupt(0)
        np.random.seed(999);np.random.random(1000)
        b=corrupt(0);c=corrupt(1)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertNotEqual(a[0].tobytes(),c[0].tobytes())
        for epoch in range(20):
            self.assertIn(weather_for('sample','agent',epoch,17,'W'),('normal','fog'))

    def test_empty_source_and_unknown_grid_do_not_create_positive_labels(self):
        empty=np.empty((0,4),np.float32)
        targets=provenance_targets(empty,empty,np.empty(0,np.int64),self.bounds,self.voxel)
        self.assertTrue(all(value.sum()==0 for key,value in targets.items() if key!='stats'))
        ordinary=normal_targets(np.array([[1.2,1.2,0]],np.float32),self.bounds,self.voxel)
        self.assertEqual(ordinary['miss_valid'].sum(),1)
        self.assertEqual(ordinary['spur_valid'].sum(),1)
        self.assertEqual(ordinary['miss_valid'][0,0,0],0)


if __name__=='__main__':unittest.main()
