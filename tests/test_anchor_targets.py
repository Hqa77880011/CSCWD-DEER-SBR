"""Metre-coordinate matching and forced-target collision counterexamples."""
import unittest
import numpy as np

from opencood.project.anchor_targets import AnchorTargets,continuous_aabb_iou


def anchors(xs):
    return np.asarray([[x,0,0,2,2,2,0] for x in xs],np.float32).reshape(1,len(xs),1,7)


def gt(xs):
    return np.asarray([[x,0,0,2,2,2,0] for x in xs],np.float32)


def decoded_centres(targets,base):
    deltas=targets['targets'].reshape(-1,7)
    base=base.reshape(-1,7)
    diagonal=np.sqrt(base[:,4]**2+base[:,5]**2)
    return deltas[:,:2]*diagonal[:,None]+base[:,:2]


class AnchorTargetTests(unittest.TestCase):
    def test_continuous_geometry_no_pixel_plus_one(self):
        result=continuous_aabb_iou([[0,0,2,2]],[[2.5,0,4.5,2],[2,0,4,2],[1,0,3,2]])
        np.testing.assert_allclose(result,[[0,0,1/3]],rtol=1e-7)

    def test_distinct_best_anchors_force_their_own_gt(self):
        # A=0 is GT1's best (.4545), yet prefers GT2 (.6). GT2's own
        # best is B=.5 (1.0). Both GT must receive their distinct best anchor.
        base=anchors([0,.5]);assigner=AnchorTargets(base,positive=.7)
        targets=assigner(gt([-.75,.5]))
        np.testing.assert_array_equal(targets['pos_equal_one'].reshape(-1),[1,1])
        np.testing.assert_allclose(decoded_centres(targets,base)[:,0],[-.75,.5],atol=1e-7)
        self.assertEqual(assigner.last_diagnostics['unrepresented_gt_indices'],[])
        self.assertEqual(assigner.last_diagnostics['forced_collision_count'],0)

    def test_true_collision_uses_highest_iou_and_reports_uncovered_gt(self):
        base=anchors([0]);assigner=AnchorTargets(base)
        targets=assigner(gt([-.75,.5]))
        self.assertAlmostEqual(decoded_centres(targets,base)[0,0],.5)
        self.assertEqual(assigner.last_diagnostics['forced_collision_count'],1)
        self.assertEqual(assigner.last_diagnostics['unrepresented_gt_indices'],[0])

    def test_equal_iou_collision_uses_stable_gt_order(self):
        base=anchors([0]);assigner=AnchorTargets(base)
        targets=assigner(gt([-.5,.5]))
        self.assertAlmostEqual(decoded_centres(targets,base)[0,0],-.5)
        self.assertEqual(assigner.last_diagnostics['unrepresented_gt_indices'],[1])

    def test_iou_ignore_band_stays_neither_positive_nor_negative(self):
        base=anchors([0,.65,2.5]);assigner=AnchorTargets(base)
        targets=assigner(gt([0]))
        np.testing.assert_array_equal(targets['pos_equal_one'].reshape(-1),[1,0,0])
        np.testing.assert_array_equal(targets['neg_equal_one'].reshape(-1),[0,0,1])
        self.assertEqual(assigner.last_diagnostics['num_ignored'],1)

    def test_disjoint_gt_is_reported_not_forced_to_distant_anchor(self):
        assigner=AnchorTargets(anchors([0]))
        targets=assigner(gt([2.5]))
        self.assertEqual(targets['pos_equal_one'].sum(),0)
        self.assertEqual(targets['neg_equal_one'].sum(),1)
        self.assertEqual(assigner.last_diagnostics['zero_overlap_gt_indices'],[0])
        self.assertEqual(assigner.last_diagnostics['unrepresented_gt_indices'],[0])

    def test_empty_gt_preserves_all_negatives_and_resets_diagnostics(self):
        assigner=AnchorTargets(anchors([0,.5]))
        assigner(gt([0]))
        targets=assigner(np.empty((0,7)))
        self.assertEqual(targets['pos_equal_one'].sum(),0)
        self.assertEqual(targets['neg_equal_one'].sum(),2)
        self.assertEqual(assigner.last_diagnostics['num_gt'],0)
        self.assertEqual(assigner.last_diagnostics['unrepresented_gt_indices'],[])


if __name__=='__main__':unittest.main()
