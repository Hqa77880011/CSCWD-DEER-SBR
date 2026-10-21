"""PointPillars target assignment, with explicit negatives and no GT truncation."""
import numpy as np
from opencood.utils import box_utils


def continuous_aabb_iou(boxes, queries):
    """Continuous metre-coordinate stand-up IoU; no image-pixel ``+1``.

    This preserves the baseline's stand-up assignment design. Evaluation uses
    the separate exact rotated geometry. Touching or disjoint BEV rectangles
    have zero intersection even if their gap is less than one metre.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or queries.ndim != 2 or queries.shape[1] != 4:
        raise ValueError('AABB inputs must have shape (N,4)')
    if not np.isfinite(boxes).all() or not np.isfinite(queries).all():
        raise ValueError('AABB coordinates must be finite')
    if np.any(boxes[:,2:] <= boxes[:,:2]) or np.any(queries[:,2:] <= queries[:,:2]):
        raise ValueError('AABB dimensions must be positive')
    intersection_size = np.maximum(0., np.minimum(boxes[:,None,2:],queries[None,:,2:])
                                   - np.maximum(boxes[:,None,:2],queries[None,:,:2]))
    intersection = intersection_size.prod(axis=2)
    area = (boxes[:,2:]-boxes[:,:2]).prod(axis=1)
    query_area = (queries[:,2:]-queries[:,:2]).prod(axis=1)
    return intersection/(area[:,None]+query_area[None,:]-intersection)

class AnchorTargets:
    def __init__(self, anchors, positive=0.6, negative=0.45):
        if np.asarray(anchors).ndim != 4 or np.asarray(anchors).shape[-1] != 7:
            raise ValueError('Anchors must have shape (H,W,A,7) in xyzhwl-yaw order')
        self.shape=anchors.shape[:3]
        self.anchors=np.asarray(anchors,dtype=np.float32).reshape(-1,7)
        self.positive=float(positive);self.negative=float(negative)
        if not len(self.anchors) or not np.isfinite(self.anchors).all() or np.any(self.anchors[:,3:6]<=0):
            raise ValueError('Anchors must be finite, nonempty and have positive dimensions')
        if not 0 <= self.negative < self.positive <= 1:
            raise ValueError('Require 0 <= negative IoU < positive IoU <= 1')
        corners=box_utils.boxes_to_corners_3d(self.anchors,order='hwl')
        self.standup=np.ascontiguousarray(box_utils.corner2d_to_standup_box(corners),dtype=np.float32)
        self.last_diagnostics={}
    def __call__(self, boxes_lwh):
        gt=np.asarray(boxes_lwh,dtype=np.float32).reshape(-1,7)[:,[0,1,2,5,4,3,6]]
        if not np.isfinite(gt).all() or np.any(gt[:,3:6]<=0):
            raise ValueError('GT boxes must be finite with positive dimensions')
        n=len(self.anchors)
        pos=np.zeros(n,np.float32);neg=np.ones(n,np.float32);target=np.zeros((n,7),np.float32)
        forced_count=collision_count=0
        zero_overlap=[];unrepresented=[]
        if len(gt):
            corners=box_utils.boxes_to_corners_3d(gt,order='hwl')
            standup=np.ascontiguousarray(box_utils.corner2d_to_standup_box(corners),dtype=np.float32)
            iou=continuous_aabb_iou(self.standup,standup)
            assignment=iou.argmax(axis=1)
            maximum=iou[np.arange(n),assignment]
            positive=maximum >= self.positive
            neg=(maximum < self.negative).astype(np.float32)
            # A distinct best anchor MUST regress to the GT that forced it,
            # even when that anchor has a larger overlap with another GT whose
            # own best anchor is elsewhere. Resolve genuine same-anchor
            # collisions by IoU, then stable GT input order. Do not invent a
            # distant replacement assignment to conceal uncovered targets.
            best_anchor=iou.argmax(axis=0)
            forced={}
            for j,a in enumerate(best_anchor):
                if iou[a,j]>0:
                    if a in forced:
                        collision_count+=1
                        if iou[a,j]>iou[a,forced[a]]:forced[a]=j
                    else:forced[a]=j
                else:zero_overlap.append(j)
            for a,j in forced.items():
                positive[a]=True
                assignment[a]=j
            forced_count=len(forced)
            pos[positive]=1;neg[positive]=0
            ids=np.flatnonzero(positive);a=self.anchors[ids];g=gt[assignment[ids]]
            unrepresented=sorted(set(range(len(gt)))-set(assignment[ids].tolist()))
            diagonal=np.sqrt(a[:,4]**2+a[:,5]**2)
            target[ids,0:2]=(g[:,0:2]-a[:,0:2])/diagonal[:,None]
            target[ids,2]=(g[:,2]-a[:,2])/a[:,3]
            target[ids,3:6]=np.log(g[:,3:6]/a[:,3:6])
            target[ids,6]=g[:,6]-a[:,6]
        self.last_diagnostics={
            'assignment_version':'continuous_aabb_forced_best_v1',
            'num_gt':len(gt),'num_positive':int(pos.sum()),'num_negative':int(neg.sum()),
            'num_ignored':int(((pos==0)&(neg==0)).sum()),'forced_anchor_count':forced_count,
            'forced_collision_count':collision_count,'zero_overlap_gt_indices':zero_overlap,
            'unrepresented_gt_indices':unrepresented,
        }
        h,w,count=self.shape
        return {'pos_equal_one':pos.reshape(h,w,count),'neg_equal_one':neg.reshape(h,w,count),
                'targets':target.reshape(h,w,count*7)}
