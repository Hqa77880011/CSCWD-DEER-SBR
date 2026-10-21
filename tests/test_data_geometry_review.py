"""Independent geometry review: reflection equivariance and frustum edges."""
import unittest
import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np

from opencood.project.data_geometry import (augment_boxes,augmentation_matrix,
    bev_fov_mask,observation_maps,point_fov_mask,prepare_points)
from opencood.project.geometry import (OPTICAL_FROM_CARLA,boxes_to_corners,
                                      points_in_fov,transform_points)
from opencood.project.data_io import ByteLRU,read_annotation,read_pcd


class DataGeometryReviewTests(unittest.TestCase):
    def test_reflection_rotation_scale_transform_all_corners_consistently(self):
        rng=np.random.default_rng(43)
        boxes=np.column_stack((rng.normal(size=(20,3))*10,rng.uniform(.5,5,(20,3)),rng.uniform(-np.pi,np.pi,20)))
        original=boxes_to_corners(boxes)
        determinants=[]
        for _ in range(16):
            matrix=augmentation_matrix(rng)
            determinants.append(np.linalg.det(matrix[:3,:3]))
            transformed=transform_points(original.reshape(-1,3),matrix).reshape(-1,8,3)
            augmented=boxes_to_corners(augment_boxes(boxes,matrix))
            # Reflection changes corner winding, so compare the geometric sets.
            squared=((transformed[:,:,None,:]-augmented[:,None,:,:])**2).sum(-1)
            self.assertLess(float(squared.min(2).max()),1e-20)
        self.assertTrue(any(d<0 for d in determinants))
        self.assertTrue(any(d>0 for d in determinants))

    def test_point_frustum_mask_matches_fixed_gt_camera_geometry(self):
        camera={'intrinsic':[[335.64,0,400],[0,335.64,300],[0,0,1]],
                'lidar_to_camera':OPTICAL_FROM_CARLA.tolist(),'image_width':800,'image_height':600}
        rng=np.random.default_rng(29)
        points=rng.uniform([-10,-100,-20],[150,100,20],(30000,3))
        np.testing.assert_array_equal(point_fov_mask(points,camera),points_in_fov(points,camera))

    def test_bev_lower_pixel_boundary_inclusive_upper_boundary_exclusive(self):
        # Optical depth=x, u=y/x, v=-z/x. y=0 is exactly the legal u=0
        # boundary along the complete vertical line, while y=2x is excluded.
        camera={'intrinsic':np.eye(3).tolist(),'lidar_to_camera':OPTICAL_FROM_CARLA.tolist(),
                'image_width':2,'image_height':2}
        grid=(np.array([[1.,1.,1.,1.]]),np.array([[0.,1.999,2.,-.001]]))
        bounds=[-10,-10,-1,10,10,0]
        np.testing.assert_array_equal(bev_fov_mask(camera,np.eye(4),bounds,grid),[[[1,1,0,0]]])
        # A camera looking vertically makes both lower u/v edges constant in z.
        vertical={'intrinsic':np.eye(3).tolist(),'lidar_to_camera':np.eye(4).tolist(),
                  'image_width':2,'image_height':2}
        grid=(np.array([[0.,.5,0.]]),np.array([[0.,0.,-.01]]))
        np.testing.assert_array_equal(bev_fov_mask(vertical,np.eye(4),[-1,-1,1,1,1,2],grid),[[[1,1,0]]])

    def test_points_filtered_in_original_camera_then_augmented_once(self):
        camera={'intrinsic':[[100,0,100],[0,100,100],[0,0,1]],
                'lidar_to_camera':OPTICAL_FROM_CARLA.tolist(),'image_width':200,'image_height':200}
        points=np.array([[10,2,0,.7],[-10,0,0,.8],[0,0,0,.9]],np.float32)
        transform=np.eye(4);transform[0,3]=3
        augmentation=np.diag([1.,-1.,1.,1.])
        result=prepare_points(points,transform,camera,augmentation,[-100,-100,-10,100,100,10])
        np.testing.assert_allclose(result,[[13,-2,0,.7]],atol=1e-7)

    def test_real_boundary_point_remains_valid_observation_if_cell_center_outside(self):
        camera={'intrinsic':np.eye(3).tolist(),'lidar_to_camera':OPTICAL_FROM_CARLA.tolist(),
                'image_width':1,'image_height':2}
        bounds=[0,0,-1,2,2,1];voxel=[1,1,2]
        grid=np.meshgrid([.5,1.5],[.5,1.5])
        # x=.99,y=.5 has u<1, but its cell centre x=.5,y=.5 has u=1.
        point=np.array([[.99,.5,0,.3]],np.float32)
        maps=observation_maps(point,point,[True,True],[[0,0,0],[0,0,0]],camera,np.eye(4),bounds,voxel,grid)
        self.assertEqual(maps['radar_count'][0,0,0],1)
        self.assertEqual(maps['radar_valid'][0,0,0],1)

    def test_pcd_byte_tampering_and_changed_hash_cannot_bypass_cache(self):
        header=(b'# synthetic test PCD\nVERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n'
                b'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n')
        payload=np.array([[1.,2.,3.,0.]],dtype='<f4')
        payload[:,3]=np.array([128 << 16],dtype='<u4').view('<f4')
        raw=header+payload.tobytes()
        digest=hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.pcd';path.write_bytes(raw)
            points=read_pcd(path,digest)
            np.testing.assert_allclose(points,[[1,2,3,128/255]],atol=1e-7)
            cache=ByteLRU(1024);cache.expected_sha256[str(path)]=digest
            cache.get(path)
            cache.expected_sha256[str(path)]='0'*64
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):cache.get(path)
            cache.expected_sha256[str(path)]=digest
            old_stat=path.stat();payload[0,0]=2.
            path.write_bytes(header+payload.tobytes())
            os.utime(path,ns=(old_stat.st_atime_ns,old_stat.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):cache.get(path)

    def test_annotation_hash_checked_before_parsing(self):
        raw=b'lidar_pose: [0, 0, 0, 0, 0, 0]\nvehicles: {}\n'
        digest=hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.yaml';path.write_bytes(raw)
            self.assertEqual(read_annotation(path,digest)['lidar_pose'],[0]*6)
            path.write_bytes(raw.replace(b'[0, 0',b'[1, 0'))
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):read_annotation(path,digest)


if __name__=='__main__':unittest.main()
