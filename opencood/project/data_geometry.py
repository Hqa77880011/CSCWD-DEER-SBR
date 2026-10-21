"""Deterministic scene augmentation and real-observation BEV maps."""
import numpy as np
from opencood.project.geometry import transform_points


def point_fov_mask(points, camera):
    xyz = transform_points(np.asarray(points)[:,:3], camera['lidar_to_camera'])
    uvw = xyz @ np.asarray(camera['intrinsic'], dtype=np.float64).T
    depth = xyz[:,2]
    # Division-free half-space tests avoid zero-depth NaNs.
    return ((depth > 0) & (uvw[:,0] >= 0) & (uvw[:,0] < camera['image_width']*depth)
            & (uvw[:,1] >= 0) & (uvw[:,1] < camera['image_height']*depth))


def augmentation_matrix(rng, enabled=True):
    if not enabled:
        return np.eye(4,dtype=np.float64)
    flip = -1.0 if rng.random() < 0.5 else 1.0
    angle = rng.uniform(-np.pi/4,np.pi/4)
    scale = rng.uniform(0.95,1.05)
    c,s = np.cos(angle),np.sin(angle)
    out = np.eye(4)
    out[:3,:3] = scale * np.array([[c,-s*flip,0],[s,c*flip,0],[0,0,1.]])
    return out


def augment_boxes(boxes, matrix):
    boxes = np.asarray(boxes,dtype=np.float64).reshape(-1,7).copy()
    if not len(boxes):return boxes
    boxes[:,:3] = transform_points(boxes[:,:3],matrix)
    scale = np.linalg.norm(matrix[:3,0])
    boxes[:,3:6] *= scale
    headings = np.column_stack((np.cos(boxes[:,6]),np.sin(boxes[:,6]),np.zeros(len(boxes))))
    headings = headings @ matrix[:3,:3].T
    boxes[:,6] = np.arctan2(headings[:,1],headings[:,0])
    return boxes


def prepare_points(points, agent_to_ego, camera, augmentation, bounds, remove_self=True):
    points = np.asarray(points,dtype=np.float32)
    if remove_self:
        own = ((points[:,0] >= -1.95) & (points[:,0] <= 2.95)
               & (points[:,1] >= -1.1) & (points[:,1] <= 1.1))
        points = points[~own]
    xyz = transform_points(points[:,:3],agent_to_ego)
    keep = point_fov_mask(xyz,camera)
    out = np.empty((int(keep.sum()),4),dtype=np.float32)
    out[:,:3] = transform_points(xyz[keep],augmentation)
    out[:,3] = points[keep,3]
    bounds = np.asarray(bounds)
    keep = (out[:,:3] >= bounds[:3]).all(1) & (out[:,:3] < bounds[3:]).all(1)
    return np.ascontiguousarray(out[keep])


def bev_grid(bounds, voxel):
    bounds,voxel = np.asarray(bounds),np.asarray(voxel)
    width,height = np.rint((bounds[3:5]-bounds[:2])/voxel[:2]).astype(int)
    x = bounds[0]+(np.arange(width)+0.5)*voxel[0]
    y = bounds[1]+(np.arange(height)+0.5)*voxel[1]
    return np.meshgrid(x,y)


def count_map(points, bounds, voxel, shape):
    if not len(points):return np.zeros((1,*shape),np.float32)
    xy = np.floor((points[:,:2]-np.asarray(bounds[:2]))/np.asarray(voxel[:2])).astype(np.int64)
    valid = (xy[:,0]>=0)&(xy[:,0]<shape[1])&(xy[:,1]>=0)&(xy[:,1]<shape[0])
    ids = xy[valid,1]*shape[1]+xy[valid,0]
    return np.bincount(ids,minlength=shape[0]*shape[1]).reshape(1,*shape).astype(np.float32)


def bev_fov_mask(camera, augmentation, bounds, grid):
    """Exact vertical-line/frustum intersection at each BEV cell centre.

    This is a BEV geometric availability map, not a claim that every point in
    that cell is visible. Actual points are filtered individually beforehand.
    """
    xx,yy = grid
    camera_t = np.asarray(camera['lidar_to_camera']) @ np.linalg.inv(augmentation)
    proj = np.asarray(camera['intrinsic']) @ camera_t[:3]
    depth = camera_t[2]
    # Each inequality a*x+b*y+c*z+d >= 0. Pixel upper bounds are half-open.
    rows = [(depth,True),(proj[0],False),(camera['image_width']*depth-proj[0],True),
            (proj[1],False),(camera['image_height']*depth-proj[1],True)]
    lower = np.full(xx.shape,bounds[2],dtype=np.float64)
    upper = np.full(xx.shape,bounds[5],dtype=np.float64)
    possible = np.ones(xx.shape,dtype=bool)
    for row,strict in rows:
        base = row[0]*xx+row[1]*yy+row[3]
        if abs(row[2]) < 1e-12:
            possible &= (base > 0) if strict else (base >= 0)
        elif row[2]>0:
            lower = np.maximum(lower,-base/row[2])
        else:
            upper = np.minimum(upper,-base/row[2])
    return (possible & (lower < upper))[None].astype(np.float32)


def observation_maps(lidar, radar, valid, origins, camera, augmentation, bounds, voxel, grid):
    shape=grid[0].shape
    lc=count_map(lidar,bounds,voxel,shape);rc=count_map(radar,bounds,voxel,shape)
    geometric=bev_fov_mask(camera,augmentation,bounds,grid)
    # A real accepted measurement is always a valid donor, even when its cell
    # centre lies just outside the frustum boundary.
    lv=np.maximum(geometric,lc>0)*float(valid[0]);rv=np.maximum(geometric,rc>0)*float(valid[1])
    scale=max(float(np.linalg.norm(np.asarray(bounds[3:5])-np.asarray(bounds[:2]))),1.)
    distance=[]
    for origin in origins:
        distance.append((np.sqrt((grid[0]-origin[0])**2+(grid[1]-origin[1])**2)/scale)[None].astype(np.float32))
    return {'lidar_count':lc,'radar_count':rc,'lidar_occupancy':(lc>0).astype(np.float32),
            'radar_occupancy':(rc>0).astype(np.float32),'lidar_valid':lv,'radar_valid':rv,
            'lidar_distance':distance[0],'radar_distance':distance[1]}
