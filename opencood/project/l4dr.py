"""Four-channel L4DR adaptation retaining FAD, bidirectional MME and MSGF.

Source: ylwhxht/L4DR at fcf2d7f0b7c632f9c23f75f720fe691be519d732
(Apache-2.0). Equations and layer sizes follow cfg_PP_L4DR.yml,
PointNet2MSG, PointHeadPreMask and BiDF_PillarVFE. The shared cooperative
detector/GT protocol is an explicit adaptation. No pretrained FAD is required.
"""
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.common_modules.pillar_vfe import PFNLayer


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, value):
        if self.training and value.numel() // value.shape[1] == 1:
            return F.batch_norm(value, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0., self.eps)
        return super().forward(value)


def _mlp(widths):
    layers = []
    for left, right in zip(widths, widths[1:]):
        layers.extend([nn.Conv1d(left, right, 1, bias=False),
                       SafeBatchNorm1d(right), nn.ReLU()])
    return nn.Sequential(*layers)


def _farthest(xyz, count):
    if xyz.is_cuda:
        from opencood.pcdet_utils.pointnet2.pointnet2_batch.pointnet2_utils import furthest_point_sample
        return furthest_point_sample(xyz[None].contiguous(), count)[0].long()
    selected = torch.empty(count, dtype=torch.long, device=xyz.device)
    distances = xyz.new_full((len(xyz),), float('inf'))
    farthest = 0
    for index in range(count):
        selected[index] = farthest
        distances = torch.minimum(distances, (xyz - xyz[farthest]).square().sum(-1))
        farthest = distances.argmax()
    return selected


def _ball(xyz, centers, radius, count):
    if xyz.is_cuda:
        from opencood.pcdet_utils.pointnet2.pointnet2_batch.pointnet2_utils import ball_query
        return ball_query(radius, count, xyz[None].contiguous(), centers[None].contiguous())[0].long()
    # Official ball query keeps the first in-radius source indices and repeats
    # the first valid neighbor to fill a short group. Centers are source points.
    distances = (centers[:, None] - xyz[None]).square().sum(-1)
    indices = torch.arange(len(xyz), device=xyz.device).expand(len(centers), -1)
    indices = indices.masked_fill(distances >= radius ** 2, len(xyz)).sort(-1).values
    indices = indices[:, :min(count, len(xyz))]
    indices = torch.where(indices == len(xyz), indices[:, :1], indices)
    if indices.shape[1] < count:
        indices = torch.cat((indices, indices[:, :1].expand(-1, count - indices.shape[1])), 1)
    return indices


class ForegroundDenoising(nn.Module):
    """PointNet++ MSG/feature propagation and the native foreground head."""
    NPOINTS = (4096, 1024, 256, 64)
    RADII = ((.1, .5), (.5, 1.), (1., 2.), (2., 4.))
    WIDTHS = (((16, 16, 32), (32, 32, 64)),
              ((64, 64, 128), (64, 96, 128)),
              ((128, 196, 256), (128, 196, 256)),
              ((256, 256, 512), (256, 384, 512)))

    def __init__(self):
        super().__init__()
        self.sa = nn.ModuleList()
        widths = [1]
        for scales in self.WIDTHS:
            self.sa.append(nn.ModuleList(_mlp([widths[-1] + 3, *s]) for s in scales))
            widths.append(sum(s[-1] for s in scales))
        fp_widths = ((128, 128), (256, 256), (512, 512), (512, 512))
        self.fp = nn.ModuleList(_mlp([(fp_widths[i + 1][-1] if i < 3 else widths[4]) + widths[i], *fp_widths[i]])
                                for i in range(4))
        self.head = nn.Sequential(nn.Linear(128, 256, bias=False), SafeBatchNorm1d(256), nn.ReLU(),
                                  nn.Linear(256, 256, bias=False), SafeBatchNorm1d(256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, points):
        if not len(points):
            return points.new_empty((0, 1))
        positions, features = [points[:, :3]], [points[:, 3:4]]
        for level, modules in enumerate(self.sa):
            xyz, feature = positions[-1], features[-1]
            centers = xyz[_farthest(xyz, min(len(xyz), self.NPOINTS[level]))]
            scales = []
            # Chunk groups to bound temporary neighbor tensors on real clouds.
            for radius, count, module in zip(self.RADII[level], (16, 32), modules):
                chunks = []
                for start in range(0, len(centers), 128):
                    current = centers[start:start + 128]
                    indices = _ball(xyz, current, radius, count)
                    grouped = torch.cat((xyz[indices] - current[:, None], feature[indices]), -1)
                    chunks.append(grouped)
                grouped = torch.cat(chunks)
                shape = grouped.shape
                # Flatten only spatial axes: one shared BN over all centers
                # and neighbors, exactly as PointNet++'s Conv2d MLP.
                encoded = module(grouped.reshape(1, -1, shape[-1]).transpose(1, 2))
                scales.append(encoded.reshape(-1, len(centers), count).amax(-1).T)
            positions.append(centers)
            features.append(torch.cat(scales, -1))
        for level in range(3, -1, -1):
            source, target = positions[level + 1], positions[level]
            pieces = []
            for start in range(0, len(target), 128):
                query = target[start:start + 128]
                if query.is_cuda and len(source) >= 3:
                    from opencood.pcdet_utils.pointnet2.pointnet2_batch.pointnet2_utils import three_nn
                    distance, index = three_nn(query[None].contiguous(), source[None].contiguous())
                    distance, index = distance[0], index[0].long()
                else:
                    distances = torch.cdist(query[None], source[None], compute_mode='donot_use_mm_for_euclid_dist')[0]
                    distance, index = distances.topk(min(3, len(source)), largest=False, sorted=True)
                weight = (distance + 1e-8).reciprocal()
                weight = weight / weight.sum(-1, keepdim=True)
                pieces.append((features[level + 1][index] * weight[..., None]).sum(1))
            combined = torch.cat((torch.cat(pieces), features[level]), -1)
            features[level] = self.fp[level](combined.T[None])[0].T
        return self.head(features[0])


def voxelize_radar(points, scores, agent, hypes, training):
    """Hard voxelize after FAD; caps follow the registered data preprocessing."""
    args = hypes['args']
    # Native MMESparseProcessor detaches before CPU hard voxelization. FAD
    # learns from its foreground loss, not detection gradients through scores.
    features = torch.cat((points, scores), -1).detach()
    coords = ((points[:, :3] - points.new_tensor(hypes['cav_lidar_range'][:3])) /
              points.new_tensor(args['voxel_size'])).floor().long()[:, [2, 1, 0]]
    unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
    first = torch.full((len(unique),), len(points), device=points.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(points), device=points.device), reduce='amin')
    order = first.argsort()[:args['max_voxel_train' if training else 'max_voxel_test']]
    remap = torch.full((len(unique),), -1, device=points.device, dtype=torch.long)
    remap[order] = torch.arange(len(order), device=points.device)
    groups = remap[inverse]
    source = torch.where(groups >= 0)[0]
    source = source[groups[source].argsort(stable=True)]
    counts = torch.bincount(groups[source], minlength=len(order))
    starts = counts.cumsum(0) - counts
    ranks = torch.arange(len(source), device=points.device) - starts[groups[source]]
    valid = ranks < args['max_points_per_voxel']
    voxels = points.new_zeros((len(order), args['max_points_per_voxel'], 5))
    voxels[groups[source[valid]], ranks[valid]] = features[source[valid]]
    coords = torch.cat((torch.full((len(order), 1), agent, device=points.device, dtype=torch.long), unique[order]), 1)
    return dict(voxel_features=voxels, voxel_coords=coords,
                voxel_num_points=counts.clamp_max(args['max_points_per_voxel']))


class BidirectionalMME(nn.Module):
    """Native 15-dimensional four-channel + foreground-score BiDF encoding."""
    def __init__(self, args):
        super().__init__()
        channels = args['pillar_vfe']['num_filters'][-1]
        self.lidar = PFNLayer(15, channels, use_norm=True, last_layer=True)
        self.radar = PFNLayer(15, channels, use_norm=True, last_layer=True)
        self.register_buffer('voxel', torch.tensor(args['voxel_size']), persistent=False)
        self.register_buffer('origin', torch.tensor(args['lidar_range'][:3]), persistent=False)

    def forward(self, lidar, radar):
        means, centers, masks = [], [], []
        for data in (lidar, radar):
            points, n, coords = (data[k] for k in ('voxel_features', 'voxel_num_points', 'voxel_coords'))
            mask = torch.arange(points.shape[1], device=points.device)[None] < n[:, None]
            mean = (points * mask[..., None]).sum(1) / n.clamp_min(1)[:, None]
            means.append(mean)
            centers.append(coords[:, [3, 2, 1]] * self.voxel + self.origin + self.voxel / 2)
            masks.append(mask)
        coords = torch.cat((lidar['voxel_coords'], radar['voxel_coords']))
        unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
        left, right = inverse[:len(means[0])], inverse[len(means[0]):]
        lmap = torch.full((len(unique),), -1, device=coords.device, dtype=torch.long)
        rmap = lmap.clone()
        lmap[left], rmap[right] = torch.arange(len(left), device=coords.device), torch.arange(len(right), device=coords.device)
        encoded = []
        for index, data in enumerate((lidar, radar)):
            points = data['voxel_features']
            other_index = rmap[left] if index == 0 else lmap[right]
            other = points.new_zeros((len(points), means[1 - index].shape[-1]))
            present = other_index >= 0
            other[present] = means[1 - index][other_index[present]]
            cluster = points[..., :3] - means[index][:, None, :3]
            center = points[..., :3] - centers[index][:, None]
            cross = (points[..., :3] - other[:, None, :3]) * present[:, None, None]
            if index == 0:
                features = torch.cat((points, cluster, center, cross,
                                      other[:, None, 3:].expand(-1, points.shape[1], -1)), -1)
            else:
                features = torch.cat((points[..., :3], other[:, None, 3:4].expand(-1, points.shape[1], -1),
                                      cross, center, cluster, points[..., 3:]), -1)
            features = features * masks[index][..., None]
            layer = self.lidar if index == 0 else self.radar
            value = layer(features).squeeze(1) if len(points) else points.new_zeros((0, layer.linear.out_features))
            encoded.append({**data, 'pillar_features': value})
        return encoded


class L4DRFrontEnd(nn.Module):
    def __init__(self, args, preprocessing):
        super().__init__()
        self.preprocessing = preprocessing
        self.fad = ForegroundDenoising()
        self.mme = BidirectionalMME(args)
        self.threshold = .1

    def forward(self, inputs):
        if 'raw_radar' not in inputs:
            raise ValueError('complete L4DR requires actual raw Radar points for FAD before voxelization')
        logits, coordinates, entries = [], [], []
        for agent, points in enumerate(inputs['raw_radar']):
            prediction = self.fad(points)
            scores = prediction.sigmoid()
            mask = scores[:, 0] > self.threshold
            logits.append(prediction)
            coordinates.append(torch.cat((points.new_full((len(points), 1), agent), points[:, :3]), -1))
            entries.append(voxelize_radar(points[mask], scores[mask], agent, self.preprocessing, self.training))
        radar = {k: torch.cat([entry[k] for entry in entries]) for k in entries[0]}
        lidar, radar = self.mme(inputs['processed_lidar'], radar)
        return lidar, radar, dict(fad_logits=torch.cat(logits), fad_coordinates=torch.cat(coordinates))


def foreground_labels(coordinates, boxes):
    """GT-only train target: inside=1; the native 0.2m enlarged shell is ignored."""
    result = torch.zeros(len(coordinates), device=coordinates.device, dtype=torch.long)
    for agent, targets in enumerate(boxes):
        indices = torch.where(coordinates[:, 0] == agent)[0]
        if not len(indices) or not len(targets):
            continue
        points = coordinates[indices, 1:4]
        offset = points[:, None] - targets[None, :, :3]
        c, s = targets[:, 6].cos(), targets[:, 6].sin()
        local = torch.stack((offset[..., 0] * c + offset[..., 1] * s,
                             -offset[..., 0] * s + offset[..., 1] * c, offset[..., 2]), -1).abs()
        inside = (local <= targets[None, :, 3:6] / 2).all(-1).any(-1)
        expanded = (local <= targets[None, :, 3:6] / 2 + .2).all(-1).any(-1)
        result[indices] = torch.where(inside, 1, torch.where(expanded, -1, 0))
    return result
