"""SBR v2: bounded, task-trained correction from current real Radar support."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .evidence_router import observation_rms, split_pointwise


class SupportedResidual(nn.Module):
    def __init__(self, channels=64, hidden_channels=16, attention_channels=8,
                 rho=0.25, g_min=0.25, hard_support=True, bounded=True,
                 ordinary=False, voxel_size=(.4, .4), support_radius_m=.8,
                 scale_floor=1e-3, scale_ceiling=1e3,
                 scale_mode='current_observation_rms',
                 type_prior_mode='multiplicative', efficient_features=False):
        super().__init__()
        if not 0 <= rho or not 0 <= g_min <= 1:
            raise ValueError('rho must be nonnegative; g_min must be in [0,1]')
        if scale_mode != 'current_observation_rms':
            raise ValueError('SBR v2 requires current_observation_rms; old calibration belongs to v1')
        if not 0 < scale_floor <= scale_ceiling or not math.isfinite(scale_ceiling):
            raise ValueError('scale bounds must be finite positive ordered values')
        self.channels, self.rho, self.g_min = channels, float(rho), float(g_min)
        self.scale_floor, self.scale_ceiling = float(scale_floor), float(scale_ceiling)
        self.scale_mode = scale_mode
        if type_prior_mode not in ('multiplicative', 'input_only'):
            raise ValueError('type_prior_mode must be multiplicative/input_only')
        self.type_prior_mode = type_prior_mode
        self.efficient_features = bool(efficient_features)
        self.collect_diagnostics = False
        self.hard_support, self.bounded, self.ordinary = bool(hard_support), bool(bounded), bool(ordinary)
        self.support_radius_m = float(support_radius_m)
        vx, vy = map(float, voxel_size[:2])
        if min(vx, vy) <= 0 or not math.isfinite(self.support_radius_m) or self.support_radius_m < 0:
            raise ValueError('support radius and voxel sizes must be finite and physically valid')
        self.padding_x = math.floor(self.support_radius_m / vx + 1e-7)
        self.padding_y = math.floor(self.support_radius_m / vy + 1e-7)
        self.offsets = tuple((dy, dx) for dy in range(-self.padding_y, self.padding_y + 1)
                             for dx in range(-self.padding_x, self.padding_x + 1)
                             if (dy * vy) ** 2 + (dx * vx) ** 2 <= self.support_radius_m ** 2 + 1e-7)
        if len(self.offsets) > 49:
            raise ValueError('support neighbourhood exceeds the 49-candidate compute budget')
        if ordinary:
            self.query = self.key = self.null_logit = None
        else:
            self.query = nn.Conv2d(channels, attention_channels, 1, bias=False)
            self.key = nn.Conv2d(channels, attention_channels, 1, bias=False)
            self.null_logit = nn.Conv2d(channels, 1, 1)
            nn.init.zeros_(self.null_logit.weight)
            nn.init.zeros_(self.null_logit.bias)
        self.input_project = nn.Conv2d(2 * channels + 5, hidden_channels, 1)
        self.residual_net = nn.Sequential(
            nn.SiLU(), nn.Conv2d(hidden_channels, hidden_channels, 3,
                                padding=1, groups=hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 1))
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def _valid_neighbours(self, valid):
        height, width = valid.shape[-2:]
        padded = F.pad(valid, (self.padding_x, self.padding_x,
                               self.padding_y, self.padding_y), value=False)
        return [padded[:, :, dy + self.padding_y:dy + self.padding_y + height,
                       dx + self.padding_x:dx + self.padding_x + width]
                for dy, dx in self.offsets]

    def local_donor(self, z, radar, radar_occupancy, radar_valid, available,
                    feature_scale=None, radar_scale=None, return_normalized=False):
        """Finite metric neighbourhood; no C*K*H*W feature unfolding."""
        valid = (radar_occupancy > 0) & (radar_valid > 0) & available.bool()
        neighbours = self._valid_neighbours(valid)
        count = torch.stack(neighbours).sum(dim=0).float()
        support = (count > 0).to(z.dtype)
        if radar_scale is None:
            radar_scale = observation_rms(radar, valid, self.scale_floor, self.scale_ceiling)
        if feature_scale is None:
            feature_scale, overall = observation_rms(
                z, valid, self.scale_floor, self.scale_ceiling, return_overall=True)
            fallback = (overall <= self.scale_floor) & valid.flatten(2).any(dim=2, keepdim=True).unsqueeze(-1)
            feature_scale = torch.where(fallback, radar_scale, feature_scale)
        normalized_z = z / feature_scale.to(z.dtype)
        normalized_radar = radar / radar_scale.to(radar.dtype)
        height, width = z.shape[-2:]
        # Unit-ball projections bound both attention logits and the derivative
        # at zero-query cells, where sparse LiDAR has no activation.
        query = F.normalize(self.query(normalized_z).float(), dim=1, eps=1.)
        key = F.pad(F.normalize(self.key(normalized_radar).float(), dim=1, eps=1.),
                    (self.padding_x, self.padding_x, self.padding_y, self.padding_y))
        # Correct null prior for candidate count. With equal real scores its
        # initial mass is 1/2 regardless of measurement density, and 1 if empty.
        logits = [self.null_logit(normalized_z).float() + count.clamp_min(1).log()]
        for (dy, dx), candidate_valid in zip(self.offsets, neighbours):
            yy, xx = dy + self.padding_y, dx + self.padding_x
            score = 2 * (query * key[:, :, yy:yy + height, xx:xx + width]).sum(1, keepdim=True)
            logits.append(score.masked_fill(~candidate_valid, -torch.inf))
        weights = torch.cat(logits, dim=1).softmax(dim=1).to(z.dtype)
        radar_pad = F.pad(normalized_radar, (self.padding_x, self.padding_x,
                                           self.padding_y, self.padding_y))
        donor = torch.zeros_like(radar)
        for i, (dy, dx) in enumerate(self.offsets, 1):
            yy, xx = dy + self.padding_y, dx + self.padding_x
            donor = donor + weights[:, i:i + 1] * radar_pad[:, :, yy:yy + height, xx:xx + width]
        # This changes only B's donor units. The original Radar bypass is intact.
        return (donor if return_normalized else donor * feature_scale.to(z.dtype)), support, weights[:, :1]

    def forward(self, z, radar, observations, modality_valid, miss=None, spur=None):
        available = modality_valid[:, 1:2, None, None].to(z.device)
        occupied = (((observations['lidar_occupancy'] > 0) &
                     (observations['lidar_valid'] > 0) &
                     modality_valid[:, :1, None, None].bool()) |
                    ((observations['radar_occupancy'] > 0) &
                     (observations['radar_valid'] > 0) & available.bool()))
        scale, overall = observation_rms(z, occupied, self.scale_floor,
                                         self.scale_ceiling, return_overall=True)
        valid = (observations['radar_occupancy'] > 0) & (observations['radar_valid'] > 0) & available.bool()
        radar_scale = observation_rms(radar, valid, self.scale_floor, self.scale_ceiling)
        # Phi_base starts as [I,0]: a missing LiDAR frame makes all of Z zero.
        # Use a genuine Radar reference only for an entirely dead agent Z;
        # otherwise its own channel scale (including dead-channel guard) wins.
        fallback = (overall <= self.scale_floor) & valid.flatten(2).any(dim=2, keepdim=True).unsqueeze(-1)
        scale = torch.where(fallback, radar_scale, scale)
        if self.ordinary:
            # SUP shares the dynamic units and losses, with an ordinary current
            # Radar donor, no local attention, hard support or tanh bound.
            donor = radar / radar_scale.to(radar.dtype)
            if not self.efficient_features:
                donor = donor * scale.to(radar.dtype)
            support = torch.stack(self._valid_neighbours(valid)).any(dim=0).to(z.dtype)
            null_weight = torch.zeros_like(support)
            miss = spur = None
        else:
            donor, support, null_weight = self.local_donor(
                z, radar, observations['radar_occupancy'], observations['radar_valid'],
                available, scale, radar_scale, return_normalized=self.efficient_features)
        if (miss is None) != (spur is None):
            raise ValueError('provide both type scores or neither')
        if miss is None:
            miss = spur = torch.zeros_like(support)
            has_a, prior = torch.zeros_like(support), torch.ones_like(support)
        else:
            miss, spur = miss.detach(), spur.detach()
            has_a = torch.ones_like(support)
            prior = (self.g_min + (1 - self.g_min) * torch.maximum(miss, spur)
                     if self.type_prior_mode == 'multiplicative' else torch.ones_like(support))
        small = torch.cat((miss, spur, observations['lidar_occupancy'].to(z.dtype),
                           support, has_a), dim=1)
        raw = self.residual_net(split_pointwise(self.input_project,
                               (z / scale.to(z.dtype), donor if self.efficient_features else donor / scale.to(z.dtype), small)))
        raw_delta = torch.tanh(raw) if self.bounded else raw
        delta = prior * self.rho * scale.to(z.dtype) * raw_delta
        if self.hard_support:
            delta = support * delta
        diagnostics = {'delta': delta, 'support': support,
                       'null_weight': null_weight, 'feature_scale': scale,
                       'scale_fallback': fallback, 'type_prior': prior}
        if self.collect_diagnostics:
            # Opt-in detached summaries: no full donor/raw graph is returned,
            # and normal training pays none of these reduction costs.
            with torch.no_grad():
                diagnostics.update(
                    raw_abs_max=raw.detach().abs().amax(),
                    tanh_saturation_fraction=(raw.detach().abs() > 2).float().mean(),
                    donor_rms_normalized=(donor.detach() if self.efficient_features else donor.detach() / scale).square().mean(dim=1, keepdim=True).sqrt(),
                    donor_z_cosine=F.cosine_similarity(
                        donor.detach() if self.efficient_features else donor.detach() / scale,
                        z.detach() / scale, dim=1).unsqueeze(1))
        return z + delta, diagnostics
