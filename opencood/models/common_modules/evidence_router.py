"""DEER: supervised degradation evidence with detached control connections.

Only current measurements enter this module. Targets belong to ProjectLoss.
The low-dimensional encoder avoids concatenating two full-resolution BEVs.
"""
import torch
from torch import nn
from torch.nn import functional as F


OBSERVATION_KEYS = (
    'lidar_count', 'radar_count', 'lidar_occupancy', 'radar_occupancy',
    'lidar_valid', 'radar_valid', 'lidar_distance', 'radar_distance',
)


def validate_observations(lidar, radar, observations, modality_valid):
    if lidar.ndim != 4 or lidar.shape != radar.shape:
        raise ValueError('LiDAR and Radar must share [agents,C,H,W]')
    expected = (lidar.shape[0], 1, *lidar.shape[-2:])
    for key in OBSERVATION_KEYS:
        if key not in observations or tuple(observations[key].shape) != expected:
            raise ValueError(f'{key} must have Scatter-grid shape {expected}')
    if tuple(modality_valid.shape) != (lidar.shape[0], 2):
        raise ValueError('modality_valid must be [sum_agents,2]')


def split_pointwise(layer, features):
    """Equivalent to conv(cat(features)), without the large input allocation."""
    offset = 0
    result = None
    for feature in features:
        end = offset + feature.shape[1]
        value = F.conv2d(feature, layer.weight[:, offset:end],
                         layer.bias if offset == 0 else None)
        result = value if result is None else result + value
        offset = end
    if offset != layer.in_channels:
        raise ValueError('pointwise projection channel mismatch')
    return result


@torch.no_grad()
def observation_rms(features, occupied, floor=1e-3, ceiling=1e3,
                    return_overall=False):
    """Detached current-agent/channel RMS over genuine occupied cells only.

    Channel floors/ceilings relative to the agent's RMS prevent a nearly dead
    channel from amplifying numerical noise. Empty agents get a numerical
    floor, which never creates observation support. There is no running state,
    clean reference, calibration pass, or train/eval-dependent update.
    """
    if not 0 < floor <= ceiling:
        raise ValueError('RMS floor/ceiling must be finite positive ordered values')
    mask = occupied.to(device=features.device, dtype=torch.float32)
    energy = features.detach().float().square()
    energy.mul_(mask)
    energy = energy.sum(dim=(2, 3), keepdim=True) / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    raw_overall = energy.mean(dim=1, keepdim=True).sqrt()
    overall = raw_overall.clamp(floor, ceiling)
    channel = energy.sqrt()
    scale = torch.maximum(torch.minimum(channel, overall * 10), overall * .1).clamp(floor, ceiling)
    return (scale, raw_overall) if return_overall else scale


class EvidenceRouter(nn.Module):
    def __init__(self, channels=64, hidden_channels=16,
                 beta_spur=1.0, beta_miss=1.0, type_mode='independent',
                 type_control=True, route_strength=.5,
                 scale_floor=1e-3, scale_ceiling=1e3,
                 amplitude_context='none'):
        super().__init__()
        # The head variants consume different amounts of random initialization.
        # Capture their common seed before constructing any optional layers.
        amplitude_seed = (torch.initial_seed() + 7103) % (2 ** 63 - 1)
        self.channels = channels
        self.beta_spur, self.beta_miss = float(beta_spur), float(beta_miss)
        if not 0 <= route_strength < 1:
            raise ValueError('route_strength must be in [0,1) to retain each available branch')
        self.route_strength = float(route_strength)
        self.scale_floor, self.scale_ceiling = float(scale_floor), float(scale_ceiling)
        if amplitude_context not in ('none', 'rms', 'zero', 'normalized_mean'):
            raise ValueError('amplitude_context must be none/rms/zero/normalized_mean')
        self.amplitude_context = amplitude_context
        if type_mode not in ('independent', 'shared', 'exclusive'):
            raise ValueError('unknown type head mode')
        self.type_mode, self.type_control = type_mode, bool(type_control)
        # Eight current-observation maps and two frame availability flags.
        self.input_project = nn.Conv2d(2 * channels + 10, hidden_channels, 1)
        self.encoder = nn.Sequential(
            nn.SiLU(), nn.Conv2d(hidden_channels, hidden_channels, 3,
                                padding=1, groups=hidden_channels),
            nn.SiLU(), nn.Conv2d(hidden_channels, hidden_channels, 1), nn.SiLU())
        head_channels = {'independent': 2, 'shared': 1, 'exclusive': 3}[type_mode]
        self.type_head = nn.Conv2d(hidden_channels, head_channels, 1)
        self.route_head = nn.Conv2d(hidden_channels, 2, 1)
        nn.init.zeros_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)
        nn.init.zeros_(self.route_head.weight)
        with torch.no_grad():
            initial_score = (1 / 3 if type_mode == 'exclusive' else 1 / 2) if type_control else 0
            self.route_head.bias.copy_(torch.tensor([self.beta_spur * initial_score,
                                                    -self.beta_miss * initial_score]))
        # Keep the spatial control path normalized, while allowing it to read
        # current per-agent/channel amplitude. This tiny 1x1 descriptor never
        # expands to a second dense 2*C BEV. Construct AFTER legacy layers so
        # adding the descriptor preserves their seeded initialization.
        self.amplitude_project = None
        if amplitude_context != 'none':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(amplitude_seed)
                self.amplitude_project = nn.Conv2d(2 * channels, hidden_channels, 1, bias=False)

    def forward(self, lidar, radar, observations, modality_valid, inputs_masked=False):
        validate_observations(lidar, radar, observations, modality_valid)
        shape = (lidar.shape[0], 2, *lidar.shape[-2:])
        available = modality_valid.to(device=lidar.device).bool()[:, :, None, None]
        # Modality dropout invalidates its whole branch, not individual cells.
        if not inputs_masked:
            lidar = lidar * available[:, :1].to(lidar.dtype)
            radar = radar * available[:, 1:].to(radar.dtype)
        maps = [observations[k].to(lidar.dtype) for k in OBSERVATION_KEYS]
        maps[:2] = [torch.log1p(x.clamp_min(0)) for x in maps[:2]]
        obs = torch.cat(maps + [available.expand(shape).to(lidar.dtype)], dim=1)
        lidar_scale = observation_rms(lidar, (observations['lidar_occupancy'] > 0) &
                                      (observations['lidar_valid'] > 0) & available[:, :1],
                                      self.scale_floor, self.scale_ceiling)
        radar_scale = observation_rms(radar, (observations['radar_occupancy'] > 0) &
                                      (observations['radar_valid'] > 0) & available[:, 1:],
                                      self.scale_floor, self.scale_ceiling)
        projected = split_pointwise(self.input_project,
                               (lidar / lidar_scale.to(lidar.dtype),
                                radar / radar_scale.to(radar.dtype), obs))
        amplitude_descriptor = None
        if self.amplitude_project is not None:
            if self.amplitude_context == 'normalized_mean':
                # Active capacity control: the same parameters and broadcast
                # receive a sample-dependent, amplitude-invariant descriptor.
                # A batched dot product avoids materializing a dense masked BEV.
                with torch.no_grad():
                    descriptors = []
                    for feature, scale, occupied in (
                            (lidar, lidar_scale, (observations['lidar_occupancy'] > 0) &
                             (observations['lidar_valid'] > 0) & available[:, :1]),
                            (radar, radar_scale, (observations['radar_occupancy'] > 0) &
                             (observations['radar_valid'] > 0) & available[:, 1:])):
                        mean = torch.bmm(feature.detach().float().flatten(2),
                                         occupied.float().flatten(2).transpose(1, 2)).unsqueeze(-1)
                        mean = mean / occupied.sum(dim=(2, 3), keepdim=True).clamp_min(1)
                        descriptors.append(mean / scale)
                    amplitude_descriptor = torch.cat(descriptors, dim=1)
            else:
                amplitude_descriptor = torch.cat((lidar_scale.log1p(), radar_scale.log1p()), dim=1)
                if self.amplitude_context == 'zero':
                    amplitude_descriptor = torch.zeros_like(amplitude_descriptor)
            projected = projected + self.amplitude_project(amplitude_descriptor.to(lidar.dtype))
        evidence = self.encoder(projected)
        type_logits = self.type_head(evidence)
        if self.type_mode == 'exclusive':
            # Binary logits for p_miss and p_spur preserve two independent BCEs
            # with the ORIGINAL two labels, even when both targets equal one.
            type_logits = type_logits.float()
            miss_logits = type_logits[:, 1:2] - torch.logsumexp(type_logits[:, [0, 2]], dim=1, keepdim=True)
            spur_logits = type_logits[:, 2:3] - torch.logsumexp(type_logits[:, :2], dim=1, keepdim=True)
        elif self.type_mode == 'shared':
            miss_logits = spur_logits = type_logits
        else:
            miss_logits, spur_logits = type_logits.split(1, dim=1)
        miss, spur = miss_logits.sigmoid(), spur_logits.sigmoid()
        route_logits = self.route_head(evidence)
        # Do not detach evidence: task gradients must still train shared encoding.
        type_bias = (torch.cat((-self.beta_spur * spur.detach(),
                                self.beta_miss * miss.detach()), dim=1)
                     if self.type_control else torch.zeros_like(route_logits))
        logits = route_logits + type_bias
        # A finite sentinel plus explicit zeroing handles BOTH missing safely.
        weights = logits.float().masked_fill(~available, -1e4).softmax(dim=1)
        weights = weights * available
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights.to(lidar.dtype)
        # Residual routing retains a shared feature floor. A saturated softmax
        # can no longer erase a valid sensor branch; unavailable ones stay zero.
        gains = (1 + self.route_strength * (2 * weights - 1)) * available
        return gains[:, :1] * lidar, gains[:, 1:] * radar, {
            'miss': miss, 'spur': spur,
            'miss_logits': miss_logits, 'spur_logits': spur_logits,
            'route_weights': weights, 'route_gains': gains,
            'lidar_scale': lidar_scale, 'radar_scale': radar_scale,
            # Detached views support diagnostics without retaining another
            # control graph. Type biases are a decomposition, not a new gate.
            'generic_route_logits': route_logits.detach(),
            'type_route_bias': type_bias.detach(),
            'final_route_logits': logits.detach(),
            'amplitude_descriptor': amplitude_descriptor,
        }
