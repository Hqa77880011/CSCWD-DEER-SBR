"""Shared PointPillar/AttFuse baseline with optional DEER and SBR.

Dataset projection happens exactly once (proj_first=True). This model never
warps features, reads weather labels, or calls a clean teacher during forward.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.common_modules.evidence_router import (
    EvidenceRouter, validate_observations, split_pointwise)
from opencood.models.common_modules.supported_residual import SupportedResidual


MODEL_INPUT_KEYS = frozenset({
    'processed_lidar', 'processed_radar', 'record_len', 'pairwise_t_matrix',
    'observation_maps', 'modality_valid', 'batch_size',
})


def select_model_inputs(ego):
    """Explicit inference allowlist; labels/teacher inputs stay outside the model."""
    return {key: value for key, value in ego.items() if key in MODEL_INPUT_KEYS}


class PointPillarDeerSbr(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = args.get('innovation', {})
        if 'require_calibrated_scale' in config:
            raise ValueError('require_calibrated_scale is a v1 option; v2 uses current-observation RMS without calibration')
        self.use_a, self.use_b = bool(config.get('use_a', False)), bool(config.get('use_b', False))
        self.activation_checkpoint = bool(config.get('activation_checkpoint', True))
        self.efficient_features = bool(config.get('efficient_features', False))
        variant = config.get('variant')
        variants = {'S0': (False, False), 'A': (True, False), 'B': (False, True),
                    'AB': (True, True), 'SUP': (True, True),
                    **{f'M{i}': (True, True) for i in range(1, 7)},
                    'N1': (True, True), 'N2': (True, True)}
        if variant is not None:
            if variant not in variants:
                raise ValueError(f'unknown innovation variant {variant!r}; expected {list(variants)}')
            self.use_a, self.use_b = variants[variant]
        self.variant = variant or ('AB' if self.use_a and self.use_b else 'A' if self.use_a else 'B' if self.use_b else 'S0')
        self.channels = int(args['pillar_vfe']['num_filters'][-1])
        if self.channels != int(args['point_pillar_scatter']['num_features']):
            raise ValueError('PillarVFE and Scatter channels disagree')
        if args.get('proj_first', True) is not True:
            raise ValueError('DEER/SBR requires loader proj_first=True')
        # All shared layers are built BEFORE optional layers under an isolated
        # RNG. Adding A/B cannot change any shared parameter initialization.
        self.shared_init_seed = int(args.get('shared_init_seed', 17))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.shared_init_seed)
            vfe_args = dict(num_point_features=4, voxel_size=args['voxel_size'],
                            point_cloud_range=args['lidar_range'])
            self.lidar_pillar_vfe = PillarVFE(args['pillar_vfe'], **vfe_args)
            self.radar_pillar_vfe = PillarVFE(args['pillar_vfe'], **vfe_args)
            self.scatter = PointPillarScatter(args['point_pillar_scatter'])
            self.phi_base = nn.Conv2d(2 * self.channels, self.channels, 1, bias=False)
            self.phi_out = nn.Conv2d(2 * self.channels, 2 * self.channels, 1, bias=False)
            self.backbone = AttBEVBackbone(args['base_bev_backbone'], 2 * self.channels)
            anchors = int(args.get('anchor_number', args.get('anchor_num', 2)))
            if 'anchor_num' in args and anchors != int(args['anchor_num']):
                raise ValueError('anchor_number and anchor_num disagree')
            self.cls_head = nn.Conv2d(self.backbone.num_bev_features, anchors, 1)
            self.reg_head = nn.Conv2d(self.backbone.num_bev_features, 7 * anchors, 1)
            with torch.no_grad():
                self.phi_base.weight.zero_()
                self.phi_base.weight[:, :self.channels, 0, 0].copy_(torch.eye(self.channels))
                self.phi_out.weight.copy_(torch.eye(2 * self.channels)[:, :, None, None])
            hidden = int(config.get('hidden_channels', 16))
            torch.random.default_generator.manual_seed(self.shared_init_seed + 1001)
            self.evidence_router = EvidenceRouter(
                self.channels, hidden, config.get('beta_spur', 1),
                config.get('beta_miss', 1),
                type_mode={'M1': 'shared', 'M2': 'exclusive'}.get(self.variant, 'independent'),
                type_control=self.variant != 'SUP',
                route_strength=config.get('route_strength', .5),
                scale_floor=config.get('scale_floor', .001),
                scale_ceiling=config.get('scale_ceiling', 1000.),
                amplitude_context=config.get('amplitude_context', 'none')) if self.use_a else None
            torch.random.default_generator.manual_seed(self.shared_init_seed + 1002)
            self.supported_residual = SupportedResidual(
                self.channels, hidden, int(config.get('attention_channels', 8)),
                config.get('rho', .25), config.get('g_min', .25),
                hard_support=self.variant not in ('M3', 'SUP'),
                bounded=self.variant not in ('M4', 'SUP'),
                ordinary=self.variant == 'SUP', voxel_size=args['voxel_size'],
                support_radius_m=config.get('support_radius_m', .8),
                scale_floor=config.get('scale_floor', .001),
                scale_ceiling=config.get('scale_ceiling', 1000.),
                scale_mode=config.get('scale_mode', 'current_observation_rms'),
                type_prior_mode=config.get('type_prior_mode', 'multiplicative'),
                efficient_features=self.efficient_features) if self.use_b else None

    def _encode(self, inputs):
        forbidden = set(inputs) - MODEL_INPUT_KEYS
        if forbidden:
            raise ValueError(f'non-model input keys: {sorted(forbidden)}; use select_model_inputs')
        record_len = inputs['record_len']
        agents = int(record_len.sum().item())
        if agents <= 0 or bool((record_len <= 0).any()):
            raise ValueError('record_len must retain at least one agent per scene')
        branches = []
        for modality, encoder in [('lidar', self.lidar_pillar_vfe), ('radar', self.radar_pillar_vfe)]:
            data = dict(inputs['processed_' + modality])
            data.update(batch_size=agents, record_len=record_len)
            device_type=data['voxel_features'].device.type
            amp_enabled=torch.is_autocast_enabled() if device_type=='cuda' else torch.is_autocast_cpu_enabled()
            amp_dtype=torch.get_autocast_gpu_dtype() if device_type=='cuda' else torch.get_autocast_cpu_dtype()
            # Absolute coordinates and within-pillar max selection are sensitive
            # to BF16 rounding. Preserve the PFN calculation/argmax in FP32;
            # the large BEV backbone and local modules still use autocast.
            with torch.autocast(device_type=device_type,enabled=False):
                data['voxel_features']=data['voxel_features'].float()
                encoded=encoder(data)
            if amp_enabled:
                encoded['pillar_features']=encoded['pillar_features'].to(amp_dtype)
            branches.append(self.scatter(encoded)['spatial_features'])
        lidar, radar = branches
        obs, available = inputs['observation_maps'], inputs['modality_valid']
        validate_observations(lidar, radar, obs, available)
        lidar = lidar * available[:, :1, None, None].to(lidar.dtype)
        radar = radar * available[:, 1:2, None, None].to(radar.dtype)
        return lidar, radar

    def extract_z0(self, inputs):
        """Current-input shared features for diagnostics, before collaboration."""
        lidar, radar = self._encode(inputs)
        return split_pointwise(self.phi_base, (lidar, radar))

    def _route_and_project(self, lidar, radar, observations, available):
        routed_l, routed_r, evidence = self.evidence_router(lidar, radar, observations, available,
                                                                      inputs_masked=self.efficient_features)
        return split_pointwise(self.phi_base, (routed_l, routed_r)), evidence

    def local_features(self, inputs, include_z0=True):
        lidar, radar = self._encode(inputs)
        obs, available = inputs['observation_maps'], inputs['modality_valid']
        aux = {'use_a': self.use_a, 'use_b': self.use_b,
               'disable_identity': self.variant == 'M5', 'variant': self.variant}
        z0 = split_pointwise(self.phi_base, (lidar, radar)) if include_z0 or not self.use_a else None
        if self.use_a:
            # Only stateless optional branches are recomputed. VFE/backbone BN
            # updates still happen exactly once; inference pays no recompute.
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                z, evidence = checkpoint(self._route_and_project, lidar, radar,
                                         obs, available, use_reentrant=False)
            else:
                z, evidence = self._route_and_project(lidar, radar, obs, available)
            aux.update(evidence)
        else:
            z = z0
            zero = z[:, :1] * 0
            aux.update(miss=zero, spur=zero, miss_logits=zero, spur_logits=zero)
        if self.use_b:
            use_prior = self.use_a and self.variant not in ('M6', 'SUP')
            residual_args = (z, radar, obs, available,
                             aux['miss'] if use_prior else None, aux['spur'] if use_prior else None)
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                z, residual = checkpoint(self.supported_residual, *residual_args,
                                          use_reentrant=False)
            else:
                z, residual = self.supported_residual(*residual_args)
            aux.update(residual)
        else:
            # Expand a scalar zero view rather than allocating a second dense BEV.
            aux.update(delta=z.new_zeros(()).expand_as(z), support=z[:, :1] * 0,
                       null_weight=z[:, :1] * 0 + 1,
                       feature_scale=z.new_ones((1, self.channels, 1, 1)))
        aux['z_local'] = z
        if include_z0:
            aux['z0'] = z0
        # The final bypass is the ORIGINAL available Radar branch.
        return split_pointwise(self.phi_out, (z, radar)), aux

    def forward(self, inputs, return_aux=None):
        if return_aux is None:
            return_aux = self.training
        local, aux = self.local_features(inputs, include_z0=False)
        output = self.backbone({'spatial_features': local, 'record_len': inputs['record_len']})
        features = output['spatial_features_2d']
        result = {'psm': self.cls_head(features), 'rm': self.reg_head(features)}
        if return_aux:
            result['aux'] = aux
        return result
