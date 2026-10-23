"""Adapters around the repository's complete native external methods.

No approximate gate replaces MDD, MGF, SwapFusion or Where2comm communication.
Only data/return signatures and dataset-specific detection heads are adapted.
"""
from copy import deepcopy
import importlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.point_pillar_deer_sbr import PointPillarDeerSbr
from opencood.loss.project_loss import ProjectLoss
from opencood.project.supplement import resolve_final

NATIVE_MODELS = {
    'attfuse': ('point_pillar_intermediate_lrf', 'PointPillarIntermediateLRF'),
    'mdd': ('point_pillar_intermediate_lrf_MDD', 'PointPillarIntermediateLRFMDD'),
    'l4dr': ('point_pillar_L4DR', 'PointPillarL4DR'),
    'cobevt': ('point_pillar_cobevt_lrf', 'PointPillarCoBEVTLRF'),
    'where2comm': ('point_pillar_where2comm_lrf', 'PointPillarWhere2commLRF'),
}


class LocalFusion(PointPillarDeerSbr):
    """Reuse DEER/SBR without retaining a second encoder or collaboration net."""
    def __init__(self, args):
        super().__init__(args)
        for name in ('lidar_pillar_vfe', 'radar_pillar_vfe', 'scatter', 'backbone', 'cls_head', 'reg_head'):
            delattr(self, name)


class WindowPaddedFusion(nn.Module):
    """Keep native SwapFusion on non-window-divisible project geometry."""
    def __init__(self, fusion, window_size):
        super().__init__()
        self.fusion, self.window_size = fusion, window_size

    def forward(self, x, mask=None):
        height, width = x.shape[-2:]
        ph, pw = (-height) % self.window_size, (-width) % self.window_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
            if mask is not None:
                mask = F.pad(mask.permute(0, 3, 4, 1, 2), (0, pw, 0, ph)).permute(0, 3, 4, 1, 2)
        return self.fusion(x, mask)[..., :height, :width]


class SingleAgentFusion(nn.Module):
    """Explicitly bypass collaboration for K-Radar's one genuine sensor rig."""
    def forward(self, features, record_len):
        if not bool((record_len == 1).all()):
            raise ValueError('K-Radar cannot contain replicated or additional agents')
        return features


class SupplementalModel(nn.Module):
    def __init__(self, config, hypes):
        super().__init__()
        self.method = config['experiment']['method']
        self.dataset_id = config['experiment']['dataset_id']
        self.classes = ('Sedan', 'Bus or Truck') if self.dataset_id == 'kradar' else ('Vehicle',)
        args = deepcopy(hypes['model']['args'])
        variant = config['experiment']['variant']
        if variant == 'F':
            variant = resolve_final(config['experiment']['final_model_file'])
        args['shared_init_seed'] = config['experiment']['seed']
        args['innovation'] = {**config['model'], 'variant': variant}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args['shared_init_seed'])
            if self.method == 'deer_sbr':
                self.native = PointPillarDeerSbr(args)
            else:
                module, name = NATIVE_MODELS[self.method]
                cls = getattr(importlib.import_module('opencood.models.' + module), name)
                self.native = cls(args)
        if self.method == 'cobevt':
            self.native.fusion_net = WindowPaddedFusion(self.native.fusion_net, args['fax_fusion']['window_size'])
        self.l4dr_frontend = None
        if self.method == 'l4dr':
            from opencood.project.l4dr import L4DRFrontEnd
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(args['shared_init_seed'] + 3007)
                self.l4dr_frontend = L4DRFrontEnd(args, hypes['preprocess'])
            del self.native.lidar_pillar_vfe, self.native.radar_pillar_vfe
        if self.dataset_id == 'kradar':
            self.native.backbone.fuse_modules = nn.ModuleList(
                SingleAgentFusion() for _ in self.native.backbone.fuse_modules)
        # All complete methods keep their native feature width and head stride.
        self.extra_cls = nn.ModuleDict()
        self.extra_reg = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args['shared_init_seed'] + 2003)
            for name in self.classes[1:]:
                self.extra_cls[name] = nn.Conv2d(self.native.cls_head.in_channels, self.native.cls_head.out_channels, 1)
                self.extra_reg[name] = nn.Conv2d(self.native.reg_head.in_channels, self.native.reg_head.out_channels, 1)
        self._features = None
        self.native.cls_head.register_forward_pre_hook(self._capture_features)
        self.local = None
        self._local_inputs = None
        self._local_aux = None
        if self.method == 'where2comm' and config['model']['local_variant'] != 'none':
            local_args = deepcopy(args)
            selected = config['model']['local_variant']
            if selected == 'F':
                selected = resolve_final(config['experiment']['final_model_file'])
            local_args['innovation'].update(variant=selected,
                type_prior_mode='multiplicative' if selected == 'N2' else 'input_only')
            self.local = LocalFusion(local_args)
            self.native.backbone.register_forward_pre_hook(self._apply_local)

    def _capture_features(self, module, args):
        self._features = args[0]

    def _apply_local(self, module, args):
        batch = dict(args[0])
        lidar, radar = batch['spatial_features'].chunk(2, dim=1)
        batch['spatial_features'], self._local_aux = self.local.fuse_encoded(
            lidar, radar, self._local_inputs, include_z0=False)
        return (batch,)

    def forward(self, inputs, return_aux=None):
        if self.dataset_id == 'kradar' and not bool((inputs['record_len'] == 1).all()):
            raise ValueError('K-Radar is a single-agent local fusion task')
        self._local_inputs = inputs
        try:
            if self.method == 'deer_sbr':
                output = self.native(inputs, return_aux=return_aux)
            elif self.method == 'l4dr':
                lidar, radar, fad = self.l4dr_frontend(inputs)
                lidar = self.native.scatter({**lidar, 'record_len': inputs['record_len']})['spatial_features']
                radar = self.native.scatter({**radar, 'record_len': inputs['record_len']})['spatial_features']
                features = self.native.backbone(dict(lidar_spatial_features=lidar,
                    radar_spatial_features=radar, record_len=inputs['record_len']))['spatial_features_2d']
                output = dict(psm=self.native.cls_head(features), rm=self.native.reg_head(features))
                if self.training:
                    output.update(fad)
            else:
                data = dict(inputs)
                if self.method == 'mdd':
                    data['train'] = self.training
                    if self.training and 'processed_de_lidar' not in data:
                        raise ValueError('MDD training requires explicit, verified reference LiDAR; no clean target is inferred')
                output = self.native(data)
                if self._local_aux is not None and (self.training if return_aux is None else return_aux):
                    output['aux'] = self._local_aux
            if len(self.classes) > 1:
                features = self._features
                output['class_outputs'] = {self.classes[0]: {'psm': output['psm'], 'rm': output['rm']}}
                for name in self.classes[1:]:
                    output['class_outputs'][name] = {'psm': self.extra_cls[name](features), 'rm': self.extra_reg[name](features)}
            return output
        finally:
            self._features = self._local_inputs = self._local_aux = None


class SupplementalLoss(ProjectLoss):
    """Shared fixed-GT detection loss; native MDD reference loss is retained."""
    def __init__(self, args, method):
        super().__init__(args)
        self.method = method
        self.epoch = 0

    def forward(self, output_dict, target_dict, train_targets=None, aux_weight=1.):
        if 'class_outputs' in output_dict:
            terms, saved = [], {}
            for name, prediction in output_dict['class_outputs'].items():
                terms.append(self.detection_loss(prediction, target_dict['class_labels'][name]))
                saved.update({name + '/' + key: value for key, value in self.loss_dict.items()})
            loss = sum(terms)
            self.loss_dict = saved
        else:
            loss = self.detection_loss(output_dict, target_dict)
        if 'aux' in output_dict:
            loss = loss + self.auxiliary_loss(output_dict['aux'], train_targets, aux_weight)
        if self.method == 'mdd':
            if 'gt_feature' not in output_dict:
                raise ValueError('native MDD reference feature target is missing')
            # PointPillarLossMDD.diff_loss_func and sigmoid_weight, verbatim
            # algebra: channel-summed squared error with epoch-dependent weight.
            weight = 1.5 * (1 - math.tanh((self.epoch + 1) / 4 - 5))
            diff = (output_dict['gt_feature'].detach() - output_dict['pred_feature']).square().sum(dim=1).mean() * weight
            self.loss_dict['mdd_reference_loss'] = diff.detach()
            loss = loss + diff
        if self.method == 'l4dr':
            from opencood.project.l4dr import foreground_labels
            if train_targets is None or 'fad_boxes' not in train_targets:
                raise ValueError('L4DR FAD training requires separate train-only foreground box targets')
            logits = output_dict['fad_logits']
            labels = foreground_labels(output_dict['fad_coordinates'], train_targets['fad_boxes'])
            weights = (labels >= 0).float() / (labels > 0).sum().clamp_min(1)
            if len(logits):
                fad = self.cls_loss_func(logits[None], (labels > 0).float()[None, :, None], weights[None]).sum() * 2.
            else:
                fad = loss * 0
            self.loss_dict['fad_foreground_loss'] = fad.detach()
            self.loss_dict['fad_foreground_points'] = (labels > 0).sum().detach()
            loss = loss + fad
        self.loss_dict['total_loss'] = loss.detach()
        return loss


def set_intervention(model, name, seed):
    """Inference-only interventions never change a trained variant's identity."""
    if model.training and name != 'none':
        raise ValueError('signal interventions are inference-only; use a trained ablation for structural claims')
    for module in model.modules():
        module.signal_intervention = name
        module.intervention_seed = int(seed)


def shuffle_signal(value, seed):
    """Deterministic within-agent spatial shuffle, independent of model RNG."""
    flat = value.flatten(2)
    axis = -1 if flat.shape[-1] > 1 else 1
    perm = torch.randperm(flat.shape[axis], generator=torch.Generator().manual_seed(seed)).to(value.device)
    return flat.index_select(axis, perm).reshape_as(value)
