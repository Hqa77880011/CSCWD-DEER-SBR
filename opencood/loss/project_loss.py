"""Shared anchor detection loss and separately supplied DEER/SBR targets."""
import torch
from torch import nn
from torch.nn import functional as F
from .point_pillar_loss import PointPillarLoss


def masked_mean(values, mask):
    mask = mask.to(device=values.device, dtype=values.dtype).expand_as(values)
    # Invalid cells may contain NaN placeholders. They must contribute ZERO.
    safe = torch.where(mask > 0, values, torch.zeros_like(values))
    return (safe * mask).sum() / mask.sum().clamp_min(1)


def masked_bce_logits(logits, targets, mask):
    mask = mask.to(logits.device)
    targets = targets.to(device=logits.device, dtype=torch.float32)
    target = torch.where(mask > 0, targets, torch.zeros_like(targets))
    return masked_mean(F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none'), mask)


def supported_residual_losses(delta, scale, support, normal=None):
    """Same weighted channel means, gathering only genuinely supported cells.

    The old dense path builds normalization, square/abs, where and multiply
    tensors over every channel and cell. Sparse support allows those temporary
    tensors to have [supported_cells, channels] shape instead. Invalid cells
    are excluded before nonlinear operations, including NaN placeholders.
    """
    delta = delta.float()
    scale = scale.detach().float().clamp_min(1e-6)
    support = support.detach().to(device=delta.device, dtype=delta.dtype)
    if support.shape != (delta.shape[0], 1, *delta.shape[-2:]):
        raise ValueError('residual support must be [agents,1,H,W]')
    agent, yy, xx = (support[:, 0] > 0).nonzero(as_tuple=True)
    values = delta.permute(0, 2, 3, 1)[agent, yy, xx]
    scales = scale.expand(delta.shape[0], delta.shape[1], 1, 1)[agent, :, 0, 0]
    normalized = values / scales
    weights = support[agent, 0, yy, xx]
    correction = (normalized.square() * weights[:, None]).sum() / (
        support.sum() * delta.shape[1]).clamp_min(1)
    # An empty gather still supplies a differentiable zero to the residual.
    identity = normalized[:0].sum()
    if normal is not None:
        normal = normal.detach().reshape(-1).to(device=delta.device, dtype=delta.dtype)
        identity_weights = weights * normal[agent]
        selected = identity_weights > 0
        identity = (normalized[selected].abs() * identity_weights[selected, None]).sum() / (
            (support * normal[:, None, None, None]).sum() * delta.shape[1]).clamp_min(1)
    return correction, identity


class ProjectLoss(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        self.lambda_a = float(args.get('lambda_a', args.get('lambda_A', .10)))
        if 'lambda_repair' in args:
            raise ValueError('teacher repair loss belongs to archived teacher_v1; use lambda_correction')
        self.lambda_correction = float(args.get('lambda_correction', .001))
        self.lambda_id = float(args.get('lambda_id', .01))
        self.efficient_features = bool(args.get('efficient_features', False))
        self.training_line = str(args.get('training_line', 'clean')).lower()
        if self.training_line not in ('clean', 'weather', 'c', 'w'):
            raise ValueError('training_line must be Clean/C or Weather/W')

    def detection_loss(self, output, target):
        psm, rm = output['psm'].float(), output['rm'].float()
        batch = psm.shape[0]
        positive = target['pos_equal_one'].reshape(batch, -1).float()
        negative = target['neg_equal_one'].reshape(batch, -1).float()
        # Ignore IoU-band anchors where BOTH masks are zero. The upstream loss
        # incorrectly treated every non-positive anchor as a negative.
        normalizer = positive.sum(1, keepdim=True).clamp_min(1)
        cls_weights = (positive + negative).clamp_max(1) / normalizer
        cls_predictions = psm.permute(0, 2, 3, 1).reshape(batch, -1, 1)
        cls = self.cls_loss_func(cls_predictions, positive.unsqueeze(-1), cls_weights).sum() / batch
        boxes = rm.permute(0, 2, 3, 1).reshape(batch, -1, 7)
        targets = target['targets'].float().reshape(batch, -1, 7)
        # NaN regression placeholders on ignored anchors cannot poison sine.
        targets = torch.where(positive.unsqueeze(-1) > 0, targets, boxes.detach())
        pred_sin, target_sin = self.add_sin_difference(boxes, targets)
        regression = self.reg_loss_func(pred_sin, target_sin, positive / normalizer).sum() / batch
        cls, regression = cls * self.cls_weight, regression * self.reg_coe
        self.loss_dict = {'conf_loss': cls.detach(), 'reg_loss': regression.detach()}
        return cls + regression

    def auxiliary_loss(self, aux, targets=None, aux_weight=1.0):
        targets = {} if targets is None else targets
        # A full fp16 BEV sum can overflow even when each feature is finite.
        # One scalar keeps a differentiable zero without a dense reduction.
        zero = aux['z_local'][0, 0, 0, 0] * 0
        type_loss = zero
        if aux['use_a']:
            for name in ('miss', 'spur'):
                if name in targets and name + '_valid' in targets:
                    type_loss = type_loss + masked_bce_logits(
                        aux[name + '_logits'], targets[name], targets[name + '_valid'])
        # v2 has no teacher feature target. The task loss learns the signed
        # residual; weak size and normal-identity penalties constrain its cost.
        if 'clean_z0' in targets:
            raise ValueError('teacher features are forbidden in teacher_free_v2')
        correction_loss, identity_loss = zero, zero
        if aux['use_b']:
            support = aux['support'].detach()
            normal = targets.get('normal_mask') if not aux.get('disable_identity', False) else None
            if self.efficient_features:
                correction_loss, identity_loss = supported_residual_losses(
                    aux['delta'], aux['feature_scale'], support, normal)
            else:
                scale = aux['feature_scale'].detach().float().clamp_min(1e-6)
                normalized = aux['delta'].float() / scale
                correction_loss = masked_mean(normalized.square(), support)
                if normal is not None:
                    normal = normal.reshape(-1, 1, 1, 1).to(support.device)
                    identity_loss = masked_mean(normalized.abs(), normal * support)
        self.loss_dict.update(type_loss=type_loss.detach(), correction_loss=correction_loss.detach(),
                              identity_loss=identity_loss.detach())
        for name in ('miss', 'spur'):
            mask = targets.get(name + '_valid')
            self.loss_dict[name + '_coverage'] = zero.detach() if mask is None else (mask > 0).float().mean().detach()
        return float(aux_weight) * (self.lambda_a * type_loss + self.lambda_correction * correction_loss + self.lambda_id * identity_loss)

    def forward(self, output_dict, target_dict, train_targets=None, aux_weight=1.0):
        loss = self.detection_loss(output_dict, target_dict)
        if 'aux' in output_dict:
            loss = loss + self.auxiliary_loss(output_dict['aux'], train_targets, aux_weight)
        self.loss_dict['total_loss'] = loss.detach()
        return loss
