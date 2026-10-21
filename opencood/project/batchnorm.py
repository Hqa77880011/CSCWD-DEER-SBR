"""Scoped train-only BN statistics for deterministic, complete-eval inference.

Training EMA buffers remain resumable state. A refit uses the fixed model's
current inputs and PyTorch CMA: equal weight per microbatch for its mean and
unbiased within-microbatch variance. It is deliberately not a pooled variance.
"""
from contextlib import contextmanager
import hashlib
import itertools
import math
import time

import torch

from opencood.project.training import (restore_rng, rng_state, select_model_inputs,
                                       to_device)


class BatchNormRefitError(RuntimeError):
    """A bounded statistics pass failed; caller's original state is restored."""


def _batchnorms(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}


def configure_batchnorm(model, policy=None):
    """Apply the common protocol's eps/momentum without changing mode/buffers.

    Other policy fields describe the runner's calibration dataset and are not
    interpreted here. Legacy callers may omit policy for eps=.001/momentum=.01.
    """
    policy = {} if policy is None else policy
    if not isinstance(policy, dict):
        raise ValueError("BN policy must be a mapping or None")
    eps, momentum = policy.get("eps", .001), policy.get("momentum", .01)
    if (isinstance(eps, bool) or not isinstance(eps, (int, float)) or
            not math.isfinite(eps) or eps <= 0 or isinstance(momentum, bool) or
            not isinstance(momentum, (int, float)) or not math.isfinite(momentum) or
            not 0 < momentum <= 1):
        raise ValueError("BN eps must be positive and momentum must be in (0,1]")
    modules = _batchnorms(model)
    if any(not module.track_running_stats for module in modules.values()):
        raise ValueError("the registered BN policy requires tracked running statistics")
    for module in modules.values():
        module.eps, module.momentum = float(eps), float(momentum)
    return model


def _finite(value):
    if torch.is_tensor(value):
        if not value.numel() or not (value.is_floating_point() or value.is_complex()):
            return True
        if value.is_complex():
            return bool(torch.isfinite(value).all())
        low, high = torch.aminmax(value.detach())
        return bool(torch.isfinite(low) & torch.isfinite(high))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def batchnorm_statistics_hash(module):
    """Stable hash of the actual BN buffers, shared by fitting and export."""
    digest = hashlib.sha256()
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        value = getattr(module, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"None")
        else:
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def batchnorm_statistics_hashes(model):
    """Map every BN's exact module name to its running-buffer hash."""
    return {name: batchnorm_statistics_hash(module) for name, module in _batchnorms(model).items()}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def refit_batchnorm(model, batches, device):
    """Temporarily install reset-CMA statistics from at most eight microbatches.

    ``batches`` contains the runner's fixed train-partition calibration batches.
    Only inference-allowlisted ego fields enter the model. Labels and metadata
    are never passed to forward. All non-BN modules stay eval during fitting;
    every module is eval while yielding the JSON-compatible report. Finally,
    original buffers, per-module modes/momenta and all RNG streams are restored,
    including when fitting or the caller's evaluation raises. Parameters and
    their existing gradients are untouched. No optimizer or checkpoint is made.

    An unused BN (for example an empty modality VFE) keeps its original stats
    and is reported explicitly. Generic singleton BN inputs raise and roll back;
    PFN's explicit singleton fallback bypasses its BN and is therefore unused.
    """
    device = torch.device(device)
    modules = _batchnorms(model)
    if not modules:
        raise BatchNormRefitError("statistics refit requires at least one BN")
    if any(not module.track_running_stats for module in modules.values()):
        raise BatchNormRefitError("statistics refit requires tracked BN buffers")
    if hasattr(batches, "__len__") and len(batches) > 8:
        raise BatchNormRefitError("statistics refit is bounded to eight microbatches")
    original_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    original_modes = {name: module.training for name, module in model.named_modules()}
    original_momenta = {name: module.momentum for name, module in modules.items()}
    original_bn = {name: {key: value.detach().clone() for key, value in module.named_buffers(recurse=False)}
                   for name, module in modules.items()}
    original_rng = rng_state()
    counts = {name: 0 for name in modules}
    element_counts = {name: [] for name in modules}
    handles = []
    report = {"policy": "train_refit_cma_v1", "statistics_definition":
              "PyTorch reset CMA: microbatch means and unbiased within-microbatch variances averaged with equal microbatch weight; not pooled",
              "optimizer_updates": 0, "max_micro_batches": 8, "micro_batches": 0,
              "sample_count": 0, "input_sample_ids": [], "input_agent_weather": [],
              "bn": {}, "unused_bn": [], "all_modules_eval_at_yield": False,
              "original_buffers_restored_on_exit": None, "rng_restored_on_exit": None}

    def before_bn(name):
        def check(module, inputs):
            value = inputs[0]
            count = value.numel() // value.shape[1]
            if count <= 1:
                raise BatchNormRefitError(f"BN {name} has {count} observation per channel")
            if not _finite(value):
                raise BatchNormRefitError(f"nonfinite calibration input at BN {name}")
            counts[name] += 1
            element_counts[name].append(count)
        return check

    _synchronize(device)
    started = time.perf_counter()
    try:
        model.eval()
        for name, module in modules.items():
            module.reset_running_stats()
            module.momentum = None
            module.train()
            handles.append(module.register_forward_pre_hook(before_bn(name)))
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            # islice enforces the same bound for unsized generators without
            # pulling a ninth batch. Sized inputs longer than eight are rejected.
            for batch in itertools.islice(batches, 8):
                if "ego" not in batch:
                    raise BatchNormRefitError("calibration batches must contain ego")
                ego = batch["ego"]
                sample_ids = list(ego.get("sample_ids", []))
                if any(not str(sample_id).startswith("train/") for sample_id in sample_ids):
                    raise BatchNormRefitError("calibration sample IDs must belong to train")
                metadata = ego.get("metadata", [])
                weather = [[agent.get("weather") for agent in agents] for agents in metadata]
                if any(item not in ("normal", "fog", None) for agents in weather for item in agents):
                    raise BatchNormRefitError("calibration permits only train Normal/Fog")
                inputs = to_device(select_model_inputs(ego), device)
                output = model(inputs, return_aux=False)
                if not _finite(output):
                    raise BatchNormRefitError("nonfinite calibration model output")
                report["micro_batches"] += 1
                report["sample_count"] += len(sample_ids) if sample_ids else len(ego["record_len"])
                report["input_sample_ids"].extend(sample_ids)
                report["input_agent_weather"].append(weather)
                del inputs, output
        if not report["micro_batches"]:
            raise BatchNormRefitError("calibration batches are empty")
        for name, module in modules.items():
            if not counts[name]:
                for key, value in module.named_buffers(recurse=False):
                    value.copy_(original_bn[name][key])
                report["unused_bn"].append(name)
            if (not _finite(module.running_mean) or not _finite(module.running_var) or
                    bool((module.running_var < 0).any())):
                raise BatchNormRefitError(f"invalid fitted statistics at BN {name}")
            report["bn"][name] = {"calls": counts[name], "elements_per_channel_by_call": element_counts[name],
                                  "num_batches_tracked": int(module.num_batches_tracked),
                                  "statistics_sha256": batchnorm_statistics_hash(module),
                                  "used_original_statistics": not bool(counts[name])}
        model.eval()
        for handle in handles:
            handle.remove()
        handles.clear()
        _synchronize(device)
        report["seconds"] = time.perf_counter() - started
        report["all_modules_eval_at_yield"] = not any(module.training for module in model.modules())
        yield report
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(original_buffers[name])
        for name, module in model.named_modules():
            module.training = original_modes[name]
        for name, module in modules.items():
            module.momentum = original_momenta[name]
        restore_rng(original_rng)
        report["original_buffers_restored_on_exit"] = all(torch.equal(value, original_buffers[name])
                                                          for name, value in model.named_buffers())
        report["rng_restored_on_exit"] = True
