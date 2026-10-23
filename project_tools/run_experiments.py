#!/usr/bin/env python3
"""Project protocol runner. Default mode only inspects; never starts training."""
import argparse
from contextlib import nullcontext, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
import platform
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from opencood.project.config import load_config, validate_config, config_hash, file_sha256
from opencood.project.training import (EpochBatchSampler, TrainState, seed_everything, select_model_inputs, to_device,
    save_checkpoint, load_checkpoint, freeze_batchnorm, consider_checkpoint,
    atomic_torch_save, tensor_state_hash, FP16_INITIAL_SCALE, rng_state, restore_rng)

REPO = Path(__file__).resolve().parents[1]


def timestamp_utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def cuda_memory_snapshot(device):
    """Allocator counters include recovered internal OOMs, not a failure verdict."""
    device = torch.device(device)
    if device.type != "cuda":
        return {key: 0 for key in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
                                  "peak_reserved_bytes", "allocator_retries", "allocator_ooms")}
    stats = torch.cuda.memory_stats(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocator_retries": stats.get("num_alloc_retries", 0),
            "allocator_ooms": stats.get("num_ooms", 0)}


def batchnorm_snapshot(model):
    """Read small running buffers without changing modes, gradients or RNG."""
    rows, invalid = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        mean = None if module.running_mean is None else module.running_mean.detach().cpu()
        variance = None if module.running_var is None else module.running_var.detach().cpu()
        finite = mean is not None and variance is not None and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(variance).all())
        negative = int((variance < 0).sum()) if variance is not None else 0
        finite_variance = variance[torch.isfinite(variance)] if variance is not None else torch.empty(0)
        rows[name] = {"finite": finite, "negative_variance_count": negative,
                      "running_var_min": float(finite_variance.min()) if finite_variance.numel() else None,
                      "running_var_max": float(finite_variance.max()) if finite_variance.numel() else None,
                      "num_batches_tracked": int(module.num_batches_tracked) if module.num_batches_tracked is not None else None}
        if not finite or negative:
            invalid.append(name)
    minima = [row["running_var_min"] for row in rows.values() if row["running_var_min"] is not None]
    maxima = [row["running_var_max"] for row in rows.values() if row["running_var_max"] is not None]
    return {"module_count": len(rows), "all_finite": all(row["finite"] for row in rows.values()),
            "invalid_modules": invalid, "running_var_min": min(minima) if minima else None,
            "running_var_max": max(maxima) if maxima else None, "modules": rows}


def monitor_batchnorm(model, telemetry=None, *, phase):
    summary = batchnorm_snapshot(model)
    if telemetry is not None:
        telemetry("batchnorm", phase=phase, batchnorm=summary)
    if summary["invalid_modules"]:
        raise FloatingPointError("invalid online/refitted BN statistics: " + json.dumps(summary))
    return summary


def resolve_repo(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def read_hypes(config):
    from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
    with resolve_repo(config["data"]["base_hypes"]).open() as stream:
        hypes = yaml.safe_load(stream)
    if hypes.get("yaml_parser") != "load_point_pillar_params":
        raise ValueError("only the verified PointPillar base hypes parser is accepted")
    if config['version'] == 5:
        from opencood.project.supplement_runner import configure_hypes
        hypes = configure_hypes(config, hypes)
    hypes = load_point_pillar_params(hypes)
    hypes["root_dir"] = str(Path(config["data"]["root"]) / "train")
    hypes["validate_dir"] = str(Path(config["data"]["root"]) / "validate")
    hypes["postprocess"]["target_args"]["score_threshold"] = config["scoring"]["score_floor"]
    hypes["load_rgb"] = False
    return hypes


def build_model(config, mode="smoke", variant=None):
    if config['version'] == 5:
        from opencood.project.supplement_models import SupplementalModel
        return SupplementalModel(config, read_hypes(config))
    from opencood.models.point_pillar_deer_sbr import PointPillarDeerSbr
    hypes = read_hypes(config)
    args = deepcopy(hypes["model"]["args"])
    args["shared_init_seed"] = config["experiment"]["seed"]
    args["innovation"] = {**config["model"], "variant": variant or config["experiment"]["variant"]}
    model = PointPillarDeerSbr(args)
    from opencood.project.batchnorm import configure_batchnorm
    configure_batchnorm(model, config["training"].get("batchnorm"))
    return model


def build_components(config, mode="smoke"):
    if config['version'] == 5:
        from opencood.project.supplement_runner import build_dataset
        from opencood.project.supplement_models import SupplementalLoss
        hypes = read_hypes(config)
        dataset = build_dataset(config, hypes, 'train', training=True, augment=True)
        loss = {**hypes['loss']['args'], 'training_line': config['experiment']['track'],
                'lambda_a': config['loss']['type_weight'],
                'lambda_correction': config['loss']['correction_weight'],
                'lambda_id': config['loss']['identity_weight'],
                'efficient_features': config['model']['efficient_features']}
        return dataset, build_model(config, mode), SupplementalLoss(loss, config['experiment']['method'])
    from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
    from opencood.loss.project_loss import ProjectLoss
    hypes = read_hypes(config)
    d, e, r = config["data"], config["experiment"], config["resources"]
    dataset = ProjectV2XRDataset(hypes, root=d["root"], split="train", training=True, track=e["track"],
        seed=e["seed"], paired=d["paired_supervision"], augment=True,
        cache_bytes=int(r["raw_cache_gib_per_worker"] * 1024 ** 3), supervision_source=d["supervision_source"],
        gt_manifest=str(Path(d["manifest_dir"]) / "eval_gt_manifest.json"), supervision_audit=d["supervision_audit"],
        matching=d["matching"], controlled_corruption=d["controlled_corruption"])
    model = build_model(config, mode)
    loss_config = {**hypes["loss"]["args"], "training_line": e["track"], "lambda_a": config["loss"]["type_weight"],
                   "lambda_correction": config["loss"]["correction_weight"], "lambda_id": config["loss"]["identity_weight"],
                   "efficient_features": config["model"]["efficient_features"]}
    return dataset, model, ProjectLoss(loss_config)


def _worker_init(worker_id):
    torch.set_num_threads(1)
    # Dataset geometry/weather are sample-keyed; this seed only covers libraries.
    np.random.seed(torch.initial_seed() % (2 ** 32))


def make_loader(dataset, config, *, sampler=None, evaluation=False):
    r = config["resources"]
    kwargs = dict(num_workers=r["num_workers"], pin_memory=r["pin_memory"], worker_init_fn=_worker_init,
                  collate_fn=dataset.collate_batch, generator=torch.Generator().manual_seed(config["experiment"]["seed"]))
    if r["num_workers"]:
        kwargs.update(prefetch_factor=r["prefetch_factor"], persistent_workers=r["persistent_workers"])
    if sampler is not None:
        kwargs["batch_sampler"] = sampler
    else:
        kwargs.update(batch_size=1 if evaluation else config["training"]["micro_batch_size"], shuffle=False)
    return DataLoader(dataset, **kwargs)


def make_eval_loader(config, split="validate", weather="normal"):
    split = "validate" if split == "validation" else split
    if config['version'] == 5:
        from opencood.project.supplement_runner import build_dataset
        if weather == 'snow' and split != 'test':
            raise ValueError('Snow is excluded from model selection')
        dataset = build_dataset(config, read_hypes(config), split, weather=weather)
        return dataset, make_loader(dataset, config, evaluation=True)
    from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
    if weather == "snow" and split != "test":
        raise ValueError("Snow is forbidden in model selection")
    dataset = ProjectV2XRDataset(read_hypes(config), root=config["data"]["root"], split=split,
        training=False, weather=weather, seed=config["experiment"]["seed"], paired=False, augment=False,
        cache_bytes=0, supervision_source="none", gt_manifest=str(Path(config["data"]["manifest_dir"]) / "eval_gt_manifest.json"))
    return dataset, make_loader(dataset, config, evaluation=True)


def make_bn_refit_loader(config):
    """Separate fixed train input stream; never mutates the training loader."""
    policy = config['training']['batchnorm']
    d, e, r = config['data'], config['experiment'], config['resources']
    if config['version'] == 5:
        from opencood.project.supplement_runner import build_dataset
        dataset = build_dataset(config, read_hypes(config), 'train', training=True, augment=policy['refit_augment'])
    else:
        from opencood.data_utils.datasets.project_v2xr_dataset import ProjectV2XRDataset
        dataset = ProjectV2XRDataset(read_hypes(config), root=d['root'], split='train', training=True,
            track=e['track'], seed=e['seed'], paired=d['paired_supervision'], augment=policy['refit_augment'],
            cache_bytes=int(r['raw_cache_gib_per_worker'] * 1024**3), supervision_source=d['supervision_source'],
            gt_manifest=str(Path(d['manifest_dir'])/'eval_gt_manifest.json'), supervision_audit=d['supervision_audit'],
            matching=d['matching'], controlled_corruption=d['controlled_corruption'])
    dataset.set_epoch(policy['refit_epoch'])
    count, micro = policy['refit_samples'], policy['refit_micro_batch_size']
    if count > len(dataset) or count % micro:
        raise ValueError('fixed BN refit samples must fit train and divide its microbatch')
    indices = [int((i + .5) * len(dataset) / count) for i in range(count)]
    if len(set(indices)) != count:
        raise ValueError('BN refit indices must be unique')
    sampler = [indices[start:start+micro] for start in range(0,count,micro)]
    loader = make_loader(dataset, config, sampler=sampler)
    return dataset, loader, indices


@contextmanager
def batchnorm_evaluation(model, config, device, telemetry=None):
    """One train-only frozen-weight refit, shared by all evaluation weather."""
    policy = config['training'].get('batchnorm', {})
    if policy.get('evaluation_statistics', 'running') == 'running':
        yield None
        return
    from opencood.project.batchnorm import refit_batchnorm, BatchNormRefitError, batchnorm_statistics_hashes
    random_state = rng_state()
    original_hashes = batchnorm_statistics_hashes(model)
    monitor_batchnorm(model, telemetry, phase="before_refit_online")
    started = time.perf_counter()
    resources, report = {}, None
    try:
        dataset, loader, indices = make_bn_refit_loader(config)
        resources['loader'] = loader
        if telemetry is not None:
            telemetry("refit_start", dataset_split="train", sample_indices=indices,
                      refit_samples=policy['refit_samples'], refit_micro_batch_size=policy['refit_micro_batch_size'],
                      refit_epoch=policy['refit_epoch'], sampling=policy['refit_sampling'],
                      input_augmentation=policy['refit_augment'])
        refit_options = {}
        if config['version'] == 5 and config['experiment']['dataset_id'] != 'v2xr':
            refit_options = dict(training_sample_ids=set(dataset.sample_ids),
                allowed_weather={s['natural_weather'] for s in dataset.manifest['samples'].values() if s['split'] == 'train'})
        with refit_batchnorm(model, loader, device, max_batches=(len(loader) if config['version'] == 5 else 8), **refit_options) as report:
            report.update(sample_indices=indices, sampling=policy['refit_sampling'],
                          refit_epoch=policy['refit_epoch'], input_augmentation=policy['refit_augment'],
                          training_protocol=config['experiment']['training_protocol'],
                          dataset_split='train', training_track=config['experiment']['track'],
                          configuration_policy=deepcopy(policy), data_contract=data_contract_metadata(config))
            if (report['sample_count'] != policy['refit_samples'] or
                    report['micro_batches'] != policy['refit_samples']//policy['refit_micro_batch_size'] or
                    report['unused_bn'] or not report['all_modules_eval_at_yield']):
                raise BatchNormRefitError('incomplete fixed training-partition statistics pass')
            # The independent fitting workers have finished their job; release
            # them before creating the validation loader.
            shutdown_loader(resources)
            report['pre_evaluation_wall_seconds'] = time.perf_counter() - started
            monitor_batchnorm(model, telemetry, phase="refitted_for_validation")
            if telemetry is not None:
                telemetry("refit_complete", sample_count=report['sample_count'], micro_batches=report['micro_batches'],
                          input_sample_ids=report['input_sample_ids'],
                          input_agent_weather=report['input_agent_weather'], unused_bn=report['unused_bn'],
                          seconds=report['pre_evaluation_wall_seconds'])
            yield report
    finally:
        shutdown_loader(resources)
        restore_rng(random_state)
        restored = batchnorm_statistics_hashes(model) == original_hashes
        if telemetry is not None:
            telemetry("refit_restore", original_bn_hashes_restored=restored,
                      original_buffers_restored_on_exit=report.get('original_buffers_restored_on_exit') if report else None,
                      rng_restored_on_exit=True, seconds=time.perf_counter() - started)
        if not restored:
            raise FloatingPointError("online BN statistics were not restored after validation refit")
        monitor_batchnorm(model, telemetry, phase="after_refit_online_restored")


def data_contract_metadata(config):
    if config['version'] == 5 and config['experiment']['dataset_id'] != 'v2xr':
        return {'base_hypes_sha256': file_sha256(resolve_repo(config['data']['base_hypes'])),
                'gt_manifest_sha256': file_sha256(Path(config['data']['manifest_dir']) / 'supplement_manifest.json')}
    manifest = Path(config["data"]["manifest_dir"]) / "eval_gt_manifest.json"
    metadata = {"base_hypes_sha256": file_sha256(resolve_repo(config["data"]["base_hypes"])),
                "gt_manifest_sha256": file_sha256(manifest)}
    for name in ("data_manifest", "split_manifest", "weather_manifest", "protocol_manifest", "image_sizes"):
        path = Path(config["data"]["manifest_dir"]) / (name + ".json")
        metadata[name + "_sha256"] = file_sha256(path) if path.exists() else None
        source_path = Path(config["data"]["root"]) / path.name
        if name != "protocol_manifest" and source_path.resolve() != path.resolve():
            if not source_path.is_file() or file_sha256(source_path) != metadata[name + "_sha256"]:
                raise ValueError(f"data.root and manifest_dir refer to different datasets: {name}")
    return metadata


def artifact_metadata(config):
    metadata = data_contract_metadata(config)
    for key in ("amp_audit",):
        path = config["artifacts"][key]
        metadata[key + "_sha256"] = file_sha256(path) if path else None
    audit = config["data"]["supervision_audit"]
    metadata["supervision_audit_sha256"] = file_sha256(audit) if audit else None
    metadata["training_protocol"] = config["experiment"]["training_protocol"]
    metadata["supervision_contract"] = supervision_contract(config)
    metadata["source_sha256"] = source_tree_hash()
    if config['version'] == 5:
        from opencood.project.supplement import resolve_final
        e = config['experiment']
        path = e['final_model_file']
        metadata['final_model_sha256'] = file_sha256(path) if path else None
        metadata['resolved_variant'] = resolve_final(path) if e['variant'] == 'F' else e['variant']
        local = config['model']['local_variant']
        metadata['resolved_local_variant'] = resolve_final(path) if local == 'F' else local
    return metadata


def source_tree_hash():
    source_files = list((REPO / "opencood").rglob("*.py")) + list((REPO / "project_tools").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(source_files, key=lambda p: p.relative_to(REPO).as_posix()):
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


AMP_TOLERANCES = {"loss_relative_error": 0.03, "psm_relative_l2": 0.05, "rm_relative_l2": 0.05,
                  "gradient_cosine_min": 0.98, "gradient_relative_l2": 0.1}


def amp_audit_contract(config):
    """Bind numerical evidence to the implementation and actual micro-batch policy."""
    training_keys = ("micro_batch_size", "world_size", "accumulation_steps", "effective_batch_size",
                     "normalization", "batchnorm", "deterministic", "precision")
    artifacts = {"training_protocol": config["experiment"]["training_protocol"]}
    return {"source_sha256": source_tree_hash(), **data_contract_metadata(config), **artifacts,
            "training_contract": {key: config["training"][key] for key in training_keys},
            "model": config["model"], "loss": config["loss"], "supervision": supervision_contract(config),
            "seed": config["experiment"]["seed"], "track": config["experiment"]["track"],
            "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version()}}


def validate_amp_audit(config, audit, *, require_stability=True):
    if (audit.get("schema") != "project_amp_consistency_v1" or audit.get("passed") is not True
            or audit.get("precision") != config["training"]["precision"] or audit.get("reference_precision") != "fp32"
            or audit.get("dataset_id") != "v2xr" or audit.get("thresholds") != AMP_TOLERANCES
            or not audit.get("evidence")):
        return "AMP audit schema, precision, evidence or preregistered thresholds mismatch"
    if audit.get("contract") != amp_audit_contract(config):
        return "AMP audit source/data/model/matching/micro-batch/normalization/runtime contract mismatch"
    required = {"S0", "AB", config["experiment"]["variant"]}
    cases = audit.get("cases", [])
    if not required <= {case.get("variant") for case in cases}:
        return "AMP audit must cover S0, AB, and the selected variant"
    for case in cases:
        identity = case.get("batch_identity_sha256", "")
        if case.get("finite") is not True or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            return "AMP case lacks finite outputs/gradients or a stable sample-agent-augmentation batch identity"
        for metric, maximum in AMP_TOLERANCES.items():
            metric = "gradient_cosine" if metric == "gradient_cosine_min" else metric
            value = case.get(metric)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                return f"AMP case is missing finite {metric}"
            if metric == "gradient_cosine":
                if not maximum <= value <= 1.000001:
                    return "AMP gradient cosine failed the frozen numerical tolerance"
            elif not 0 <= value <= maximum:
                return f"AMP {metric} failed the frozen numerical tolerance"
    if require_stability:
        evidence = audit.get("evidence", {})
        if not isinstance(evidence, dict) or evidence.get("stability_validated") is not True:
            return "AMP numerical agreement alone is insufficient: at least 20 finite real training updates are required"
        reference = evidence.get("stability_report", {})
        path = Path(reference.get("path", ""))
        if not path.is_file() or reference.get("sha256") != file_sha256(path):
            return "AMP bounded-training stability report is missing or its SHA256 changed"
        stability = json.loads(path.read_text())
        if (stability.get("passed") is not True or stability.get("all_losses_and_gradients_finite") is not True
                or stability.get("status") != "diagnostic_completed" or stability.get("mode") not in ("smoke", "benchmark")
                or stability.get("formal_experiment_started") is not False
                or stability.get("completed_optimizer_updates", 0) < 20):
            return "AMP stability report must contain at least 20 successful bounded real training updates"
        if stability.get("variant") != config["experiment"]["variant"] or stability.get("audit_contract") != amp_audit_contract(config):
            return "AMP stability report variant/source/training contract mismatch"
    return None


def supervision_contract(config):
    """Bind train-only type targets to their actual source, never density labels."""
    d = config["data"]
    if config['version'] == 5:
        return dict(dataset_id=config['experiment']['dataset_id'], split='train',
            supervision_source=d['supervision_source'], controlled_corruption=d['controlled_corruption'],
            seed=config['experiment']['seed'], natural_unknown_types='masked',
            real_reference_policy='verified_clear_only',
            source_sha256=source_tree_hash(), data_contract=data_contract_metadata(config))
    result = {"dataset_id": "v2xr", "split": "train", "supervision_source": d["supervision_source"],
              "source_sha256": source_tree_hash(), "data_contract": data_contract_metadata(config)}
    if d["supervision_source"] == "provenance_v1":
        result["controlled_corruption"] = d["controlled_corruption"]
        result.update(seed=config["experiment"]["seed"],epoch=0,audit_samples=32)
    else:
        result["matching"] = d["matching"]
    return result


@lru_cache(maxsize=4)
def _bounded_provenance_replay(config_json, contract_json):
    """Actual train-input replay, cached only for this exact source/data contract."""
    from project_tools.audit_pair_supervision import audit_provenance
    config=json.loads(config_json)
    replay=audit_provenance(config,32)
    replay_contract={key:value for key,value in replay['contract'].items() if key!='sample_indices'}
    if replay_contract!=json.loads(contract_json):
        raise ValueError('source or data changed during the bounded provenance replay')
    return replay


def provenance_supervision_blockers(config,audit):
    """A report's booleans never authorize source labels without actual replay."""
    evidence=audit.get('evidence',{})
    required_evidence=('source_ids_verified','normal_train_only','all_natural_fog_labels_masked',
                       'no_snow_or_validation_test','no_teacher_network')
    if not (audit.get('schema')=='project_provenance_supervision_audit_v2'
            and audit.get('split')=='train' and audit.get('supervision_source')=='provenance_v1'
            and audit.get('formal_positive_supervision_ready') is True
            and isinstance(evidence,dict) and evidence.get('kind')=='controlled_source_ids'
            and all(evidence.get(key) is True for key in required_evidence)
            and audit.get('positive_miss_cells',0)>0 and audit.get('positive_spur_cells',0)>0):
        return ['supervision audit has not verified both positive types and source evidence']
    contract=supervision_contract(config)
    report_contract=audit.get('contract',{})
    if not isinstance(report_contract,dict) or {key:value for key,value in report_contract.items() if key!='sample_indices'}!=contract:
        return ['supervision audit source/data/seed/augmentation contract mismatch']
    indices=audit.get('sample_indices')
    if (audit.get('seed')!=contract['seed'] or audit.get('epoch')!=0 or audit.get('samples')!=32
            or not isinstance(indices,list) or len(indices)!=32 or any(type(index) is not int for index in indices)
            or len(set(indices))!=32
            or indices!=report_contract.get('sample_indices') or not audit.get('rows')):
        return ['provenance audit must bind seed, epoch and 32 unique bounded train sample indices']
    try:
        replay=_bounded_provenance_replay(json.dumps(config,sort_keys=True),json.dumps(contract,sort_keys=True))
    except Exception as error:
        return [f'bounded train provenance replay failed: {type(error).__name__}: {error}']
    # Timing alone may differ. Every coverage count, source-ID/target digest,
    # agent/weather identity and sampled index must equal the real train replay.
    expected={key:value for key,value in replay.items() if key!='seconds'}
    supplied={key:value for key,value in audit.items() if key!='seconds'}
    if supplied!=expected or replay.get('formal_positive_supervision_ready') is not True:
        return ['provenance audit does not match actual bounded train replay (source IDs, targets, masks or totals)']
    return []


def formal_blockers(config):
    if config['version'] == 5:
        from opencood.project.supplement_runner import blockers
        return blockers(config)
    e, a, d = config["experiment"], config["artifacts"], config["data"]
    blockers = []
    uses_a = e["variant"] not in ("S0", "B")
    if e["track"] == "W" and uses_a:
        audit_path = d["supervision_audit"]
        if not audit_path or not Path(audit_path).is_file():
            blockers.append("W auxiliary supervision needs an audited source/matching semantics and positive coverage")
        else:
            audit = json.loads(Path(audit_path).read_text())
            if d["supervision_source"] == "provenance_v1":
                blockers.extend(provenance_supervision_blockers(config,audit))
            else:
                valid_source = audit.get("unmatched_semantics_verified") is True
                if not (audit.get("split") == "train" and valid_source
                        and audit.get("positive_miss_cells", 0) > 0 and audit.get("positive_spur_cells", 0) > 0
                        and audit.get("evidence")):
                    blockers.append("supervision audit has not verified both positive types and source evidence")
                expected_supervision_contract = supervision_contract(config)
                if audit.get("contract") != expected_supervision_contract:
                    blockers.append("supervision audit source/data/matching thresholds contract mismatch")
    if config["training"]["precision"] != "fp32":
        if not a["amp_audit"] or not Path(a["amp_audit"]).is_file():
            blockers.append("AMP requires a recorded FP32 consistency audit before formal comparisons")
        else:
            audit = json.loads(Path(a["amp_audit"]).read_text())
            reason = validate_amp_audit(config, audit)
            if reason:
                blockers.append(reason)
    return blockers


def preflight(config):
    paths = {"root": Path(config["data"]["root"]), "base_hypes": resolve_repo(config["data"]["base_hypes"]),
             "gt_manifest": Path(config["data"]["manifest_dir"]) / "eval_gt_manifest.json"}
    if config['version'] == 5 and config['experiment']['dataset_id'] != 'v2xr':
        paths['gt_manifest'] = Path(config['data']['manifest_dir']) / 'supplement_manifest.json'
    report = {"status": "preflight", "config_sha256": config_hash(config), "resolved_config": config,
              "paths": {key: {"path": str(path), "exists": path.exists()} for key, path in paths.items()},
              "formal_blockers": formal_blockers(config), "cuda_available": torch.cuda.is_available(),
              "gpu_count": torch.cuda.device_count(), "torch": torch.__version__, "python": platform.python_version(),
              "formal_experiment_started": False}
    for name in ("memory.max", "memory.current", "memory.stat", "memory.events", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            report.setdefault("cgroup", {})[name] = path.read_text().strip()
    return report


def validate_epoch(model, config, device, telemetry=None):
    from project_tools.export_predictions import export_loader
    from opencood.project.evaluation import evaluate_predictions
    scores, details = {}, {}
    domains = config['selection']['domains'] if config['version'] == 5 else (["normal"] if config["experiment"]["track"] == "C" else ["normal", "fog"])
    for weather in domains:
        started = time.perf_counter()
        stage_seconds, stage_calls = Counter(), Counter()
        def record_profile(row):
            stage_seconds[row["stage"]] += row["seconds"]
            stage_calls[row["stage"]] += 1
        dataset, loader = make_eval_loader(config, "validate", weather)
        if telemetry is not None:
            telemetry("validation_weather_start", split="validate", weather=weather, total_samples=len(dataset))
        progress_resources = {"loader": loader}
        def progress_batches():
            samples, batches = 0, 0
            iterator = iter(loader)
            progress_resources["iterator"] = iterator
            for batch in iterator:
                yield batch
                samples += len(batch['ego']['sample_ids'])
                batches += 1
                if telemetry is not None and (batches == 1 or batches % 25 == 0 or samples == len(dataset)):
                    telemetry("validation_progress", split="validate", weather=weather,
                              completed_samples=samples, total_samples=len(dataset), batches=batches,
                              seconds=time.perf_counter() - started,
                              stage_seconds=dict(stage_seconds), stage_calls=dict(stage_calls))
        try:
            records = export_loader(model, progress_batches() if telemetry is not None else loader, dataset, device,
                                    profile_callback=record_profile if telemetry is not None else None)
        finally:
            shutdown_loader(progress_resources)
        if telemetry is not None:
            telemetry("validation_scoring_start", split="validate", weather=weather, evaluated_samples=len(records),
                      seconds=time.perf_counter() - started)
        scoring_started = time.perf_counter()
        if config['version'] == 5:
            from opencood.project.supplement_evaluation import evaluate_supplement
            result = evaluate_supplement(dataset.manifest, records, 'validate', config['scoring'], sample_ids=dataset.sample_ids)
        else:
            result = evaluate_predictions(dataset.manifest, records, "validate")
        stage_seconds["scoring"] += time.perf_counter() - scoring_started
        stage_calls["scoring"] += 1
        scores[weather] = result["metrics"]["3d"]["0.7"]["ap"]
        details[weather] = result
        if telemetry is not None:
            telemetry("validation_weather_complete", split="validate", weather=weather,
                      evaluated_samples=len(records), ap_3d_07=scores[weather], seconds=time.perf_counter() - started,
                      stage_seconds=dict(stage_seconds), stage_calls=dict(stage_calls))
    return scores, details


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def shutdown_loader(resources):
    """Close workers before an exception traceback keeps its loader alive.

    PyTorch 2.1 has no public DataLoader.close; iterator shutdown is the same
    internal cleanup used by its destructor, invoked deterministically here.
    """
    iterator = resources.get("iterator")
    loader = resources.get("loader")
    if iterator is None and loader is not None:
        iterator = getattr(loader, "_iterator", None)
    try:
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
    finally:
        if loader is not None and hasattr(loader, "_iterator"):
            loader._iterator = None
        resources.clear()


def run(config, args):
    resources = {}
    try:
        return _run(config, args, resources)
    except Exception as error:
        telemetry = resources.get("telemetry")
        if telemetry is not None:
            telemetry("failure", error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        resources.pop("telemetry", None)
        for sig, handler in resources.pop("signal_handlers", {}).items():
            signal.signal(sig, handler)
        try:
            shutdown_loader(resources)
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def _run(config, args, resources):
    if args.mode == "train" and not args.allow_formal_run:
        raise ValueError("formal training is disabled by default; --allow-formal-run is required")
    blockers = formal_blockers(config)
    if args.mode == "train" and blockers:
        raise ValueError("formal readiness gates: " + "; ".join(blockers))
    if args.mode in ("smoke", "benchmark") and (args.steps <= 0 or args.warmup < 0 or args.steps + args.warmup > 200):
        raise ValueError("diagnostic mode is limited to 200 total optimizer updates including warmup")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t, e = config["training"], config["experiment"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if world != t["world_size"]:
        raise ValueError("configured world_size must match torchrun WORLD_SIZE")
    if config['version'] == 5:
        from opencood.project.supplement_runner import select_device
        device = select_device(config, local_rank)
    else:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(config["resources"]["cpu_threads"])
    seed_everything(e["seed"] + rank, t["deterministic"])
    dataset, model, criterion = build_components(config, args.mode)
    initialize = getattr(args, 'initialize_from', None)
    if config['version'] == 5:
        transfer = e['transfer']
        if transfer == 'zero_shot':
            raise ValueError('zero_shot is evaluation-only; target training is forbidden')
        if bool(initialize) != (transfer == 'fine_tuned'):
            raise ValueError('fine_tuned requires --initialize-from; retrained recipes start from random initialization')
        if initialize:
            saved = torch.load(initialize, map_location='cpu', weights_only=False)
            if saved.get('metadata', {}).get('checkpoint_role') != 'selected':
                raise ValueError('fine-tuning initialization requires selected source weights')
            model.load_state_dict(saved['model'], strict=True)
    if args.mode == "smoke":
        # Keep the dataset/collate contract; only shorten the sample manifest.
        subset = min(32, len(dataset))
    else:
        subset = len(dataset)
    sampler = EpochBatchSampler(subset, t["micro_batch_size"], t["accumulation_steps"], e["seed"], rank=rank, world_size=world)
    if sampler.total_batches == 0:
        raise ValueError("not enough samples for one complete effective batch")
    loader = make_loader(dataset, config, sampler=sampler)
    resources["loader"] = loader
    model.to(device)
    criterion.to(device)
    shared_state = {key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value) for key, value in model.state_dict().items()
                    if not key.startswith(("evidence_router.", "supported_residual.", "supervision_control.",
                        "native.evidence_router.", "native.supported_residual.", "local."))}
    initial_hash = tensor_state_hash(shared_state)
    optimizer_class = torch.optim.AdamW if t.get('optimizer', 'adam') == 'adamw' else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"], eps=t["adam_eps"])
    if config['version'] == 5:
        from opencood.project.supplement_runner import scheduler_for
        scheduler = scheduler_for(optimizer, config)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=t["milestones"], gamma=t["gamma"])
    amp_enabled = t["precision"] != "fp32"
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP protocol is CUDA-only")
    amp_dtype = torch.float16 if t["precision"] == "amp_fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(init_scale=FP16_INITIAL_SCALE, enabled=t["precision"] == "amp_fp16")
    state, metadata = TrainState(), artifact_metadata(config)
    if initialize:
        metadata['initialization_checkpoint_sha256'] = file_sha256(initialize)
    output = Path(config["artifacts"]["output_root"]) / e["run_id"] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    telemetry_started = time.perf_counter()
    telemetry_path = output / f"telemetry.rank{rank}.jsonl"
    def telemetry(event, **fields):
        if event == "failure":
            fields["batchnorm"] = batchnorm_snapshot(model)
        row = {"schema": "project_training_telemetry_v1", "event": event,
               "timestamp_utc": timestamp_utc(), "elapsed_seconds": time.perf_counter() - telemetry_started,
               "pid": os.getpid(), "rank": rank, "epoch": state.epoch,
               "optimizer_step": state.optimizer_steps, "micro_step": state.micro_steps,
               "samples_seen": state.samples_seen, "sampler_cursor": sampler.state_dict(),
               "cuda_memory": cuda_memory_snapshot(device), **fields}
        with telemetry_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    resources["telemetry"] = telemetry
    telemetry("startup", mode=args.mode, run_id=e["run_id"], variant=e["variant"], track=e["track"], seed=e["seed"],
              source_sha256=metadata["source_sha256"], config_sha256=config_hash(config),
              resume_requested=str(args.resume) if args.resume else None,
              allocator_counter_semantics="process-cumulative; recovered allocator OOMs/retries are not training failures")
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / f"latest.rank{rank}.pt"
        state = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, sampler, config, metadata)
        if world > 1:
            cursors = [None] * world
            torch.distributed.all_gather_object(cursors, (state.epoch, state.optimizer_steps, sampler.next_batch))
            if len(set(cursors)) != 1:
                raise ValueError("DDP checkpoint ranks were not saved at the same completed update")
        telemetry("resume_loaded", checkpoint=str(resume_path), checkpoint_role="resume", bn_statistics_role="online_running")
    elif (output / f"latest.rank{rank}.pt").exists():
        raise ValueError("output already has a checkpoint; pass --resume or choose a new run_id")
    if rank == 0:
        common_path = Path(config["artifacts"]["output_root"]) / "initializations" / f"v2xr_{e['training_protocol']}_shared_s{e['seed']}.pt"
        if config['version'] == 5:
            common_path = common_path.with_name(f"{e['dataset_id']}_{e['method']}_shared_s{e['seed']}.pt")
        if common_path.exists():
            previous = torch.load(common_path, map_location="cpu", weights_only=False)
            if previous["sha256"] != initial_hash:
                raise ValueError("shared initialization changed within a paired comparison")
        else:
            atomic_torch_save({"shared_state": shared_state, "sha256": initial_hash, "seed": e["seed"]}, common_path)
        (output / "state.json").write_text(json.dumps({"status": "training" if args.mode == "train" else "diagnostic_running", "mode": args.mode}))
        (output / "resolved_config.json").write_text(json.dumps(config, indent=2))
        (output / "run_metadata.json").write_text(json.dumps({**metadata, "shared_initialization_sha256": initial_hash,
            "diagnostic_only": args.mode != "train", "feature_scale_mode": config["model"]["scale_mode"],
            "teacher_networks": 0, "clean_feature_forwards": 'one extra reference encoder pass per microbatch' if config['version'] == 5 and e['method'] == 'mdd' else 0,
            "parameter_count": sum(p.numel() for p in model.parameters()), "torch": torch.__version__}, indent=2))
    telemetry("training_ready", shared_initialization_sha256=initial_hash)
    monitor_batchnorm(model, telemetry, phase="resume_loaded_online" if args.resume else "initialized_online")
    def commit_checkpoint(path, checkpoint_metadata=None):
        checkpoint_metadata = metadata if checkpoint_metadata is None else checkpoint_metadata
        role = checkpoint_metadata.get("checkpoint_role", "resume")
        monitor_batchnorm(model, telemetry, phase="checkpoint_" + role)
        save_checkpoint(path, model, optimizer, scheduler, scaler, state, sampler, config, checkpoint_metadata)
        telemetry("checkpoint_commit", checkpoint=str(path), checkpoint_role=role,
                  bn_statistics_role=checkpoint_metadata.get("bn_statistics_role", "online_running"),
                  checkpoint_bytes=Path(path).stat().st_size)
    network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
        find_unused_parameters=True) if world > 1 else model
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings, losses, completed = [], [], 0
    measured_phases = {name: [] for name in ("data", "compute", "optimizer")}
    stop_requested = [False]
    def request_stop(signum, frame):
        stop_requested[0] = True
    resources["signal_handlers"] = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    max_updates = args.warmup + args.steps if args.mode != "train" else None
    log_path = output / f"steps.rank{rank}.jsonl"
    while state.epoch < t["epochs"] and not stop_requested[0]:
        dataset.set_epoch(state.epoch)
        if hasattr(criterion, 'epoch'):
            criterion.epoch = state.epoch
        model.train()
        if t["normalization"] == "frozen_batchnorm":
            freeze_batchnorm(model)
        iterator = iter(loader)
        resources["iterator"] = iterator
        update_start = time.perf_counter()
        data_time = compute_time = optimizer_time = 0.0
        update_losses = []
        loss_term_sums = Counter()
        target_summary, weather_counts, sensor_points = Counter(), Counter(), Counter()
        agents_per_micro, unrepresented_samples = [], []
        assignment_versions = set()
        type_coverage = Counter()
        for batch in iterator:
            data_time += time.perf_counter() - update_start
            _sync(device)
            compute_start = time.perf_counter()
            ego = batch["ego"]
            agents_per_micro.append(int(ego["record_len"].sum()))
            for sample_id, diagnostic in zip(ego["sample_ids"], ego.get("target_diagnostics", [])):
                assignment_versions.add(diagnostic["assignment_version"])
                for key in ("num_gt", "num_positive", "num_negative", "num_ignored", "forced_anchor_count", "forced_collision_count"):
                    target_summary[key] += diagnostic[key]
                target_summary["zero_overlap_gt_count"] += len(diagnostic["zero_overlap_gt_indices"])
                target_summary["unrepresented_gt_count"] += len(diagnostic["unrepresented_gt_indices"])
                if diagnostic["unrepresented_gt_indices"]:
                    unrepresented_samples.append({"sample_id": sample_id, "indices": diagnostic["unrepresented_gt_indices"]})
            if len(assignment_versions) > 1:
                raise ValueError("mixed anchor-assignment protocols in one optimizer update")
            for agent_list in ego.get("metadata", []):
                for agent in agent_list:
                    if agent["weather"] not in ("normal", "fog") and not (config['version'] == 5 and e['dataset_id'] != 'v2xr'):
                        raise ValueError("a forbidden training weather reached the runner")
                    weather_counts[agent["weather"]] += 1
                    for modality in ("lidar", "radar"):
                        sensor_points[modality] += agent[modality + "_points"]
            for name in ("miss", "spur"):
                raw_targets = batch.get("train_targets", {})
                if name + "_valid" in raw_targets:
                    mask = raw_targets[name + "_valid"] > 0
                    type_coverage[name + "_cells"] += mask.numel()
                    type_coverage[name + "_valid_cells"] += int(mask.sum())
                    type_coverage[name + "_positive_cells"] += int(((raw_targets[name] > 0) & mask).sum())
            inputs = to_device(select_model_inputs(ego), device)
            if config['version'] == 5 and e['method'] == 'mdd':
                inputs['processed_de_lidar'] = to_device(ego['reference_lidar'], device)
            targets = to_device(batch.get("train_targets", {}), device)
            labels = to_device(ego["label_dict"], device)
            is_last_micro = (sampler.next_batch + 1) % t["accumulation_steps"] == 0
            sync_context = network.no_sync() if world > 1 and not is_last_micro else nullcontext()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    prediction = network(inputs)
                    loss = criterion(prediction, labels, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                scaler.scale(loss / t["accumulation_steps"]).backward()
            _sync(device)
            compute_time += time.perf_counter() - compute_start
            update_losses.append(float(loss.detach()))
            loss_term_sums.update({key: float(value) for key, value in criterion.loss_dict.items()})
            # Do not keep the previous micro-batch's dense aux feature maps
            # alive while constructing the next forward graph.
            del prediction, loss, inputs, targets, labels
            sampler.advance()
            state.micro_steps += 1
            state.samples_seen += t["micro_batch_size"] * world
            if is_last_micro:
                optimizer_start = time.perf_counter()
                scaler.unscale_(optimizer)
                bad_gradients = [name for name, parameter in model.named_parameters()
                                 if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                if bad_gradients:
                    raise FloatingPointError("nonfinite training gradients: " + json.dumps({
                        "parameters": bad_gradients, "scale": scaler.get_scale(), "epoch": state.epoch,
                        "optimizer_steps_completed": state.optimizer_steps, "micro_steps": state.micro_steps,
                        "precision": t["precision"]}))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _sync(device)
                optimizer_time += time.perf_counter() - optimizer_start
                state.optimizer_steps += 1
                completed += 1
                if world > 1:
                    stop_flag = torch.tensor(int(stop_requested[0]), device=device)
                    torch.distributed.all_reduce(stop_flag, op=torch.distributed.ReduceOp.MAX)
                    stop_requested[0] = bool(stop_flag.item())
                if args.mode == "train" and state.optimizer_steps % t["checkpoint_interval_updates"] == 0:
                    commit_checkpoint(output / f"latest.rank{rank}.pt")
                row = {"timestamp_utc": timestamp_utc(), "elapsed_seconds": time.perf_counter() - telemetry_started,
                       "cuda_memory": cuda_memory_snapshot(device),
                       "epoch": state.epoch, "optimizer_step": state.optimizer_steps, "loss": float(np.mean(update_losses)),
                       "data_seconds": data_time, "compute_seconds": compute_time, "optimizer_seconds": optimizer_time,
                       "effective_batch": t["effective_batch_size"], "lr": optimizer.param_groups[0]["lr"],
                       "clean_reference_agents_last_micro": int(ego['record_len'].sum()) if config['version'] == 5 and e['method'] == 'mdd' else 0,
                       "loss_terms": {key: value / len(update_losses) for key, value in loss_term_sums.items()},
                       "loss_term_aggregation": "mean_across_micro_batches", "counts_scope": "rank_local", "rank": rank,
                       "target_diagnostics": dict(target_summary), "unrepresented_samples": unrepresented_samples,
                       "assignment_versions": sorted(assignment_versions), "agent_weather_counts": dict(weather_counts),
                       "agents_per_micro": agents_per_micro, "sensor_points": dict(sensor_points), "type_coverage": dict(type_coverage)}
                with log_path.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                if completed > args.warmup:
                    timings.append(data_time + compute_time + optimizer_time)
                    losses.append(row["loss"])
                    for name, value in (("data", data_time), ("compute", compute_time), ("optimizer", optimizer_time)):
                        measured_phases[name].append(value)
                if stop_requested[0] or (max_updates and completed >= max_updates):
                    break
                data_time = compute_time = optimizer_time = 0.0
                update_losses = []
                loss_term_sums = Counter()
                target_summary, weather_counts, sensor_points = Counter(), Counter(), Counter()
                agents_per_micro, unrepresented_samples = [], []
                assignment_versions = set()
                type_coverage = Counter()
            update_start = time.perf_counter()
        if stop_requested[0] or (max_updates and completed >= max_updates):
            break
        if sampler.next_batch != sampler.total_batches:
            raise RuntimeError("epoch ended before full effective batch budget")
        state.epoch += 1
        scheduler.step()
        sampler.set_epoch(state.epoch)
        if args.mode == "train":
            if world > 1:
                torch.distributed.barrier()
            if rank == 0:
                # Refit only current-weight BN statistics on the registered
                # train stream. Both validation weather use this single state.
                # Save selected while those buffers are live; resume state is
                # saved after the context restores the original online EMA.
                telemetry("validation_start", split="validate", weather=["normal"] if e["track"] == "C" else ["normal", "fog"])
                with batchnorm_evaluation(model, config, device, telemetry) as bn_report:
                    scores, details = validate_epoch(model, config, device, telemetry)
                    selected = consider_checkpoint(state, e["track"], scores, state.epoch,
                        domains=config['selection']['domains'] if config['version'] == 5 else None)
                    if selected:
                        selected_metadata = {**metadata, "checkpoint_role": "selected"}
                        if config['version'] == 5 and bn_report is None:
                            selected_metadata['bn_statistics_role'] = 'native_running_frozen'
                        if bn_report is not None:
                            selected_metadata.update(bn_statistics_role="train_refit_cma_v1",
                                                     batchnorm_refit=deepcopy(bn_report))
                        commit_checkpoint(output / "selected.pt", selected_metadata)
                    if config['version'] == 5 and state.epoch in t['retain_epochs']:
                        retained = {**metadata, 'checkpoint_role': 'budget_snapshot',
                            'bn_statistics_role': 'train_refit_cma_v1' if bn_report else 'native_running_frozen',
                            'recipe': t['recipe'], 'snapshot_epoch': state.epoch,
                            'snapshot_kind': 'independent_recipe_endpoint' if state.epoch == t['epochs'] else 'long_recipe_intermediate'}
                        if bn_report:
                            retained['batchnorm_refit'] = deepcopy(bn_report)
                        commit_checkpoint(output / f"budget_epoch{state.epoch:03d}.pt", retained)
                if bn_report is not None:
                    details['batchnorm_refit'] = bn_report
                (output / f"validation_epoch{state.epoch:03d}.json").write_text(json.dumps(details, indent=2))
                telemetry("validation_complete", scores=scores, selected=selected,
                          selected_epoch=state.selected_epoch, best_score=state.best_score)
            if world > 1:
                selection = [state.best_score, state.selected_epoch, selected] if rank == 0 else [None, None, None]
                torch.distributed.broadcast_object_list(selection, src=0)
                state.best_score, state.selected_epoch, selected = selection
            commit_checkpoint(output / f"latest.rank{rank}.pt")

        if max_updates:
            # Diagnostic modes may reuse their small train-only sample set.
            continue
    commit_checkpoint(output / f"latest.rank{rank}.pt")
    report = {"status": "interrupted_resumable" if stop_requested[0] else ("diagnostic_completed" if args.mode != "train" else "checkpoint_selected"),
              "formal_experiment_started": args.mode == "train", "mode": args.mode,
              "passed": not stop_requested[0], "all_losses_and_gradients_finite": True,
              "variant": e["variant"], "completed_optimizer_updates": completed,
              "training_protocol": e["training_protocol"], "teacher_networks": 0,
              "clean_feature_forwards": state.micro_steps if config['version'] == 5 and e['method'] == 'mdd' else 0,
              "feature_scale_mode": config["model"]["scale_mode"],
              "audit_contract": amp_audit_contract(config),
              "measured_optimizer_updates": len(timings), "warmup_optimizer_updates": min(args.warmup, completed),
              "mean_optimizer_step_seconds": float(np.mean(timings)) if timings else None,
              "effective_samples_per_second": t["effective_batch_size"] / float(np.mean(timings)) if timings else None,
              "mean_phase_seconds": {name: float(np.mean(values)) if values else None for name, values in measured_phases.items()},
              "data_wait_fraction": float(sum(measured_phases["data"]) / sum(timings)) if timings else None,
              "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0,
              "training_micro_steps": state.micro_steps, "optimizer_steps": state.optimizer_steps, "output": str(output),
              "formal_blockers": blockers,
              "timing_scope": "real data + transfer + forward/backward + optimizer; excludes validation/checkpoint writes"}
    report["source_unchanged"] = report["audit_contract"]["source_sha256"] == metadata["source_sha256"]
    report["passed"] = report["passed"] and report["source_unchanged"]
    (output / f"report.rank{rank}.json").write_text(json.dumps(report, indent=2))
    telemetry("run_complete", status=report["status"], passed=report["passed"], selected_epoch=state.selected_epoch)
    if rank == 0:
        (output / "state.json").write_text(json.dumps({"status": report["status"], "mode": args.mode, "selected_epoch": state.selected_epoch}, indent=2))
        print(json.dumps(report, indent=2))
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "cscwd_configs/project_v4.yaml"))
    parser.add_argument("--mode", choices=["preflight", "smoke", "benchmark", "train"], default="preflight")
    parser.add_argument("--allow-formal-run", action="store_true")
    parser.add_argument("--resume", help="checkpoint file or run-mode directory containing per-rank checkpoints")
    parser.add_argument('--initialize-from', help='v5 fine_tuned source selected weights; never optimizer/resume state')
    parser.add_argument("--steps", type=int, default=None, help="measured optimizer updates; smoke=4, benchmark=100")
    parser.add_argument("--warmup", type=int, default=None, help="warmup optimizer updates; benchmark=20")
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (100 if args.mode == "benchmark" else 4)
    args.warmup = args.warmup if args.warmup is not None else (20 if args.mode == "benchmark" else 0)
    if args.steps <= 0 or args.warmup < 0:
        parser.error("steps must be positive and warmup nonnegative")
    config = load_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2))
    else:
        try:
            run(config, args)
        except Exception as exc:
            output = Path(config["artifacts"]["output_root"]) / config["experiment"]["run_id"] / args.mode
            if output.is_dir() and int(os.environ.get("RANK", "0")) == 0:
                checkpoints = sorted(str(path) for path in output.glob("latest.rank*.pt"))
                (output / "state.json").write_text(json.dumps({"status": "failed", "stage": args.mode,
                    "reason": str(exc), "last_recoverable_checkpoints": checkpoints}, indent=2))
            raise



if __name__ == "__main__":
    main()
