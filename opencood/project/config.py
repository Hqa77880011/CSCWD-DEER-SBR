"""Strict, canonical project configuration. Legacy hypes live in a separate file."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import yaml

DEFAULTS = {
    "version": 2,
    "experiment": {"run_id": "v2xr_tf2_C_AB_s17", "dataset_id": "v2xr", "protocol": "project_fix_v1",
                   "training_protocol": "teacher_free_v2", "track": "C", "seed": 17, "variant": "AB"},
    "data": {"root": "/root/autodl-tmp/CSCWD/datasets/v2xr", "manifest_dir": "/root/autodl-tmp/CSCWD/datasets/v2xr",
             "base_hypes": "cscwd_configs/baseline_full_data_check.yaml", "paired_supervision": False,
             "supervision_source": "none", "supervision_audit": None,
             "controlled_corruption": {"probability": 0.5, "drop_fraction": 0.2, "inject_fraction": 0.05, "max_offset_m": 0.3},
             "matching": {"max_distance": 0.002, "ambiguity_margin": 0.001, "unmatched_min_distance": 0.05}},
    "training": {"epochs": 30, "effective_batch_size": 8, "micro_batch_size": 4, "world_size": 1, "accumulation_steps": 2,
                 "lr": 0.002, "weight_decay": 0.0001, "adam_eps": 1e-10, "milestones": [10, 15], "gamma": 0.1,
                 "checkpoint_interval_updates": 100, "precision": "fp32", "normalization": "batchnorm", "deterministic": True,
                 "batchnorm": {"eps": 0.001, "momentum": 0.01, "evaluation_statistics": "running",
                               "refit_samples": 0, "refit_sampling": "uniform_midpoints_v1", "refit_epoch": 0,
                               "refit_augment": False, "refit_micro_batch_size": 4}},
    "resources": {"num_workers": 8, "prefetch_factor": 2, "pin_memory": True, "persistent_workers": True,
                  "raw_cache_gib_per_worker": 0.5, "ram_budget_gib": 64, "cpu_threads": 4},
    "model": {"hidden_channels": 16, "attention_channels": 8, "rho": 0.25, "g_min": 0.25,
              "scale_mode": "current_observation_rms", "scale_floor": 0.001, "scale_ceiling": 1000.0,
              "support_radius_m": 0.8, "route_strength": 0.5, "activation_checkpoint": True,
              "amplitude_context": "none", "type_prior_mode": "multiplicative", "efficient_features": False},
    "loss": {"type_weight": 0.1, "correction_weight": 0.001, "identity_weight": 0.01},
    "artifacts": {"output_root": "/root/autodl-tmp/CSCWD/outputs/teacher_free_v2", "amp_audit": None},
    "selection": {"metric": "ap3d", "iou": 0.7, "tie_tolerance": 1e-6},
    "scoring": {"iou_thresholds": [0.3, 0.5, 0.7], "score_floor": 0.001, "nms_bev_iou": 0.15, "max_detections": 500},
}


def _merge(default, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a mapping")
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError(f"unknown configuration key(s): {', '.join(prefix + str(k) for k in sorted(unknown))}")
    result = deepcopy(default)
    for key, value in supplied.items():
        if isinstance(default[key], dict):
            result[key] = _merge(default[key], value, prefix + key + ".")
        else:
            expected = default[key]
            if expected is not None:
                if isinstance(expected, bool):
                    valid = type(value) is bool
                elif isinstance(expected, int):
                    valid = type(value) is int
                elif isinstance(expected, float):
                    valid = type(value) in (int, float)
                else:
                    valid = isinstance(value, type(expected))
                if not valid:
                    raise ValueError(f"{prefix + key} has the wrong value type")
            elif value is not None and not isinstance(value, str):
                raise ValueError(f"{prefix + key} must be a path string or null")
            result[key] = value
    return result


def validate_config(config):
    defaults = deepcopy(DEFAULTS)
    if config.get("version") in (3, 4):
        version = config["version"]
        defaults["version"] = version
        defaults["experiment"].update(training_protocol=f"teacher_free_v{version}", run_id=f"v2xr_tf{version}_C_AB_s17")
        defaults["model"].update(amplitude_context="rms", type_prior_mode="input_only", efficient_features=True)
        defaults["artifacts"]["output_root"] = f"/root/autodl-tmp/CSCWD/outputs/teacher_free_v{version}"
        if version == 4:
            defaults["training"]["batchnorm"].update(evaluation_statistics="train_refit_cma_v1", refit_samples=32)
    c = _merge(defaults, config)
    def finite_values(value):
        if isinstance(value, dict):
            return all(finite_values(item) for item in value.values())
        if isinstance(value, list):
            return all(finite_values(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite_values(c):
        raise ValueError("configuration contains a nonfinite number")
    e, t, d = c["experiment"], c["training"], c["data"]
    if (c["version"] not in (2, 3, 4) or e["protocol"] != "project_fix_v1" or e["dataset_id"] != "v2xr"
            or e["training_protocol"] != f"teacher_free_v{c['version']}"):
        raise ValueError("config version must match teacher_free_v2/v3/v4; scoring remains project_fix_v1; teacher v1 is archived")
    if e["seed"] not in (17, 29, 43) or e["track"] not in ("C", "W") or e["variant"] not in ("S0", "A", "B", "AB", "SUP", "M1", "M2", "M3", "M4", "M5", "M6", "N1", "N2"):
        raise ValueError("invalid registered seed/track/variant")
    if not isinstance(e["run_id"], str) or not e["run_id"] or Path(e["run_id"]).name != e["run_id"]:
        raise ValueError("run_id must be a single nonempty directory name")
    for key in ("epochs", "effective_batch_size", "micro_batch_size", "world_size", "accumulation_steps", "checkpoint_interval_updates"):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    registered_epochs = (20, 30) if c["version"] == 4 else (30,)
    if t["epochs"] not in registered_epochs or t["effective_batch_size"] != 8:
        budget = "20 or 30" if c["version"] == 4 else "30"
        raise ValueError(f"registered V2X-R v{c['version']} budget is {budget} epochs / effective batch 8; smoke limits are CLI-only")
    if t["micro_batch_size"] * t["world_size"] * t["accumulation_steps"] != t["effective_batch_size"]:
        raise ValueError("micro_batch_size * world_size * accumulation_steps must equal effective_batch_size")
    if t["precision"] not in ("fp32", "amp_fp16", "amp_bf16") or t["normalization"] not in ("batchnorm", "frozen_batchnorm"):
        raise ValueError("unsupported precision/normalization")
    if t["batchnorm"] != defaults["training"]["batchnorm"]:
        raise ValueError("batchnorm policy must match the separately registered training protocol")
    if c["version"] == 4 and (t["precision"] != "fp32" or t["normalization"] != "batchnorm"):
        raise ValueError("teacher_free_v4 requires FP32 and online batchnorm with train-only evaluation refit")
    if t["lr"] <= 0 or t["weight_decay"] < 0 or not 0 < t["gamma"] <= 1:
        raise ValueError("invalid optimizer values")
    if not isinstance(t["milestones"], list) or any(type(m) is not int or m <= 0 or m >= t["epochs"] for m in t["milestones"]) or t["milestones"] != sorted(set(t["milestones"])):
        raise ValueError("milestones must be increasing unique epochs before final epoch")
    if e["track"] == "C" and d["paired_supervision"]:
        raise ValueError("C line is clean only and must not use paired weather supervision")
    matching = d["matching"]
    if not 0 < matching["max_distance"] < matching["unmatched_min_distance"] or matching["ambiguity_margin"] < 0:
        raise ValueError("invalid frozen correspondence thresholds")
    if d["supervision_source"] not in ("none", "mutual_nn_v1", "provenance_v1"):
        raise ValueError("unknown/unimplemented supervision source")
    if d["paired_supervision"] and d["supervision_source"] != "mutual_nn_v1":
        raise ValueError("clean point-array pairs are only used for mutual_nn_v1 type targets; no teacher inputs exist")
    if e["track"] == "C" and d["supervision_source"] != "none":
        raise ValueError("C line must remain clean; provenance perturbations belong to the separately registered W protocol")
    if e["track"] == "W" and e["variant"] not in ("S0", "B") and d["supervision_source"] == "none":
        raise ValueError("W DEER branches require an explicit trustworthy type supervision source")
    corruption = d["controlled_corruption"]
    if (not 0 <= corruption["probability"] <= 1 or not 0 <= corruption["drop_fraction"] < 1
            or not 0 <= corruption["inject_fraction"] <= 1 or corruption["max_offset_m"] <= 0):
        raise ValueError("invalid registered train-only provenance perturbation parameters")
    if d["supervision_source"] == "provenance_v1" and (corruption["probability"] <= 0 or corruption["drop_fraction"] <= 0 or corruption["inject_fraction"] <= 0):
        raise ValueError("provenance_v1 must permit both removed and injected source-ID labels")
    if c["selection"] != DEFAULTS["selection"] or c["scoring"] != DEFAULTS["scoring"]:
        raise ValueError("project_fix_v1 selection/scoring settings are frozen")
    model = c["model"]
    if c["version"] == 2:
        if (model["amplitude_context"] != "none" or model["type_prior_mode"] != "multiplicative"
                or model["efficient_features"] or e["variant"] in ("N1", "N2")):
            raise ValueError("v3 mechanisms require a separately registered version 3 or 4 training protocol")
    else:
        expected_context = "normalized_mean" if e["variant"] == "N1" else "rms"
        expected_prior = "multiplicative" if e["variant"] == "N2" else "input_only"
        if model["amplitude_context"] != expected_context or model["type_prior_mode"] != expected_prior:
            raise ValueError("v3/v4 mechanism changes must use the registered N1/N2 ablations")
        if t["precision"] != "fp32":
            raise ValueError(f"teacher_free_v{c['version']} is registered only for stable FP32")
    r = c["resources"]
    if type(r["num_workers"]) is not int or r["num_workers"] < 0 or r["prefetch_factor"] <= 0 or r["raw_cache_gib_per_worker"] < 0 or r["cpu_threads"] <= 0:
        raise ValueError("invalid resource settings")
    if max(1, r["num_workers"]) * r["raw_cache_gib_per_worker"] > r["ram_budget_gib"] / 2:
        raise ValueError("worker raw caches must fit within half the host RAM budget, leaving room for prefetch/model")
    if r["ram_budget_gib"] <= 0 or t["adam_eps"] <= 0 or any(v < 0 for v in c["loss"].values()):
        raise ValueError("RAM budget / Adam epsilon must be positive and losses nonnegative")
    if c["model"]["hidden_channels"] <= 0 or c["model"]["attention_channels"] <= 0 or c["model"]["rho"] <= 0 or not 0 <= c["model"]["g_min"] <= 1:
        raise ValueError("invalid module dimensions or residual bounds")
    model = c["model"]
    if (model["scale_mode"] != "current_observation_rms" or not 0 < model["scale_floor"] < model["scale_ceiling"]
            or not 0 <= model["route_strength"] < 1 or model["support_radius_m"] <= 0):
        raise ValueError("invalid teacher-free dynamic scale, route floor or physical support radius")
    if r["num_workers"] == 0:
        r["persistent_workers"] = False
    for key in ("paired_supervision",):
        if type(d[key]) is not bool:
            raise ValueError(f"data.{key} must be boolean")
    return c


def load_config(path):
    with open(path, encoding="utf8") as stream:
        return validate_config(yaml.safe_load(stream))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SCIENTIFIC_TRAINING_KEYS = (
    "epochs", "effective_batch_size", "micro_batch_size", "world_size", "accumulation_steps",
    "lr", "weight_decay", "adam_eps", "milestones", "gamma", "precision", "normalization", "batchnorm", "deterministic",
)


def scientific_training_contract(config):
    """Settings that must agree across matched method comparisons.

    Variant, run identifiers and checkpoint frequency intentionally differ;
    the training line and data perturbations must agree within a comparison.
    Geometry and implementation hashes are validated separately.
    """
    try:
        return {"dataset_id": config["experiment"]["dataset_id"], "protocol": config["experiment"]["protocol"],
                "training_protocol": config["experiment"]["training_protocol"],
                "seed": config["experiment"]["seed"], "track": config["experiment"]["track"],
                "model": deepcopy(config["model"]),
                "loss": deepcopy(config["loss"]),
                "data": {key: deepcopy(config["data"][key]) for key in ("paired_supervision", "supervision_source", "matching", "controlled_corruption")},
                "training": {key: deepcopy(config["training"][key]) for key in SCIENTIFIC_TRAINING_KEYS}}
    except KeyError as exc:
        raise ValueError(f"scientific training contract is missing {exc.args[0]}") from exc


def require_matching_training_contract(reference, current, artifact):
    if scientific_training_contract(reference) != scientific_training_contract(current):
        raise ValueError(f"{artifact} scientific training contract mismatch (budget, micro-batch, optimizer, precision, normalization or loss)")
