#!/usr/bin/env python3
"""Export every sample, including empty predictions, with stable anchor IDs.

Decoding intentionally does not call the legacy postprocessor, which applies
additional box filters and a score threshold that differ from project_fix_v1.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from opencood.project.training import select_model_inputs, to_device, seed_everything
from opencood.project.config import file_sha256


def validate_selected_batchnorm_checkpoint(saved, config, model):
    """Reject online/unverified v4 buffers before any formal inference or I/O.

    The caller has loaded the checkpoint strictly and checked its source/config
    artifact identity. This check binds its refit manifest to the actual buffers
    and registered input/statistics policy. Historical protocols keep their
    existing export behavior.
    """
    if config["version"] != 4:
        return None
    from opencood.project.batchnorm import batchnorm_statistics_hashes
    metadata = saved.get("metadata", {})
    policy = config["training"]["batchnorm"]
    report = metadata.get("batchnorm_refit")
    if (metadata.get("checkpoint_role") != "selected" or
            metadata.get("bn_statistics_role") != "train_refit_cma_v1" or
            policy.get("evaluation_statistics") != "train_refit_cma_v1" or
            not isinstance(report, dict)):
        raise ValueError("v4 export requires a selected checkpoint with a train_refit_cma_v1 manifest")
    expected_micro = policy["refit_samples"] // policy["refit_micro_batch_size"]
    if (report.get("policy") != "train_refit_cma_v1" or
            report.get("configuration_policy") != policy or
            report.get("dataset_split") != "train" or
            report.get("training_protocol") != config["experiment"]["training_protocol"] or
            report.get("training_track") != config["experiment"]["track"] or
            report.get("sampling") != policy["refit_sampling"] or
            report.get("refit_epoch") != policy["refit_epoch"] or
            report.get("input_augmentation") is not policy["refit_augment"] or
            report.get("sample_count") != policy["refit_samples"] or
            report.get("micro_batches") != expected_micro or
            report.get("optimizer_updates") != 0 or
            report.get("unused_bn") != [] or report.get("all_modules_eval_at_yield") is not True):
        raise ValueError("v4 selected BN manifest does not match the registered statistics/input policy")
    sample_ids, indices = report.get("input_sample_ids"), report.get("sample_indices")
    if (not isinstance(sample_ids, list) or len(sample_ids) != policy["refit_samples"] or
            any(not isinstance(sid, str) or not sid.startswith("train/") for sid in sample_ids) or
            len(set(sample_ids)) != len(sample_ids) or not isinstance(indices, list) or
            len(indices) != policy["refit_samples"] or any(type(i) is not int or i < 0 for i in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError("v4 selected BN manifest requires unique fixed training sample IDs/indices")
    weather = report.get("input_agent_weather")
    if (not isinstance(weather, list) or len(weather) != expected_micro or
            any(not isinstance(micro, list) or len(micro) != policy["refit_micro_batch_size"] for micro in weather) or
            any(not isinstance(scene, list) or not scene or
                any(item not in ("normal", "fog") for item in scene)
                for micro in weather for scene in micro)):
        raise ValueError("v4 selected BN manifest requires the original train Normal/Fog weather stream")
    contract = report.get("data_contract")
    required_contract = ("base_hypes_sha256", "gt_manifest_sha256", "data_manifest_sha256",
                         "split_manifest_sha256", "weather_manifest_sha256", "protocol_manifest_sha256",
                         "image_sizes_sha256")
    if (not isinstance(contract, dict) or set(contract) != set(required_contract) or
            any(key not in metadata or metadata[key] != contract[key] for key in required_contract)):
        raise ValueError("v4 selected BN manifest data contract does not match checkpoint artifacts")
    if any(module.training for module in model.modules()):
        raise ValueError("v4 selected export requires every module in eval mode")
    hashes = batchnorm_statistics_hashes(model)
    rows = report.get("bn")
    if not hashes or not isinstance(rows, dict) or set(rows) != set(hashes):
        raise ValueError("v4 selected BN manifest does not cover exactly the model's BN modules")
    modules = dict(model.named_modules())
    for name, digest in hashes.items():
        module, row = modules[name], rows[name]
        elements = row.get("elements_per_channel_by_call") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("statistics_sha256") != digest or
                row.get("calls") != expected_micro or row.get("num_batches_tracked") != expected_micro or
                row.get("used_original_statistics") is not False or
                int(module.num_batches_tracked) != expected_micro or module.eps != policy["eps"] or
                not isinstance(elements, list) or len(elements) != expected_micro or
                any(type(count) is not int or count <= 1 for count in elements) or
                not torch.isfinite(module.running_mean).all() or not torch.isfinite(module.running_var).all() or
                bool((module.running_var < 0).any())):
            raise ValueError(f"v4 selected BN statistics/count/policy mismatch: {name}")
    return report


def reference_decode_predictions(output, anchors, sample_ids, score_floor=0.001):
    """Unchanged dictionary decode, retained for exact validation comparisons."""
    deltas = output["rm"].float().permute(0, 2, 3, 1).contiguous().reshape(len(sample_ids), -1, 7)
    scores = output["psm"].float().sigmoid().permute(0, 2, 3, 1).contiguous().reshape(len(sample_ids), -1)
    anchors = torch.as_tensor(anchors, device=deltas.device, dtype=deltas.dtype).reshape(1, -1, 7)
    if anchors.shape[1] != deltas.shape[1] or scores.shape[1] != deltas.shape[1]:
        raise ValueError("prediction / anchor geometry mismatch")
    diagonal = torch.sqrt(anchors[..., 4].square() + anchors[..., 5].square())
    boxes = torch.empty_like(deltas)
    boxes[..., :2] = deltas[..., :2] * diagonal[..., None] + anchors[..., :2]
    boxes[..., 2] = deltas[..., 2] * anchors[..., 3] + anchors[..., 2]
    boxes[..., 3:6] = deltas[..., 3:6].exp() * anchors[..., 3:6]
    boxes[..., 6] = deltas[..., 6] + anchors[..., 6]
    boxes = boxes[..., [0, 1, 2, 5, 4, 3, 6]]  # xyzhwl-yaw -> xyzlwh-yaw
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("nonfinite detector output; never silently drop bad boxes")
    records = []
    for index, sample_id in enumerate(sample_ids):
        ids = (scores[index] >= score_floor).nonzero(as_tuple=False).flatten()
        kept_boxes = boxes[index, ids].detach().cpu().tolist()
        kept_scores = scores[index, ids].detach().cpu().tolist()
        records.append({"sample_id": str(sample_id), "predictions": [
            {"prediction_id": f"anchor_{anchor:07d}", "class_name": "Vehicle", "score": score, "box": box}
            for anchor, score, box in zip(ids.cpu().tolist(), kept_scores, kept_boxes)]})
    return records


def decode_prediction_arrays(output, anchors, sample_ids, score_floor=0.001):
    """Decode at the same FP32 precision without materializing candidate dicts."""
    deltas = output["rm"].float().permute(0, 2, 3, 1).contiguous().reshape(len(sample_ids), -1, 7)
    scores = output["psm"].float().sigmoid().permute(0, 2, 3, 1).contiguous().reshape(len(sample_ids), -1)
    anchors = torch.as_tensor(anchors, device=deltas.device, dtype=deltas.dtype).reshape(1, -1, 7)
    if anchors.shape[1] != deltas.shape[1] or scores.shape[1] != deltas.shape[1]:
        raise ValueError("prediction / anchor geometry mismatch")
    diagonal = torch.sqrt(anchors[..., 4].square() + anchors[..., 5].square())
    boxes = torch.empty_like(deltas)
    boxes[..., :2] = deltas[..., :2] * diagonal[..., None] + anchors[..., :2]
    boxes[..., 2] = deltas[..., 2] * anchors[..., 3] + anchors[..., 2]
    boxes[..., 3:6] = deltas[..., 3:6].exp() * anchors[..., 3:6]
    boxes[..., 6] = deltas[..., 6] + anchors[..., 6]
    boxes = boxes[..., [0, 1, 2, 5, 4, 3, 6]]
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("nonfinite detector output; never silently drop bad boxes")
    records = []
    for index, sample_id in enumerate(sample_ids):
        ids = (scores[index] >= score_floor).nonzero(as_tuple=False).flatten()
        records.append({"sample_id": str(sample_id),
                        "prediction_ids": [f"anchor_{anchor:07d}" for anchor in ids.cpu().tolist()],
                        "scores": scores[index, ids].detach().cpu().numpy(),
                        "boxes": boxes[index, ids].detach().cpu().numpy()})
    return records


def decode_predictions(output, anchors, sample_ids, score_floor=0.001):
    from opencood.project.evaluation import _predictions_from_arrays
    records = []
    for arrays in decode_prediction_arrays(output, anchors, sample_ids, score_floor):
        records.append({"sample_id": arrays["sample_id"], "predictions": _predictions_from_arrays(
            arrays["boxes"], arrays["scores"], arrays["prediction_ids"], range(len(arrays["boxes"])))})
    return records


@torch.inference_mode()
def export_loader(model, loader, dataset, device, *, postprocess=True,
                  implementation="optimized", profile_callback=None):
    """Export with optional exact, synchronized stage timings for diagnosis.

    The callback receives {stage, seconds}; CUDA synchronization happens only
    when profiling is requested. Both implementations emit identical records.
    """
    from opencood.project.evaluation import (postprocess_prediction_arrays,
                                            reference_postprocess_predictions)
    if implementation not in ("optimized", "reference"):
        raise ValueError("Unknown export implementation")

    def stage(name, operation):
        if profile_callback is None:
            return operation()
        import time
        synchronize = name in ("h2d", "forward", "decode") and torch.device(device).type == "cuda"
        if synchronize:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        value = operation()
        if synchronize:
            torch.cuda.synchronize(device)
        profile_callback({"stage": name, "seconds": time.perf_counter() - started})
        return value

    original_modes = {name: module.training for name, module in model.named_modules()}
    model.eval()
    records = []
    iterator = None
    try:
        iterator = iter(loader)
        while True:
            try:
                batch = stage("loader_wait", lambda: next(iterator))
            except StopIteration:
                break
            if any(module.training for module in model.modules()):
                raise ValueError("prediction export requires every module in eval mode")
            ego = batch["ego"]
            inputs = stage("h2d", lambda: to_device(select_model_inputs(ego), device))
            output = stage("forward", lambda: model(inputs, return_aux=False))
            array_path = postprocess and implementation == "optimized"
            decoder = decode_prediction_arrays if array_path else reference_decode_predictions
            decoded = stage("decode", lambda: decoder(output, dataset.anchor_box, ego["sample_ids"]))
            for batch_index, record in enumerate(decoded):
                if array_path:
                    # Keep serialized field order identical to the old decode.
                    record["predictions"] = None
                sample = dataset.manifest["samples"][record["sample_id"]]
                record.update({key: sample[key] for key in ("split", "sequence", "timestamp", "ego_id", "agent_ids", "gt_sha256")})
                record.update(dataset="v2xr", evaluation_weather=dataset.weather,
                              protocol_sha256=dataset.manifest["protocol_sha256"])
                agent_metadata = ego["metadata"][batch_index]
                record.update(source_agent_ids=sample["agent_ids"], agent_ids=[item["agent_id"] for item in agent_metadata],
                              evaluated_agent_ids=[item["agent_id"] for item in agent_metadata],
                              agent_weather={item["agent_id"]: item["weather"] for item in agent_metadata})
                if postprocess:
                    # The evaluator owns the single frozen postprocessing contract.
                    if array_path:
                        record["predictions"], record["filter_counts"] = stage("postprocess", lambda:
                            postprocess_prediction_arrays(record["boxes"], record["scores"], record["prediction_ids"], sample))
                        for key in ("boxes", "scores", "prediction_ids"):
                            del record[key]
                    else:
                        record["predictions"], record["filter_counts"] = stage("postprocess", lambda:
                            reference_postprocess_predictions(record["predictions"], sample))
            records.extend(decoded)
    finally:
        from project_tools.run_experiments import shutdown_loader
        try:
            for name, module in model.named_modules():
                module.training = original_modes[name]
        finally:
            shutdown_loader({"loader": loader, "iterator": iterator})
    return records


def write_predictions(records, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    metadata = {**metadata, "predictions_file_sha256": file_sha256(path)}
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cscwd_configs/project_v4.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validate", "validation", "test"], default="validate")
    parser.add_argument("--weather", choices=["normal", "fog", "snow"], default="normal")
    parser.add_argument("--execute", action="store_true", help="without this flag only verify arguments")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    args = parser.parse_args()
    args.split = "validate" if args.split == "validation" else args.split
    from opencood.project.config import load_config, file_sha256, config_hash
    from project_tools.run_experiments import build_model, make_eval_loader, artifact_metadata
    config = load_config(args.config)
    if args.weather == "snow" and args.split != "test":
        parser.error("Snow cannot be used for validation/model selection")
    if args.split == "test" and not args.allow_test_evaluation:
        parser.error("final test requires --allow-test-evaluation after method freezing")
    if not args.execute:
        print(json.dumps({"status": "preflight", **vars(args)}, indent=2))
        return
    # Standalone export must use the same deterministic FP32 reference as
    # in-run validation, including the explicit TF32 policy.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["resources"]["cpu_threads"])
    seed_everything(config["experiment"]["seed"], config["training"]["deterministic"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config, mode="export")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema_version") != config["version"] or saved.get("training_protocol") != config["experiment"]["training_protocol"] or saved["config_sha256"] != config_hash(config) or saved["train_state"]["selected_epoch"] < 0 or saved.get("metadata", {}).get("checkpoint_role") != "selected" or saved["train_state"]["epoch"] != saved["train_state"]["selected_epoch"] or saved.get("sampler", {}).get("next_batch") != 0:
        raise ValueError("export requires a selected checkpoint with matching configuration")
    expected_artifacts = artifact_metadata(config)
    for key, value in expected_artifacts.items():
        if saved.get("metadata", {}).get(key) != value:
            raise ValueError(f"checkpoint source/configured artifact mismatch: {key}")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    validate_selected_batchnorm_checkpoint(saved, config, model)
    model.to(device)
    dataset, loader = make_eval_loader(config, args.split, args.weather)
    records = export_loader(model, loader, dataset, device)
    from opencood.project.evaluation import default_protocol, postprocess_hash
    write_predictions(records, args.output, {"config_sha256": config_hash(config), "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_protocol": config["experiment"]["training_protocol"], "protocol": "project_fix_v1", "protocol_sha256": default_protocol()["protocol_sha256"], "postprocess_sha256": postprocess_hash(),
        "gt_manifest_sha256": file_sha256(Path(config["data"]["manifest_dir"]) / "eval_gt_manifest.json"),
        "source_kind": "retrained", "dataset": "v2xr", "split": args.split, "evaluation_weather": args.weather,
        "checkpoint_epoch": saved["train_state"]["selected_epoch"], "selected_epoch": saved["train_state"]["selected_epoch"],
        "postprocessed": True, "sample_count": len(records), "inference_precision": "fp32", "tf32": False})
    print(json.dumps({"status": "predictions_exported", "samples": len(records), "output": args.output}))


if __name__ == "__main__":
    main()
