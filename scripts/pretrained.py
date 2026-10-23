#!/usr/bin/env python3
"""Download and evaluate the original selected checkpoints without retraining."""
import argparse
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
VARIANTS = ("S0", "A", "B", "AB", "SUP", "N1", "N2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"Missing file or SHA256/size mismatch: {path}")


def load_release():
    release = json.loads((REPO / "repro/pretrained_release.json").read_text())
    if release["schema"] != "cscwd_pretrained_release_v1" or set(release["variants"]) != set(VARIANTS):
        raise ValueError("Unsupported pretrained release manifest")
    return release


def download(release, directory, variants):
    directory.mkdir(parents=True, exist_ok=True)
    names = [release["metadata_asset"], *(release["variants"][v]["asset"] for v in variants)]
    for name in names:
        target = directory / name
        if target.exists():
            verify_file(target, release["assets"][name])
            print(f"Already verified: {name}", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix=".download-", dir=directory) as temporary:
            subprocess.run(["gh", "release", "download", release["tag"], "--repo", release["repository"],
                            "--pattern", name, "--dir", temporary], check=True)
            path = Path(temporary) / name
            verify_file(path, release["assets"][name])
            path.replace(target)
        print(f"Downloaded and verified: {name}", flush=True)


def unpack_metadata(release, directory):
    archive = directory / release["metadata_asset"]
    verify_file(archive, release["assets"][archive.name])
    destination = directory / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) != len(release["metadata_files"]) or {m.name for m in members} != set(release["metadata_files"]):
            raise ValueError("Metadata archive members differ from the release manifest")
        for member in members:
            expected = release["metadata_files"][member.name]
            if not member.isfile() or Path(member.name).name != member.name or member.size != expected["bytes"]:
                raise ValueError("Unsafe or unexpected metadata member")
            target = destination / member.name
            if target.exists():
                verify_file(target, expected)
                continue
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
                    temporary = Path(output.name)
                    with source.extractfile(member) as stream:
                        shutil.copyfileobj(stream, output)
                    output.flush()
                # Windows cannot rename a NamedTemporaryFile while its handle
                # remains open. Verify and commit after closing that handle.
                verify_file(temporary, expected)
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return destination


def dataset_view(metadata, raw_root, output_root):
    """Keep original metadata bytes while relocating only the raw-data directory."""
    raw_root = raw_root.resolve()
    if not (raw_root / "test").is_dir():
        raise FileNotFoundError(f"Missing original test point clouds/annotations: {raw_root / 'test'}")
    view = output_root / "data"
    view.mkdir(parents=True, exist_ok=True)
    targets = {path.name: path.resolve() for path in metadata.glob("*.json")}
    targets.update({split: raw_root / split for split in ("train", "validate", "test") if (raw_root / split).is_dir()})
    for name, target in targets.items():
        link = view / name
        if link.exists() or link.is_symlink():
            if not link.samefile(target):
                raise ValueError(f"Existing dataset view points elsewhere: {link}; choose a new output root")
        else:
            try:
                link.symlink_to(target, target_is_directory=target.is_dir())
            except OSError as error:
                if os.name != 'nt' or error.winerror != 1314:
                    raise
                # Junctions and file hard links preserve the same source data
                # without requiring Windows' symbolic-link privilege.
                if target.is_dir():
                    import _winapi
                    _winapi.CreateJunction(str(target), str(link))
                else:
                    os.link(target, link)
    return view.resolve()


def relocated_config(original, view, metadata, output_root):
    config = deepcopy(original)
    config["data"].update(root=str(view), manifest_dir=str(view),
                          supervision_audit=str(metadata / "provenance_audit.json"))
    config["artifacts"]["output_root"] = str(output_root)
    return config


def verify_parallel_pipeline(release):
    proof = json.loads((REPO / "repro/pretrained_parallel_equivalence.json").read_text())
    if (not proof.get("passed") or proof.get("source_sha256") != release["source_sha256"]
            or proof.get("pipeline_sha256") != sha256(REPO / "scripts/parallel_export.py")
            or any(not proof.get("weather", {}).get(weather, {}).get("exact_full_result_equal")
                   for weather in ("normal", "fog"))):
        raise ValueError("Parallel export source/equivalence evidence changed")


def load_selected(release, directory, variant, config):
    """Prove the original checkpoint/config identity before accepting path relocation."""
    import torch
    from opencood.project.config import config_hash, load_config
    from project_tools.run_experiments import artifact_metadata, build_model, source_tree_hash
    from project_tools.export_predictions import validate_selected_batchnorm_checkpoint
    entry = release["variants"][variant]
    path = directory / entry["asset"]
    verify_file(path, release["assets"][entry["asset"]])
    config_path = REPO / "repro/configs" / f"{variant}.yaml"
    if sha256(config_path) != entry["config_file_sha256"]:
        raise ValueError("Original experiment configuration snapshot changed")
    original = load_config(config_path)
    if config_hash(original) != entry["config_sha256"] or source_tree_hash() != release["source_sha256"]:
        raise ValueError("Original configuration or scientific Python source identity changed")
    # Only three data paths and the output path may differ from the frozen config.
    restored = deepcopy(config)
    for key in ("root", "manifest_dir", "supervision_audit"):
        restored["data"][key] = original["data"][key]
    restored["artifacts"]["output_root"] = original["artifacts"]["output_root"]
    if restored != original:
        raise ValueError("Pretrained evaluation permits path relocation only")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("train_state", {})
    if (saved.get("schema_version") != 4 or saved.get("training_protocol") != "teacher_free_v4"
            or saved.get("config") != original or saved.get("config_sha256") != entry["config_sha256"]
            or state.get("epoch") != entry["selected_epoch"] or state.get("selected_epoch") != entry["selected_epoch"]
            or saved.get("sampler", {}).get("next_batch") != 0):
        raise ValueError("Checkpoint is not the recorded selected experiment state")
    for key, value in artifact_metadata(config).items():
        if saved.get("metadata", {}).get(key) != value:
            raise ValueError(f"Original checkpoint artifact identity mismatch: {key}")
    model = build_model(config, mode="export")
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    validate_selected_batchnorm_checkpoint(saved, config, model)
    return model


def evaluate(release, directory, metadata, variants, args):
    import time
    import torch
    from opencood.project.config import config_hash, load_config
    from opencood.project.training import seed_everything
    from opencood.project.batchnorm import batchnorm_statistics_hashes
    from opencood.project.evaluation import default_protocol, postprocess_hash, evaluate_predictions, validate_archive_metadata
    from project_tools.export_predictions import export_loader, write_predictions
    from project_tools.run_experiments import make_eval_loader
    if not torch.cuda.is_available():
        raise RuntimeError("Pretrained full evaluation requires a CUDA GPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    view = dataset_view(metadata, args.data_root, args.output_root)
    device = torch.device("cuda:0")
    weather = ("normal", "fog", "snow") if args.weather == "all" else (args.weather,)
    for variant in variants:
        original = load_config(REPO / "repro/configs" / f"{variant}.yaml")
        config = relocated_config(original, view, metadata, args.output_root)
        torch.set_num_threads(config["resources"]["cpu_threads"])
        seed_everything(config["experiment"]["seed"], True)
        model = load_selected(release, directory, variant, config).to(device)
        bn_before = batchnorm_statistics_hashes(model)
        for condition in weather:
            started = time.perf_counter()
            output = args.output_root / variant / condition
            output.mkdir(parents=True, exist_ok=True)
            predictions = output / "predictions.jsonl"
            if predictions.exists() or (output / "summary.json").exists():
                raise FileExistsError(f"Existing evaluation output: {output}; choose another --output-root")
            dataset, loader = make_eval_loader(config, "test", condition)
            completed = [0]
            def progress(row):
                if row["stage"] == "postprocess":
                    completed[0] += 1
                    if completed[0] % 100 == 0:
                        print(f"{variant}/{condition}: {completed[0]}/{len(dataset)} samples", flush=True)
            if args.postprocess_workers == 1:
                records = export_loader(model, loader, dataset, device, profile_callback=progress)
            else:
                verify_parallel_pipeline(release)
                from parallel_export import export_loader_parallel
                records = export_loader_parallel(model, loader, dataset, device, profile_callback=progress,
                    workers=args.postprocess_workers, max_inflight=4 * args.postprocess_workers)
            if batchnorm_statistics_hashes(model) != bn_before:
                raise ValueError("Selected BN buffers changed during inference")
            entry = release["variants"][variant]
            archive_metadata = {"config_sha256": config_hash(original), "runtime_config_sha256": config_hash(config),
                "checkpoint_sha256": entry["checkpoint_sha256"], "source_sha256": release["source_sha256"],
                "training_protocol": "teacher_free_v4", "protocol": "project_fix_v1",
                "protocol_sha256": default_protocol()["protocol_sha256"], "postprocess_sha256": postprocess_hash(),
                "gt_manifest_sha256": release["metadata_files"]["eval_gt_manifest.json"]["sha256"],
                "source_kind": "retrained", "dataset": "v2xr", "split": "test", "evaluation_weather": condition,
                "checkpoint_epoch": entry["selected_epoch"], "selected_epoch": entry["selected_epoch"],
                "postprocessed": True, "sample_count": len(records), "inference_precision": "fp32", "tf32": False,
                "release": release["tag"], "path_relocation_only": True}
            write_predictions(records, predictions, archive_metadata)
            validate_archive_metadata(archive_metadata, records, dataset.manifest, "test", archive_metadata["gt_manifest_sha256"])
            result = evaluate_predictions(dataset.manifest, records, "test")
            if result["num_samples"] != release["expected_test_samples"] or result["num_gt"] != release["expected_test_gt"]:
                raise ValueError("Incomplete evaluation sample/GT coverage")
            summary = {"variant": variant, "weather": condition, "num_samples": result["num_samples"],
                "num_gt": result["num_gt"], "selected_epoch": entry["selected_epoch"],
                "checkpoint_sha256": entry["checkpoint_sha256"], "bn_unchanged": True,
                "AP3D": {key: value["ap"] for key, value in result["metrics"]["3d"].items()},
                "APBEV": {key: value["ap"] for key, value in result["metrics"]["bev"].items()},
                "elapsed_seconds": time.perf_counter() - started}
            with (REPO / "results/reference/paper_test_summary.csv").open() as stream:
                reference = next(row for row in csv.DictReader(stream) if row["variant"] == variant)
            reference_ap = float(reference[f"{condition.title()}_AP3D07_percent"]) / 100
            summary["reference_AP3D07"] = reference_ap
            summary["AP3D07_delta_pp"] = 100 * (summary["AP3D"]["0.7"] - reference_ap)
            summary["reference_AP3D07_matches_1e_6"] = abs(summary["AP3D"]["0.7"] - reference_ap) <= 1e-6
            result["archive_metadata"] = archive_metadata
            with gzip.open(output / "scores.json.gz", "wt", encoding="utf8") as stream:
                json.dump(result, stream, allow_nan=False)
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
            del records, result, dataset, loader
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "verify", "evaluate"))
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="AB")
    parser.add_argument("--weather", choices=("normal", "fog", "snow", "all"), default="all")
    parser.add_argument("--weights-dir", type=Path, default=REPO / "checkpoints/v1.0.0")
    parser.add_argument("--data-root", type=Path, default=REPO / "datasets/v2xr")
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/pretrained_evaluation")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument("--postprocess-workers", type=int, default=8,
                        help="Exact CPU postprocessing workers; 1 uses the serial reference exporter")
    args = parser.parse_args()
    if args.postprocess_workers < 1:
        parser.error("--postprocess-workers must be positive")
    if args.action == "evaluate" and not args.allow_test_evaluation:
        parser.error("Frozen final test requires --allow-test-evaluation")
    args.weights_dir, args.output_root = args.weights_dir.resolve(), args.output_root.resolve()
    release = load_release()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.action == "download":
        download(release, args.weights_dir, variants)
    metadata = unpack_metadata(release, args.weights_dir)
    for variant in variants:
        name = release["variants"][variant]["asset"]
        verify_file(args.weights_dir / name, release["assets"][name])
        print(f"PASS original {variant} checkpoint SHA256", flush=True)
    if args.action == "evaluate":
        evaluate(release, args.weights_dir, metadata, variants, args)


if __name__ == "__main__":
    from legacy_runtime import execute_original
    execute_original('pretrained.py')
