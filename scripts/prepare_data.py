#!/usr/bin/env python3
"""Download the pinned release and generate the project's exact manifest schemas."""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
REVISION = "4c9ff1773b49517464714368d30778b5252bdbd0"
SUFFIXES = {"normal": ".pcd", "radar": "_radar.pcd", "fog": "_fog_0.060.pcd",
            "snow": "_snow_2.5_2.0.pcd", "annotation": ".yaml"}


def normalize_manifests(root, reference=REPO / "repro/metadata"):
    """Convert verified archive receipts to the frame schema consumed by fixed GT."""
    from opencood.project.fixed_gt import atomic_json
    entries, images = {}, {}
    archives = sorted((root / "manifests/archives").glob("*.json"))
    remote = json.loads((reference / "remote_files.json").read_text())
    expected = {entry["path"]: entry for entry in remote["files"] if entry["path"].endswith(".7z")}
    received = set()
    for archive in archives:
        receipt = json.loads(archive.read_text())
        name = receipt["archive"]
        if name in received or name not in expected or receipt["revision"] != REVISION:
            raise ValueError(f"Unexpected or duplicate archive receipt: {name}")
        source = expected[name]
        if source.get("lfs") and receipt["archive_sha256"] != source["lfs"]["oid"]:
            raise ValueError(f"Archive identity mismatch: {name}")
        received.add(name)
        for entry in receipt["retained_files"]:
            path = entry["path"]
            if path in entries:
                raise ValueError(f"Duplicate source file: {path}")
            if not (root / path).is_file() or (root / path).stat().st_size != entry["bytes"]:
                raise ValueError(f"Missing or truncated retained file: {path}")
            entries[path] = entry
        images.update(receipt["image_sizes"])
    if received != set(expected):
        raise ValueError(f"Incomplete archives: {len(received)}/{len(expected)}")
    frames = []
    counts = collections.defaultdict(set)
    sensor_counts = collections.Counter()
    for name in sorted(entries):
        if not name.endswith(".yaml"):
            continue
        path = Path(name)
        if path.name == "data_protocol.yaml":
            continue
        split, sequence, agent, filename = path.parts
        stamp = path.stem
        prefix = path.with_suffix("").as_posix()
        modalities = {key: entries[prefix + suffix] for key, suffix in SUFFIXES.items()}
        for index in range(4):
            if prefix + f"_camera{index}.png" not in images:
                raise ValueError(f"Missing camera dimensions: {prefix}")
        sample_id = f"{split}/{sequence}/{stamp}"
        frames.append({"sample_id": sample_id, "split": split, "sequence": f"{split}/{sequence}",
                       "agent": agent, "timestamp": stamp, "modalities": modalities})
        counts[split].add(sample_id)
        sensor_counts[split] += 1
    splits = json.loads((reference / "split_manifest.json").read_text())
    for split, spec in splits["splits"].items():
        if len(counts[split]) != spec["cooperative_samples"] or sensor_counts[split] != spec["sensor_frames"]:
            raise ValueError(f"Official split coverage differs: {split}")
    for name in ("split_manifest.json", "weather_manifest.json", "calibration_manifest.json", "source_manifest.json"):
        shutil.copy2(reference / name, root / name)
    atomic_json(root / "image_sizes.json", {"schema": "png_ihdr_sizes_v1", "root": str(root),
                "source_revision": REVISION, "rgb_pixels_saved": False, "images": images})
    atomic_json(root / "data_manifest.json", {"source_repo": "hx24/V2X-R", "source_revision": REVISION,
                "root": str(root), "frames": frames, "missing_files": [],
                "archive_receipts": str(root / "manifests/archives")})
    print(json.dumps({"revision": REVISION, "frames": len(frames),
                      "samples": {key: len(value) for key, value in counts.items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "datasets/v2xr")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gt-workers", type=int, default=8)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--manifests-only", action="store_true", help="Use already verified archive receipts")
    args = parser.parse_args()
    if args.workers < 1 or args.gt_workers < 1:
        parser.error("Worker counts must be positive")
    root = args.data_root.resolve()
    if not args.manifests_only:
        work = root.parent / ".v2xr_preparation"
        (work / "setup").mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "repro/metadata/remote_files.json", work / "setup/v2xr_full_remote_manifest.json")
        spec = importlib.util.spec_from_file_location("v2xr_downloader", REPO / "project_tools/download_full_v2xr.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT, module.DATA, module.REVISION = work, root, REVISION
        module.ARCHIVES = work / "archives"
        module.STATE = work / "setup/full_v2xr_state.json"
        module.PROGRESS = work / "setup/full_v2xr_progress.json"
        module.LOGS = work / "setup/full_v2xr_logs"
        # The legacy summary schema differs from the project's training schema.
        module.finalize = lambda: normalize_manifests(root)
        sys.argv = [str(spec.origin), "--workers", str(args.workers)] + (["--probe-only"] if args.probe_only else [])
        module.main()
        if args.probe_only:
            return
    else:
        normalize_manifests(root)
    from opencood.project.fixed_gt import atomic_json, build_manifest
    manifest, protocol = build_manifest(root, args.gt_workers, lambda n, total: print(f"Fixed GT {n}/{total}", flush=True))
    atomic_json(root / "protocol_manifest.json", protocol)
    atomic_json(root / "eval_gt_manifest.json", manifest)
    expected = json.loads((REPO / "repro/metadata/gt_summary.json").read_text())
    if manifest["summary"] != expected:
        raise ValueError("Rebuilt GT differs from the recorded experiment summary")
    print("PASS: pinned archives, paired modalities, official splits and fixed GT summary")


if __name__ == "__main__":
    main()
