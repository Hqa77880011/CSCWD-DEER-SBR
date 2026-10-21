#!/usr/bin/env python3
"""Replay saved model predictions against complete, frozen project_fix_v1 GT."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencood.project.fixed_gt import atomic_json, file_sha256, load_manifest
from opencood.project.evaluation import evaluate_predictions, read_prediction_archive, validate_archive_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.gt_manifest)
    metadata, records = read_prediction_archive(args.predictions)
    gt_sha256 = file_sha256(args.gt_manifest)
    prediction_sha256 = file_sha256(args.predictions)
    validate_archive_metadata(metadata, records, manifest, args.split, gt_sha256, prediction_sha256)
    result = evaluate_predictions(manifest, records, args.split)
    if metadata["postprocessed"] and result["postprocess_counts"]["input"] != result["postprocess_counts"]["output"]:
        raise ValueError("Archive claims frozen postprocessing but contains excluded predictions")
    result.update({"archive_metadata": metadata, "predictions_file_sha256": file_sha256(args.predictions),
                   "predictions_sha256": prediction_sha256, "gt_manifest_sha256": gt_sha256})
    atomic_json(args.output, result)
    print(json.dumps({"num_samples": result["num_samples"], "num_gt": result["num_gt"],
                      "AP3D": {k: v["ap"] for k,v in result["metrics"]["3d"].items()},
                      "APBEV": {k: v["ap"] for k,v in result["metrics"]["bev"].items()},
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
