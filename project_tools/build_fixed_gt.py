#!/usr/bin/env python3
"""Build fixed GT from existing verified metadata; never run training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencood.project.fixed_gt import atomic_json, build_manifest, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest, protocol = build_manifest(args.data_root, args.workers,
                                         lambda n, total: print(f"Fixed GT {n}/{total}", flush=True))
    atomic_json(args.data_root/"protocol_manifest.json", protocol)
    atomic_json(args.data_root/"eval_gt_manifest.json", manifest)
    print(json.dumps({"status": "passed", "summary": manifest["summary"],
                      "eval_gt_manifest_sha256": file_sha256(args.data_root/"eval_gt_manifest.json"),
                      "protocol_sha256": protocol["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
