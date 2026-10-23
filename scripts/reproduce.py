#!/usr/bin/env python3
"""Portable entrypoint for the executed seven-variant, seed-17 experiment."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("S0", "A", "B", "AB", "SUP", "N1", "N2")
sys.path.insert(0, str(REPO))


def execute(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True)


def configure(data_root, output_root):
    import yaml
    from opencood.project.config import validate_config
    data_root, output_root = data_root.resolve(), output_root.resolve()
    directory = output_root / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        config = yaml.safe_load((REPO / "repro/configs" / f"{variant}.yaml").read_text())
        config["data"].update(root=str(data_root), manifest_dir=str(data_root),
                              supervision_audit=str(output_root / "provenance_audit.json"))
        config["artifacts"]["output_root"] = str(output_root)
        config = validate_config(config)
        path = directory / f"{variant}.yaml"
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError(f"Existing configuration differs: {path}. Choose a new output root.")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Configured {len(VARIANTS)} variants in {directory}")


def check_results():
    with (REPO / "results/reference/paper_test_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if tuple(row["variant"] for row in rows) != VARIANTS:
        raise ValueError("Unexpected result variants")
    baseline = Decimal(rows[0]["Fog_Snow_mean_percent"])
    for row in rows:
        mean = (Decimal(row["Fog_AP3D07_percent"]) + Decimal(row["Snow_AP3D07_percent"])) / 2
        if abs(mean - Decimal(row["Fog_Snow_mean_percent"])) > Decimal("1e-12"):
            raise ValueError(f"Incorrect weather mean: {row['variant']}")
        if abs(mean - baseline - Decimal(row["Fog_Snow_delta_vs_S0_pp"])) > Decimal("1e-12"):
            raise ValueError(f"Incorrect baseline delta: {row['variant']}")
        print(f"{row['variant']:>3}  Normal {float(row['Normal_AP3D07_percent']):.2f}"
              f"  Fog {float(row['Fog_AP3D07_percent']):.2f}"
              f"  Snow {float(row['Snow_AP3D07_percent']):.2f}  Mean {float(mean):.2f}")
    print("PASS: seven reference rows, means and deltas (no inference performed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("configure", "audit", "preflight", "smoke", "train", "evaluate", "results"))
    parser.add_argument("--data-root", type=Path, default=REPO / "datasets/v2xr")
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/reproduction")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="AB")
    parser.add_argument("--resume", type=Path, help="Online resume checkpoint for one train variant")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    args = parser.parse_args()
    if args.action == "results":
        check_results()
        return
    if args.resume and (args.action != "train" or args.variant == "all"):
        parser.error("--resume requires train and one explicit variant")
    if args.action == "configure":
        configure(args.data_root, args.output_root)
        return
    output = args.output_root.resolve()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    if args.action == "evaluate" and not args.allow_test_evaluation:
        parser.error("Final test requires --allow-test-evaluation; freeze all methods first")
    for variant in variants:
        config = output / "configs" / f"{variant}.yaml"
        if not config.is_file():
            parser.error("Run configure with the same --output-root first")
        if args.action == "audit":
            execute(REPO / "project_tools/audit_pair_supervision.py", "--config", config,
                    "--samples", "32", "--output", output / "provenance_audit.json")
            break  # All seven variants share the same source/data/augmentation contract.
        if args.action == "evaluate":
            from opencood.project.config import load_config
            run = load_config(config)["experiment"]["run_id"]
            checkpoint = output / run / "train/selected.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for weather in ("normal", "fog", "snow"):
                predictions = output / "final_test" / variant / f"{weather}.jsonl"
                predictions.parent.mkdir(parents=True, exist_ok=True)
                scores = predictions.with_suffix(".scores.json")
                if predictions.exists() or scores.exists():
                    raise FileExistsError(f"Final-test output already exists: {predictions}")
                execute(REPO / "project_tools/export_predictions.py", "--config", config,
                        "--checkpoint", checkpoint, "--split", "test", "--weather", weather,
                        "--output", predictions, "--execute", "--allow-test-evaluation")
                data = Path(load_config(config)["data"]["manifest_dir"])
                execute(REPO / "project_tools/evaluate_predictions.py", "--gt-manifest",
                        data / "eval_gt_manifest.json", "--predictions", predictions,
                        "--split", "test", "--output", scores)
            continue
        options = ["--config", config, "--mode", args.action]
        if args.action == "train":
            options.append("--allow-formal-run")
            if args.resume:
                options.extend(("--resume", args.resume.resolve()))
        if args.action == "smoke":
            options.extend(("--steps", "1", "--warmup", "0"))
        execute(REPO / "project_tools/run_experiments.py", *options)


if __name__ == "__main__":
    from legacy_runtime import execute_original
    execute_original('reproduce.py')
