"""Portable entrypoints must preserve the executed budget and reject incomplete data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import yaml

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortableReproductionTests(unittest.TestCase):
    def test_relocated_configs_preserve_scientific_settings_and_refuse_overwrite(self):
        tool = script("reproduce")
        with tempfile.TemporaryDirectory(prefix="cscwd portable ") as directory:
            root = Path(directory)
            output = root / "runs"
            tool.configure(root / "custom data", output)
            for variant in tool.VARIANTS:
                generated = yaml.safe_load((output / "configs" / f"{variant}.yaml").read_text())
                original = yaml.safe_load((REPO / "repro/configs" / f"{variant}.yaml").read_text())
                for key in ("training", "resources", "model", "loss", "selection", "scoring", "experiment"):
                    self.assertEqual(generated[key], original[key])
                self.assertEqual(generated["training"]["epochs"], 20)
                self.assertEqual(generated["data"]["root"], str(root / "custom data"))
                self.assertEqual(generated["artifacts"]["output_root"], str(output))
            tool.configure(root / "custom data", output)
            with self.assertRaisesRegex(ValueError, "differs"):
                tool.configure(root / "different data", output)

    def test_normalize_receipts_preserves_modalities_and_checks_archive_identity(self):
        tool = script("prepare_data")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            reference = Path(directory) / "reference"
            receipts = root / "manifests/archives"
            receipts.mkdir(parents=True)
            reference.mkdir()
            prefix = "train/sequence/1/000001"
            entries = []
            for suffix in tool.SUFFIXES.values():
                path = root / (prefix + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
                entries.append({"path": prefix + suffix, "bytes": 4, "sha256": "f" * 64, "crc32": "00000000"})
            receipt = {"archive": "train/sequence/1.7z", "revision": tool.REVISION,
                       "archive_sha256": "a" * 64, "retained_files": entries,
                       "image_sizes": {prefix + f"_camera{i}.png": [800, 600] for i in range(4)}}
            path = receipts / "archive.json"
            path.write_text(json.dumps(receipt))
            (reference / "remote_files.json").write_text(json.dumps({"revision": tool.REVISION, "files": [
                {"path": receipt["archive"], "lfs": {"oid": "a" * 64}}]}))
            (reference / "split_manifest.json").write_text(json.dumps({"splits": {
                "train": {"cooperative_samples": 1, "sensor_frames": 1}}}))
            for name in ("weather_manifest", "calibration_manifest", "source_manifest"):
                (reference / f"{name}.json").write_text("{}")
            tool.normalize_manifests(root, reference)
            frame = json.loads((root / "data_manifest.json").read_text())["frames"][0]
            self.assertEqual(frame["sample_id"], "train/sequence/000001")
            self.assertEqual(set(frame["modalities"]), set(tool.SUFFIXES))
            self.assertEqual(frame["modalities"]["fog"]["path"], prefix + "_fog_0.060.pcd")
            receipt["archive_sha256"] = "b" * 64
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                tool.normalize_manifests(root, reference)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete archives"):
                tool.normalize_manifests(root, reference)


if __name__ == "__main__":
    unittest.main()
