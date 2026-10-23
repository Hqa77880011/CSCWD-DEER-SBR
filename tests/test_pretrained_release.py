"""Release integrity, safe extraction and checkpoint relocation regressions."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pretrained", ROOT / "scripts/pretrained.py")
pretrained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrained)


class PretrainedReleaseTests(unittest.TestCase):
    def test_parallel_pipeline_matches_the_full_exact_equivalence_evidence(self):
        pretrained.verify_parallel_pipeline(pretrained.load_release())

    def test_release_pins_the_actual_seven_selected_checkpoint_hashes(self):
        release = pretrained.load_release()
        selections = json.loads((ROOT / "results/reference/manifest.json").read_text())["selections"]
        for entry in selections:
            variant = release["variants"][entry["variant"]]
            self.assertEqual(variant["checkpoint_sha256"], entry["checkpoint_sha256"])
            self.assertEqual(variant["config_sha256"], entry["config_sha256"])
            self.assertEqual(variant["selected_epoch"], entry["selected_epoch"])
            self.assertEqual(release["assets"][variant["asset"]]["sha256"], entry["checkpoint_sha256"])

    def test_metadata_unpack_checks_content_and_rejects_traversal_or_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "metadata.tar.gz"
            def packed(name="data.json", symlink=False):
                with tarfile.open(archive, "w:gz") as target:
                    member = tarfile.TarInfo(name)
                    member.size = 2
                    if symlink:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/etc/passwd"
                    target.addfile(member, None if symlink else io.BytesIO(b"{}"))
                return {"metadata_asset": archive.name,
                    "assets": {archive.name: {"bytes": archive.stat().st_size, "sha256": pretrained.sha256(archive)}},
                    "metadata_files": {name: {"bytes": 2, "sha256": __import__('hashlib').sha256(b'{}').hexdigest()}}}
            release = packed()
            result = pretrained.unpack_metadata(release, directory)
            self.assertEqual((result / "data.json").read_text(), "{}")
            (result / "data.json").write_text("xx")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                pretrained.unpack_metadata(release, directory)
            for name, link in (("../escape.json", False), ("link.json", True)):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    pretrained.unpack_metadata(packed(name, link), directory)

    def test_dataset_view_keeps_raw_data_and_original_metadata_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, metadata = root / "raw data", root / "metadata"
            (raw / "test").mkdir(parents=True)
            metadata.mkdir()
            (metadata / "data_manifest.json").write_text("{}")
            view = pretrained.dataset_view(metadata, raw, root / "output")
            self.assertEqual((view / "test").resolve(), raw / "test")
            self.assertTrue((view / "data_manifest.json").samefile(metadata / "data_manifest.json"))
            self.assertFalse((raw / "data_manifest.json").exists())
            other = root / "other"
            (other / "test").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "elsewhere"):
                pretrained.dataset_view(metadata, other, root / "output")


if __name__ == "__main__":
    unittest.main()
