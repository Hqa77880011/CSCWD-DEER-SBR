"""CPU evidence for scoped CMA math, true eval, isolation and exact rollback."""
import copy
import json
import random
import unittest

import numpy as np
import torch
from torch import nn

from opencood.models.common_modules.pillar_vfe import PFNLayer
from opencood.project.batchnorm import (BatchNormRefitError, configure_batchnorm,
                                       refit_batchnorm, batchnorm_statistics_hashes)
from opencood.project.config import validate_config
from opencood.project.training import restore_rng, rng_state
from project_tools.export_predictions import validate_selected_batchnorm_checkpoint


class ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2, eps=.001, momentum=.01)
        self.radar_bn = nn.BatchNorm1d(2, eps=.001, momentum=.01)
        self.dropout = nn.Dropout(.9)
        self.head = nn.Linear(2, 2)
        self.register_buffer("forwards", torch.zeros((), dtype=torch.int64))
        self.seen_keys = []
        self.fail = False

    def forward(self, inputs, return_aux=False):
        self.seen_keys.append(set(inputs))
        self.forwards.add_(1)
        random.random()
        np.random.random()
        torch.rand(())
        lidar = inputs["processed_lidar"]
        output = self.bn(lidar)
        radar = inputs.get("processed_radar")
        if radar is not None and radar.shape[0]:
            self.radar_bn(radar)
        output = self.head(self.dropout(output))
        if self.fail:
            raise RuntimeError("deliberate forward failure")
        return {"psm": output}


def batch(lidar, radar=None, index=0):
    count = lidar.shape[0]
    ego = {"processed_lidar": lidar, "record_len": torch.ones(count, dtype=torch.long),
           "sample_ids": [f"train/toy/{index}_{i}" for i in range(count)],
           "metadata": [[{"weather": "normal" if i % 2 else "fog"}] for i in range(count)],
           "label_dict": {"forbidden": torch.tensor(float("nan"))},
           "gt_boxes": "must not enter forward", "weather": "not an inference input"}
    if radar is not None:
        ego["processed_radar"] = radar
    return {"ego": ego, "train_targets": {"forbidden": torch.tensor(123)}}


class BatchNormRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = ToyDetector().train()
        self.model.dropout.eval()  # Preserve mixed per-module modes exactly.
        self.model.radar_bn.momentum = .2
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, .75)
        self.batches = [batch(torch.tensor([[0., 2.], [2., 6.]]), index=0),
                        batch(torch.tensor([[10., 20.], [12., 24.], [14., 28.], [16., 32.]]), index=1)]

    def snapshot(self):
        return {"state": copy.deepcopy(self.model.state_dict()),
                "modes": {n: m.training for n, m in self.model.named_modules()},
                "momenta": {n: m.momentum for n, m in self.model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)},
                "grads": {n: p.grad.clone() for n, p in self.model.named_parameters()},
                "rng": rng_state()}

    def assert_restored(self, before):
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before["state"][name], rtol=0, atol=0)
        self.assertEqual({n: m.training for n, m in self.model.named_modules()}, before["modes"])
        self.assertEqual({n: m.momentum for n, m in self.model.named_modules()
                          if isinstance(m, nn.modules.batchnorm._BatchNorm)}, before["momenta"])
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter.grad, before["grads"][name], rtol=0, atol=0)
        now = rng_state()
        self.assertEqual(now["python"], before["rng"]["python"])
        self.assertEqual(now["numpy"][0], before["rng"]["numpy"][0])
        np.testing.assert_array_equal(now["numpy"][1], before["rng"]["numpy"][1])
        self.assertEqual(now["numpy"][2:], before["rng"]["numpy"][2:])
        torch.testing.assert_close(now["torch"], before["rng"]["torch"], rtol=0, atol=0)
        if now["cuda"] is not None:
            for current, old in zip(now["cuda"], before["rng"]["cuda"]):
                torch.testing.assert_close(current, old, rtol=0, atol=0)

    def test_cma_equal_micro_math_allowlist_and_full_restoration(self):
        before = self.snapshot()
        with refit_batchnorm(self.model, self.batches, "cpu") as report:
            values = [item["ego"]["processed_lidar"] for item in self.batches]
            expected_mean = torch.stack([x.mean(0) for x in values]).mean(0)
            expected_var = torch.stack([x.var(0, unbiased=True) for x in values]).mean(0)
            torch.testing.assert_close(self.model.bn.running_mean, expected_mean)
            torch.testing.assert_close(self.model.bn.running_var, expected_var)
            self.assertFalse(torch.allclose(expected_mean, torch.cat(values).mean(0)))
            self.assertEqual(int(self.model.bn.num_batches_tracked), 2)
            self.assertEqual(report["bn"]["bn"]["elements_per_channel_by_call"], [2, 4])
            self.assertEqual(report["sample_count"], 6)
            self.assertEqual(len(report["bn"]["bn"]["statistics_sha256"]), 64)
            self.assertTrue(report["all_modules_eval_at_yield"])
            self.assertFalse(any(m.training for m in self.model.modules()))
            for keys in self.model.seen_keys:
                self.assertEqual(keys, {"processed_lidar", "record_len"})
            for name, parameter in self.model.named_parameters():
                torch.testing.assert_close(parameter, before["state"][name], rtol=0, atol=0)
                torch.testing.assert_close(parameter.grad, before["grads"][name], rtol=0, atol=0)
            json.dumps(report, allow_nan=False)
        self.assertTrue(report["original_buffers_restored_on_exit"])
        self.assertTrue(report["rng_restored_on_exit"])
        self.assert_restored(before)

    def test_eval_outputs_independent_of_probe_batch_companions(self):
        with refit_batchnorm(self.model, self.batches, "cpu"):
            x = torch.tensor([[4., 8.]])
            first = self.model({"processed_lidar": x})["psm"]
            together = self.model({"processed_lidar": torch.cat([x, x * 100])})["psm"][:1]
            torch.testing.assert_close(first, together, rtol=1e-6, atol=1e-6)

    def test_exception_in_caller_restores_fitted_buffers_modes_grads_and_rng(self):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with refit_batchnorm(self.model, self.batches, "cpu"):
                random.random(); np.random.random(); torch.rand(())
                raise RuntimeError("caller failure")
        self.assert_restored(before)

    def test_forward_failure_and_nonfinite_or_singleton_roll_back(self):
        for case in ("forward", "nonfinite", "singleton"):
            with self.subTest(case=case):
                before = self.snapshot()
                self.model.fail = case == "forward"
                values = torch.tensor([[float("nan"), 1.], [2., 3.]]) if case == "nonfinite" else torch.ones(1 if case == "singleton" else 2, 2)
                with self.assertRaises((RuntimeError, BatchNormRefitError)):
                    with refit_batchnorm(self.model, [batch(values)], "cpu"):
                        self.fail("invalid calibration must not yield")
                self.assert_restored(before)
                self.model.fail = False

    def test_unused_empty_modality_keeps_original_stats(self):
        self.model.radar_bn.running_mean.fill_(7.)
        self.model.radar_bn.running_var.fill_(3.)
        self.model.radar_bn.num_batches_tracked.fill_(9)
        before = self.snapshot()
        entries = [batch(torch.tensor([[0., 2.], [2., 6.]]), radar=torch.empty(0, 2))]
        with refit_batchnorm(self.model, entries, "cpu") as report:
            self.assertEqual(report["unused_bn"], ["radar_bn"])
            self.assertTrue(report["bn"]["radar_bn"]["used_original_statistics"])
            self.assertEqual(report["bn"]["radar_bn"]["num_batches_tracked"], 9)
            torch.testing.assert_close(self.model.radar_bn.running_mean, torch.full((2,), 7.))
        self.assert_restored(before)

    def test_each_bn_counts_only_its_actual_calls(self):
        entries = [batch(torch.tensor([[0., 2.], [2., 6.]]), radar=torch.empty(0, 2)),
                   batch(torch.tensor([[1., 3.], [4., 8.]]), radar=torch.tensor([[2., 4.], [6., 12.], [10., 20.]]))]
        with refit_batchnorm(self.model, entries, "cpu") as report:
            self.assertEqual(report["bn"]["bn"]["calls"], 2)
            self.assertEqual(report["bn"]["radar_bn"]["calls"], 1)
            self.assertEqual(report["bn"]["radar_bn"]["elements_per_channel_by_call"], [3])
            self.assertEqual(int(self.model.radar_bn.num_batches_tracked), 1)

    def test_eval_parent_pfn_singleton_uses_explicit_bn_mode_guard(self):
        layer = PFNLayer(2, 2, use_norm=True, last_layer=True).eval()
        layer.norm.train()
        layer.norm.running_mean.fill_(2.)
        layer.norm.running_var.fill_(3.)
        output = layer(torch.ones(1, 1, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(int(layer.norm.num_batches_tracked), 0)
        # Conversely a training parent with frozen BN must use eval statistics.
        layer.train(); layer.norm.eval()
        self.assertTrue(torch.isfinite(layer(torch.ones(1, 1, 2))).all())
        self.assertEqual(int(layer.norm.num_batches_tracked), 0)

    def test_saved_eval_stats_and_resume_stats_have_distinct_scopes(self):
        original = copy.deepcopy(self.model.state_dict())
        with refit_batchnorm(self.model, self.batches, "cpu"):
            selected = copy.deepcopy(self.model.state_dict())
            self.assertFalse(torch.equal(selected["bn.running_mean"], original["bn.running_mean"]))
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertEqual(int(selected["bn.num_batches_tracked"]), 2)
        self.assertEqual(int(self.model.bn.num_batches_tracked), 0)

    def test_bounds_forbidden_partition_empty_stream_and_configuration(self):
        before = self.snapshot()
        forbidden = batch(torch.ones(2, 2))
        forbidden["ego"]["sample_ids"] = ["validate/x/0", "validate/x/1"]
        for entries in ([], self.batches * 5, [forbidden]):
            with self.assertRaises(BatchNormRefitError):
                with refit_batchnorm(self.model, entries, "cpu"):
                    self.fail("invalid stream")
            self.assert_restored(before)
        configure_batchnorm(self.model, {"eps": .002, "momentum": .1, "evaluation_statistics": "train_refit_cma_v1"})
        self.assertEqual(self.model.bn.eps, .002)
        self.assertEqual(self.model.radar_bn.momentum, .1)
        with self.assertRaises(ValueError):
            configure_batchnorm(self.model, {"momentum": float("nan")})

    def selected_fixture(self):
        config = validate_config({"version": 4})
        policy = config["training"]["batchnorm"]
        entries = [batch(torch.tensor([[0., 2.], [2., 6.], [3., 7.], [4., 8.]]) + i,
                         radar=torch.tensor([[1., 2.], [2., 4.], [3., 6.], [4., 8.]]) + i,
                         index=i) for i in range(8)]
        contract = {key: "a" * 64 for key in ("base_hypes_sha256", "gt_manifest_sha256", "data_manifest_sha256",
                    "split_manifest_sha256", "weather_manifest_sha256", "protocol_manifest_sha256", "image_sizes_sha256")}
        with refit_batchnorm(self.model, entries, "cpu") as report:
            selected_model = copy.deepcopy(self.model).eval()
            manifest = copy.deepcopy(report)
        manifest.update(sample_indices=list(range(32)), sampling=policy["refit_sampling"],
                        refit_epoch=policy["refit_epoch"], input_augmentation=policy["refit_augment"],
                        training_protocol=config["experiment"]["training_protocol"], dataset_split="train",
                        training_track=config["experiment"]["track"], configuration_policy=copy.deepcopy(policy),
                        data_contract=contract)
        saved = {"metadata": {"checkpoint_role": "selected", "bn_statistics_role": "train_refit_cma_v1",
                              "batchnorm_refit": manifest, **contract}}
        return saved, config, selected_model

    def test_selected_export_accepts_only_manifest_bound_to_exact_refit_buffers(self):
        saved, config, model = self.selected_fixture()
        manifest = validate_selected_batchnorm_checkpoint(saved, config, model)
        self.assertEqual({name: row["statistics_sha256"] for name, row in manifest["bn"].items()},
                         batchnorm_statistics_hashes(model))
        model.bn.running_mean[0].add_(1.)
        with self.assertRaisesRegex(ValueError, "statistics/count/policy mismatch"):
            validate_selected_batchnorm_checkpoint(saved, config, model)

    def test_selected_export_rejects_missing_online_wrong_policy_and_incomplete_manifests(self):
        saved, config, model = self.selected_fixture()
        cases = {
            "missing_manifest": lambda m: m.pop("batchnorm_refit"),
            "resume": lambda m: m.update(checkpoint_role="resume"),
            "online": lambda m: m.update(bn_statistics_role="online_running"),
            "wrong_policy": lambda m: m["batchnorm_refit"]["configuration_policy"].update(momentum=.1),
            "uncalibrated": lambda m: m["batchnorm_refit"].update(policy="running"),
            "missing_bn": lambda m: m["batchnorm_refit"]["bn"].pop("radar_bn"),
            "unused_bn": lambda m: m["batchnorm_refit"].update(unused_bn=["radar_bn"]),
            "wrong_count": lambda m: m["batchnorm_refit"]["bn"]["bn"].update(calls=7),
            "wrong_split": lambda m: m["batchnorm_refit"].update(dataset_split="validate"),
            "wrong_aug": lambda m: m["batchnorm_refit"].update(input_augmentation=True),
            "wrong_source": lambda m: m["batchnorm_refit"]["data_contract"].update(gt_manifest_sha256="b" * 64),
            "wrong_weather": lambda m: m["batchnorm_refit"]["input_agent_weather"][0][0].append("snow"),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                bad = copy.deepcopy(saved)
                mutate(bad["metadata"])
                with self.assertRaises(ValueError):
                    validate_selected_batchnorm_checkpoint(bad, config, model)
        model.bn.train()
        with self.assertRaisesRegex(ValueError, "every module in eval"):
            validate_selected_batchnorm_checkpoint(saved, config, model)


if __name__ == "__main__":
    unittest.main()
