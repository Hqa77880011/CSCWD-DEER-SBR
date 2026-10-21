import copy
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from opencood.project.config import DEFAULTS, validate_config, config_hash, require_matching_training_contract
from opencood.project.weather import weather_for, augmentation_seed
from opencood.project.supervision import mutual_nn_targets
from opencood.project.training import (EpochBatchSampler, TrainState, save_checkpoint, load_checkpoint,
    seed_everything, consider_checkpoint, selection_score, select_model_inputs)


class ProtocolTests(unittest.TestCase):
    def test_weather_rng_worker_independence_and_no_snow(self):
        before = [weather_for(str(i), "2", 5, 17, "W") for i in range(10000)]
        random.seed(956)
        np.random.seed(299)
        after = [weather_for(str(i), "2", 5, 17, "W") for i in range(10000)]
        self.assertEqual(before, after)
        self.assertLess(abs(before.count("fog") / len(before) - 3 / 7), .02)
        self.assertEqual({weather_for(str(i), "2", 2, 43, "C") for i in range(100)}, {"normal"})
        self.assertNotEqual(augmentation_seed("a", 1, 17), augmentation_seed("a", 2, 17))

    def test_strict_config_budget_and_leakage(self):
        self.assertEqual(validate_config({})["training"]["effective_batch_size"], 8)
        for bad in ({"trining": {}}, {"training": {"unknown": True}}, {"training": {"micro_batch_size": 2}},
                    {"data": {"paired_supervision": True}}, {"selection": {"iou": .5}}):
            with self.assertRaises(ValueError):
                validate_config(bad)
        self.assertEqual(select_model_inputs({"record_len": 1, "train_targets": "secret", "clean_z0": 2, "gt_boxes": 3}), {"record_len": 1})
        with self.assertRaises(ValueError):
            validate_config({"experiment": {"track": "W", "variant": "B"}, "data": {"paired_supervision": True}})

    def test_sampler_prefetch_is_not_consumption_and_rank_budget(self):
        sampler = EpochBatchSampler(35, 2, 2, 17, rank=0, world_size=2)
        batches = list(sampler)
        self.assertEqual(sampler.next_batch, 0)
        sampler.advance(2)
        self.assertEqual(list(sampler), batches[2:])
        other = list(EpochBatchSampler(35, 2, 2, 17, rank=1, world_size=2))
        self.assertFalse(set(sum(batches, [])) & set(sum(other, [])))
        self.assertEqual(len(sum(batches + other, [])), 32)

    def test_exact_checkpoint_resume_after_update(self):
        seed_everything(17)
        model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Dropout(.3), torch.nn.Linear(7, 1))
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, [2], gamma=.1)
        sampler = EpochBatchSampler(24, 2, 4, 17)
        state, config = TrainState(), validate_config({})
        def step():
            opt.zero_grad(set_to_none=True)
            losses = []
            for _ in range(4):
                x = torch.randn(2, 3) + random.random() + float(np.random.random())
                loss = model(x).square().mean() / 4
                loss.backward()
                losses.append(float(loss.detach()))
                sampler.advance()
                state.micro_steps += 1
            opt.step()
            opt.zero_grad(set_to_none=True)
            state.optimizer_steps += 1
            return losses
        step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resume.pt"
            save_checkpoint(path, model, opt, sch, None, state, sampler, config)
            expected = step()
            expected_params = copy.deepcopy(model.state_dict())
            torch.rand(100)
            state = load_checkpoint(path, model, opt, sch, None, sampler, config)
            actual = step()
            self.assertEqual(expected, actual)
            for key in expected_params:
                self.assertTrue(torch.equal(expected_params[key], model.state_dict()[key]))
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, opt, sch, None, sampler, {**config, "version": 1})

    def test_selection_never_snow_and_tie_earlier(self):
        state = TrainState()
        self.assertTrue(consider_checkpoint(state, "W", {"normal": .8, "fog": .4}, 1))
        self.assertFalse(consider_checkpoint(state, "W", {"normal": .8, "fog": .400001}, 2))
        self.assertEqual(state.selected_epoch, 1)
        with self.assertRaises(ValueError):
            selection_score("W", {"normal": .8, "fog": .4, "snow": .9})

    def test_unmatched_and_ambiguous_points_are_not_labels(self):
        kwargs = dict(point_cloud_range=[0, 0, 0, 4, 4, 4], voxel_size=[1, 1, 4])
        clean = np.array([[.1, .1, .1], [1.1, .1, .1]])
        observed = np.array([[.1, .1, .1], [2.1, .1, .1]])
        result = mutual_nn_targets(clean, observed, **kwargs)
        self.assertEqual(result["miss_valid"].sum(), 1)
        self.assertEqual(result["spur_valid"].sum(), 1)
        self.assertEqual(result["deg_valid"].sum(), 0)
        verified = mutual_nn_targets(clean, observed, allow_unmatched=True, **kwargs)
        self.assertEqual(verified["deg_valid"].sum(), 2)
        duplicate = mutual_nn_targets(clean[:1].repeat(2, 0), observed[:1], **kwargs)
        self.assertEqual(duplicate["miss_valid"].sum(), 0)

    def test_amp_evidence_binds_contract_and_recomputes_acceptance(self):
        from project_tools.run_experiments import validate_amp_audit, AMP_TOLERANCES
        config = validate_config({"training": {"precision": "amp_bf16"}})
        contract = {"micro_batch_size": 4, "normalization": "batchnorm", "source_sha256": "current"}
        case = {"finite": True, "batch_identity_sha256": "a" * 64, "loss_relative_error": .01,
                "psm_relative_l2": .01, "rm_relative_l2": .01, "gradient_cosine": .999, "gradient_relative_l2": .02}
        audit = {"schema": "project_amp_consistency_v1", "passed": True, "dataset_id": "v2xr", "precision": "amp_bf16",
                 "reference_precision": "fp32", "contract": contract, "thresholds": AMP_TOLERANCES, "evidence": "paired real batch",
                 "cases": [{**case, "variant": variant} for variant in ("S0", "AB")]}
        with patch("project_tools.run_experiments.amp_audit_contract", return_value=contract):
            self.assertIsNone(validate_amp_audit(config, audit, require_stability=False))
            self.assertIn("20 finite", validate_amp_audit(config, audit))
            bad = copy.deepcopy(audit)
            bad["cases"][1]["gradient_relative_l2"] = .9
            self.assertIsNotNone(validate_amp_audit(config, bad))
            bad = copy.deepcopy(audit)
            bad["contract"]["micro_batch_size"] = 8
            self.assertIsNotNone(validate_amp_audit(config, bad))
            self.assertIsNotNone(validate_amp_audit(config, {"passed": True}))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "stable.json"
                stable = {"passed": True, "all_losses_and_gradients_finite": True, "status": "diagnostic_completed",
                          "mode": "benchmark", "formal_experiment_started": False, "completed_optimizer_updates": 20,
                          "variant": "AB", "audit_contract": contract}
                import json
                from opencood.project.config import file_sha256
                path.write_text(json.dumps(stable))
                audited = copy.deepcopy(audit)
                audited["evidence"] = {"stability_validated": True, "stability_report": {"path": str(path), "sha256": file_sha256(path)}}
                self.assertIsNone(validate_amp_audit(config, audited))
                stable["completed_optimizer_updates"] = 19
                path.write_text(json.dumps(stable))
                audited["evidence"]["stability_report"]["sha256"] = file_sha256(path)
                self.assertIn("at least 20", validate_amp_audit(config, audited))

    def test_teacher_free_config_rejects_legacy_artifacts_and_mixed_protocol(self):
        for bad in ({"version": 1}, {"experiment": {"training_protocol": "teacher_v1"}},
                    {"artifacts": {"teacher_checkpoint": "old.pt"}},
                    {"artifacts": {"feature_scale": "old_rms.pt"}},
                    {"model": {"require_calibrated_scale": False}},
                    {"loss": {"repair_weight": 0.05}}):
            with self.assertRaises(ValueError):
                validate_config(bad)
        config = validate_config({"experiment": {"track": "W", "variant": "B"}})
        self.assertFalse(config["data"]["paired_supervision"])
        self.assertEqual(config["training"]["micro_batch_size"], 4)
        self.assertEqual(config["training"]["accumulation_steps"], 2)
        self.assertEqual(config["resources"]["num_workers"], 8)
        with self.assertRaises(ValueError):
            validate_config({"experiment": {"track": "W", "variant": "A"}})
        with self.assertRaises(ValueError):
            validate_config({"data": {"supervision_source": "provenance_v1"}})
        provenance = validate_config({"experiment": {"track": "W", "variant": "AB"},
                                      "data": {"supervision_source": "provenance_v1"}})
        self.assertFalse(provenance["data"]["paired_supervision"])

    def test_provenance_formal_gate_requires_source_evidence_and_current_contract(self):
        import json
        from project_tools.run_experiments import formal_blockers, supervision_contract
        base = {"experiment": {"track": "W", "variant": "AB"},
                "data": {"supervision_source": "provenance_v1"}}
        contract = {"gt_manifest_sha256": "fixed"}
        evidence = {"kind": "controlled_source_ids", "source_ids_verified": True,
                    "normal_train_only": True, "all_natural_fog_labels_masked": True,
                    "no_snow_or_validation_test": True, "no_teacher_network": True}
        with tempfile.TemporaryDirectory() as folder, \
                patch("project_tools.run_experiments.data_contract_metadata", return_value=contract), \
                patch("project_tools.run_experiments.source_tree_hash", return_value="source"):
            path = Path(folder) / "audit.json"
            config = validate_config({**base, "data": {**base["data"], "supervision_audit": str(path)}})
            self.assertTrue(formal_blockers(config))
            audit = {"schema": "project_provenance_supervision_audit_v2", "split": "train",
                     "supervision_source": "provenance_v1", "formal_positive_supervision_ready": True,
                     "positive_miss_cells": 4, "positive_spur_cells": 2,
                     "seed":17,"epoch":0,"samples":32,"sample_indices":list(range(32)),
                     "rows":[{"sample_id":"train/tiny","sample_index":0,"epoch":0,"agent_id":"0",
                              "source_id_sha256":"exact-source-ids","target_sha256":"exact-target-masks",
                              "positive_miss_cells":4,"positive_spur_cells":2}],
                     "evidence": evidence, "contract": {**supervision_contract(config),"sample_indices":list(range(32))}}
            path.write_text(json.dumps(audit))
            # Isolate the costly dataset replay in this unit test. Data tests
            # independently verify exact source events and geometry; the real
            # 32-scene report is replayed again by the preflight command.
            with patch('project_tools.run_experiments._bounded_provenance_replay',return_value=copy.deepcopy(audit)) as replay:
                self.assertEqual(formal_blockers(config), [])
                altered_row=copy.deepcopy(audit['rows']);altered_row[0]['target_sha256']='forged-mask'
                for mutation in (dict(positive_spur_cells=0), dict(unmatched_semantics_verified=True, evidence={}),
                                 dict(contract={**audit["contract"], "source_sha256": "old"}),
                                 dict(contract={**audit["contract"], "seed":29}),
                                 dict(rows=altered_row),dict(positive_miss_cells=5),dict(sample_indices=[0]*32),
                                 dict(rows=[])):
                    path.write_text(json.dumps({**audit, **mutation}))
                    self.assertTrue(formal_blockers(config))
                self.assertGreaterEqual(replay.call_count,3)
                # A collection of positive counts + true flags is no proof.
                forged={key:value for key,value in audit.items() if key not in ('rows','sample_indices','seed','epoch')}
                path.write_text(json.dumps(forged))
                self.assertTrue(formal_blockers(config))
            # B's current-observation training has no type target / teacher gate.
            self.assertEqual(formal_blockers(validate_config({"experiment": {"track": "W", "variant": "B"}})), [])

    def test_training_contract_and_run_cleanup(self):
        from project_tools.run_experiments import run
        from types import SimpleNamespace
        import signal
        current = validate_config({})
        reference = copy.deepcopy(current)
        reference["experiment"].update(variant="B", run_id="another")
        reference["training"]["checkpoint_interval_updates"] = 1
        require_matching_training_contract(reference, current, "matched method")
        for change in ({"normalization": "frozen_batchnorm"}, {"micro_batch_size": 2, "accumulation_steps": 4}):
            bad = copy.deepcopy(reference)
            bad["training"].update(change)
            with self.assertRaisesRegex(ValueError, "training contract mismatch"):
                require_matching_training_contract(bad, current, "matched method")
        bad = copy.deepcopy(reference)
        bad["data"]["controlled_corruption"]["drop_fraction"] = .3
        with self.assertRaisesRegex(ValueError, "training contract mismatch"):
            require_matching_training_contract(bad, current, "matched method")
        closed = []
        iterator = SimpleNamespace(_shutdown_workers=lambda: closed.append(True))
        loader = SimpleNamespace(_iterator=iterator)
        old_handler = signal.getsignal(signal.SIGTERM)
        def failing_run(config, args, resources):
            resources.update(loader=loader, iterator=iterator)
            resources["signal_handlers"] = {signal.SIGTERM: signal.signal(signal.SIGTERM, lambda s, f: None)}
            raise FloatingPointError("intentional diagnostic failure")
        with patch("project_tools.run_experiments._run", side_effect=failing_run):
            with self.assertRaises(FloatingPointError):
                run(current, SimpleNamespace())
        self.assertEqual(closed, [True])
        self.assertIsNone(loader._iterator)
        self.assertIs(signal.getsignal(signal.SIGTERM), old_handler)

    def test_prediction_decode_dimensions_empty_and_finite(self):
        from project_tools.export_predictions import decode_predictions
        anchor = torch.tensor([[[[1., 2., 3., 2., 4., 6., .5]]]])
        output = {"rm": torch.zeros(2, 7, 1, 1), "psm": torch.tensor([[[[0.]]], [[[-100.]]]])}
        records = decode_predictions(output, anchor, ["positive", "empty"])
        self.assertEqual(records[0]["predictions"][0]["box"], [1., 2., 3., 6., 4., 2., .5])
        self.assertEqual(records[0]["predictions"][0]["prediction_id"], "anchor_0000000")
        self.assertEqual(records[1]["predictions"], [])
        output["rm"][0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            decode_predictions(output, anchor, ["positive", "empty"])

    def test_checkpoint_rejects_legacy_schema_even_with_same_config_hash(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
        sampler = EpochBatchSampler(8, 4, 2, 17)
        config = validate_config({})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, scheduler, None, TrainState(), sampler, config)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["schema_version"], 2)
            self.assertEqual(checkpoint["training_protocol"], "teacher_free_v2")
            for mutation in (dict(schema_version=1), dict(training_protocol="teacher_v1")):
                torch.save({**checkpoint, **mutation}, path)
                with self.assertRaisesRegex(ValueError, "configuration/schema mismatch"):
                    load_checkpoint(path, model, optimizer, scheduler, None, sampler, config)


if __name__ == "__main__":
    unittest.main()
