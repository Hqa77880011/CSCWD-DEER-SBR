"""Historical v3 regression reads its verified archive, without live YAML copies."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import torch
import yaml
from opencood.project.config import validate_config, require_matching_training_contract
from opencood.project.training import (save_checkpoint, load_checkpoint, TrainState, EpochBatchSampler)


@lru_cache(maxsize=1)
def archived_v3_configs():
    root = Path(__file__).resolve().parents[1] / 'repro/archive/teacher_free_v3_archive'
    archive = root / 'teacher_free_v3_source.tar.gz'
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = '31d509c5696eeab1d9ecdbb4f0499a0e331b5356887add76c4de6f0cf7af92ec'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected or manifest['archive_sha256'] != expected:
        raise ValueError('historical v3 archive SHA256 changed')
    recorded = {entry['path']: entry for entry in manifest['files']}
    configs = {}
    with tarfile.open(archive, 'r:gz') as source:
        for member in source.getmembers():
            path = Path(member.name)
            if (not member.isfile() or path.parent.as_posix() != 'V2X-R_project/cscwd_configs'
                    or not path.name.startswith('project_v3') or path.suffix != '.yaml'):
                continue
            content = source.extractfile(member).read()
            if (len(content) != recorded[member.name]['size_bytes']
                    or hashlib.sha256(content).hexdigest() != recorded[member.name]['sha256']):
                raise ValueError(f'historical v3 configuration changed: {member.name}')
            configs[path.name] = validate_config(yaml.safe_load(content))
    if len(configs) != 19:
        raise ValueError('historical v3 archive must contain the complete 19-config baseline')
    return configs


class V3ProtocolTests(unittest.TestCase):
    def test_version_mechanism_precision_and_ablation_registration(self):
        c = validate_config({'version': 3})
        self.assertEqual(c['experiment']['training_protocol'], 'teacher_free_v3')
        self.assertEqual(c['model']['amplitude_context'], 'rms')
        self.assertEqual(c['model']['type_prior_mode'], 'input_only')
        for bad in ({'version': 2, 'model': {'amplitude_context': 'rms'}},
                    {'version': 3, 'experiment': {'training_protocol': 'teacher_free_v2'}},
                    {'version': 3, 'model': {'type_prior_mode': 'multiplicative'}},
                    {'version': 3, 'model': {'amplitude_context': 'normalized_mean'}},
                    {'version': 3, 'training': {'precision': 'amp_bf16'}}):
            with self.assertRaises(ValueError):
                validate_config(bad)
        n1 = validate_config({'version': 3, 'experiment': {'variant': 'N1'},
                              'model': {'amplitude_context': 'normalized_mean'}})
        n2 = validate_config({'version': 3, 'experiment': {'variant': 'N2'},
                              'model': {'type_prior_mode': 'multiplicative'}})
        self.assertEqual(n1['model']['hidden_channels'], n2['model']['hidden_channels'])

    def test_archived_configs_keep_common_input_budget(self):
        archived = archived_v3_configs()
        for track in ('C', 'W'):
            configs = [archived[f'project_v3_{track}_{v}.yaml'] for v in
                       (('S0', 'A', 'B', 'AB') if track == 'C' else ('S0', 'A', 'B', 'AB', 'SUP', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'))]
            for c in configs[1:]:
                require_matching_training_contract(configs[0], c, 'v3 matrix')
        self.assertEqual(len(archived), 19)

    def test_schema3_roundtrip_rejects_v2_checkpoint(self):
        c = validate_config({'version': 3})
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10])
        sampler = EpochBatchSampler(32, 4, 2, 17)
        state = TrainState()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'latest.pt'
            save_checkpoint(p, model, optimizer, scheduler, None, state, sampler, c)
            self.assertEqual(torch.load(p, weights_only=False)['schema_version'], 3)
            load_checkpoint(p, model, optimizer, scheduler, None, sampler, c)
            old = copy.deepcopy(c)
            old['version'] = 2
            old['experiment']['training_protocol'] = 'teacher_free_v2'
            with self.assertRaises(ValueError):
                load_checkpoint(p, model, optimizer, scheduler, None, sampler, old)
            save_checkpoint(p, model, optimizer, scheduler, None, state, sampler, old)
            with self.assertRaises(ValueError):
                load_checkpoint(p, model, optimizer, scheduler, None, sampler, c)


if __name__ == '__main__':
    unittest.main()
