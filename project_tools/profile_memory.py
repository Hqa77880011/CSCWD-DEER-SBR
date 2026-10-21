#!/usr/bin/env python3
"""Bounded FP32/train-only allocator and tensor-lifetime diagnosis, never AP.

Every phase synchronizes CUDA and samples allocator state. These timings are
diagnostic overhead, not throughput measurements; use benchmark_resources.py
for paired throughput. No empty_cache calls or allocator changes are made.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from opencood.project.config import load_config, config_hash
from opencood.project.training import EpochBatchSampler, seed_everything, select_model_inputs, to_device
from project_tools.run_experiments import (build_components, make_loader, shutdown_loader,
                                          source_tree_hash, data_contract_metadata)


def snapshot_summary(segments, device_index=0):
    """Separate whole free segments, mixed-segment holes and pending frees.

    Inactive blocks in mixed segments are evidence of split cached blocks,
    not proof that fragmentation prevents a particular allocation. Raw traces
    retain requested allocation sizes and retries/OOMs for that assessment.
    """
    result = Counter()
    inactive_sizes, stack_bytes = [], Counter()
    for segment in segments:
        if segment.get('device', device_index) != device_index:
            continue
        blocks = segment.get('blocks', [])
        mixed = any(block['state'] != 'inactive' for block in blocks)
        result['segments'] += 1
        result['segment_bytes'] += segment['total_size']
        for block in blocks:
            state, size = block['state'], block['size']
            result[state + '_bytes'] += size
            result[state + '_blocks'] += 1
            if state == 'inactive':
                inactive_sizes.append(size)
                result['mixed_segment_inactive_bytes' if mixed else 'wholly_free_segment_bytes'] += size
            elif state == 'active_allocated':
                result['active_requested_bytes'] += block.get('requested_size', size)
                frames = block.get('frames', [])
                local = next((f for f in frames if '/opencood/' in f.get('filename', '')), None)
                frame = local or (frames[0] if frames else {})
                name = f"{frame.get('filename', 'unrecorded')}:{frame.get('line', 0)}:{frame.get('name', '')}"
                stack_bytes[name] += block.get('requested_size', size)
    result['largest_inactive_block_bytes'] = max(inactive_sizes, default=0)
    for label, low, high in [('lt_1m', 0, 2**20), ('1m_16m', 2**20, 16*2**20),
                             ('16m_256m', 16*2**20, 256*2**20), ('ge_256m', 256*2**20, float('inf'))]:
        result['inactive_' + label + '_bytes'] = sum(s for s in inactive_sizes if low <= s < high)
        result['inactive_' + label + '_blocks'] = sum(low <= s < high for s in inactive_sizes)
    return dict(result) | {'top_active_allocation_sites': [
        {'site': site, 'requested_bytes': size} for site, size in stack_bytes.most_common(12)]}


def tensor_storage_inventory(groups):
    """Count unique CUDA storage once; aliases/views must not look like copies.

    This describes explicit Python references only. The unexplained allocated
    bytes also include saved autograd activations and allocator rounding; they
    must not be labelled a leak or temporary allocation without lifecycle data.
    """
    seen, results = set(), {}
    def tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
    for name, values in groups.items():
        count = 0
        for tensor in tensors(values):
            if tensor.device.type != 'cuda' or tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                count += storage.nbytes()
        results[name + '_unique_storage_bytes'] = count
    results['known_unique_storage_bytes'] = sum(results.values())
    return results


def memory_phase_hook(recorder, name):
    def record_hook(module, inputs, result):
        recorder.record('module:' + name)
        # A non-None return value replaces the real module output in PyTorch.
    return record_hook


class PhaseRecorder:
    def __init__(self, device, model, optimizer, output_dir, snapshot_update, raw_snapshots):
        self.device, self.model, self.optimizer = device, model, optimizer
        self.output_dir, self.snapshot_update = output_dir, snapshot_update
        self.raw_snapshots = raw_snapshots
        self.rows, self.update, self.micro = [], 0, 0
        self.peak_allocated = self.peak_reserved = 0
        self.oom_count = 0
        self.history_enabled = False
        self.started = time.perf_counter()

    def record(self, phase, references=None):
        torch.cuda.synchronize(self.device)
        stats = torch.cuda.memory_stats(self.device)
        self.peak_allocated = max(self.peak_allocated, stats['allocated_bytes.all.peak'])
        self.peak_reserved = max(self.peak_reserved, stats['reserved_bytes.all.peak'])
        groups = {'parameters': list(self.model.parameters()), 'buffers': list(self.model.buffers()),
                  'gradients': [p.grad for p in self.model.parameters()],
                  'optimizer_state': self.optimizer.state, **(references or {})}
        inventory = tensor_storage_inventory(groups)
        row = {'update': self.update, 'micro': self.micro, 'phase': phase,
               'elapsed_seconds_with_instrumentation': time.perf_counter() - self.started,
               'stats': {key: value for key, value in stats.items() if key.endswith('.current') or key in
                         ('num_alloc_retries', 'num_ooms', 'num_sync_all_streams', 'num_device_alloc', 'num_device_free')},
               'interval_peak_allocated_bytes': stats['allocated_bytes.all.peak'],
               'interval_peak_reserved_bytes': stats['reserved_bytes.all.peak'],
               'inventory': inventory,
               'allocated_minus_known_storage_bytes': stats['allocated_bytes.all.current'] - inventory['known_unique_storage_bytes']}
        # memory_snapshot is public in installed PyTorch 2.1.2; history export
        # below uses the version-checked private API only for trace persistence.
        row['snapshot'] = snapshot_summary(torch.cuda.memory_snapshot(), self.device.index or 0)
        if stats.get('num_ooms', 0) > self.oom_count:
            # A library can catch an allocator OOM and choose another kernel.
            # Persist it immediately, before the bounded history ring wraps.
            raw = torch.cuda.memory._snapshot()
            row['new_allocator_oom_events'] = [event for trace in raw['device_traces'] for event in trace
                                               if event['action'] == 'oom']
            row['allocator_oom_count_increase'] = stats['num_ooms'] - self.oom_count
            if self.raw_snapshots:
                path = self.output_dir / f'u{self.update}_m{self.micro}_allocator_oom.pickle'
                with path.open('wb') as stream:
                    pickle.dump(raw, stream)
                row['allocator_oom_snapshot'] = str(path)
            self.oom_count = stats['num_ooms']
        free, total = torch.cuda.mem_get_info(self.device)
        row['device_free_bytes'], row['device_total_bytes'] = free, total
        if (self.raw_snapshots and self.update == self.snapshot_update and
                phase in ('forward', 'loss', 'backward', 'optimizer_zero_grad', 'release') and
                (self.micro == 1 or phase == 'optimizer_zero_grad')):
            path = self.output_dir / f'u{self.update}_m{self.micro}_{phase}.pickle'
            with path.open('wb') as stream:
                pickle.dump(torch.cuda.memory._snapshot(), stream)
            row['raw_snapshot'] = str(path)
        self.rows.append(row)
        torch.cuda.reset_peak_memory_stats(self.device)
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO/'cscwd_configs/project_v4_W_AB.yaml'))
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--sample-index', type=int, help='repeat this real train frame; otherwise use the registered sampler')
    parser.add_argument('--output', required=True, help='JSON report; raw snapshots use its suffix-free sibling directory')
    parser.add_argument('--snapshot-update', type=int, default=2, help='one bounded update to persist full allocation traces')
    parser.add_argument('--no-raw-snapshots', action='store_true')
    parser.add_argument('--workspace-release-probe', action='store_true',
                        help='once after all updates, release cuBLAS workspaces to attribute persistent allocations; never a training optimization')
    args = parser.parse_args()
    if not 1 <= args.steps <= 4 or not 1 <= args.snapshot_update <= args.steps:
        parser.error('allow 1..4 updates with snapshot-update inside that range')
    cfg = load_config(args.config)
    if cfg['training']['precision'] != 'fp32' or cfg['training']['world_size'] != 1:
        parser.error('requires the registered single-GPU FP32 configuration')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for allocator lifetime diagnosis')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output.with_suffix('')
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(cfg['resources']['cpu_threads'])
    seed_everything(cfg['experiment']['seed'], cfg['training']['deterministic'])
    report = {'schema': 'train_allocator_lifetime_v1', 'diagnostic_only': True, 'formal_experiment_started': False,
              'split': 'train', 'precision': 'fp32', 'torch_version': torch.__version__,
              'allocator_backend': torch.cuda.memory.get_allocator_backend(),
              'allocator_environment': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
              'config_sha256': config_hash(cfg), 'source_sha256': source_tree_hash(),
              'data_contract': data_contract_metadata(cfg), 'requested_optimizer_updates': args.steps,
              'sample_mode': 'repeated_train_frame' if args.sample_index is not None else 'registered_epoch_sampler',
              'sample_index': args.sample_index, 'micro_batch_size': cfg['training']['micro_batch_size'],
              'accumulation_steps': cfg['training']['accumulation_steps'], 'num_workers': cfg['resources']['num_workers'],
              'samples': [], 'updates': [], 'passed': False,
              'interpretation': {
                  'allocated': 'allocator bytes held by active tensor allocations, including saved graph tensors',
                  'active': 'allocated bytes plus blocks awaiting stream completion',
                  'reserved': 'CUDA segments held by PyTorch, including reusable cache and split inactive blocks',
                  'fragmentation': 'reserved-allocated is not fragmentation; inspect mixed inactive blocks, block sizes, retries and OOM request traces',
                  'inventory': 'deduplicated explicit CUDA storage; remainder includes unenumerated saved activations and rounding',
                  'timing': 'phase synchronization/snapshots add overhead; do not compare these timings as throughput',
                  'driver_memory': 'device free includes other processes and CUDA allocations outside this allocator'}}
    handles, resources, recorder = [], {}, None
    try:
        # Inspect the installed API signature before use: PyTorch 2.1.2 accepts
        # enabled/context/stacks/max_entries. It is diagnostic, never training.
        torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=20000)
        report['history_api'] = '_record_memory_history(enabled=all,context=all,stacks=python,max_entries=20000)'
        dataset, model, criterion = build_components(cfg)
        model.to(device).train()
        criterion.to(device)
        t = cfg['training']
        optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], eps=t['adam_eps'], weight_decay=t['weight_decay'])
        recorder = PhaseRecorder(device, model, optimizer, snapshot_dir, args.snapshot_update, not args.no_raw_snapshots)
        recorder.record('model_optimizer_before_state')
        if args.sample_index is None:
            sampler = EpochBatchSampler(len(dataset), t['micro_batch_size'], t['accumulation_steps'], cfg['experiment']['seed'])
        else:
            if not 0 <= args.sample_index < len(dataset):
                raise ValueError('sample-index outside train')
            sampler = [[args.sample_index] * t['micro_batch_size']] * (args.steps * t['accumulation_steps'])
        loader = make_loader(dataset, cfg, sampler=sampler)
        resources['loader'] = loader
        iterator = iter(loader)
        resources['iterator'] = iterator
        for name in ('lidar_pillar_vfe', 'radar_pillar_vfe', 'evidence_router', 'supported_residual', 'backbone'):
            module = getattr(model, name, None)
            if module is not None:
                handles.append(module.register_forward_hook(memory_phase_hook(recorder, name)))
        optimizer.zero_grad(set_to_none=True)
        for update in range(1, args.steps + 1):
            recorder.update = update
            loss_sum, loading_seconds = 0., 0.
            for micro in range(1, t['accumulation_steps'] + 1):
                recorder.micro = micro
                started = time.perf_counter()
                batch = next(iterator)
                loading_seconds += time.perf_counter() - started
                ego = batch['ego']
                weather = [agent['weather'] for scene in ego['metadata'] for agent in scene]
                if any(item not in ('normal', 'fog') for item in weather):
                    raise ValueError('forbidden weather in train-only resource diagnosis')
                report['samples'].append({'update': update, 'micro': micro, 'sample_ids': ego['sample_ids'],
                                           'agents': ego['record_len'].tolist(), 'weather': weather})
                inputs = to_device(select_model_inputs(ego), device)
                labels = to_device(ego['label_dict'], device)
                targets = to_device(batch.get('train_targets', {}), device)
                recorder.record('transfer', {'inputs': inputs, 'labels': labels, 'targets': targets})
                prediction = model(inputs, return_aux=True)
                references = {'inputs': inputs, 'labels': labels, 'targets': targets, 'prediction': prediction}
                recorder.record('forward', references)
                loss = criterion(prediction, labels, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite diagnostic loss')
                recorder.record('loss', references)
                (loss/t['accumulation_steps']).backward()
                recorder.record('backward', references)
                loss_sum += float(loss.detach()) / t['accumulation_steps']
                del references, prediction, loss, inputs, labels, targets, ego, batch
                recorder.record('release')
                if isinstance(sampler, EpochBatchSampler):
                    sampler.advance()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite diagnostic gradient')
            optimizer.step()
            recorder.record('optimizer_before_zero_grad')
            optimizer.zero_grad(set_to_none=True)
            recorder.record('optimizer_zero_grad')
            report['updates'].append({'update': update, 'loss': loss_sum,
                                      'loading_wait_seconds_with_instrumentation': loading_seconds})
            print(json.dumps(report['updates'][-1]), flush=True)
        if args.workspace_release_probe:
            if not hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
                raise RuntimeError('local PyTorch lacks the diagnostic cuBLAS workspace-release API')
            before = torch.cuda.memory_allocated(device)
            torch._C._cuda_clearCublasWorkspaces()
            after = recorder.record('diagnostic_clear_blas_workspaces')
            report['workspace_release_probe'] = {
                'api': 'torch._C._cuda_clearCublasWorkspaces', 'torch_version': torch.__version__,
                'released_allocated_bytes': before - after['stats']['allocated_bytes.all.current'],
                'interpretation': 'single post-diagnostic attribution only; not applied during training and not a resource/throughput benefit'}
        report['passed'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        if recorder is not None:
            try:
                recorder.record('failure')
                with (snapshot_dir/'failure.pickle').open('wb') as stream:
                    pickle.dump(torch.cuda.memory._snapshot(), stream)
            except Exception as snapshot_error:
                report['failure_snapshot_error'] = repr(snapshot_error)
        raise
    finally:
        for handle in handles:
            handle.remove()
        shutdown_loader(resources)
        if recorder is not None:
            report.update(phases=recorder.rows, peak_allocated_bytes=recorder.peak_allocated,
                          peak_reserved_bytes=recorder.peak_reserved)
        report['source_sha256_end'] = source_tree_hash()
        report['source_unchanged'] = report['source_sha256'] == report['source_sha256_end']
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == '__main__':
    main()
