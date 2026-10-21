"""Allocator accounting and noninvasive hooks can be checked without CUDA."""
import unittest
import torch
from project_tools.profile_memory import memory_phase_hook, snapshot_summary


class MemoryProfileTests(unittest.TestCase):
    def test_forward_hook_never_replaces_real_outputs(self):
        class Recorder:
            phases = []
            def record(self, phase):
                self.phases.append(phase)
                return {'diagnostic': 'must not become the output'}
        recorder = Recorder()
        module = torch.nn.Identity()
        handle = module.register_forward_hook(memory_phase_hook(recorder, 'vfe'))
        value = {'pillar_features': torch.ones(2, 3)}
        try:
            self.assertIs(module(value), value)
            self.assertEqual(recorder.phases, ['module:vfe'])
        finally:
            handle.remove()

    def test_split_cache_and_pending_free_are_distinct(self):
        segments = [
            dict(device=0, total_size=100, blocks=[
                dict(state='active_allocated', size=50, requested_size=48, frames=[]),
                dict(state='active_awaiting_free', size=10),
                dict(state='inactive', size=40)]),
            dict(device=0, total_size=80, blocks=[dict(state='inactive', size=80)]),
            dict(device=1, total_size=999, blocks=[dict(state='inactive', size=999)])]
        summary = snapshot_summary(segments, device_index=0)
        self.assertEqual(summary['segment_bytes'], 180)
        self.assertEqual(summary['active_requested_bytes'], 48)
        self.assertEqual(summary['active_awaiting_free_bytes'], 10)
        self.assertEqual(summary['mixed_segment_inactive_bytes'], 40)
        self.assertEqual(summary['wholly_free_segment_bytes'], 80)
        self.assertEqual(summary['largest_inactive_block_bytes'], 80)
        self.assertEqual(summary['inactive_lt_1m_bytes'], 120)


if __name__ == '__main__':
    unittest.main()
