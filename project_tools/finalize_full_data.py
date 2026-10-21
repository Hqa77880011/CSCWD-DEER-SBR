"""Save the download outcome after the detached worker finishes its full audit."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/CSCWD')


def main():
    v2xr = json.loads((ROOT / 'datasets/v2xr/data_audit.json').read_text())
    astyx = json.loads((ROOT / 'datasets/astyx/data_audit.json').read_text())
    success = (v2xr['all_source_files_verified'] and not v2xr['pairing_errors']
               and astyx['status'] == 'passed' and astyx['sample_count'] == 546)
    result = {'updated_at': datetime.now(timezone.utc).isoformat(),
              'status': 'complete' if success else 'requires_attention',
              'v2xr': {'all_source_files_verified': v2xr['all_source_files_verified'],
                       'cooperative_samples': {s: d['cooperative_samples'] for s, d in v2xr['splits'].items()},
                       'agent_frames': sum(d['agent_frames'] for d in v2xr['splits'].values()),
                       'pairing_errors': len(v2xr['pairing_errors']),
                       'paper_sample_counts_match': v2xr['paper_sample_counts_match'],
                       'empty_source_archives': v2xr['empty_source_archives']},
              'astyx': {'samples': 546, 'sensor_files': astyx['verified_sensor_files'], 'status': astyx['status']},
              'rgb_downloaded': False, 'training_started': False,
              'paths': {'v2xr': str(ROOT / 'datasets/v2xr'),
                        'astyx': str(ROOT / 'datasets/astyx/dataset_astyx_hires2019')}}
    (ROOT / 'setup/full_data_status.json').write_text(json.dumps(result, indent=2) + '\n')
    path = ROOT / 'setup/setup_status.json'
    setup = json.loads(path.read_text())
    setup['phase'] = 'full_source_data_downloaded' if success else 'data_audit_requires_attention'
    setup['full_data_download'] = result
    setup['formal_training_ready'] = False
    setup['pending'] = [x for x in setup.get('pending', []) if x != 'full_data_storage_and_official_splits']
    path.write_text(json.dumps(setup, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not success:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
