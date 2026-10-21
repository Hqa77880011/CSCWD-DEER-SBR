"""Show the live state of the persistent full-data download."""
import json
from pathlib import Path

root = Path('/root/autodl-tmp/CSCWD')
complete = root / 'setup/full_data_status.json'
if complete.exists():
    print(complete.read_text())
else:
    progress = root / 'setup/full_v2xr_progress.json'
    if progress.exists():
        data = json.loads(progress.read_text())
        print('V2X-R:', data['status'])
        print('Source bytes verified: %.2f / %.2f GiB' %
              (data.get('verified_source_bytes', 0) / 2**30, data.get('total_source_bytes', 0) / 2**30))
        if 'approximate_transferred_bytes' in data:
            print('Approximate bytes received: %.2f GiB' % (data['approximate_transferred_bytes'] / 2**30))
        print('File states:', data.get('file_states', {}))
        print('Updated (UTC):', data['updated_at'])
        if 'error' in data:
            print('Error:', data['error'])
    astyx = root / 'datasets/astyx/data_audit.json'
    if astyx.exists():
        data = json.loads(astyx.read_text())
        print('Astyx:', data['status'], data['sample_count'], 'samples,', data['verified_sensor_files'], 'sensor files')
