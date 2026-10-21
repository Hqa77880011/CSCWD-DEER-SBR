"""Download the pinned V2X-R release, stream-extract and audit, then remove archives.

Uses aria2 for resumable transfer and native 7z for a single decompression pass.
Each source archive is verified against the Hub LFS SHA256 (or Git blob SHA1),
each unpacked file against its archive CRC32, and retained files get SHA256s.
PNG pixels are discarded while PNG IHDR dimensions are retained. Cleanup is
explicitly authorized for this task, after verification and durable manifests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler
import zlib

ROOT = Path('/root/autodl-tmp/CSCWD')
REVISION = '4636ce397b214aee910cfc8ccf25a2dfac1a7906'
REPO = 'hx24/V2X-R'
ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
ARCHIVES = ROOT / 'datasets/v2xr_archives'
DATA = ROOT / 'datasets/v2xr'
STATE = ROOT / 'setup/full_v2xr_state.json'
PROGRESS = ROOT / 'setup/full_v2xr_progress.json'
LOGS = ROOT / 'setup/full_v2xr_logs'
LOCK = threading.RLock()
RECORDS = {}
ENTRIES = []
STOP = threading.Event()


class Redirect308Handler(HTTPRedirectHandler):
    """Python 3.10 needs an explicit handler for permanent HTTP 308 redirects."""

    def http_error_308(self, request, response, code, message, headers):
        # 307 preserves the method and request headers with the same GET semantics.
        return self.http_error_302(request, response, 307, message, headers)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(threading.get_ident()))
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def update(path, **fields):
    with LOCK:
        RECORDS.setdefault(path, {}).update(fields, updated_at=now())
        atomic_json(STATE, {'repo_id': REPO, 'revision': REVISION, 'endpoint': ENDPOINT,
                            'updated_at': now(), 'files': RECORDS})


def key(path):
    return path.replace('/', '__').replace('.7z', '')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_source(entry, path):
    if path.stat().st_size != entry['size']:
        raise RuntimeError('Wrong source size: ' + entry['path'])
    digest = sha256(path)
    if entry.get('lfs'):
        if digest != entry['lfs']['oid']:
            raise RuntimeError('Source SHA256 mismatch: ' + entry['path'])
    else:
        blob = hashlib.sha1(b'blob ' + str(entry['size']).encode() + b'\0' + path.read_bytes()).hexdigest()
        if blob != entry['oid']:
            raise RuntimeError('Source Git blob mismatch: ' + entry['path'])
    return digest


def http_download(entry, dest):
    partial = dest.with_suffix(dest.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == entry['size']:
        verify_source(entry, partial)
        partial.replace(dest)
        return
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    url = f'{ENDPOINT}/datasets/{REPO}/resolve/{REVISION}/{entry["path"]}?download=true'
    with build_opener(Redirect308Handler()).open(Request(url, headers=headers), timeout=45) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Invalid resumed Content-Range')
        with partial.open('ab' if append else 'wb') as stream:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
    partial.replace(dest)


def fetch(entry):
    name = entry['path']
    with LOCK:
        previous = dict(RECORDS.get(name, {}))
    if previous.get('status') == 'extracted_verified' and previous.get('manifest'):
        manifest_path = Path(previous['manifest'])
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if all((DATA / x['path']).is_file() and (DATA / x['path']).stat().st_size == x['bytes']
                   for x in manifest['retained_files']):
                return None
    if previous.get('status') == 'metadata_verified':
        target = DATA / name
        if target.is_file() and sha256(target) == previous['sha256']:
            return None
    dest = ARCHIVES / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    control = Path(str(dest) + '.aria2')
    if dest.is_file() and not control.exists():
        try:
            digest = verify_source(entry, dest)
            update(name, status='downloaded_verified', source_bytes=entry['size'], sha256=digest)
            return dest
        except RuntimeError:
            # Keep any unexpected pre-existing file for inspection.
            dest.rename(dest.with_name(dest.name + '.invalid-' + str(int(time.time()))))
    if shutil.disk_usage(ROOT).free < entry['size'] + 2 * 2**30:
        raise RuntimeError('Insufficient staging space for ' + name)
    update(name, status='downloading', source_bytes=entry['size'])
    url = f'{ENDPOINT}/datasets/{REPO}/resolve/{REVISION}/{name}?download=true'
    log_path = LOGS / (key(name) + '.download.log')
    for attempt in range(1, 7):
        try:
            if entry['size'] > 1024 * 1024:
                command = ['aria2c', '--continue=true', '--auto-file-renaming=false',
                           '--file-allocation=none', '--max-connection-per-server=8', '--split=8',
                           '--min-split-size=8M', '--max-tries=4', '--retry-wait=3',
                           '--timeout=45', '--connect-timeout=20', '--console-log-level=warn',
                           '--summary-interval=0', '--show-console-readout=false',
                           '--download-result=hide', '--dir=' + str(dest.parent),
                           '--out=' + dest.name, url]
                with log_path.open('ab') as log:
                    result = subprocess.run(command, stdout=log, stderr=log)
                if result.returncode:
                    raise RuntimeError('aria2 transfer failed: ' + str(result.returncode))
            else:
                http_download(entry, dest)
            digest = verify_source(entry, dest)
            update(name, status='downloaded_verified', source_bytes=entry['size'], sha256=digest)
            print(now(), 'DOWNLOADED', name, entry['size'], flush=True)
            return dest
        except Exception as exc:
            update(name, status='retrying', attempt=attempt, error=str(exc))
            if attempt == 6:
                raise
            time.sleep(min(30, 2 * attempt))
            update(name, status='downloading')


def archive_listing(archive):
    result = subprocess.run(['7z', 'l', '-slt', str(archive)], capture_output=True, text=True, check=True)
    content = result.stdout.split('----------\n', 1)[1]
    entries = []
    for block in content.strip().split('\n\n'):
        fields = dict(line.split(' = ', 1) for line in block.splitlines() if ' = ' in line)
        if not fields.get('Path'):
            continue
        name = PurePosixPath(fields['Path'])
        if name.is_absolute() or '..' in name.parts or '\\' in str(name):
            raise RuntimeError('Unsafe archive path')
        if fields.get('Attributes', '').startswith('D'):
            continue
        entries.append({'path': str(name), 'bytes': int(fields['Size']), 'crc32': fields.get('CRC', '')})
    return entries


def extract(entry, archive):
    name = entry['path']
    if archive is None:
        return
    if not name.endswith('.7z'):
        dest = DATA / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, dest)
        update(name, status='metadata_verified', sha256=sha256(dest), source_bytes=entry['size'])
        return
    members = archive_listing(archive)
    keep_bytes = sum(x['bytes'] for x in members if not x['path'].lower().endswith('.png'))
    if shutil.disk_usage(ROOT).free < keep_bytes + 2 * 2**30:
        raise RuntimeError('Insufficient extraction space for ' + name)
    update(name, status='extracting', retained_bytes=keep_bytes, member_count=len(members))
    retained = []
    images = {}
    sequence = PurePosixPath(name).parent
    log_path = LOGS / (key(name) + '.extract.log')
    started = time.monotonic()
    with log_path.open('wb') as log:
        process = subprocess.Popen(['7z', 'x', '-so', '-bd', '-bb0', '-mmt=2', str(archive)],
                                   stdout=subprocess.PIPE, stderr=log)
        try:
            for member in members:
                image = member['path'].lower().endswith('.png')
                relative = str(sequence / member['path'])
                target = DATA / relative
                crc = 0
                digest = hashlib.sha256()
                remaining = member['bytes']
                header = bytearray()
                output = None
                temp = None
                if not image:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(target.suffix + '.extracting')
                    output = temp.open('wb')
                try:
                    while remaining:
                        block = process.stdout.read(min(1024 * 1024, remaining))
                        if not block:
                            raise RuntimeError('Early archive EOF: ' + relative)
                        remaining -= len(block)
                        crc = zlib.crc32(block, crc)
                        if image:
                            if len(header) < 24:
                                header.extend(block[:24 - len(header)])
                        else:
                            digest.update(block)
                            output.write(block)
                finally:
                    if output:
                        output.close()
                if member['crc32'] and f'{crc:08X}' != member['crc32']:
                    raise RuntimeError('Unpacked CRC32 mismatch: ' + relative)
                if image:
                    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
                        raise RuntimeError('Invalid PNG IHDR: ' + relative)
                    width, height = struct.unpack('>II', header[16:24])
                    if width <= 0 or height <= 0:
                        raise RuntimeError('Invalid PNG dimensions')
                    images[relative] = [width, height]
                else:
                    temp.replace(target)
                    retained.append({'path': relative, 'bytes': member['bytes'],
                                     'crc32': member['crc32'], 'sha256': digest.hexdigest()})
            if process.stdout.read(1):
                raise RuntimeError('Unexpected unlisted archive bytes')
            if process.wait() != 0:
                raise RuntimeError('7z reported an archive integrity error')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    manifest_path = DATA / 'manifests/archives' / (key(name) + '.json')
    with LOCK:
        source_sha = RECORDS[name]['sha256']
    manifest = {'source_repo': REPO, 'revision': REVISION, 'archive': name,
                'archive_bytes': entry['size'], 'archive_sha256': source_sha,
                'verified_at': now(), 'retained_files': retained, 'image_sizes': images,
                'empty_source_archive': not members, 'rgb_pixels_saved': False}
    atomic_json(manifest_path, manifest)
    # Flush data and manifests before cleaning an authorized, verified source archive.
    os.sync()
    archive.unlink()
    update(name, status='extracted_verified', manifest=str(manifest_path), archive_removed=True,
           retained_bytes=keep_bytes, retained_file_count=len(retained),
           image_header_count=len(images), empty_source_archive=not members,
           extraction_seconds=round(time.monotonic() - started, 1))
    print(now(), 'EXTRACTED', name, len(retained), 'files;', len(images),
          'image headers; archive cleaned', flush=True)


def report_progress():
    while not STOP.wait(15):
        with LOCK:
            statuses = {}
            verified_bytes = 0
            approximate_transferred = 0
            for entry in ENTRIES:
                record = RECORDS.get(entry['path'], {})
                status = record.get('status', 'pending')
                statuses[status] = statuses.get(status, 0) + 1
                if status in ('extracted_verified', 'metadata_verified', 'extracting', 'downloaded_verified'):
                    verified_bytes += entry['size']
                    approximate_transferred += entry['size']
                elif status in ('downloading', 'retrying'):
                    path = ARCHIVES / entry['path']
                    try:
                        approximate_transferred += min(entry['size'], path.stat().st_blocks * 512)
                    except FileNotFoundError:
                        pass
            total = sum(x['size'] for x in ENTRIES)
            snapshot = {'updated_at': now(), 'status': 'running', 'total_source_bytes': total,
                        'endpoint': ENDPOINT,
                        'verified_source_bytes': verified_bytes,
                        'approximate_transferred_bytes': approximate_transferred,
                        'source_verified_percent': round(100 * verified_bytes / total, 2),
                        'file_states': statuses, 'disk_free_bytes': shutil.disk_usage(ROOT).free}
            atomic_json(PROGRESS, snapshot)


def finalize():
    archives = []
    images = {}
    for record in RECORDS.values():
        if record.get('status') == 'extracted_verified':
            path = Path(record['manifest'])
            manifest = json.loads(path.read_text())
            images.update(manifest['image_sizes'])
            archives.append({'manifest': str(path.relative_to(DATA)), 'archive': manifest['archive'],
                             'source_sha256': manifest['archive_sha256'],
                             'empty_source_archive': manifest['empty_source_archive']})
    atomic_json(DATA / 'image_sizes.json', {'schema': 'png_ihdr_sizes_v1', 'root': str(DATA),
                'source_revision': REVISION, 'images': images, 'rgb_pixels_saved': False})
    split_report = {}
    paired_errors = []
    for split in ('train', 'validate', 'test'):
        samples = 0
        frames = 0
        sequences = []
        for seq in sorted((DATA / split).iterdir()):
            if not seq.is_dir():
                continue
            agents = []
            for agent in sorted(seq.iterdir()):
                if not agent.is_dir():
                    continue
                timestamps = sorted(x.stem for x in agent.glob('*.yaml'))
                if not timestamps:
                    continue
                for stamp in timestamps:
                    for suffix in ('.pcd', '_radar.pcd', '_fog_0.060.pcd', '_snow_2.5_2.0.pcd'):
                        p = agent / (stamp + suffix)
                        if not p.is_file():
                            paired_errors.append(str(p.relative_to(DATA)))
                    image_key = str((agent / (stamp + '_camera0.png')).relative_to(DATA))
                    if image_key not in images:
                        paired_errors.append('image_size:' + image_key)
                agents.append({'agent': agent.name, 'frames': len(timestamps), 'timestamps': timestamps})
                frames += len(timestamps)
            if agents:
                reference = agents[0]['timestamps']
                if any(a['timestamps'] != reference for a in agents):
                    paired_errors.append('agent_timestamp_alignment:' + str(seq.relative_to(DATA)))
                samples += len(reference)
                sequences.append({'sequence': seq.name, 'cooperative_samples': len(reference),
                                  'agents': [{k: v for k, v in a.items() if k != 'timestamps'} for a in agents]})
        split_report[split] = {'cooperative_samples': samples, 'agent_frames': frames, 'sequences': sequences}
    expected = {'train': 8084, 'validate': 829, 'test': 3166}
    audit = {'source_repo': REPO, 'revision': REVISION, 'completed_at': now(),
             'all_source_files_verified': all(RECORDS.get(e['path'], {}).get('status')
                  in ('extracted_verified', 'metadata_verified') for e in ENTRIES),
             'splits': split_report, 'paper_expected_cooperative_samples': expected,
             'paper_sample_counts_match': all(split_report[k]['cooperative_samples'] == n for k, n in expected.items()),
             'pairing_errors': paired_errors, 'image_metadata_entries': len(images),
             'empty_source_archives': [x['archive'] for x in archives if x['empty_source_archive']],
             'rgb_pixels_saved': False, 'fixed_evaluation_gt_created': False}
    atomic_json(DATA / 'data_audit.json', audit)
    atomic_json(DATA / 'data_manifest.json', {'source_repo': REPO, 'revision': REVISION,
                 'archives': archives, 'checksums': 'per-archive manifests include each retained file SHA256 and CRC32',
                 'image_sizes': 'image_sizes.json', 'audit': 'data_audit.json'})
    complete = audit['all_source_files_verified'] and not paired_errors
    atomic_json(PROGRESS, {'updated_at': now(), 'status': 'complete' if complete else 'audit_attention',
                           'endpoint': ENDPOINT,
                           'verified_source_bytes': sum(x['size'] for x in ENTRIES),
                           'total_source_bytes': sum(x['size'] for x in ENTRIES),
                           'source_verified_percent': 100, 'audit': str(DATA / 'data_audit.json'),
                           'disk_free_bytes': shutil.disk_usage(ROOT).free})
    print(now(), 'FINISHED', json.dumps({k: v['cooperative_samples'] for k, v in split_report.items()}),
          'pairing_errors', len(paired_errors), flush=True)


def main():
    global ENTRIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--probe-only', action='store_true',
                        help='Download and verify one small source metadata file, then exit')
    args = parser.parse_args()
    if not ENDPOINT.startswith('https://'):
        raise ValueError('HF_ENDPOINT must be an HTTPS Hugging Face compatible endpoint')
    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    job_lock = (ROOT / 'setup/full_v2xr.lock').open('w')
    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'setup/full_v2xr.pid').write_text(str(os.getpid()) + '\n')
    source = json.loads((ROOT / 'setup/v2xr_full_remote_manifest.json').read_text())
    if source['revision'] != REVISION:
        raise RuntimeError('Pinned source revision changed')
    ENTRIES = source['files']
    if STATE.is_file():
        old = json.loads(STATE.read_text())
        if old['revision'] != REVISION:
            raise RuntimeError('Cannot resume a different data revision')
        RECORDS.update(old['files'])
    print(now(), 'CONFIG', json.dumps({'endpoint': ENDPOINT, 'revision': REVISION,
                                      'workers': args.workers, 'repo_id': REPO}), flush=True)
    if args.probe_only:
        entry = next(e for e in ENTRIES if e['path'] == '.gitattributes')
        archive = fetch(entry)
        extract(entry, archive)
        atomic_json(ROOT / 'setup/v2xr_endpoint_probe.json',
                    {'endpoint': ENDPOINT, 'revision': REVISION, 'file': entry['path'],
                     'status': 'passed', 'source_checksum_verified': True, 'checked_at': now()})
        print('PASS endpoint metadata download and pinned source checksum', flush=True)
        return
    cached = [e for e in ENTRIES if (ARCHIVES / e['path']).is_file()]
    other = [e for e in ENTRIES if e not in cached]
    ordered = sorted(cached, key=lambda e: e['size']) + sorted(other, key=lambda e: -e['size'])
    progress_thread = threading.Thread(target=report_progress, daemon=True)
    progress_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            iterator = iter(ordered)
            for _ in range(args.workers):
                entry = next(iterator, None)
                if entry is not None:
                    pending[pool.submit(fetch, entry)] = entry
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    entry = pending.pop(future)
                    archive = future.result()
                    extract(entry, archive)
                    following = next(iterator, None)
                    if following is not None:
                        pending[pool.submit(fetch, following)] = following
    except BaseException as exc:
        atomic_json(PROGRESS, {'updated_at': now(), 'status': 'failed', 'error': str(exc),
                               'endpoint': ENDPOINT,
                               'resume_command': 'python project_tools/download_full_v2xr.py'})
        raise
    finally:
        STOP.set()
        progress_thread.join(timeout=20)
    finalize()


if __name__ == '__main__':
    main()
