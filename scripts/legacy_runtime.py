"""Run the original seven-group code without weakening its release identity."""
import hashlib
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import zipfile

ORIGINAL_SOURCE = '71459fab18d8ad489c84cf3002bd283750b15444964f09f359b0772fcbf697e1'


def execute_original(script):
    repository = Path(__file__).resolve().parents[1]
    archive = repository / 'repro/archive/legacy_v4_runtime.zip'
    # Each invocation owns its source directory, including concurrent GPU runs.
    with tempfile.TemporaryDirectory(prefix='cscwd_legacy_v4_') as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(archive) as sources:
            for info in sources.infolist():
                target = (directory / info.filename).resolve()
                if directory.resolve() not in target.parents:
                    raise ValueError('unsafe path in original runtime archive')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(sources.read(info))
        # Extensions are rebuilt on the target host, outside this source-only
        # archive. Make those ABI-specific binaries visible to the original
        # package without importing the modified primary Python sources.
        for binary in (repository / 'opencood').rglob('*'):
            if binary.is_file() and binary.suffix in ('.so', '.pyd'):
                target = directory / binary.relative_to(repository)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(binary, target)
        source_hash = hashlib.sha256()
        files = list((directory / 'opencood').rglob('*.py')) + list((directory / 'project_tools').glob('*.py'))
        for path in sorted(files, key=lambda p: p.relative_to(directory).as_posix()):
            source_hash.update(path.relative_to(directory).as_posix().encode())
            source_hash.update(hashlib.sha256(path.read_bytes()).digest())
        if source_hash.hexdigest() != ORIGINAL_SOURCE:
            raise ValueError('original scientific source does not match the recorded seven-group release')
        sys.argv[0] = str(directory / 'scripts' / script)
        # The source lives temporarily elsewhere; user-facing defaults still
        # refer to the checkout and must survive after this process exits.
        defaults = {
            '--data-root': repository / 'datasets/v2xr',
            '--output-root': repository / 'outputs' / (
                'pretrained_evaluation' if script == 'pretrained.py' else 'reproduction'),
        }
        if script == 'pretrained.py':
            defaults['--weights-dir'] = repository / 'checkpoints/v1.0.0'
        for option, path in defaults.items():
            if not any(arg == option or arg.startswith(option + '=') for arg in sys.argv[1:]):
                sys.argv.extend([option, str(path)])
        sys.path.insert(0, str(directory))
        runpy.run_path(sys.argv[0], run_name='__main__')
