"""Bounded CPU readers; RGB bit fields are decoded without reading images."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
import yaml

class AnnotationLoader(yaml.CSafeLoader):
    pass

def _dtype(loader, node):
    d = loader.construct_mapping(node, deep=True)
    dtype = np.dtype(d['args'][0])
    if dtype.kind not in 'fiu' or dtype.itemsize > 8:
        raise ValueError('Unsupported numeric YAML dtype')
    return dtype.newbyteorder(d['state'][1])

def _scalar(loader, node):
    dtype, data = loader.construct_sequence(node, deep=True)
    if len(data) != dtype.itemsize:
        raise ValueError('Invalid numeric scalar payload')
    return np.frombuffer(data, dtype=dtype, count=1)[0].item()

AnnotationLoader.add_constructor('tag:yaml.org,2002:python/tuple', lambda l,n: tuple(l.construct_sequence(n,deep=True)))
AnnotationLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.dtype', _dtype)
AnnotationLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', _scalar)

def read_annotation(path, expected_sha256=None):
    payload=Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError(f'Annotation source SHA256 mismatch: {path}')
    return yaml.load(payload, Loader=AnnotationLoader)

def read_pcd(path, expected_sha256=None):
    """Same xyz + red-channel feature as upstream; never expose green labels.

    The red channel is the source framework's encoded scalar, not a newly
    calibrated velocity in m/s. No re-normalisation or radar transform occurs.
    """
    with open(path, 'rb') as stream:
        header = {}
        digest=hashlib.sha256() if expected_sha256 else None
        while True:
            raw = stream.readline()
            if digest is not None:digest.update(raw)
            if not raw:
                raise ValueError(f'Truncated PCD: {path}')
            line = raw.decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, _, value = line.partition(' ')
            header[key] = value
            if key == 'DATA':
                break
        if (header['FIELDS'] != 'x y z rgb' or header['SIZE'] != '4 4 4 4'
                or header['TYPE'] != 'F F F F' or header['COUNT'] != '1 1 1 1'):
            raise ValueError(f'Unsupported PCD schema: {path}')
        payload = stream.read()
        if digest is not None:
            digest.update(payload)
            if digest.hexdigest()!=expected_sha256:
                raise ValueError(f'PCD source SHA256 mismatch: {path}')
    if header['DATA'] == 'ascii':
        raw = np.fromstring(payload, sep=' ', dtype=np.float32)
    elif header['DATA'] == 'binary':
        raw = np.frombuffer(payload, dtype='<f4')
    else:
        raise ValueError(f'Unsupported PCD encoding: {header["DATA"]}')
    if raw.size != int(header['POINTS']) * 4:
        raise ValueError(f'PCD point count mismatch: {path}')
    raw = raw.reshape(-1,4)
    if not np.isfinite(raw).all():
        raise ValueError(f'Nonfinite PCD: {path}')
    points = np.empty(raw.shape, dtype=np.float32)
    points[:,:3] = raw[:,:3]
    rgb = np.ascontiguousarray(raw[:,3]).view(np.uint32)
    points[:,3] = ((rgb >> 16) & 255).astype(np.float32) / 255.0
    return points

class ByteLRU:
    """A per-worker cache with a byte budget, never an unbounded frame cache."""
    def __init__(self, max_bytes=0):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self.items = OrderedDict()
        self.expected_sha256 = {}
    def get(self, path, kind='pcd'):
        path = Path(path)
        stat = path.stat()
        expected=self.expected_sha256.get(str(path))
        key = (str(path), stat.st_size, stat.st_mtime_ns, kind, expected)
        if key in self.items:
            value, size = self.items.pop(key)
            self.items[key] = (value,size)
            return value
        value = read_pcd(path,expected) if kind == 'pcd' else read_annotation(path,expected)
        size = value.nbytes if kind == 'pcd' else max(1024,stat.st_size*6)
        if self.max_bytes and size <= self.max_bytes:
            while self.items and self.bytes + size > self.max_bytes:
                _, (_, old_size) = self.items.popitem(last=False)
                self.bytes -= old_size
            self.items[key] = (value,size)
            self.bytes += size
        return value
