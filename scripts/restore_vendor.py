"""Restore the pinned, patched Dynamax source for research (no network)."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
archive = root / 'research/dynamax-source.tar.gz'
manifest = json.loads((root / 'research/dynamax-source.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
destination = root / 'research/vendor/dynamax'
if destination.exists():
    print('Existing vendor checkout preserved:', destination)
else:
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter='data')
    print('Restored pinned source:', destination)
