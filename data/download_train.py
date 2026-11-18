"""Fetch the 32 benchmark training MAT files from the authors' OSF Features folders."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def verify(path, entry):
    if path.stat().st_size != entry['size']:
        raise ValueError(f'Unexpected size: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'Content differs from the audited train export: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parent/'local/data/train')
    parser.add_argument('--verify-only', action='store_true', help='Check local files without contacting OSF')
    args = parser.parse_args()
    entries = json.loads(Path(__file__).with_name('train_files.json').read_text())
    if not args.verify_only:
        args.destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        target = args.destination / entry['name']
        if target.exists() or args.verify_only:
            verify(target, entry)
        else:
            with tempfile.TemporaryDirectory(dir=args.destination) as staging:
                payload = Path(staging)/target.name
                subprocess.run(['osf', '-p', entry['project'], 'fetch',
                                'osfstorage/Features/' + target.name, str(payload)], check=True)
                verify(payload, entry)
                payload.replace(target)
        print('Verified', target.name, flush=True)


if __name__ == '__main__':
    main()
