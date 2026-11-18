#!/usr/bin/env python3
"""Download ZuCo benchmark test files, resuming .part files after interruptions.

No Python packages are required. Files receive their final .mat name only after
their size and Dropbox content hash match the published metadata.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("zuco_test_manifest.json")
BASE_URL = "https://files.osf.io/v1/resources/d7frw/providers/dropbox/"
STOP = threading.Event()
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def file_size(path):
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def content_hash(path):
    # https://www.dropbox.com/developers/reference/content-hash
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            if STOP.is_set():
                raise InterruptedError("Download stopped; partial files are preserved")
            digest.update(hashlib.sha256(block).digest())
    return digest.hexdigest()


def verify(path, entry):
    if file_size(path) != entry["size"]:
        raise RuntimeError(f"Unexpected size: {path}")
    if content_hash(path) != entry["dropbox_hash"]:
        raise RuntimeError(f"Checksum mismatch: {path}; file preserved for inspection")


def download(entry, destination):
    target = destination / entry["name"]
    partial = target.with_suffix(".mat.part")
    if target.exists():
        verify(target, entry)
        log(f"VERIFIED {entry['name']} (already downloaded)")
        return

    failures = 0
    while not STOP.is_set():
        offset = file_size(partial)
        if offset > entry["size"]:
            raise RuntimeError(f"Partial file is larger than expected: {partial}")
        if offset == entry["size"]:
            verify(partial, entry)
            partial.rename(target)
            log(f"VERIFIED {entry['name']}")
            return

        log(f"DOWNLOAD {entry['name']} from {offset:,} / {entry['size']:,} bytes")
        # A fresh curl after each failure makes -C - use the newly saved offset.
        # curl rejects a server that ignores the requested resume range.
        command = [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--continue-at", "-", "--connect-timeout", "20",
            "--max-time", "300", "--speed-limit", "16384", "--speed-time", "60",
            "--output", str(partial), BASE_URL + entry["name"],
        ]
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as process:
            while True:
                try:
                    _, error = process.communicate(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if STOP.is_set():
                        process.terminate()
                        process.communicate()
                        raise InterruptedError("Download stopped; partial files are preserved")

        saved = file_size(partial)
        if saved == entry["size"]:
            continue
        if process.returncode in (23, 33):
            raise RuntimeError(f"Cannot save/resume {entry['name']}: {error.strip()}")
        failures = 0 if saved > offset else failures + 1
        if failures >= 20:
            raise RuntimeError(f"20 attempts without progress: {entry['name']}: {error.strip()}")
        log(f"RETRY {entry['name']}; saved {saved:,} bytes; {error.strip()}")
        STOP.wait(min(30, 2 ** min(failures + 1, 5)))

    raise InterruptedError("Download stopped; partial files are preserved")


def progress(entries, destination):
    total = sum(entry["size"] for entry in entries)
    saved = 0
    complete = 0
    for entry in entries:
        target = destination / entry["name"]
        if target.exists():
            complete += file_size(target) == entry["size"]
            saved += min(file_size(target), entry["size"])
        else:
            saved += min(file_size(target.with_suffix(".mat.part")), entry["size"])
    return saved, total, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/local/data/test")
    parser.add_argument("--jobs", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--status", action="store_true", help="Show saved bytes without network access")
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text())["files"]
    destination = args.destination.resolve()
    if args.status:
        saved, total, complete = progress(entries, destination)
        print(f"{saved / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * saved / total:.1f}%); "
              f"{complete}/{len(entries)} completed files\n{destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another downloader is already running for this directory")
        saved, total, _ = progress(entries, destination)
        if shutil.disk_usage(destination).free < total - saved + 2 * 1024 ** 3:
            raise SystemExit("Not enough free disk space (including a 2 GiB reserve)")
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: STOP.set())
        log(f"START {len(entries)} files, {total / 1e9:.2f} GB; destination: {destination}")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = {pool.submit(download, entry, destination) for entry in entries}
            previous, _, _ = progress(entries, destination)
            previous_time = time.monotonic()
            errors = []
            while pending:
                done, pending = wait(pending, timeout=20)
                for future in done:
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(str(error))
                        log(f"ERROR {error}")
                        STOP.set()
                saved, total, complete = progress(entries, destination)
                now = time.monotonic()
                speed = (saved - previous) / max(now - previous_time, 0.001) / 1e6
                log(f"PROGRESS {saved / 1e9:.2f}/{total / 1e9:.2f} GB "
                    f"({100 * saved / total:.1f}%); {speed:.2f} MB/s; {complete}/{len(entries)} verified")
                previous, previous_time = saved, now
        if errors or STOP.is_set():
            raise SystemExit("Stopped. Run this command again to resume saved .part files.")
        log("COMPLETE: all files match their published Dropbox content hashes")


if __name__ == "__main__":
    main()
