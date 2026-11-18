# Data acquisition

The benchmark splits and NEMAR release have different coverage. In particular, a download from OSF project d7frw alone provides the benchmark test set. The benchmark authors name separate OSF projects for the training files. [Authors’ download script](https://github.com/norahollenstein/zuco-benchmark/blob/main/get_data.sh)

| Export | Source | Contents and integrity check |
| --- | --- | --- |
| MATLAB train | [NR](https://osf.io/2abup/) and [TSR](https://osf.io/6etg8/), OSF Storage, Features folder | 16 participants, one file per task and participant. The SHA-256 values in train_files.json identify the local files audited here. |
| MATLAB test | [OSF d7frw](https://osf.io/d7frw/), Dropbox provider | 10 X participants, 37.32 GB. zuco_test_manifest.json preserves the provider sizes and Dropbox content hashes. |
| NEMAR | [nm000279 v1.0.0 manifest](https://data.nemar.org/nm000279/v1.0.0/manifest.json) | 18 Y participants, 19,698,373,353 bytes. nemar_manifest.json is the published manifest, including the SHA-256 or Git blob checksum for each file. |

Run from the repository root with Python 3.11 or later, curl and a POSIX shell. The audit dependency versions in requirements.txt are those used for the release check with Python 3.14. osfclient supplies the osf executable used for train.

```sh
python -m pip install -r requirements.txt
python data/download_train.py
python data/download_nemar_missing.py
# Optional for the historical train + test MATLAB inventory
python data/download_zuco.py
```

By default, files go under `data/local/`, with subpaths `data/train/`, `data/test/` and `data.nemar.org/nm000279/v1.0.0/`. For an existing collection, set `ZUCO2_INPUT_ROOT` to the directory containing those subpaths before running the audits. To change a downloader destination, pass `--destination` explicitly.

The Dropbox and NEMAR downloaders are copies of the tools used during acquisition, with local paths adapted. Both use resumable transfers and verify a file before installing it. The NEMAR downloader preserves any replaced file and requires a fresh `--out` directory after a completed run. The small train wrapper follows the authors’ OSF layout and fetches individual files. It verifies existing files and downloads missing ones to temporary files. An interrupted train file restarts from the beginning.

For this release the train wrapper rehashed all 32 local files and the NEMAR downloader verified all 2,411 local entries with zero transfers. The archived test verification matches the ten provider hashes. The X files were unavailable for a fresh read. Network transfers from an empty directory were outside this reproduction.
