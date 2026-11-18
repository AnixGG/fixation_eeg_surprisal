# Audit sequence and reproduction

Start with the [data layout](../data/README.md). The [MATLAB](matlab/README.md) and [NEMAR](nemar/README.md) inventories can run independently. The [bounded alignment](alignment/README.md) reads both exports. Its confirmed pairs and calibrated clocks are inputs to the full NR alignment. The last check follows the saved physical event rows back into the original TSV files.

The scripts read records from the local input root. They write full evidence into new, ignored directories beside each audit. The release contains smaller extracts with the original identifiers, counts and units. Recreating the large tables is optional for reading the results.

## Source of the code and tables

| Public component | Origin and adaptation |
| --- | --- |
| MATLAB inventory | audit_zuco2_n400.py and its saved evidence.json. The reader and probe are retained. An explicit train-only option allows replay without X files. |
| NEMAR inventory | audit_nemar_nm000279.py and its saved source manifest, plus the later download-completion inventory. The manifest is bundled as acquisition metadata. Output paths are separate from inputs. |
| Event-interval figure | Event reader, presentation rule and histogram from audit_nemar_n400_feasibility.py. The original figure and its three event summaries were reproduced without loading EEG. |
| Bounded alignment | audit_zuco2_alignment.py. The old report renderer and its references to two earlier narrative reports were removed. The candidate, calibration, validation, ambiguity and EEG pointer functions were copied unchanged. |
| Full NR alignment | integration.py from the later main pipeline, its matching settings and the small I/O helpers from common.py. The physical-event check is the event-only section of diagnostics.py. Stage logs, redundant checkpoints and unrelated configuration were omitted. The field names in the generated tables retain the original conventions. |

## Equivalence check, 18 November 2026

| Check on the available data | Comparison |
| --- | --- |
| MATLAB train | All 32 filenames and file records, every field of the YAG schema and sample, and the canonical text counts agreed with the archived evidence. For the 16 NR input files, the new SHA-256 values also agreed with the archived full-alignment inputs. |
| MATLAB test | The ten inventory rows, byte counts and recorded provider hashes were cross-checked between the historical audit, acquisition manifest and verification record. No X files were available for a new array-level comparison. |
| NEMAR early snapshot | The original file list and the two preserved truncated binaries reproduced the early inventory. All 2,096 dataset paths, sizes and mtimes agreed. Header, marker, event, word, coordinate, sample and integrity results matched exactly. A Finder file, absolute local root and run time were left out of the comparison. |
| NEMAR completed release | All 2,411 files agreed with the published checksums and the earlier completion record. All 252 run identities, binary targets, sizes, frame counts, durations, channel counts, rates and file-presence checks agreed with the completion inventory. New event totals refer to the completed release. |
| Bounded alignment | All scientific evidence sections agreed, including every per-fixation decision, validation row, clock and negative control. All four output CSVs were byte-identical. |
| Full NR alignment | Both event IDs, all word links, candidate decisions and reasons, errors, physical neighbours, hashes, EEG sample pointers and flags matched across every output row. The full match and decision tables, two catalogues, block diagnostics, negative controls and physical checks were byte-identical. The summary agreed after excluding elapsed time. |
| Figures | The three-run interval statistics and PNG matched the historical outputs exactly. The NR coverage plot is a new rendering of the checked block counts. |

The audits and comparisons are deterministic. The reference checksums are taken from the original full tables. They allow a new checkout to check its output without access to the research workspace. [Reference checksums](alignment/reference_tables.sha256)
