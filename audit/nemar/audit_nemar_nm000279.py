"""Read-only nm000279 audit: metadata inventory and one small EEG window.

No other ZuCo export is read. No models, surprisal, filters or bulk EEG epochs.
All output is written to the selected output directory, outside the dataset.
"""

from collections import Counter
import configparser
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import argparse

import numpy as np

PUBLIC = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('ZUCO2_INPUT_ROOT', PUBLIC / 'data/local')).resolve()
DATA = ROOT / "data.nemar.org/nm000279/v1.0.0"
OUT = Path(__file__).with_name("generated")


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def header(path):
    content = path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(content[content.index("[Common Infos]"):])
    common = parser["Common Infos"]
    channels = [parser["Channel Infos"][f"Ch{i}"].split(",")
                for i in range(1, int(common["NumberOfChannels"]) + 1)]
    return {"channels": [item[0] for item in channels],
            "resolution": [float(item[2]) for item in channels],
            "units": [item[3] for item in channels],
            "frequency": 1e6 / float(common["SamplingInterval"]),
            "orientation": common["DataOrientation"],
            "binary_format": parser["Binary Infos"]["BinaryFormat"],
            "data_file": common["DataFile"], "marker_file": common["MarkerFile"]}


def count_events(path, last_sample_seconds=None):
    counter = Counter()
    out_of_bounds = Counter()
    maximum_onset = float('-inf')
    maximum_trigger_onset = float('-inf')
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        for row in reader:
            counter[row["trial_type"]] += 1
            onset = float(row['onset'])
            maximum_onset = max(maximum_onset, onset)
            if row['trial_type'].startswith('trigger_'):
                maximum_trigger_onset = max(maximum_trigger_onset, onset)
            if last_sample_seconds is not None and onset > last_sample_seconds + .1:
                out_of_bounds[row['trial_type']] += 1
    return {"columns": fields, "counts": dict(counter), "rows": sum(counter.values()),
        "maximum_onset_seconds": maximum_onset,
        "maximum_trigger_onset_seconds": maximum_trigger_onset,
        "onsets_beyond_eeg_end_plus_100ms": dict(out_of_bounds)}


def sample():
    base = DATA / "sub-YAG/eeg/sub-YAG_task-nr_run-1"
    rows = table(Path(str(base) + "_events.tsv"))
    events = [{**row, "row_1based_including_header": i + 2,
               "onset_seconds": float(row["onset"]), "duration_seconds": float(row["duration"])}
              for i, row in enumerate(rows)]
    fixations = [event for event in events if event["trial_type"] == "fixation"]
    saccades = [event for event in events if event["trial_type"] == "saccade"]
    starts = [event["onset_seconds"] for event in events if event["trial_type"] == "trigger_10"]
    stops = [event["onset_seconds"] for event in events if event["trial_type"] == "trigger_11"]
    intervals = []
    all_deltas = []
    eligible_next = []
    for index, start in enumerate(starts[:5]):
        stop = next(value for value in stops if value > start)
        current = [event for event in fixations if start <= event["onset_seconds"] < stop]
        deltas = np.diff([event["onset_seconds"] for event in current])
        all_deltas.extend(deltas.tolist())
        for event, next_event in zip(current, current[1:]):
            if event["duration_seconds"] >= 0.1 and event["onset_seconds"] + 0.5 <= stop:
                eligible_next.append(next_event["onset_seconds"] - event["onset_seconds"])
        intervals.append({"interval_1based": index + 1, "start_trigger_10_seconds": start,
            "stop_trigger_11_seconds": stop, "sentence_id": None,
            "interpretation": "Candidate reading interval; trigger semantics and text key not documented in events",
            "fixations": current, "adjacent_pairs": len(deltas),
            "next_before_500ms": int(np.sum(deltas < .5 - 1e-9)),
            "next_before_300ms": int(np.sum(deltas < .3 - 1e-9))})

    # A real fixation within the first candidate reading interval. The row has
    # no word identity; duration similarity to word measures is not a word key.
    event = next(event for event in fixations if abs(event["onset_seconds"] - 2.701) < 1e-9)
    position = fixations.index(event)
    preceding = fixations[position - 1]
    following = fixations[position + 1]
    incoming = max((saccade for saccade in saccades
                    if saccade["onset_seconds"] + saccade["duration_seconds"] <= event["onset_seconds"] + 1e-9),
                   key=lambda saccade: saccade["onset_seconds"] + saccade["duration_seconds"])
    hdr = header(Path(str(base) + "_eeg.vhdr"))
    assert hdr["orientation"] == "MULTIPLEXED" and hdr["binary_format"] == "IEEE_FLOAT_32"
    binary_path = base.parent / hdr["data_file"]
    width = len(hdr["channels"])
    assert binary_path.stat().st_size % (4 * width) == 0
    binary = np.memmap(binary_path, mode="r", dtype="<f4").reshape(-1, width)
    frequency = hdr["frequency"]
    first = int(np.ceil((event["onset_seconds"] + .3) * frequency - 1e-8))
    last = int(np.floor((event["onset_seconds"] + .5) * frequency + 1e-8))
    channels = [hdr["channels"].index("E55"), hdr["channels"].index("E62")]
    raw_window = np.asarray(binary[first:last + 1, channels], dtype=np.float64)
    uv_window = raw_window * np.array(hdr["resolution"])[channels]
    trigger_markers = []
    for line in Path(str(base) + "_eeg.vmrk").read_text().splitlines():
        if line.startswith("Mk"):
            parts = line.split("=", 1)[1].split(",")
            trigger_markers.append({"description": parts[1], "sample_0based": int(parts[2]) - 1})
    triggers = [event for event in events if event["trial_type"].startswith("trigger_")]
    marker_errors = [abs(marker["sample_0based"] / frequency - trigger["onset_seconds"])
                     for marker, trigger in zip(trigger_markers, triggers)]
    probe = {"participant": "YAG", "task": "nr", "run": 1,
        "events_file": str(Path(str(base) + "_events.tsv").relative_to(DATA)),
        "event_row_1based_including_header": event["row_1based_including_header"],
        "onset_seconds": event["onset_seconds"], "end_seconds": event["onset_seconds"] + event["duration_seconds"],
        "fixation_duration_ms": event["duration_seconds"] * 1000,
        "previous_fixation_duration_ms": preceding["duration_seconds"] * 1000,
        "next_fixation_onset_seconds": following["onset_seconds"],
        "next_fixation_delta_ms": 1000 * (following["onset_seconds"] - event["onset_seconds"]),
        "incoming_saccade": incoming,
        "incoming_saccade_end_to_fixation_gap_ms": 1000 * (event["onset_seconds"] - incoming["onset_seconds"] - incoming["duration_seconds"]),
        "sentence_id": None, "word_id": None, "word": None,
        "sentence_word_blocker": "events.tsv has no sentence/word/AOI identifier or screen-word bounds; wordmeasures has no run/onset join key",
        "eeg_shape_samples_channels": list(binary.shape), "frequency_hz": frequency,
        "window_sample_indices_0based_inclusive": [first, last],
        "window_times_seconds": [first / frequency, last / frequency],
        "window_relative_to_fixation_ms": [1000 * (first / frequency - event["onset_seconds"]),
                                           1000 * (last / frequency - event["onset_seconds"])],
        "window_shape": list(uv_window.shape), "channels": [hdr["channels"][i] for i in channels],
        "header_resolution_uV_per_stored_unit": [hdr["resolution"][i] for i in channels],
        "means_uV_per_header": uv_window.mean(axis=0).tolist(),
        "combined_mean_uV_per_header": float(uv_window.mean()),
        "all_probe_values_finite": bool(np.isfinite(uv_window).all()),
        "named_Cz_CPz_Pz_mean": None,
        "named_ROI_blocker": "Cz is not a stored channel; E55/E62 are diagnostic channels, no virtual reference or interpolation applied",
        "processing": "None: header calibration only; no baseline, filtering, cleaning or rereferencing",
        "trigger_rows": len(triggers), "marker_rows": len(trigger_markers),
        "trigger_marker_max_timing_difference_seconds": max(marker_errors),
        "trigger_code_mapping": dict(sorted({(trigger["trial_type"], marker["description"])
             for trigger, marker in zip(triggers, trigger_markers)}))}
    all_deltas = np.asarray(all_deltas)
    eligible_next = np.asarray(eligible_next)
    return {"scope": "YAG nr run-1: first five trigger_10-to-trigger_11 intervals; one small signal window",
        "intervals": intervals, "probe": probe,
        "overlap": {"fixations": sum(len(item["fixations"]) for item in intervals),
            "pairs": len(all_deltas), "before_500ms": int(np.sum(all_deltas < .5 - 1e-9)),
            "percent_before_500ms": float(100 * np.mean(all_deltas < .5 - 1e-9)),
            "before_300ms": int(np.sum(all_deltas < .3 - 1e-9)),
            "median_onset_delta_ms": float(1000 * np.median(all_deltas)),
            "eligible_current_duration_ge_100ms_and_500ms_window_in_interval": {
                "count": len(eligible_next), "before_500ms": int(np.sum(eligible_next < .5 - 1e-9)),
                "percent_before_500ms": float(100 * np.mean(eligible_next < .5 - 1e-9))}}}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if OUT == ROOT or OUT.is_relative_to(ROOT / 'data') or OUT.is_relative_to(ROOT / 'data.nemar.org'):
        raise ValueError('Choose an output outside the source data')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {OUT}')
    started = time.monotonic()
    paths = [path for path in DATA.rglob("*") if path.is_file()]
    before = {str(path.relative_to(DATA)): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    manifest_path = PUBLIC / "data/nemar_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {entry["path"]: entry for entry in manifest}
    missing = [entry for entry in manifest if entry["path"] not in before]
    mismatches = [{"path": name, "expected_bytes": entry["size"], "actual_bytes": before[name][0]}
                  for name, entry in expected.items() if name in before and entry["size"] != before[name][0]]
    per_run = []
    for item in manifest:
        if not item["path"].endswith("_eeg.vhdr"):
            continue
        path = DATA / item["path"]
        stem = str(path)[:-len("_eeg.vhdr")]
        needed = [stem + "_eeg" + suffix for suffix in [".vhdr", ".vmrk", ".eeg"]]
        existence = {Path(name).suffix: Path(name).is_file() for name in needed}
        good_size = all(Path(name).is_file() and Path(name).stat().st_size == expected[str(Path(name).relative_to(DATA))]["size"] for name in needed)
        record = {"header": item["path"], "participant": path.parts[-3],
                  "all_triplet_files_present": all(existence.values()), "triplet_matches_manifest_sizes": good_size,
                  "existence": existence}
        if path.exists():
            parsed = header(path)
            record["header_metadata"] = parsed
            data_path = path.parent / parsed["data_file"]
            if data_path.exists():
                byte_count = data_path.stat().st_size
                record["binary_bytes_divisible_by_frame"] = byte_count % (4 * len(parsed["channels"])) == 0
                record["sample_count_from_bytes"] = byte_count // (4 * len(parsed["channels"]))
        last_time = ((record['sample_count_from_bytes'] - 1) / record['header_metadata']['frequency']
                     if 'sample_count_from_bytes' in record else None)
        record['eeg_last_sample_seconds'] = last_time
        marker_path = Path(stem + "_eeg.vmrk")
        if marker_path.exists() and 'sample_count_from_bytes' in record:
            positions = [int(line.split('=', 1)[1].split(',')[2]) - 1
                         for line in marker_path.read_text().splitlines() if line.startswith('Mk')]
            record['marker_positions_beyond_eeg_samples'] = sum(
                position >= record['sample_count_from_bytes'] for position in positions)
            record['maximum_marker_position_0based'] = max(positions) if positions else None
        event_path = Path(stem + "_events.tsv")
        if event_path.exists():
            record["event_summary"] = count_events(event_path, last_time)
        per_run.append(record)
    word_tables = []
    YAG_texts = {}
    for path in sorted((DATA / "derivatives/reading_measures").rglob("*.tsv")):
        sentence_ids = set()
        examples = []
        count = 0
        with path.open() as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = reader.fieldnames
            for row in reader:
                count += 1
                sentence_ids.add(row["sentence_id"])
                if len(examples) < 2:
                    examples.append({key: row[key] for key in ["sentence_id", "word_id", "word", "nfixations", "ffd"]})
        word_tables.append({"file": str(path.relative_to(DATA)), "rows": count,
            "unique_sentence_ids": len(sentence_ids), "minimum_sentence_id": min(map(int, sentence_ids)),
            "maximum_sentence_id": max(map(int, sentence_ids)), "columns": fields, "examples": examples})
        if path.parent.name == 'sub-YAG':
            grouped = {}
            for row in table(path):
                grouped.setdefault(row['sentence_id'], []).append((int(row['word_id']), row['word']))
            task = path.name.split('_task-')[1].split('_')[0]
            YAG_texts[task] = [' '.join(word for _, word in sorted(words)) for words in grouped.values()]
    electrodes_path = DATA / "sub-YAG/eeg/sub-YAG_space-CapTrak_electrodes.tsv"
    electrodes = table(electrodes_path)
    sample_result = sample()
    global_events = Counter()
    for run in per_run:
        global_events.update(run.get('event_summary', {}).get('counts', {}))
    coordinate_paths = sorted(DATA.rglob('*electrodes.tsv'))
    physio_paths = sorted(DATA.rglob('*physio.tsv.gz'))
    physio_summary = []
    for path in physio_paths:
        with gzip.open(path, 'rt') as stream:
            count = sum(1 for _ in stream)
        physio_summary.append({'file': str(path.relative_to(DATA)), 'rows': count})
    selected_integrity = []
    for name in ['sub-YAK/eeg/sub-YAK_task-nr_run-1_eeg.eeg',
                 'sub-YAK/eeg/sub-YAK_task-nr_run-1_events.tsv',
                 'sub-YAK/eeg/sub-YAK_task-nr_run-1_eeg.vmrk']:
        payload = (DATA / name).read_bytes()
        algorithm = expected[name]['checksum_algorithm']
        digest = (hashlib.sha256(payload).hexdigest() if algorithm == 'sha256'
                  else hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest())
        selected_integrity.append({'file': name, 'algorithm': algorithm,
                                   'matches_manifest': digest == expected[name]['checksum']})
    after = {str(path.relative_to(DATA)): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    assert before == after, "A source file changed during this read-only audit"
    evidence = {"dataset": "nm000279", "version": "v1.0.0", "local_root": str(DATA.relative_to(ROOT)),
        "remote_manifest_url": "https://data.nemar.org/nm000279/v1.0.0/manifest.json",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_files": len(manifest), "local_files_including_DS_Store": len(paths),
        "local_bytes": sum(value[0] for value in before.values()),
        "missing_files": missing, "size_mismatches": mismatches,
        "missing_extensions": dict(Counter(Path(entry["path"]).suffix for entry in missing)),
        "local_extensions": dict(Counter(path.suffix for path in paths)),
        "participants": table(DATA / "participants.tsv"), "runs": per_run,
        "word_tables": word_tables, "YAG_electrodes": electrodes,
        "global_event_counts": dict(global_events),
        "electrode_files": len(coordinate_paths),
        "unique_electrode_file_hashes": len({hashlib.sha256(path.read_bytes()).hexdigest() for path in coordinate_paths}),
        "physio_files": physio_summary,
        "selected_file_integrity_checks": selected_integrity,
        "YAG_text_counts": {**{task: {'presentation_ids': len(texts), 'unique_reconstructed_strings': len(set(texts))}
                                  for task, texts in YAG_texts.items()},
                            'total_unique_reconstructed_strings': len(set(YAG_texts['nr'] + YAG_texts['tsr']))},
        "metadata_snapshots": {name: json.loads((DATA / name).read_text()) for name in [
            'dataset_description.json', 'task-nr_eeg.json', 'task-nr_events.json',
            'sub-YAG/eeg/sub-YAG_task-nr_run-1_eeg.json',
            'sub-YAG/eeg/sub-YAG_task-nr_run-1_recording-eyetrack_physio.json',
            'derivatives/reading_measures/task-nr_desc-wordmeasures_beh.json']},
        "sample": sample_result, "source_sizes_and_mtimes_unchanged": before == after,
        "source_sizes_and_mtimes_ns": before, "runtime_seconds": time.monotonic() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"participants": len(evidence["participants"]), "expected_runs": len(per_run),
        "complete_triplets_by_size": sum(item["triplet_matches_manifest_sizes"] for item in per_run),
        "word_tables": len(word_tables), "word_rows": sum(item["rows"] for item in word_tables),
        "missing": len(missing), "size_mismatches": mismatches,
        "overlap": sample_result["overlap"], "probe": sample_result["probe"],
        "runtime_seconds": evidence["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
