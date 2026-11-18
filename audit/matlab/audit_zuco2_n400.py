"""Read-only, bounded audit: metadata for all MAT files and five YAG NR trials.

Only audit artifacts outside data/ are written. No models, filtering, or bulk
EEG extraction are performed. All HDF5 dataset files are opened with mode='r'.
"""

from collections import Counter
import json
import os
from pathlib import Path
import time
import argparse

import h5py
import numpy as np

PUBLIC = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('ZUCO2_INPUT_ROOT', PUBLIC / 'data/local')).resolve()
OUT = Path(__file__).with_name("generated")


def empty(obj):
    return isinstance(obj, h5py.Dataset) and bool(obj.attrs.get("MATLAB_empty", 0))


def obj_at(file, group, field, index):
    return file[group[field][index, 0]]


def numeric(obj):
    return np.array([]) if empty(obj) else np.asarray(obj[()]).reshape(-1)


def text(obj):
    if empty(obj) or obj.attrs.get("MATLAB_class") != b"char":
        return None
    return np.asarray(obj[()], dtype="<u2").tobytes().decode("utf-16le")


def describe(file, obj, depth=0):
    if isinstance(obj, h5py.Group):
        return {key: describe(file, obj[key], depth + 1) for key in obj}
    result = {"shape_h5py": list(obj.shape), "dtype": str(obj.dtype),
              "matlab_class": str(obj.attrs.get("MATLAB_class")),
              "matlab_empty": empty(obj), "attribute_names": list(obj.attrs)}
    if h5py.check_dtype(ref=obj.dtype) and obj.size and depth < 4:
        ref = obj[tuple(0 for _ in obj.shape)]
        if ref:
            result["first_reference_target"] = describe(file, file[ref], depth + 1)
    return result


def inventory(paths):
    records = []
    for path in paths:
        with h5py.File(path, "r") as file:
            sent = file["sentenceData"]
            raw_refs = sent["rawData"][()].reshape(-1)
            et_refs = sent["rawET"][()].reshape(-1)
            matrices = 0
            missing = 0
            scalar_nan = 0
            matched_et_shape = 0
            shapes = Counter()
            dtypes = Counter()
            for raw_ref, et_ref in zip(raw_refs, et_refs):
                raw = file[raw_ref]
                if empty(raw) or raw.size == 1:
                    missing += 1
                    if not empty(raw) and raw.size == 1:
                        scalar_nan += bool(np.isnan(raw[()]).all())
                elif len(raw.shape) == 2 and raw.shape[1] == 105:
                    matrices += 1
                    shapes[str(raw.shape[1])] += 1
                    dtypes[str(raw.dtype)] += 1
                    matched_et_shape += file[et_ref].shape == (raw.shape[0], 2)
                else:
                    raise AssertionError((str(path), raw.name, raw.shape))
            record = {
                "file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "participant": path.stem.removeprefix("results").split("_")[0],
                "subset": path.parent.name, "sentence_slots": len(raw_refs),
                "eeg_matrices": matrices, "missing_eeg_slots": missing,
                "scalar_nan_missing": scalar_nan, "matched_et_shape": matched_et_shape,
                "eeg_channel_counts": dict(shapes), "eeg_dtypes": dict(dtypes),
                "root_keys": list(file), "root_attribute_names": list(file.attrs),
                "sentence_fields": list(sent),
            }
            if "sentID" in sent:
                identifiers = [numeric(file[ref]) for ref in sent["sentID"][()].reshape(-1)]
                values = [float(value[0]) for value in identifiers if len(value) == 1 and np.isfinite(value[0])]
                record["sentID_summary"] = {"finite": len(values), "unique": len(set(values)),
                    "minimum": min(values), "maximum": max(values)}
            records.append(record)
    return records


def canonical_texts():
    texts = {}
    for task in ["NR", "TSR"]:
        with h5py.File(ROOT / "data/train" / f"resultsYAG_{task}.mat", "r") as file:
            texts[task] = [text(file[ref]) for ref in file["sentenceData/content"][()].reshape(-1)]
    return {"NR_slots": len(texts["NR"]), "TSR_slots": len(texts["TSR"]),
            "NR_unique_exact_texts": len(set(texts["NR"])),
            "TSR_unique_exact_texts": len(set(texts["TSR"])),
            "shared_exact_texts": len(set(texts["NR"]) & set(texts["TSR"])),
            "total_unique_exact_texts": len(set(texts["NR"] + texts["TSR"]))}


def sample():
    trials = []
    all_gaps = []
    with h5py.File(ROOT / "data/train/resultsYAG_NR.mat", "r") as file:
        sent = file["sentenceData"]
        for si in range(5):
            eeg = obj_at(file, sent, "rawData", si)
            et = obj_at(file, sent, "rawET", si)
            words = obj_at(file, sent, "word", si)
            fixation_table = obj_at(file, sent, "allFixations", si)
            saccade_table = obj_at(file, sent, "allSaccades", si)
            all_durations = numeric(fixation_table["duration"])
            saccade_durations = numeric(saccade_table["duration"])
            saccade_amplitudes = numeric(saccade_table["amplitude"])
            retained_indices = np.flatnonzero(all_durations >= 50)
            clock = et[:, 0].astype(float)
            events = []
            word_covariates = []
            for wi in range(words["rawET"].shape[0]):
                word_text = text(obj_at(file, words, "content", wi))
                et_cells = obj_at(file, words, "rawET", wi)
                eeg_cells = obj_at(file, words, "rawEEG", wi)
                amp = numeric(obj_at(file, words, "inSacc_amp", wi))
                fix_positions = numeric(obj_at(file, words, "fixPositions", wi))
                nfix = numeric(obj_at(file, words, "nFixations", wi))
                word_covariates.append({"word_position_1based": wi + 1, "word": word_text,
                    "nFixations": nfix.tolist(), "incoming_amplitudes": amp.tolist()})
                if not h5py.check_dtype(ref=et_cells.dtype):
                    continue
                for fi, ref in enumerate(et_cells[()].reshape(-1)):
                    fixation_et = file[ref]
                    if empty(fixation_et) or len(fixation_et.shape) != 2 or fixation_et.shape[1] != 4:
                        continue
                    start = float(fixation_et[0, 0])
                    stop = float(fixation_et[-1, 0])
                    row = int(np.argmin(np.abs(clock - start)))
                    fixation_eeg = file[eeg_cells[fi, 0]]
                    event = {"word": word_text, "word_position_1based": wi + 1,
                        "word_length_with_punctuation": len(word_text),
                        "within_word_fixation_1based": fi + 1,
                        "stored_fixPosition": float(fix_positions[fi]),
                        "onset_clock_ms": start, "last_sample_clock_ms": stop,
                        "onset_relative_to_sentence_ms": start - float(clock[0]),
                        "onset_row_0based": row, "onset_match_error_ms": float(clock[row] - start),
                        "duration_last_minus_first_ms": stop - start,
                        "fixation_et_shape": list(fixation_et.shape),
                        "fixation_eeg_shape": list(fixation_eeg.shape),
                        "nominal_duration_ms": (fixation_et.shape[0] - 1) * 2,
                        "window_300_500_in_sentence_bounds": row + 250 < eeg.shape[0]}
                    filtered_index = int(fix_positions[fi]) - 1
                    original_index = int(retained_indices[filtered_index])
                    event["original_allFixations_index_0based_under_100ms_rule"] = original_index
                    event["duration_matches_100ms_filtered_fixPosition_mapping"] = bool(
                        all_durations[original_index] == fixation_et.shape[0] - 1)
                    if len(saccade_amplitudes) == len(all_durations):
                        event["candidate_incoming_saccade_amplitude"] = float(saccade_amplitudes[original_index])
                    if original_index:
                        event["previous_original_fixation_nominal_duration_ms"] = float(all_durations[original_index - 1] * 2)
                    if original_index + 1 < min(len(all_durations), len(saccade_durations)):
                        event["next_original_onset_delta_ms_under_duration_reconstruction"] = float(
                            2 * (all_durations[original_index] + saccade_durations[original_index + 1]))
                    events.append(event)
            events.sort(key=lambda item: item["onset_clock_ms"])
            gaps = np.diff([event["onset_clock_ms"] for event in events])
            all_gaps.extend(gaps.tolist())
            for ei, event in enumerate(events):
                if ei:
                    event["previous_stored_fixation_duration_ms"] = events[ei - 1]["duration_last_minus_first_ms"]
                if ei + 1 < len(events):
                    event["next_stored_fixation_onset_delta_ms"] = float(gaps[ei])
                    i0 = event["original_allFixations_index_0based_under_100ms_rule"]
                    i1 = events[ei + 1]["original_allFixations_index_0based_under_100ms_rule"]
                    if len(saccade_durations) == len(all_durations):
                        predicted = 2 * (all_durations[i0:i1].sum() + saccade_durations[i0 + 1:i1 + 1].sum())
                        event["timing_reconstruction_error_to_next_stored_ms"] = float(predicted - gaps[ei])
            trials.append({"sentence_index_1based": si + 1,
                "sentence": text(obj_at(file, sent, "content", si)),
                "sentence_eeg_shape_h5py": list(eeg.shape),
                "sentence_et_shape_h5py": list(et.shape),
                "clock_median_step_ms": float(np.median(np.diff(clock))),
                "clock_min_step_ms": float(np.min(np.diff(clock))),
                "clock_max_step_ms": float(np.max(np.diff(clock))),
                "allFixations_duration_stored": numeric(obj_at(file, sent, "allFixations", si)["duration"]).tolist(),
                "allSaccades_duration_stored": numeric(obj_at(file, sent, "allSaccades", si)["duration"]).tolist(),
                "word_covariates": word_covariates, "timed_word_fixations": events,
                "adjacent_stored_pairs": len(gaps), "next_stored_before_500_ms": int(np.sum(gaps < 500))})

        first = next(event for event in trials[0]["timed_word_fixations"] if event["word_position_1based"] == 1)
        row = first["onset_row_0based"]
        eeg = obj_at(file, sent, "rawData", 0)
        # 101 samples include both 300 and 500 ms at the documented 500 Hz.
        window = eeg[row + 150:row + 251, :]
        finite_channels = np.flatnonzero(np.isfinite(window).all(axis=0))
        anonymous_columns = finite_channels[:3]
        word_covariates = trials[0]["word_covariates"][first["word_position_1based"] - 1]
        diagnostic = {
            "participant": "YAG", "task": "NR", "sentence_index_1based": 1,
            "sentence": trials[0]["sentence"], **first,
            "window_shape": list(window.shape), "sampling_rate_hz": 500,
            "window_sample_offsets_inclusive": [150, 250],
            "window_rows_0based_inclusive": [row + 150, row + 250],
            "fully_finite_channel_columns_1based": (finite_channels + 1).tolist(),
            "anonymous_diagnostic_columns_1based": (anonymous_columns + 1).tolist(),
            "anonymous_column_means_native_units": window[:, anonymous_columns].mean(axis=0).tolist(),
            "anonymous_three_column_mean_native_units": float(window[:, anonymous_columns].mean()),
            "named_Cz_CPz_Pz_mean": None,
            "named_ROI_blocker": "No local channel labels/order or chanlocs verified; anonymous columns are not anatomical labels",
            "baseline_correction": "none; numerical extraction check only",
            "word_level_inSacc_amp_native_units": word_covariates["incoming_amplitudes"],
            "saccade_mapping_warning": "Do not join allSaccades to allFixations by row: naive duration reconstruction fails in this sample",
        }
        raw_word_eeg = obj_at(file, obj_at(file, sent, "word", 0), "rawEEG", first["word_position_1based"] - 1)
        raw_word_eeg = file[raw_word_eeg[first["within_word_fixation_1based"] - 1, 0]]
        local = raw_word_eeg[()]
        parent = eeg[row:row + len(local), :]
        joint = np.isfinite(local) & np.isfinite(parent)
        diagnostic["word_eeg_matches_sentence_eeg_on_joint_finite_samples"] = bool(np.array_equal(local[joint], parent[joint]))
        diagnostic["word_eeg_joint_finite_values"] = int(joint.sum())
        diagnostic["word_eeg_nan_count"] = int(np.isnan(local).sum())
        diagnostic["corresponding_sentence_eeg_nan_count"] = int(np.isnan(parent).sum())
    gaps = np.asarray(all_gaps)
    return {"participant": "YAG", "task": "NR", "trials_checked": 5,
        "trials": trials, "diagnostic_example": diagnostic,
        "overlap": {"definition": "Adjacent available word.rawET fixation onsets, within sentence; not all original fixations",
            "timed_fixations": sum(len(trial["timed_word_fixations"]) for trial in trials),
            "pairs": len(gaps), "before_300_ms": int(np.sum(gaps < 300)),
            "before_500_ms": int(np.sum(gaps < 500)),
            "percent_before_500_ms": float(100 * np.mean(gaps < 500)),
            "median_onset_delta_ms": float(np.median(gaps)),
            "minimum_onset_delta_ms": float(np.min(gaps)),
            "maximum_onset_delta_ms": float(np.max(gaps))}}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--train-only', action='store_true', help='Audit train when benchmark test is unavailable')
    args = parser.parse_args()
    OUT = args.out.resolve()
    if OUT == ROOT or OUT.is_relative_to(ROOT / 'data') or OUT.is_relative_to(ROOT / 'data.nemar.org'):
        raise ValueError('Choose an output outside the source data')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {OUT}')
    started = time.monotonic()
    subsets = ["train"] if args.train_only else ["train", "test"]
    paths = sorted(p for subset in subsets for p in (ROOT / "data" / subset).glob("*.mat"))
    expected = json.loads((PUBLIC / "data/train_files.json").read_text())
    if not args.train_only:
        expected += json.loads((PUBLIC / "data/zuco_test_manifest.json").read_text())["files"]
    required = {(("train" if "_NR." in r["name"] or "_TSR." in r["name"] else "test"), r["name"]) for r in expected}
    found = {(p.parent.name, p.name) for p in paths}
    if found != required:
        raise ValueError(f"Input file set differs: missing={sorted(required-found)}, unexpected={sorted(found-required)}")
    before = {str(path.relative_to(ROOT)): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    records = inventory(paths)
    schemas = {}
    for name in (["data/train/resultsYAG_NR.mat"] if args.train_only else ["data/train/resultsYAG_NR.mat", "data/test/resultsXAH.mat"]):
        with h5py.File(ROOT / name, "r") as file:
            schemas[name] = {key: describe(file, obj_at(file, file["sentenceData"], key, 0))
                             for key in file["sentenceData"]}
    evidence = {"subsets": subsets, "scope": "Selected-file metadata only; five YAG NR sentences for timings; one waveform window",
                "inventory": records, "canonical_text_counts": canonical_texts(),
                "representative_nested_schemas": schemas, "sample": sample()}
    after = {str(path.relative_to(ROOT)): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}
    assert before == after, "Dataset size or modification time changed during audit"
    evidence["dataset_size_and_mtime_unchanged"] = True
    evidence["source_file_size_and_mtime_ns"] = before
    evidence["runtime_seconds"] = time.monotonic() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"files": len(records),
        "eeg_matrices": sum(record["eeg_matrices"] for record in records),
        "missing_eeg_slots": sum(record["missing_eeg_slots"] for record in records),
        "texts": evidence["canonical_text_counts"], "overlap": evidence["sample"]["overlap"],
        "example": evidence["sample"]["diagnostic_example"],
        "runtime_seconds": evidence["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
