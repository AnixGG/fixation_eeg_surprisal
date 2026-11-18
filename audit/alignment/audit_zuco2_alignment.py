"""Read-only, bounded ZuCo2 MATLAB–NEMAR event alignment audit.

Run: python3 audit/alignment/audit_zuco2_alignment.py
No EEG signal values are accessed. Scope and units are recorded in the generated evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import configparser
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import h5py
import numpy as np

PUBLIC = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('ZUCO2_INPUT_ROOT', PUBLIC / 'data/local')).resolve()
DATA = ROOT / 'data.nemar.org/nm000279/v1.0.0'
OUT = Path(__file__).with_name('pilot_generated')
TOL = {'time_ms': 4.0, 'duration_ms': 4.0, 'gaze_px': 2.0}
SOURCES = {}


def track(p, hash_file=True):
    key = rel(p)
    if key not in SOURCES:
        st = p.stat()
        SOURCES[key] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest() if hash_file else None}


def empty(o):
    return isinstance(o, h5py.Dataset) and bool(o.attrs.get('MATLAB_empty', 0))


def nums(o):
    return np.array([]) if empty(o) else np.asarray(o[()]).ravel()


def txt(o):
    if not isinstance(o, h5py.Dataset) or empty(o) or o.attrs.get('MATLAB_class') != b'char':
        return None
    return np.asarray(o[()], dtype='<u2').tobytes().decode('utf-16le')


def ref(f, g, field, i):
    return f[g[field][i, 0]]


def scalar(o):
    a = nums(o)
    return float(a[0]) if len(a) == 1 and np.isfinite(a[0]) else None


def finite(v):
    try:
        return float(v) if np.isfinite(float(v)) else None
    except (ValueError, TypeError):
        return None


def table(p):
    with p.open(encoding='utf-8-sig', newline='') as s:
        return list(csv.DictReader(s, delimiter='\t'))


def rel(p):
    return str(p.absolute().relative_to(ROOT))


def mat_sent(f, participant, condition, si):
    """Read ET only; keep missing fixation slots explicitly INVALID."""
    s = f['sentenceData']
    w = ref(f, s, 'word', si - 1)
    result = {'participant': participant, 'condition': condition, 'sentence_id': si,
              'text': txt(ref(f, s, 'content', si - 1)), 'events': [], 'words': []}
    if not isinstance(w, h5py.Group):
        result['missing_sentence'] = True
        return result
    bounds = ref(f, s, 'wordbounds', si - 1)
    b = np.asarray(bounds[()]) if not empty(bounds) else None
    af = ref(f, s, 'allFixations', si - 1)
    afd = nums(af['duration']) if isinstance(af, h5py.Group) else np.array([])
    afx = nums(af['x']) if isinstance(af, h5py.Group) else np.array([])
    afy = nums(af['y']) if isinstance(af, h5py.Group) else np.array([])
    retained = np.flatnonzero(afd >= 50)
    for wi in range(w['content'].shape[0]):
        word = txt(ref(f, w, 'content', wi))
        nfix = scalar(ref(f, w, 'nFixations', wi))
        positions = nums(ref(f, w, 'fixPositions', wi))
        amps = nums(ref(f, w, 'inSacc_amp', wi)).tolist()
        sdurs = nums(ref(f, w, 'inSacc_duration', wi)).tolist()
        cells = ref(f, w, 'rawET', wi)
        word_rec = {'word_id': wi + 1, 'word': word, 'nfix': nfix, 'incoming_amplitudes': amps}
        result['words'].append(word_rec)
        refs = cells[()].ravel() if h5py.check_dtype(ref=cells.dtype) and not empty(cells) else []
        count = max(len(refs), int(nfix or 0))
        for fi in range(count):
            e = {'participant_id': participant, 'condition': condition, 'sentence_id': si,
                 'word_id': wi + 1, 'word': word, 'word_length': len(word) if word else None,
                 'word_count': w['content'].shape[0], 'within_word_fixation': fi + 1,
                 'matlab_fixation_id': f'{participant}/{condition}/s{si}/w{wi+1}/f{fi+1}',
                 'source_file_matlab': rel(Path(f.filename)), 'word_hdf5_path': f'sentenceData/word[{si-1},0] -> reference',
                 'matlab_fixPosition': finite(positions[fi]) if fi < len(positions) else None,
                 'matlab_word_nfix': nfix, 'matlab_incoming_amplitudes': amps,
                 'matlab_single_fix_incoming_amp': amps[0] if nfix == 1 and len(amps) == 1 else None,
                 'matlab_single_fix_incoming_duration_samples': sdurs[0] if nfix == 1 and len(sdurs) == 1 else None,
                 'wordbounds': b[:, wi].tolist() if b is not None and b.ndim == 2 and b.shape[0] == 4 else None,
                 'status': 'INVALID', 'reason': 'missing_or_invalid_rawET'}
            if fi < len(refs) and refs[fi]:
                a = f[refs[fi]]
                e['rawET_path'] = f'sentenceData/word[{si-1},0] -> rawET[{wi},0] -> cell[{fi},0] (dereference at arrows)'
                e['rawET_hdf5_object_address'] = int(h5py.h5o.get_info(a.id).addr)
                e['rawET_shape'] = list(a.shape)
                if not empty(a) and a.ndim == 2 and a.shape[1] == 4 and a.shape[0] >= 2:
                    v = np.asarray(a[()], dtype=float)
                    e['rawET_values_sha256_float64_le'] = hashlib.sha256(v.astype('<f8').tobytes()).hexdigest()
                    if np.isfinite(v[:, 0]).all() and np.all(np.diff(v[:, 0]) > 0):
                        # Exclude the final endpoint sample from gaze/pupil summary.
                        mean = np.nanmean(v[:-1, 1:], axis=0)
                        e.update(start_ms=float(v[0, 0]), stop_ms=float(v[-1, 0]),
                                 duration_ms=float(v[-1, 0]-v[0, 0]), n_samples=len(v),
                                 gaze_x=finite(mean[0]), gaze_y=finite(mean[1]), pupil=finite(mean[2]),
                                 nominal_span_ms=(len(v)-1)*2, status='UNMATCHED', reason='not_yet_aligned')
            fp = e['matlab_fixPosition']
            if fp is not None and fp.is_integer() and 0 < fp <= len(retained):
                ai = int(retained[int(fp)-1])
                e['allFixations_index_0based_hypothesis'] = ai
                e['allFixations_duration_samples'] = finite(afd[ai])
                e['allFixations_duration_agrees'] = bool(e.get('n_samples', -1)-1 == afd[ai])
                e['corrected_gaze_x'] = finite(afx[ai]) if ai < len(afx) else None
                e['corrected_gaze_y'] = finite(afy[ai]) if ai < len(afy) else None
            result['events'].append(e)
    valid = sorted([e for e in result['events'] if 'start_ms' in e], key=lambda e: e['start_ms'])
    for i, e in enumerate(valid):
        e['previous_stored_onset_ms'] = valid[i-1]['start_ms'] if i else None
        e['next_stored_onset_ms'] = valid[i+1]['start_ms'] if i+1 < len(valid) else None
    return result


def load_run(participant, condition, run):
    p = DATA / f'sub-{participant}/eeg/sub-{participant}_task-{condition.lower()}_run-{run}_events.tsv'
    track(p)
    events = []
    for i, row in enumerate(table(p)):
        e = {'row': i+2, 'source': rel(p), 'type': row['trial_type']}
        for k, v in row.items():
            if k not in ['trial_type', 'value']:
                e[k] = finite(v)
        e['end'] = e['onset'] + e['duration']
        events.append(e)
    fixes = sorted([e for e in events if e['type'] == 'fixation'], key=lambda e: e['onset'])
    sacs = sorted([e for e in events if e['type'] == 'saccade'], key=lambda e: e['end'])
    sac_ends = np.array([x['end'] for x in sacs])
    for i, e in enumerate(fixes):
        e['fix_index'] = i
        e['previous'] = fixes[i-1] if i else None
        e['next'] = fixes[i+1] if i+1 < len(fixes) else None
        j = np.searchsorted(sac_ends, e['onset'] + 1e-9, side='right') - 1
        incoming = sacs[j] if j >= 0 else None
        e['incoming'] = incoming if incoming and 0 <= e['onset']-incoming['end'] <= .00400001 else None
    # Trigger pairings are candidate protocol intervals, validated against ET.
    intervals = []
    open_trigger = None
    for e in events:
        if e['type'] in ('trigger_10', 'trigger_12'):
            open_trigger = e
        elif e['type'] in ('trigger_11', 'trigger_13') and open_trigger:
            expected = 'trigger_11' if open_trigger['type'] == 'trigger_10' else 'trigger_13'
            if e['type'] == expected:
                intervals.append({'start': open_trigger['onset'], 'end': e['onset'],
                                  'start_row': open_trigger['row'], 'end_row': e['row'],
                                  'start_type': open_trigger['type'], 'end_type': e['type']})
            open_trigger = None
    return {'participant': participant, 'condition': condition, 'run': run, 'path': p,
            'events': events, 'fixes': fixes, 'intervals': intervals,
            'features': np.array([[e['duration']*1000, e['gaze_x'] or np.nan, e['gaze_y'] or np.nan] for e in fixes])}


def feature_candidates(e, run, duration=4., gaze=2.):
    if e['status'] == 'INVALID' or e['gaze_x'] is None or e['gaze_y'] is None:
        return []
    out = []
    v = run['features']
    de = e['duration_ms'] - (v[:, 0] + 2)
    ge = np.hypot(e['gaze_x']-v[:, 1], e['gaze_y']-v[:, 2])
    for j in np.flatnonzero((np.abs(de) <= duration + 1e-8) & (ge <= gaze + 1e-8)):
        n = run['fixes'][j]
        out.append({'nemar': n, 'duration_error_ms': float(de[j]), 'gaze_error_px': float(ge[j]),
                    'offset_ms': e['start_ms']-1000*n['onset']})
    return out


def offset_clusters(sentence, runs, tol=4.):
    """Search without sentence/run identity forcing; count unique ordered anchors."""
    groups = []
    for run in runs:
        pairs = [(e, c) for e in sentence['events'] for c in feature_candidates(e, run)]
        for e, c in pairs:
            near = [(a, b) for a, b in pairs if abs(b['offset_ms']-c['offset_ms']) <= tol]
            # Multiple possible candidates for one source event are not anchors.
            counts = Counter(a['matlab_fixation_id'] for a, b in near)
            unique = sorted([(a, b) for a, b in near if counts[a['matlab_fixation_id']] == 1], key=lambda p: p[0]['start_ms'])
            if not unique:
                continue
            ids = tuple(b['nemar']['row'] for a, b in unique)
            if len(set(ids)) != len(ids) or any(b[1]['nemar']['onset'] <= a[1]['nemar']['onset'] for a, b in zip(unique, unique[1:])):
                continue
            key = (run['run'], ids)
            if any(g['key'] == key for g in groups):
                continue
            groups.append({'key': key, 'run': run['run'], 'count': len(unique),
                           'words': len(set(a['word_id'] for a, b in unique)),
                           'span_ms': unique[-1][0]['start_ms']-unique[0][0]['start_ms'],
                           'offset_ms': float(np.median([b['offset_ms'] for a,b in unique])),
                           'offset_range_ms': float(np.ptp([b['offset_ms'] for a,b in unique])), 'pairs': unique})
    return sorted(groups, key=lambda g: -g['count'])


def basic_n(n):
    if n is None:
        return None
    return {k: n.get(k) for k in ['row', 'type', 'onset', 'duration', 'end', 'gaze_x', 'gaze_y',
                                  'pupil_size', 'sacc_amplitude', 'sacc_peakvel', 'sacc_endx', 'sacc_endy', 'source']}


def candidate_record(c, offset):
    return {**basic_n(c['nemar']), 'time_error_ms': c['offset_ms']-offset,
            'duration_error_ms': c['duration_error_ms'], 'gaze_error_px': c['gaze_error_px']}


def stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return {'n': len(v), 'min': float(v.min()), 'median': float(np.median(v)),
            'p95': float(np.quantile(v, .95)), 'max': float(v.max()), 'mean': float(v.mean())} if len(v) else {'n': 0}


def cluster_summary(g):
    return {k: v for k, v in g.items() if k not in ['pairs', 'key']}


def calibrate(sentences, runs, target_run):
    anchors = []
    diagnostics = []
    for s in sentences:
        groups = offset_clusters(s, runs)
        if not groups:
            diagnostics.append({'sentence_id': s['sentence_id'], 'accepted': False, 'reason': 'no_anchors'})
            continue
        best = groups[0]
        rivals = [g for g in groups[1:] if g['run'] != best['run'] or abs(g['offset_ms']-best['offset_ms']) > 8]
        rival = rivals[0]['count'] if rivals else 0
        ok = (best['run'] == target_run and best['count'] >= 6 and best['words'] >= 3
              and best['span_ms'] >= 1000 and rival < best['count']/2)
        diagnostics.append({'sentence_id': s['sentence_id'], 'accepted': ok,
                            'best': cluster_summary(best), 'strongest_competing_island_count': rival,
                            'competing_islands': [cluster_summary(g) for g in rivals[:3]]})
        if ok:
            anchors.extend(best['pairs'])
    offsets = [d['best']['offset_ms'] for d in diagnostics if d['accepted']]
    offset = float(np.median(offsets)) if offsets else None
    residuals = [c['offset_ms']-offset for e,c in anchors] if offset is not None else []
    span = float(np.ptp([e['start_ms'] for e,c in anchors])) if anchors else 0
    accepted = (len(offsets) == len(sentences) and len(offsets) >= 3 and span >= 60000
                and max([abs(x) for x in residuals], default=np.inf) <= TOL['time_ms'])
    return {'accepted': accepted, 'constant_offset_ms': offset, 'equation': 'NEMAR_ms = MATLAB_ms - constant_offset_ms',
            'n_anchors': len(anchors), 'anchor_span_ms': span, 'residual_ms': stats(residuals),
            'sentence_offsets_ms': offsets, 'calibration_sentences': diagnostics,
            'drift': 'No drift correction fitted; constant clock accepted only within fixed 4 ms residual bound.'}


def header_info(run, manifest):
    base = str(run['path']).removesuffix('_events.tsv')
    p = Path(base + '_eeg.vhdr')
    if not p.exists():
        return {'complete': False, 'reason': 'missing_header'}
    track(p)
    content = p.read_text(encoding='utf-8-sig')
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read_string(content[content.index('[Common Infos]'):])
    c = cp['Common Infos']
    nchan = int(c['NumberOfChannels'])
    hz = 1e6/float(c['SamplingInterval'])
    bin_path = p.parent/c['DataFile']
    vmrk_path = p.parent/c['MarkerFile']
    files = [p, bin_path, vmrk_path]
    complete = all(x.exists() and x.stat().st_size == manifest.get(str(x.relative_to(DATA)), -1) for x in files)
    for x in files:
        if x.exists():
            track(x, x.suffix != '.eeg')
    frames = bin_path.stat().st_size//(nchan*4) if bin_path.exists() else 0
    hdr = {'complete': complete, 'header': rel(p), 'binary': rel(bin_path), 'marker': rel(vmrk_path),
           'sampling_rate_hz': hz, 'channels': nchan, 'orientation': c['DataOrientation'],
           'format': cp['Binary Infos']['BinaryFormat'], 'sample_frames': frames,
           'frame_bytes': nchan*4, 'last_sample_s': (frames-1)/hz,
           'size_multiple_of_frame': bin_path.exists() and bin_path.stat().st_size % (nchan*4) == 0}
    hdr['complete'] &= hdr['size_multiple_of_frame'] and hdr['orientation'] == 'MULTIPLEXED' and hdr['format'] == 'IEEE_FLOAT_32'
    markers = []
    if vmrk_path.exists():
        for line in vmrk_path.read_text(encoding='utf-8-sig').splitlines():
            if line.startswith('Mk') and '=' in line:
                row = line.split('=', 1)[1].split(',')
                if row[0] == 'Stimulus':
                    code = row[1].removeprefix('S').strip()
                    markers.append(('S '+code, (int(row[2])-1)/hz))
    triggers = [(e['type'], e['onset']) for e in run['events'] if e['type'].startswith('trigger_')]
    hdr['marker_trigger_count'] = len(markers)
    hdr['events_trigger_count'] = len(triggers)
    # BrainVision labels are enumerated per recording, not globally semantic.
    pairs=Counter((a[0], b[0]) for a,b in zip(markers,triggers))
    forward=defaultdict(set); reverse=defaultdict(set)
    for a,b in pairs:
        forward[a].add(b); reverse[b].add(a)
    hdr['marker_code_mapping_by_time_order']=[{'brainvision':a,'events':b,'count':n} for (a,b),n in pairs.items()]
    hdr['marker_codes_bijective'] = len(markers)==len(triggers) and all(len(v)==1 for v in forward.values()) and all(len(v)==1 for v in reverse.values())
    hdr['marker_max_time_error_ms'] = max([abs(a[1]-b[1])*1000 for a,b in zip(markers,triggers)], default=None)
    hdr['EEG_clock_verified'] = hdr['marker_codes_bijective'] and hdr['marker_max_time_error_ms'] is not None and hdr['marker_max_time_error_ms'] < 1e-6
    return hdr


def align(sentence, run, clock, tol=TOL, word_lookup=None, interval=None, require_chain=True):
    output = []
    for original in sentence['events']:
        e = dict(original)
        e['run_id'] = run['run']
        e['split'] = sentence.get('split', 'control')
        e['candidates'] = []
        if e['status'] == 'INVALID':
            output.append(e)
            continue
        if not clock['accepted']:
            has_rival=any(d.get('strongest_competing_island_count',0)>=6 for d in clock.get('calibration_sentences',[]))
            e.update(status='AMBIGUOUS' if has_rival else 'UNMATCHED', reason='clock_calibration_not_accepted')
            output.append(e)
            continue
        cands = feature_candidates(e, run, tol['duration_ms'], tol['gaze_px'])
        cands = [c for c in cands if abs(c['offset_ms']-clock['constant_offset_ms']) <= tol['time_ms']+1e-8]
        e['candidates'] = [candidate_record(c, clock['constant_offset_ms']) for c in cands]
        if not cands:
            e.update(status='UNMATCHED', reason='no_candidate_within_fixed_clock_duration_gaze_bounds')
        elif len(cands) > 1:
            e.update(status='AMBIGUOUS', reason='multiple_candidates')
        else:
            c = cands[0]
            n = c['nemar']
            e.update(status='MATCHED', reason='unique_clock_duration_gaze_candidate', nemar=basic_n(n),
                     nemar_previous=basic_n(n['previous']), nemar_next=basic_n(n['next']),
                     nemar_incoming=basic_n(n['incoming']),
                     time_error_ms=c['offset_ms']-clock['constant_offset_ms'],
                     duration_error_ms=c['duration_error_ms'], gaze_error_px=c['gaze_error_px'],
                     pupil_error=(e['pupil']-n['pupil_size']) if e['pupil'] is not None and n['pupil_size'] is not None else None)
            sac = n['incoming']
            e['incoming_amp_error_deg'] = (e['matlab_single_fix_incoming_amp']-sac['sacc_amplitude']
                if sac is not None and e['matlab_single_fix_incoming_amp'] is not None and sac['sacc_amplitude'] is not None else None)
            r = word_lookup.get((e['sentence_id'], e['word_id'])) if word_lookup is not None else None
            e['word_text_agrees_with_reading_measures'] = r is not None and r['word'] == e['word']
            e['nfix_agrees_with_reading_measures'] = r is not None and finite(r['nfixations']) == e['matlab_word_nfix']
            e['in_expected_trigger_interval'] = interval is not None and interval['start'] <= n['onset'] < interval['end']
            if require_chain and (not e['word_text_agrees_with_reading_measures'] or not e['in_expected_trigger_interval']):
                e.update(status='AMBIGUOUS', reason='word_metadata_or_presentation_order_contradiction')
        output.append(e)
    # Refuse local conflicts rather than choose the first or nearest candidate.
    proposed = [e for e in output if e['status'] == 'MATCHED']
    rows = Counter(e['nemar']['row'] for e in proposed)
    bad = {e['matlab_fixation_id'] for e in proposed if rows[e['nemar']['row']] > 1}
    proposed.sort(key=lambda e: e['start_ms'])
    for a,b in zip(proposed, proposed[1:]):
        if b['nemar']['onset'] <= a['nemar']['onset']:
            bad.update([a['matlab_fixation_id'], b['matlab_fixation_id']])
    for e in output:
        if e['matlab_fixation_id'] in bad:
            e.update(status='AMBIGUOUS', reason='one_to_one_or_order_conflict')
    confirmed = [e for e in output if e['status'] == 'MATCHED']
    sufficient = (len(confirmed) >= 4 and len(set(e['word_id'] for e in confirmed)) >= 3 and
                  max([e['start_ms'] for e in confirmed], default=0)-min([e['start_ms'] for e in confirmed], default=0) >= 1000)
    for e in output:
        if e['status'] == 'MATCHED' and not sufficient:
            e.update(status='AMBIGUOUS', reason='insufficient_sequence_support')
    return output


def validation_row(rows, participant, condition, run, sentence_id, split, test, **extra):
    matched = [r for r in rows if r['status'] == 'MATCHED']
    counts = Counter(r['status'] for r in rows)
    ordered = sorted(matched, key=lambda r: r['start_ms'])
    gaps = [1000*(b['nemar']['onset']-a['nemar']['onset'])-(b['start_ms']-a['start_ms']) for a,b in zip(ordered, ordered[1:])]
    return {'participant_id': participant, 'condition': condition, 'run_id': run, 'sentence_id': sentence_id,
            'split': split, 'test': test, 'total': len(rows), **{k: counts[k] for k in ['MATCHED', 'AMBIGUOUS', 'UNMATCHED', 'INVALID']},
            'max_abs_time_error_ms': max([abs(e['time_error_ms']) for e in matched], default=None),
            'max_abs_adjacent_interval_error_ms': max([abs(x) for x in gaps], default=None),
            'order_violations': sum(b['nemar']['onset'] <= a['nemar']['onset'] for a,b in zip(ordered, ordered[1:])),
            'max_gaze_error_px': max([e['gaze_error_px'] for e in matched], default=None), **extra}


def confirmed_csv(e, block):
    n = e['nemar']; prev=e['nemar_previous']; nxt=e['nemar_next']; sac=e['nemar_incoming']; hdr=block['eeg']
    sample = int(np.floor(n['onset']*hdr['sampling_rate_hz']+.5))
    return {**{k:e.get(k) for k in ['participant_id','condition','run_id','sentence_id','word_id','word','word_length','matlab_fixation_id']},
            'nemar_fixation_id': f"{e['participant_id']}/{e['condition']}/run{e['run_id']}/events_row{n['row']}",
            'fixation_onset': n['onset'], 'fixation_duration': n['duration'],
            'previous_fixation_duration': prev['duration'] if prev else None,
            'incoming_saccade_amplitude': sac['sacc_amplitude'] if sac else None,
            'next_fixation_onset': nxt['onset'] if nxt else None,
            'match_status': e['status'], 'matching_error': e['time_error_ms'],
            'matching_evidence': 'unique_fixed_clock+duration+raw_gaze;ordered_multievent_sequence;reading_measures_text;trigger_order;vmrk_clock;complete_EEG_bytes',
            'source_file_matlab': e['source_file_matlab'], 'source_file_nemar': n['source'],
            'split': e['split'], 'matlab_onset_ms': e['start_ms'], 'matlab_span_ms': e['duration_ms'],
            'matlab_nominal_span_ms': e['nominal_span_ms'], 'duration_error_after_plus2ms': e['duration_error_ms'],
            'gaze_error_px': e['gaze_error_px'], 'pupil_error_native_units': e['pupil_error'],
            'incoming_amp_error_deg_single_fix_words': e['incoming_amp_error_deg'],
            'matlab_rawET_reference_chain': e['rawET_path'], 'matlab_rawET_object_address': e['rawET_hdf5_object_address'],
            'matlab_rawET_values_sha256_float64_le': e['rawET_values_sha256_float64_le'],
            'within_word_fixation': e['within_word_fixation'], 'matlab_fixPosition': e['matlab_fixPosition'],
            'nemar_event_row_including_header': n['row'],
            'previous_fixation_row': prev['row'] if prev else None, 'next_fixation_row': nxt['row'] if nxt else None,
            'incoming_saccade_row': sac['row'] if sac else None,
            'incoming_saccade_onset_s': sac['onset'] if sac else None,
            'incoming_saccade_duration_s': sac['duration'] if sac else None,
            'next_onset_delta_s': nxt['onset']-n['onset'] if nxt else None,
            'source_eeg_header': hdr['header'], 'source_eeg_binary': hdr['binary'],
            'eeg_frequency_hz': hdr['sampling_rate_hz'], 'eeg_sample_nearest_0based': sample,
            'eeg_onset_fractional_sample_0based': n['onset']*hdr['sampling_rate_hz'],
            'eeg_nearest_sample_error_ms': (sample/hdr['sampling_rate_hz']-n['onset'])*1000,
            'eeg_frame_byte_offset': sample*hdr['frame_bytes'],
            'clock_offset_ms': block['clock']['constant_offset_ms']}


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (list,dict)) else v for k,v in row.items()})


def rect_flags(e):
    b=e['wordbounds']
    if b is None or e.get('gaze_x') is None:
        return None, None
    inside = lambda x,y: b[0] <= x <= b[2] and b[1] <= y <= b[3]
    raw = inside(e['gaze_x'], e['gaze_y'])
    corrected = inside(e['corrected_gaze_x'], e['corrected_gaze_y']) if e.get('allFixations_duration_agrees') and e.get('corrected_gaze_x') is not None else None
    return raw, corrected


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT, help='New directory; existing output files are never overwritten.')
    args=parser.parse_args(); OUT=args.out.resolve()
    if OUT == ROOT or OUT.is_relative_to(ROOT/'data') or OUT.is_relative_to(DATA):
        raise ValueError('Output must be separate from input data and previous audits')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Choose a new --out directory: {OUT}')
    started=time.monotonic()
    manifest_path=PUBLIC/'data/nemar_manifest.json'
    manifest={x['path']:x['size'] for x in json.loads(manifest_path.read_text())}
    run_cache={}
    def get_runs(p,c):
        if (p,c) not in run_cache:
            run_cache[p,c]=[load_run(p,c,r) for r in range(1,8)]
        return run_cache[p,c]
    blocks=[]; all_rows=[]; validation=[]; selected_sentences=[]
    specs=[(p,'NR',r) for p in ['YAG','YDG','YIS'] for r in [1,4]]+[('YAG','TSR',1)]
    # Frozen after exploratory YAG sequences. No holdout is used to fit clocks.
    explored_yag_nr={1,2,3,25,49,51,75,100,152,175,201}
    for p,c,r in specs:
        runs=get_runs(p,c);run=runs[r-1]
        lo=1+sum(len(x['intervals']) for x in runs[:r-1]);hi=lo+len(run['intervals'])-1
        cal_ids=sorted(set([lo,(lo+hi)//2,hi]))
        hold_ids=sorted(set([lo+3,lo+4,lo+7,(lo+hi)//2-3,(lo+hi)//2+3,hi-2,hi-1])-set(cal_ids))
        # Fresh final holdout after loader development: never used in pilot inspection.
        final_ids=sorted(set([lo+11,lo+15,hi-4])-set(cal_ids+hold_ids))
        if p=='YAG' and c=='NR':
            hold_ids=[i for i in hold_ids if i not in explored_yag_nr]
        # Initial Henry-neighbour pilot is included as development, never validation.
        pilot_ids=[2,3] if (p,c,r)==('YAG','NR',1) else []
        mat_path=ROOT/f'data/train/results{p}_{c}.mat';track(mat_path,False)
        word_path=DATA/f'derivatives/reading_measures/sub-{p}/sub-{p}_task-{c.lower()}_desc-wordmeasures_beh.tsv';track(word_path)
        words={(int(x['sentence_id']),int(x['word_id'])):x for x in table(word_path)}
        with h5py.File(mat_path,'r') as f:
            sentences=[mat_sent(f,p,c,i) for i in sorted(set(cal_ids+hold_ids+final_ids+pilot_ids))]
        for s in sentences:
            s['split']='calibration' if s['sentence_id'] in cal_ids else ('pilot' if s['sentence_id'] in pilot_ids else ('final_holdout' if s['sentence_id'] in final_ids else 'holdout'))
        clock=calibrate([s for s in sentences if s['split']=='calibration'],runs,r)
        eeg=header_info(run,manifest)
        block={'participant':p,'condition':c,'run':r,'sentence_range_from_trigger_order':[lo,hi],
               'calibration_ids':cal_ids,'holdout_ids':hold_ids,'final_holdout_ids':final_ids,'pilot_ids':pilot_ids,'clock':clock,'eeg':eeg,
               'trigger_counts_all_runs':[len(x['intervals']) for x in runs]}
        blocks.append(block)
        print(f'{p}/{c}/run{r}: calibration accepted={clock["accepted"]}, offset={clock["constant_offset_ms"]}',flush=True)
        for s in sentences:
            si=s['sentence_id']; interval=run['intervals'][si-lo]
            s['run']=r;s['interval']=interval;selected_sentences.append(s)
            rows=align(s,run,clock,word_lookup=words,interval=interval)
            for e in rows:
                if e['status']=='MATCHED':
                    in_eeg=e['nemar']['onset'] <= eeg.get('last_sample_s',-1)
                    if not eeg.get('complete') or not eeg.get('EEG_clock_verified') or not in_eeg:
                        e.update(status='AMBIGUOUS',reason='EEG_chain_not_verified')
            all_rows.extend(rows)
            validation.append(validation_row(rows,p,c,r,si,s['split'],'baseline'))
            if s['split'] in ['holdout','final_holdout']:
                for name,tol in [('tight',{'time_ms':2.,'duration_ms':2.,'gaze_px':1.}),
                                 ('wide',{'time_ms':8.,'duration_ms':8.,'gaze_px':4.})]:
                    other=align(s,run,clock,tol,words,interval)
                    baseline_ids={(e['matlab_fixation_id'],e['nemar']['row']) for e in rows if e['status']=='MATCHED'}
                    changed=sum((e['matlab_fixation_id'],e['nemar']['row']) not in baseline_ids for e in other if e['status']=='MATCHED')
                    validation.append(validation_row(other,p,c,r,si,s['split'],'sensitivity_'+name,
                                                     new_or_changed_pairs=changed,**tol))
    # Negative controls actively search a free offset, with ID gate deliberately relaxed.
    controls=[]
    for s in [x for x in selected_sentences if x['split'] in ['holdout','final_holdout']]:
        wrong_p='YDG' if s['participant']=='YAG' else 'YAG'
        wrong_c=s['condition']; wrong_run=s['run']
        for name,p,c,r in [('wrong_participant',wrong_p,wrong_c,wrong_run),
                           ('wrong_run',s['participant'],s['condition'],2 if s['run'] != 2 else 3)]:
            run=get_runs(p,c)[r-1]
            gs=offset_clusters(s,[run]);best=gs[0] if gs else None
            count=best['count'] if best else 0
            accepted=best is not None and count>=6 and best['words']>=3 and best['span_ms']>=1000
            controls.append({'source_participant':s['participant'],'source_condition':s['condition'],
                             'source_run':s['run'],'sentence_id':s['sentence_id'],'test':name,
                             'target_participant':p,'target_condition':c,'target_run':r,
                             'best_sequence_anchors':count,'sequence_accepted':accepted,
                             'best':cluster_summary(best) if best else None})
            validation.append({'participant_id':s['participant'],'condition':s['condition'],'run_id':s['run'],
                               'sentence_id':s['sentence_id'],'split':'negative_control','test':name,
                               'target_participant':p,'target_run':r,'best_sequence_anchors':count,
                               'false_sequence_accepted':int(accepted)})
    matches=[e for e in all_rows if e['status']=='MATCHED']
    # Final global one-to-one and order checks; fail closed rather than silently publish.
    keys=[(e['participant_id'],e['condition'],e['run_id'],e['nemar']['row']) for e in matches]
    assert len(keys)==len(set(keys)), 'Duplicate NEMAR assignments across selected sentences'
    for b in blocks:
        ordered=sorted([e for e in matches if (e['participant_id'],e['condition'],e['run_id'])==(b['participant'],b['condition'],b['run'])],key=lambda e:e['start_ms'])
        assert all(a['nemar']['onset']<z['nemar']['onset'] for a,z in zip(ordered,ordered[1:])), 'Global temporal order violation'
    blookup={(b['participant'],b['condition'],b['run']):b for b in blocks}
    match_table=[confirmed_csv(e,blookup[e['participant_id'],e['condition'],e['run_id']]) for e in matches]
    unmatched=[e for e in all_rows if e['status'] in ['UNMATCHED','INVALID']]
    ambiguous=[e for e in all_rows if e['status']=='AMBIGUOUS']
    hold=[e for e in all_rows if e['split'] in ['holdout','final_holdout']]
    counts={k:sum(e['status']==k for e in all_rows) for k in ['MATCHED','AMBIGUOUS','UNMATCHED','INVALID']}
    holdcounts={k:sum(e['status']==k for e in hold) for k in counts}
    selection={}
    for status in ['MATCHED','UNMATCHED','AMBIGUOUS','INVALID']:
        es=[e for e in all_rows if e['status']==status]
        selection[status]={'count':len(es),'duration_ms':stats([e.get('duration_ms') for e in es]),
                           'word_position':stats([e['word_id'] for e in es]),
                           'relative_word_position':stats([e['word_id']/e['word_count'] for e in es]),
                           'within_word_fixation':stats([e['within_word_fixation'] for e in es]),
                           'first_fixation_count':sum(e['within_word_fixation']==1 for e in es),
                           'duration_missing':sum('duration_ms' not in e for e in es)}
    henry=next(e for e in all_rows if (e['participant_id'],e['condition'],e['sentence_id'],e['word_id'],e['within_word_fixation'])==('YAG','NR',1,1,1))
    henry_s=next(s for s in selected_sentences if (s['participant'],s['condition'],s['sentence_id'])==('YAG','NR',1))
    hrun=get_runs('YAG','NR')[0]
    henry_g=offset_clusters(henry_s,get_runs('YAG','NR'))
    same_features=[{'run':rr['run'],**candidate_record(c,blookup['YAG','NR',1]['clock']['constant_offset_ms'])} for rr in get_runs('YAG','NR') for c in feature_candidates(henry,rr)]
    neighbor_rows=sorted([e for e in matches if (e['participant_id'],e['condition'],e['sentence_id'])==('YAG','NR',1)],key=lambda e:e['start_ms'])
    neighbor_rows=neighbor_rows[:12]
    # Inspect actual nested saccade fields/counts without assuming a fixation row join.
    with h5py.File(ROOT/'data/train/resultsYAG_NR.mat','r') as f:
        s=f['sentenceData'];sc=ref(f,s,'allSaccades',0);af=ref(f,s,'allFixations',0)
        nested={}
        for key in sc:
            obj=sc[key]
            nested[key]=({'kind':'group','fields':list(obj)} if isinstance(obj,h5py.Group) else
                         {'shape':list(obj.shape),'first_values':nums(obj)[:8].tolist()})
        fixation_structure={key:{'shape':list(af[key].shape),'first_values':nums(af[key])[:8].tolist()} for key in af}
    for e in matches:
        e['raw_gaze_inside_wordbounds'],e['corrected_gaze_inside_wordbounds']=rect_flags(e)
    nemar_coverage=[]; all_unassigned_nemar_durations=[]
    for b in blocks:
        p,c,r=b['participant'],b['condition'],b['run'];run=get_runs(p,c)[r-1]
        selected=[s for s in selected_sentences if (s['participant'],s['condition'],s['run'])==(p,c,r)]
        ns=[n for n in run['fixes'] if any(s['interval']['start'] <= n['onset'] < s['interval']['end'] for s in selected)]
        assigned={e['nemar']['row'] for e in matches if (e['participant_id'],e['condition'],e['run_id'])==(p,c,r)}
        un=[n for n in ns if n['row'] not in assigned]
        all_unassigned_nemar_durations.extend(n['duration']*1000 for n in un)
        nemar_coverage.append({'participant':p,'condition':c,'run':r,'all_fixations_in_selected_intervals':len(ns),
                              'matched':sum(n['row'] in assigned for n in ns),'unassigned':len(un),
                              'unassigned_duration_below_98ms':sum(n['duration'] < .098-1e-9 for n in un),
                              'unassigned_duration_at_least_98ms':sum(n['duration'] >= .098-1e-9 for n in un),
                              'unassigned_duration_ms':stats([n['duration']*1000 for n in un])})
    # Sources checked after all reads; big binary/MAT files are stat-checked, not hashed.
    changes=[]
    for name,before in SOURCES.items():
        p=ROOT/name; st=p.stat()
        if st.st_size!=before['bytes'] or st.st_mtime_ns!=before['mtime_ns']:
            changes.append(name)
        if before['sha256'] is not None and hashlib.sha256(p.read_bytes()).hexdigest()!=before['sha256']:
            changes.append(name)
    assert not changes, f'Input changed during audit: {changes}'
    summary={'total_fixation_slots':len(all_rows),'status_counts':counts,'holdout_status_counts':holdcounts,
             'holdout_fixation_slots':len(hold),'participants':sorted(set(e['participant_id'] for e in matches)),
             'final_holdout_status_counts':dict(Counter(e['status'] for e in all_rows if e['split']=='final_holdout')),
             'split_sentence_counts':dict(Counter(s['split'] for s in selected_sentences)),
             'participant_condition_runs':len(blocks),'selected_sentences':len(selected_sentences),
             'matched_sentences':len(set((e['participant_id'],e['condition'],e['sentence_id']) for e in matches)),
             'canonical_sentence_keys':len(set((e['condition'],e['sentence_id']) for e in matches)),
             'matched_participant_word_slots':len(set((e['participant_id'],e['condition'],e['sentence_id'],e['word_id']) for e in matches)),
             'matched_canonical_sentence_word_keys':len(set((e['condition'],e['sentence_id'],e['word_id']) for e in matches)),
             'matched_first_fixations':sum(e['within_word_fixation']==1 for e in matches),
             'all_word_slots_selected':sum(len(s['words']) for s in selected_sentences),
             'word_slots_not_fixated':sum(w['nfix'] is None or w['nfix']==0 for s in selected_sentences for w in s['words']),
             'time_error_ms':stats([e['time_error_ms'] for e in matches]),
             'absolute_time_error_ms':stats([abs(e['time_error_ms']) for e in matches]),
             'duration_error_after_plus2_ms':stats([e['duration_error_ms'] for e in matches]),
             'nominal_duration_error_after_plus2_ms':stats([e['nominal_span_ms']-(e['nemar']['duration']*1000+2) for e in matches]),
             'nominal_duration_exact_plus2_count':sum(abs(e['nominal_span_ms']-(e['nemar']['duration']*1000+2))<1e-6 for e in matches),
             'gaze_error_px':stats([e['gaze_error_px'] for e in matches]),
             'pupil_error_native':stats([e['pupil_error'] for e in matches]),
             'single_fix_word_incoming_amp_error_deg':stats([e['incoming_amp_error_deg'] for e in matches]),
             'single_fix_word_amp_disagreements_over_001deg':sum(e['incoming_amp_error_deg'] is not None and abs(e['incoming_amp_error_deg'])>.010001 for e in matches),
             'reading_measures_nfix_disagreements':sum(not e['nfix_agrees_with_reading_measures'] for e in matches),
             'previous_fixation_available':sum(e['nemar_previous'] is not None for e in matches),
             'incoming_saccade_available':sum(e['nemar_incoming'] is not None for e in matches),
             'next_fixation_available':sum(e['nemar_next'] is not None for e in matches),
             'raw_gaze_outside_wordbounds':sum(e['raw_gaze_inside_wordbounds'] is False for e in matches),
             'corrected_gaze_outside_wordbounds':sum(e['corrected_gaze_inside_wordbounds'] is False for e in matches),
             'corrected_gaze_unavailable':sum(e['corrected_gaze_inside_wordbounds'] is None for e in matches),
             'allFixations_duration_mapping_disagreements':sum(e.get('allFixations_duration_agrees') is False for e in all_rows if 'n_samples' in e),
             'negative_controls':len(controls),'false_sequences_in_controls':sum(c['sequence_accepted'] for c in controls),
             'maximum_control_sequence_anchors':max(c['best_sequence_anchors'] for c in controls),
             'all_selected_nemar_fixations':sum(d['all_fixations_in_selected_intervals'] for d in nemar_coverage),
             'all_selected_nemar_unassigned':sum(d['unassigned'] for d in nemar_coverage),
             'all_selected_nemar_unassigned_below_98ms':sum(d['unassigned_duration_below_98ms'] for d in nemar_coverage),
             'all_selected_nemar_unassigned_duration_ms':stats(all_unassigned_nemar_durations),
             'matched_nemar_duration_ms':stats([e['nemar']['duration']*1000 for e in matches]),
             'input_changes':changes}
    participant_summary=[]
    for b in blocks:
        es=[e for e in all_rows if (e['participant_id'],e['condition'],e['run_id'])==(b['participant'],b['condition'],b['run'])]
        participant_summary.append({'participant':b['participant'],'condition':b['condition'],'run':b['run'],
                                    'sentences':len(set(e['sentence_id'] for e in es)), 'total':len(es),**dict(Counter(e['status'] for e in es))})
    evidence={'scope':'Bounded ET alignment only; no EEG amplitudes, linguistic predictors, or scientific models calculated',
              'protocol':{'baseline_tolerances':TOL,'duration_convention_ms':2,'gaze_summary':'mean of rawET columns x,y excluding last endpoint',
                          'calibration_anchor_minimum':6,'calibration_word_minimum':3,'calibration_span_min_ms':1000,
                          'calibration_rival_max_fraction_exclusive':.5,'calibration_sentence_count':3,'block_anchor_span_min_ms':60000,
                          'validation_sequence_minimum':4,'validation_word_minimum':3,'validation_span_min_ms':1000,
                          'no_optimization':'Unique candidates only; no dynamic-programming cost function or forced assignments.',
                          'exploratory_YAG_NR_sentences':sorted(explored_yag_nr),
                          'sampling':'Each selected block: first/middle/last calibration; fixed relative sentence offsets for holdout. YAG exploratory sentences excluded from holdout.'},
              'summary':summary,'participant_summary':participant_summary,'blocks':blocks,
              'selection_bias_descriptive':selection,'nemar_coverage':nemar_coverage,'validation':validation,'negative_controls':controls,
              'henry':{'result':henry,'feature_only_candidates_across_all_seven_runs':same_features,
                       'sentence_sequence_best':cluster_summary(henry_g[0]),
                       'neighbor_sequence':[{k:e.get(k) for k in ['word','word_id','within_word_fixation','start_ms','duration_ms','nemar','time_error_ms','gaze_error_px','incoming_amp_error_deg']} for e in neighbor_rows]},
              'schema_examples':{'allSaccades':nested,'allFixations':fixation_structure},
              'event_details':all_rows,'sources':SOURCES,'software':{'python':platform.python_version(),'numpy':np.__version__,'h5py':h5py.__version__},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'elapsed_seconds':time.monotonic()-started,
              'units':{'fixation_onset':'NEMAR seconds from run EEG start','fixation_duration':'NEMAR seconds, end minus start',
                       'previous_fixation_duration':'NEMAR seconds; previous in full fixation stream, including short events',
                       'incoming_saccade_amplitude':'NEMAR degrees; immediate preceding saccade ending 0–4 ms before fixation',
                       'next_fixation_onset':'NEMAR seconds, next in full run fixation stream','matching_error':'MATLAB_ms minus clock offset minus NEMAR_ms',
                       'matlab_span_ms':'last time minus first time in rawET, milliseconds','word_id':'1-based MATLAB word array index, text-checked against reading_measures',
                       'sentence_id':'1-based MATLAB sentence slot, text/order checked; not an event field','missing':'empty CSV cell / JSON null, never zero replacement'}}
    # Prepublication integrity checks on final tables, not EEG signal processing.
    assert all(r['match_status']=='MATCHED' for r in match_table)
    assert len(match_table)+len(unmatched)+len(ambiguous)==len(all_rows)
    assert all(abs(r['eeg_nearest_sample_error_ms'])<=1.000001 for r in match_table)
    assert all(r['eeg_sample_nearest_0based']<blookup[r['participant_id'],r['condition'],r['run_id']]['eeg']['sample_frames'] for r in match_table)
    evidence['technical_checks']={'unique_nemar_keys':True,'global_temporal_order':True,'status_partition':True,
                                  'EEG_pointer_within_complete_file':True,'maximum_sample_rounding_error_ms':max(abs(r['eeg_nearest_sample_error_ms']) for r in match_table),
                                  'source_files_unchanged':not changes,'EEG_amplitude_values_read':0}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(OUT/'matches.csv',match_table)
    issue_fields=['participant_id','condition','run_id','sentence_id','word_id','word','matlab_fixation_id','status','reason','candidates','source_file_matlab','rawET_path','rawET_shape']
    write_csv(OUT/'ambiguous.csv',[{k:e.get(k) for k in issue_fields} for e in ambiguous],issue_fields)
    write_csv(OUT/'unmatched.csv',[{k:e.get(k) for k in issue_fields} for e in unmatched],issue_fields)
    write_csv(OUT/'validation.csv',validation)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f'Output: {OUT}')


if __name__ == '__main__':
    main()
