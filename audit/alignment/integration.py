"""Extend the existing audited clock/duration/gaze matching, without EEG fits."""
from collections import Counter,defaultdict
from copy import deepcopy
import h5py
import numpy as np
import pandas as pd
import audit_zuco2_alignment as old
import argparse
import csv
import hashlib
import json
import tempfile
import time
from pathlib import Path

ROOT, DATA, PUBLIC = old.ROOT, old.DATA, old.PUBLIC
CONFIG = {'matching': json.loads(Path(__file__).with_name('matching.json').read_text())}

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def jwrite(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix=path.name+'.',suffix='.writing',delete=False) as f:
  f.write(json.dumps(clean(value),ensure_ascii=False,indent=2)+'\n');tmp=Path(f.name)
 tmp.replace(path)
def jread(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if isinstance(rows,pd.DataFrame):rows.to_csv(p,index=False);return
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 if not fields:raise ValueError('Do not create fake empty tables: '+str(p))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict)) else clean(v) for k,v in r.items()})
def log(msg):print(time.strftime('%H:%M:%S')+' '+str(msg),flush=True)


def mat_sent(f,p,sid):
 """Same rawET interpretation as the audited reader; optional fields are explicit.

 Some sentences lack rawET or inSacc fields entirely. Inspect the actual word
 struct and keep missing rawET slots INVALID; optional MAT saccade summaries
 are not needed to obtain the physical incoming NEMAR saccade.
 """
 s=f['sentenceData'];w=old.ref(f,s,'word',sid-1)
 if not isinstance(w,h5py.Group):return old.mat_sent(f,p,'NR',sid)
 needed={'inSacc_amp','inSacc_duration','rawET','fixPositions'}
 if needed.issubset(w.keys()):return old.mat_sent(f,p,'NR',sid)
 text=old.txt(old.ref(f,s,'content',sid-1));out={'participant':p,'condition':'NR','sentence_id':sid,'text':text,'events':[],'words':[],
  'missing_optional_word_fields':sorted(needed-set(w.keys())),'word_fields_checked':list(w.keys())}
 def value(field,wi):return old.ref(f,w,field,wi) if field in w else None
 def nums(field,wi):
  x=value(field,wi);return old.nums(x) if x is not None else np.array([])
 for wi in range(w['content'].shape[0]):
  word=old.txt(value('content',wi));nf=old.scalar(value('nFixations',wi));positions=nums('fixPositions',wi)
  amps=nums('inSacc_amp',wi).tolist();durs=nums('inSacc_duration',wi).tolist();cells=value('rawET',wi)
  refs=cells[()].ravel() if cells is not None and h5py.check_dtype(ref=cells.dtype) and not old.empty(cells) else []
  out['words'].append({'word_id':wi+1,'word':word,'nfix':nf,'incoming_amplitudes':amps})
  for fi in range(max(len(refs),int(nf or 0))):
   e={'participant_id':p,'condition':'NR','sentence_id':sid,'word_id':wi+1,'word':word,'word_length':len(word) if word else None,
    'word_count':w['content'].shape[0],'within_word_fixation':fi+1,'matlab_fixation_id':f'{p}/NR/s{sid}/w{wi+1}/f{fi+1}',
    'source_file_matlab':old.rel(Path(f.filename)),'matlab_fixPosition':old.finite(positions[fi]) if fi<len(positions) else None,
    'matlab_word_nfix':nf,'matlab_incoming_amplitudes':amps,'matlab_single_fix_incoming_amp':amps[0] if nf==1 and len(amps)==1 else None,
    'matlab_single_fix_incoming_duration_samples':durs[0] if nf==1 and len(durs)==1 else None,
    'wordbounds':None,'status':'INVALID','reason':'rawET_field_absent_in_checked_word_struct' if cells is None else 'missing_or_invalid_rawET'}
   if fi<len(refs) and refs[fi]:
    a=f[refs[fi]];e.update(rawET_path=f'sentenceData/word[{sid-1},0] -> rawET[{wi},0] -> cell[{fi},0] (dereference at arrows)',
     rawET_hdf5_object_address=int(h5py.h5o.get_info(a.id).addr),rawET_shape=list(a.shape))
    if not old.empty(a) and a.ndim==2 and a.shape[1]==4 and a.shape[0]>=2:
     v=np.asarray(a[()],float);e['rawET_values_sha256_float64_le']=hashlib.sha256(v.astype('<f8').tobytes()).hexdigest()
     if np.isfinite(v[:,0]).all() and np.all(np.diff(v[:,0])>0):
      m=np.nanmean(v[:-1,1:],axis=0);e.update(start_ms=float(v[0,0]),stop_ms=float(v[-1,0]),duration_ms=float(v[-1,0]-v[0,0]),
       n_samples=len(v),gaze_x=old.finite(m[0]),gaze_y=old.finite(m[1]),pupil=old.finite(m[2]),nominal_span_ms=(len(v)-1)*2,
       status='UNMATCHED',reason='not_yet_aligned')
   out['events'].append(e)
 return out

def anchor_clock(sentences,runs,run_id):
 diagnostics=[];chosen=[];lo,hi=CONFIG['matching']['canonical_sentence_ranges'][run_id-1];mid=(lo+hi)//2
 zones=[[lo+i for i in range(5)],[mid+i for i in [0,-1,1,-2,2]],[hi-i for i in range(5)]]
 for zone in zones:
  selected=None
  for sid in zone:
   s=sentences[sid];groups=old.offset_clusters(s,runs);best=groups[0] if groups else None
   rivals=[g for g in groups[1:] if g['run']!=best['run'] or abs(g['offset_ms']-best['offset_ms'])>8] if best else []
   rival=rivals[0]['count'] if rivals else 0
   ok=best is not None and best['run']==run_id and best['count']>=6 and best['words']>=3 and best['span_ms']>=1000 and rival<best['count']/2
   diagnostics.append({'sentence_id':sid,'accepted':ok,'best':old.cluster_summary(best) if best else None,'rival_count':rival})
   if ok:selected=s;break
  if selected:chosen.append(selected)
 clock=old.calibrate(chosen,runs,run_id) if len(chosen)==3 else {'accepted':False,'constant_offset_ms':None,'reason':'fewer_than3_supported_anchor_zones'}
 clock['search_diagnostics']=diagnostics;clock['selected_sentence_ids']=[s['sentence_id'] for s in chosen]
 return clock

def containing_interval(sentence,run,clock):
 valid=[e for e in sentence['events'] if e['status']!='INVALID']
 if not clock['accepted'] or not valid:return None
 predicted=np.array([e['start_ms']-clock['constant_offset_ms'] for e in valid])/1000
 hits=[i for i in run['intervals'] if np.all((predicted>=i['start']-.004)&(predicted<i['end']+.004))]
 return hits[0] if len(hits)==1 else None

def run(OUT, pilot):
 start=time.monotonic()
 paths=sorted((ROOT/'data/train').glob('*_NR.mat'));ps=[p.stem[7:-3] for p in paths]
 assert len(ps)==16,ps
 manifest=jread(PUBLIC/'data/nemar_manifest.json');manifest_size={r['path']:r['size'] for r in manifest}
 prior=pd.read_csv(pilot/'matches.csv',keep_default_na=False)
 prior=prior[prior.condition=='NR'];prior_by={r['matlab_fixation_id']:r for r in prior.to_dict('records')}
 ev=jread(pilot/'evidence.json');prior_blocks={(b['participant'],b['run']):b for b in ev['blocks'] if b['condition']=='NR'}
 decisions=[];matches=[];words_out=[];sent_out=[];blocks=[];negatives=[];source_meta=[]
 for path,p in zip(paths,ps):
  log(f'A {p}: read MATLAB ET and all NR events')
  source_meta.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'sha256':sha(path)})
  runs=[old.load_run(p,'NR',r) for r in range(1,8)]
  lookup_path=DATA/f'derivatives/reading_measures/sub-{p}/sub-{p}_task-nr_desc-wordmeasures_beh.tsv'
  old.track(lookup_path)
  word_lookup={(int(w['sentence_id']),int(w['word_id'])):w for w in old.table(lookup_path)}
  with h5py.File(path,'r') as h:
   ns=h['sentenceData']['word'].shape[0];assert ns==349,(p,ns)
   sentences={sid:mat_sent(h,p,sid) for sid in range(1,ns+1)}
  for sid,s in sentences.items():
   sent_out.append({'participant_id':p,'sentence_id':sid,'text':s['text'],'word_count':len(s['words']),'matlab_fixation_slots':len(s['events']),
    'missing_checked_word_fields':s.get('missing_optional_word_fields',[])})
   for w in s['words']:
    words_out.append({'participant_id':p,'sentence_id':sid,'word_id':w['word_id'],'word':w['word'],'sentence_text':s['text'],
     'nfix_saved':w['nfix'],'matlab_incoming_amplitudes':w['incoming_amplitudes']})
  for run_id,run0 in enumerate(runs,1):
   lo,hi=CONFIG['matching']['canonical_sentence_ranges'][run_id-1]
   clock=deepcopy(prior_blocks[p,run_id]['clock']) if (p,run_id) in prior_blocks else anchor_clock(sentences,runs,run_id)
   hdr=old.header_info(run0,manifest_size)
   block={'participant':p,'condition':'NR','run':run_id,'clock':clock,'eeg':hdr,'canonical_sentence_range':[lo,hi],
    'presentation_count':len(run0['intervals']),'expected_count':hi-lo+1,'reused_prior_clock':(p,run_id) in prior_blocks}
   blocks.append(block);log(f'A {p}/run{run_id}: clock={clock["accepted"]}, presentations={len(run0["intervals"])}')
   for sid in range(lo,hi+1):
    s=sentences[sid];s['split']='extension_holdout' if sid not in clock.get('selected_sentence_ids',[]) else 'calibration'
    interval=containing_interval(s,run0,clock)
    new_s={**s,'events':[e for e in s['events'] if e['matlab_fixation_id'] not in prior_by]}
    rows=old.align(new_s,run0,clock,word_lookup=word_lookup,interval=interval)
    by_id={e['matlab_fixation_id']:e for e in rows}
    for original in s['events']:
     mid=original['matlab_fixation_id'];r=by_id.get(mid);reused=mid in prior_by
     if reused:
      known=prior_by[mid];idx=int(known['nemar_event_row_including_header'])-2;n=run0['events'][idx]
      assert n['type']=='fixation' and abs(n['onset']-float(known['fixation_onset']))<1e-10
      assert original.get('rawET_values_sha256_float64_le')==known['matlab_rawET_values_sha256_float64_le'],mid
      row=dict(known);row['prior_match_reused']=True
      state='MATCHED';reason='existing_validated_pair_source_hash_and_event_identity_checked';matches.append(row)
     else:
      state=r['status'];reason=r['reason']
      if state=='MATCHED':
       row=old.confirmed_csv(r,block);row['prior_match_reused']=False;matches.append(row)
     if state=='MATCHED':
      row.update(sentence_text=s['text'],presentation_start_s=interval['start'] if interval else None,presentation_end_s=interval['end'] if interval else None,
       previous_fixation_source='NEMAR full physical fixation stream, previous by onset',incoming_saccade_source='NEMAR nearest preceding saccade end within4ms',
       previous_is_physical_stream_event=True,first_saved_fixation=int(original['within_word_fixation'])==1,
       EEG_chain_eligible=bool(hdr['complete'] and hdr['EEG_clock_verified'] and float(row['fixation_onset'])<=hdr['last_sample_s']))
     decisions.append({'participant_id':p,'condition':'NR','run_id':run_id,'sentence_id':sid,'word_id':original['word_id'],'word':original['word'],
      'matlab_fixation_id':mid,'within_word_fixation':original['within_word_fixation'],'status':state,'reason':reason,'prior_pair_reused':reused,
      'matlab_single_fix_incoming_amp':original.get('matlab_single_fix_incoming_amp'),
      'matlab_amp_error_deg':row.get('incoming_amp_error_deg_single_fix_words') if state=='MATCHED' else None})
   # One free-offset negative control per block, selected without EEG amplitudes.
   controls=[sentences[x] for x in range(lo,hi+1) if x not in clock.get('selected_sentence_ids',[]) and len(sentences[x]['events'])>=8]
   if controls and clock['accepted']:
    s=controls[len(controls)//2];wrong=runs[run_id%7];groups=old.offset_clusters(s,[wrong]);best=groups[0] if groups else None
    accepted=best is not None and best['count']>=6 and best['words']>=3 and best['span_ms']>=1000
    negatives.append({'participant_id':p,'run_id':run_id,'sentence_id':s['sentence_id'],'test':'wrong_run_free_clock','rival_run':wrong['run'],
     'strongest_island_count':best['count'] if best else 0,'false_sequence_accepted':accepted})
    wrongp='YDG' if p!='YDG' else 'YAG';wr=old.load_run(wrongp,'NR',run_id);gg=old.offset_clusters(s,[wr]);bb=gg[0] if gg else None
    acc=bb is not None and bb['count']>=6 and bb['words']>=3 and bb['span_ms']>=1000
    negatives.append({'participant_id':p,'run_id':run_id,'sentence_id':s['sentence_id'],'test':'wrong_participant_free_clock','rival_participant':wrongp,
     'strongest_island_count':bb['count'] if bb else 0,'false_sequence_accepted':acc})
 # Resolve cross-sentence collisions conservatively, never by amplitude or nearest distance.
 cnt=Counter(m['nemar_fixation_id'] for m in matches);bad={k for k,v in cnt.items() if v>1};bad_ids={m['matlab_fixation_id'] for m in matches if m['nemar_fixation_id'] in bad}
 for d in decisions:
  if d['matlab_fixation_id'] in bad_ids:d.update(status='AMBIGUOUS',reason='cross_sentence_duplicate_NEMAR_event')
 matches=[m for m in matches if m['matlab_fixation_id'] not in bad_ids]
 assert len(set(m['matlab_fixation_id'] for m in matches))==len(matches)
 assert len(set(m['nemar_fixation_id'] for m in matches))==len(matches)
 counts=Counter(d['status'] for d in decisions)
 csvwrite(OUT/'matched_fixations.csv',matches);csvwrite(OUT/'matching_decisions.csv',decisions)
 csvwrite(OUT/'words_catalog.csv',words_out);csvwrite(OUT/'sentences_catalog.csv',sent_out)
 csvwrite(OUT/'matching_negative_controls.csv',negatives)
 jwrite(OUT/'alignment_blocks.json',blocks);jwrite(OUT/'matlab_input_integrity.json',source_meta)
 jwrite(OUT/'matching_input_integrity.json',old.SOURCES)
 prior_ids=set(prior_by);reused_now={m['matlab_fixation_id'] for m in matches if m['prior_match_reused']}
 result={'participants':ps,'source_participants':len(ps),'source_sentences_participant':len(sent_out),'source_words_participant':len(words_out),
  'MAT_fixation_slots':len(decisions),'status_counts':counts,'matched_fixations':len(matches),'participants_matched':len(set(m['participant_id'] for m in matches)),
  'clock_accepted_runs':sum(b['clock']['accepted'] for b in blocks),'total_runs':len(blocks),'reused_prior_matches':len(reused_now),
  'prior_pairs_not_reused':sorted(prior_ids-reused_now),'cross_sentence_collisions':len(bad),'negative_control_false_accepts':sum(n['false_sequence_accepted'] for n in negatives),
  'elapsed_seconds':time.monotonic()-start}
 jwrite(OUT/'integration_summary.json',result);return result


def check_physical_events(OUT):
 matches=pd.read_csv(OUT/'matched_fixations.csv');summary=[];errors=[]
 for (p,run),g in matches.groupby(['participant_id','run_id']):
  path=ROOT/g.iloc[0].source_file_nemar;e=pd.read_csv(path,sep='\t',na_values=['n/a']);e['row']=np.arange(len(e))+2
  fixes=e[e.trial_type=='fixation'].sort_values('onset',kind='stable');where={int(v):i for i,v in enumerate(fixes.row)};f=fixes.to_dict('records')
  count_prev=count_next=count_sac=0
  for r in g.itertuples():
   i=where[int(r.nemar_event_row_including_header)];actual=f[i];prev=f[i-1] if i else None;nxt=f[i+1] if i+1<len(f) else None
   if abs(actual['duration']-r.fixation_duration)>1e-10:errors.append([r.matlab_fixation_id,'current_duration'])
   if prev:
    count_prev+=1
    if int(r.previous_fixation_row)!=prev['row'] or abs(r.previous_fixation_duration-prev['duration'])>1e-10:errors.append([r.matlab_fixation_id,'previous_physical_fixation'])
   if nxt:
    count_next+=1
    if int(r.next_fixation_row)!=nxt['row'] or abs(r.next_fixation_onset-nxt['onset'])>1e-10:errors.append([r.matlab_fixation_id,'next_physical_fixation'])
   if np.isfinite(r.incoming_saccade_row):
    sac=e.iloc[int(r.incoming_saccade_row)-2];gap=r.fixation_onset-sac.onset-sac.duration;count_sac+=1
    if sac.trial_type!='saccade' or not(-1e-9<=gap<=.0040001) or abs(r.incoming_saccade_amplitude-sac.sacc_amplitude)>1e-10:errors.append([r.matlab_fixation_id,'incoming_saccade'])
  summary.append({'participant_id':p,'run':run,'matched_events_checked':len(g),'previous_physical_checked':count_prev,'next_physical_checked':count_next,'incoming_saccade_checked':count_sac})
 csvwrite(OUT/'physical_event_covariate_checks.csv',summary);jwrite(OUT/'physical_event_check_errors.json',errors);assert not errors,errors[:5]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=old.OUT, help='Output of the bounded alignment')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('full_generated'))
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or out.is_relative_to(ROOT/'data') or out.is_relative_to(DATA) or out == args.pilot.resolve():
        raise ValueError('Choose an output outside the input datasets and pilot')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {out}')
    print(json.dumps(run(out, args.pilot), indent=2))
    check_physical_events(out)
