#!/usr/bin/env python3
"""Complete exactly nm000279/v1.0.0 from a pinned manifest; verify every file.

Only absent or checksum-invalid targets are installed. Invalid originals are
copied to the run's preserved_originals directory before atomic replacement.
Staging .part downloads resume across curl failures; originals are never used
as an unverified byte prefix. Previous audits are not modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent.parent
BASE='https://data.nemar.org/nm000279/v1.0.0/'
PRINT=threading.Lock()
def log(x):
 with PRINT:print(datetime.now().astimezone().isoformat(timespec='seconds')+' '+x,flush=True)
def checksum(path,algorithm):
 h=hashlib.sha256() if algorithm=='sha256' else hashlib.sha1()
 if algorithm=='git':h.update(f'blob {path.stat().st_size}\0'.encode())
 elif algorithm!='sha256':raise ValueError(algorithm)
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def inspect(root,r):
 p=root/r['path']
 if not p.exists():return {'path':r['path'],'status':'missing','expected_size':r['size']}
 size=p.stat().st_size
 if size!=r['size']:return {'path':r['path'],'status':'size_mismatch','actual_size':size,'expected_size':r['size']}
 actual=checksum(p,r['checksum_algorithm'])
 return {'path':r['path'],'status':'verified' if actual==r['checksum'] else 'checksum_mismatch','actual_size':size,
  'expected_size':r['size'],'actual_checksum':actual,'expected_checksum':r['checksum'],'mtime_ns':p.stat().st_mtime_ns}
def writej(p,x):
 temp=p.with_suffix(p.suffix+'.writing');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def fetch(root,out,r):
 rel=r['path'];target=root/rel;part=out/'staging'/Path(rel+'.part');part.parent.mkdir(parents=True,exist_ok=True)
 url=r.get('bytes_url') or r['url'];assert url.startswith(BASE),url
 for attempt in range(1,9):
  if part.exists() and part.stat().st_size>r['size']:
   part.rename(part.with_name(part.name+f'.oversize-{attempt}'))
  if not part.exists() or part.stat().st_size!=r['size']:
   cmd=['curl','-L','--fail','--silent','--show-error','--continue-at','-','--connect-timeout','20','--max-time','120',
     '--speed-limit','8192','--speed-time','45','--output',str(part),url]
   z=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
   if z.returncode:
    log(f'RETRY {attempt} {rel}: curl{z.returncode} {z.stderr.strip()[:250]}')
    if z.returncode==33 and part.exists():part.rename(part.with_name(part.name+f'.range-unsupported-{attempt}'))
    time.sleep(min(2*attempt,10));continue
  if not part.exists() or part.stat().st_size!=r['size']:
   log(f'RETRY {attempt} {rel}: unexpected downloaded size');continue
  actual=checksum(part,r['checksum_algorithm'])
  if actual!=r['checksum']:
   part.rename(part.with_name(part.name+f'.checksum-failed-{attempt}'));log(f'RETRY {attempt} {rel}: checksum mismatch');continue
  # Keep the old target until the new payload is completely verified.
  backup=None
  if target.exists():
   backup=out/'preserved_originals'/rel;backup.parent.mkdir(parents=True,exist_ok=True)
   if backup.exists():raise FileExistsError(backup)
   shutil.copy2(target,backup)
  target.parent.mkdir(parents=True,exist_ok=True);part.replace(target)
  log('INSTALLED '+rel)
  return {'path':rel,'status':'installed_verified','size':r['size'],'checksum_algorithm':r['checksum_algorithm'],
   'checksum':actual,'preserved_original':str(backup) if backup else None}
 raise RuntimeError('Download failed after8 attempts: '+rel)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',type=Path,default=Path(__file__).with_name('nemar_manifest.json'))
 ap.add_argument('--destination',type=Path,default=ROOT/'data/local/data.nemar.org/nm000279/v1.0.0')
 ap.add_argument('--out',type=Path,default=ROOT/'data/download_state');ap.add_argument('--jobs',type=int,default=4,choices=range(1,9));a=ap.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);root=a.destination.resolve();root.mkdir(parents=True,exist_ok=True);rows=json.loads(a.manifest.read_text())
 assert isinstance(rows,list) and len({r['path'] for r in rows})==len(rows)
 for r in rows:
  assert not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts
  assert r['checksum_algorithm'] in ['git','sha256']
 lock=(out/'download.lock').open('a+')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'completion.json').exists():raise FileExistsError('Completed run exists; use a new --out')
 shutil.copyfile(a.manifest,out/'manifest.json')
 log('VERIFYING all existing manifest files')
 before=[inspect(root,r) for r in rows];writej(out/'verification_before.json',before)
 todo=[r for r,b in zip(rows,before) if b['status']!='verified'];need=sum(r['size'] for r in todo)
 log(f'PLAN {dict(Counter(b["status"] for b in before))}; download at most{need/1e9:.3f}GB')
 if shutil.disk_usage(root).free<need+1024**3:raise RuntimeError('Not enough disk space including1GiB reserve')
 installed=[];errors=[];start=time.monotonic()
 with ThreadPoolExecutor(max_workers=a.jobs) as pool:
  futures={pool.submit(fetch,root,out,r):r for r in todo}
  for fut in as_completed(futures):
   try:installed.append(fut.result())
   except Exception as err:errors.append({'path':futures[fut]['path'],'error':str(err)});log('ERROR '+str(err))
   writej(out/'download_progress.json',{'installed':installed,'errors':errors,'requested':len(todo)})
 after=[inspect(root,r) for r in rows];writej(out/'verification_after.json',after)
 changed_good=[]
 for b,c in zip(before,after):
  if b['status']=='verified' and (c['status']!='verified' or b['mtime_ns']!=c.get('mtime_ns')):changed_good.append(b['path'])
 result={'dataset':'nm000279','version':'v1.0.0','manifest_url':BASE+'manifest.json','manifest_sha256':checksum(out/'manifest.json','sha256'),
  'destination':str(root),'manifest_files':len(rows),'expected_bytes':sum(r['size'] for r in rows),
  'before_counts':dict(Counter(b['status'] for b in before)),'after_counts':dict(Counter(b['status'] for b in after)),
  'installed_files':len(installed),'installed_payload_bytes':sum(r['size'] for r in installed),'errors':errors,
  'previously_good_files_changed':changed_good,'preserved_originals':[r for r in installed if r['preserved_original']],
  'elapsed_download_and_final_verify_s':time.monotonic()-start,'script_sha256':checksum(Path(__file__),'sha256')}
 writej(out/'completion.json',result);log('FINISH '+json.dumps(result,ensure_ascii=False))
 assert not errors and not changed_good and all(r['status']=='verified' for r in after)
if __name__=='__main__':main()
