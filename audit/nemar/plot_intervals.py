"""Within-presentation fixation intervals in the three NR/run-1 recordings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_nemar_nm000279 import DATA, table

def load_events(path):
    rr=table(path);out=[]
    for i,r in enumerate(rr):
        out.append({'row':i+2,'type':r['trial_type'],'onset':float(r['onset']),
                    'duration':float(r['duration']),'end':float(r['onset'])+float(r['duration']),
                    'amp':None if r['sacc_amplitude']=='n/a' else float(r['sacc_amplitude']),
                    'has_gaze_xy':r['gaze_x']!='n/a' and r['gaze_y']!='n/a',
                    'has_saccade_end_xy':r['sacc_endx']!='n/a' and r['sacc_endy']!='n/a'})
    return out

def intervals(events):
    out=[];start=None
    for e in events:
        if e['type'] in ['trigger_10','trigger_12']:start=e
        elif start and e['type']==('trigger_11' if start['type']=='trigger_10' else 'trigger_13'):
            out.append([start['onset'],e['onset']]);start=None
    return out

def stat(v):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    return {'n':int(len(v)),'min':float(v.min()),'median':float(np.median(v)),
            'p05':float(np.quantile(v,.05)),'p95':float(np.quantile(v,.95)),
            'max':float(v.max()),'mean':float(v.mean())} if len(v) else {'n':0}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('intervals_generated'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    all_ifi = []
    summaries = []
    for p in ('YAG', 'YDG', 'YIS'):
        events = load_events(DATA / f'sub-{p}/eeg/sub-{p}_task-nr_run-1_events.tsv')
        fixes = [e for e in events if e['type']=='fixation']
        ints = intervals(events)
        ifi=[b['onset']-a['onset'] for iv in ints for a,b in zip([e for e in fixes if iv[0]<=e['onset']<iv[1]], [e for e in fixes if iv[0]<=e['onset']<iv[1]][1:])]
        all_ifi.append((p,ifi))
        summaries.append({'participant':p, 'within_presentation_IFI_ms':stat(np.array(ifi)*1000),
                          'within_presentation_IFI_before_ms':{str(t):sum(v<t/1000-1e-9 for v in ifi) for t in [100,250,450,500]}})
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(p,ifi) in zip(axes,all_ifi):
        ax.hist(np.array(ifi)*1000,bins=np.arange(0,1201,20),color='#146b83');ax.axvline(450,c='orange');ax.axvline(500,c='red',ls='--')
        ax.set(title=f'{p}, pairs={len(ifi)}',xlabel='Within-presentation fixation IFI (ms)')
    axes[0].set_ylabel('Count (values >1200 ms omitted in plot only)');fig.tight_layout();fig.savefig(args.out/'fixation_intervals.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (args.out/'fixation_intervals.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__ == '__main__':
    main()
