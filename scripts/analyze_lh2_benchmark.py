"""Matched sensor-average errors; never compare an average with temporal max."""
import csv,json,re
from pathlib import Path
import numpy as np
from benchmark_lh2 import ROOT,OUT,rows

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);assert np.isfinite(p).all()
    positive=y>0;error=p-y
    threshold={}
    for t in [2,4]:
        actual=y>=t;pred=p>=t
        tp=int(np.sum(actual&pred));fn=int(np.sum(actual&~pred));fp=int(np.sum(~actual&pred));tn=int(np.sum(~actual&~pred))
        threshold[str(t)]={'TP':tp,'FN':fn,'FP':fp,'TN':tn,'recall':tp/(tp+fn) if tp+fn else None}
    return {'n':len(y),'MAE_volpct':float(np.mean(abs(error))),'RMSE_volpct':float(np.sqrt(np.mean(error**2))),
      'bias_volpct':float(np.mean(error)),'mean_observed':float(np.mean(y)),
      'FAC2_positive':float(np.mean((p[positive]>=y[positive]/2)&(p[positive]<=y[positive]*2))),
      'n_positive':int(positive.sum()),'thresholds_volpct':threshold}

def main():
    obs=rows();slab=json.loads((OUT/'slab.json').read_text());records=[];summary=[]
    for t in range(1,8):
        target=[r for r in obs if int(r['test'])==t];y=[float(r['average']) for r in target]
        ps={};timing={}
        for method in ['passive','lh2']:
            folder=OUT/f'test{t}_{method}_20261107'
            pred=list(csv.DictReader((folder/'receptors.csv').open()))
            assert len(pred)==len(y)
            ps[method]=[float(r['vol_pct']) for r in pred]
            timing[method]=json.loads((folder/'execution.json').read_text())['wall_s']
            log=(folder/'run.log').read_text()
            assert 'Number of rogue particles: 0' in log
            assert all(float(r['averaged_s'])==float(target[0]['window_end_s'])-float(target[0]['window_start_s']) for r in pred)
        s=next(r for r in slab if r['test']==t);ps['slab']=s['pred_volpct'];timing['slab']=s['wall_s']
        for i,r in enumerate(target):records.append({**r,**{k+'_volpct':v[i] for k,v in ps.items()}})
        summary.append({'test':t,'methods':{k:{**metrics(y,v),'wall_s':timing[k]} for k,v in ps.items()}})
    global_metrics={k:metrics([float(r['average'])for r in records],[r[k+'_volpct']for r in records]) for k in ['passive','lh2','slab']}
    result={'comparison':'Public code, matched quasi-steady sensor-box time averages; not private selected research model',
      'all_sensors':global_metrics,'per_test':summary,
      'unignited_tests_1_2_3_4_7':{k:metrics([float(r['average'])for r in records if int(r['test'])in[1,2,3,4,7]],
          [r[k+'_volpct']for r in records if int(r['test'])in[1,2,3,4,7]])for k in ['passive','lh2','slab']}}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    with (OUT/'sensor_predictions.csv').open('w',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    print(json.dumps(result,indent=2))
    plot(records,summary)

def plot(records,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors={'passive':'#7f8c8d','lh2':'#007c91','slab':'#b65b20'}
    names={'passive':'QES passive','lh2':'QES + LH2 prototype','slab':'Public SLABx-LH2'}
    y=np.array([float(r['average'])for r in records])
    for k in ['passive','lh2','slab']:
        p=np.array([r[k+'_volpct']for r in records])
        axes[0].scatter(y,p,s=17,alpha=.65,color=colors[k],label=names[k])
        axes[1].plot(range(1,8),[s['methods'][k]['RMSE_volpct']for s in summary],'o-',color=colors[k],label=names[k])
        axes[2].plot(range(1,8),[s['methods'][k]['wall_s']for s in summary],'o-',color=colors[k])
    lim=max(1,float(y.max()),max(r[k+'_volpct']for r in records for k in colors))*1.05
    axes[0].plot([0,lim],[0,lim],'k--',lw=1);axes[0].set(xlabel='Observed mean (vol%)',ylabel='Predicted mean (vol%)',xlim=(0,lim),ylim=(0,lim))
    axes[1].set(xlabel='FFI test',ylabel='RMSE (percentage points)',xticks=range(1,8))
    axes[2].set(xlabel='FFI test',ylabel='Measured wall time (s)',yscale='log',xticks=range(1,8))
    axes[0].legend(fontsize=8);axes[1].grid(alpha=.2);axes[2].grid(alpha=.2)
    fig.suptitle('FFI: 210 sensor averages | Fixed RH=75%, neutral wind | Prototype comparison',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
