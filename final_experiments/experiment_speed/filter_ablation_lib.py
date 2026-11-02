#!/usr/bin/env python3
"""Paper-grade seed-grouped offline ablations for the universal pre-relax filter."""
from __future__ import annotations

import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import filter_dataset as base

REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
DATA=Path(os.environ.get("DELAC_FILTER_DATA", str(REPO/"result/filter_development")))
FEATURE_CACHE=Path(os.environ.get("DELAC_FILTER_FEATURES", str(REPO/"result/filter_features/pre_relax_features.csv")))
OUTPUT=REPO/'result/final_experiments/experiment_speed/paper_filter_offline_ablation'
SMACT_SOURCE=FEATURE_CACHE.parent/"smact_features.csv"
SEEDS=(0,1,7,21,44,123)
GEOMETRY=('n_atoms','volume_per_atom_A3','min_distance_A','min_lattice_A','max_lattice_A','min_angle_deg','max_angle_deg','lattice_aspect_ratio')
COMPOSITION=('n_elements','density_g_cm3','mean_Z','max_Z','mean_X','range_X','mean_atomic_radius')
FULL=GEOMETRY+COMPOSITION
INPUT_EXPERIMENTS={
 'I0_geometry':(GEOMETRY,False),'I1_composition':(COMPOSITION,False),
 'I2_full':(FULL,False),'I3_full_smact':(FULL,True),
}
TARGET_EXPERIMENTS={
 'T0_stable_novel':'stable_novel','T1_joint_bad':'joint_bad',
 'T2_weighted_joint_bad':'weighted_joint_bad','T3_dual_R3':'dual',
}
MODELS=('ExtraTrees','RandomForest','HistGradientBoosting','LogisticRegression')

def truth(s): return s.astype(str).str.lower().isin(('true','1','yes'))
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)); q.replace(p)
def model(kind,seed):
 if kind=='ExtraTrees': return ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=-1)
 if kind=='RandomForest': return RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=-1)
 if kind=='HistGradientBoosting': return HistGradientBoostingClassifier(max_iter=300,l2_regularization=1,class_weight='balanced',random_state=seed)
 if kind=='LogisticRegression': return make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed))
 raise ValueError(kind)
def matrix(d,cols,med): return d[list(cols)].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(med).fillna(0)
def long_y(d,q): return (pd.to_numeric(d.relax_steps,errors='coerce').ge(q)|~truth(d.relax_converged)).astype(int)
def risk_fit(train,cal,test,cols,kind,target,q,seed):
 med=train[list(cols)].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).median().fillna(0)
 xtr,xca,xte=matrix(train,cols,med),matrix(cal,cols,med),matrix(test,cols,med)
 stable=truth(train.stable); long=long_y(train,q); started=time.perf_counter()
 if target=='dual':
  a,b=model(kind,seed),model(kind,seed+1000); a.fit(xtr,(~stable).astype(int)); b.fit(xtr,long)
  risks=[a.predict_proba(x)[:,1]*b.predict_proba(x)[:,1] for x in (xca,xte)]
 elif target=='stable_novel':
  m=model(kind,seed); m.fit(xtr,(stable&truth(train.novel)).astype(int)); risks=[1-m.predict_proba(x)[:,1] for x in (xca,xte)]
 else:
  bad=(~stable)&long.astype(bool); y=bad.astype(int); weight=None
  if target=='weighted_joint_bad': weight=np.where(bad,1+pd.to_numeric(train.relax_steps,errors='coerce').fillna(300).to_numpy()/300,1.)
  m=model(kind,seed)
  if weight is None: kwargs={}
  elif kind=='LogisticRegression': kwargs={'logisticregression__sample_weight':weight}
  else: kwargs={'sample_weight':weight}
  m.fit(xtr,y,**kwargs); risks=[m.predict_proba(x)[:,1] for x in (xca,xte)]
 return risks[0],risks[1],time.perf_counter()-started
def hard_reject(d):
 return ~(d.volume_per_atom_A3.between(3,80)&d.lattice_aspect_ratio.le(10)&d.min_distance_A.ge(.5)&d.density_g_cm3.between(.1,30))
def threshold(risk,cal,floor,forced=None):
 r=np.asarray(risk,float); valid=cal.label_valid.to_numpy(); stable=truth(cal.stable).to_numpy()&valid; forced=np.zeros(len(cal),bool) if forced is None else np.asarray(forced,bool)
 allowed=int(math.floor((1-floor)*stable.sum()+1e-9)); best=None
 for t in np.r_[np.nextafter(r.max(),np.inf),np.unique(r)[::-1]]:
  reject=forced|(r>=t); lost=int((reject&stable).sum())
  if lost>allowed: continue
  candidate=(int(reject.sum()),float(t),lost)
  if best is None or candidate[0]>best[0]: best=candidate
 if best is None: return None
 return {'value':best[1],'calibration_rejected':best[0],'calibration_stable_rejected':best[2],'calibration_stable':int(stable.sum())}
def metrics(d,reject):
 reject=np.asarray(reject,bool); valid=d.label_valid.to_numpy(); stable=truth(d.stable).to_numpy()&valid; sun=truth(d.SUN).to_numpy()&valid
 wall=pd.to_numeric(d.relax_wall_seconds,errors='coerce').fillna(0).to_numpy(); steps=pd.to_numeric(d.relax_steps,errors='coerce').fillna(0).to_numpy()
 def rec(y): return float(1-(reject&y).sum()/y.sum()) if y.sum() else None
 return {'rows':len(d),'rejected':int(reject.sum()),'rejection_fraction':float(reject.mean()),'stable':int(stable.sum()),'stable_rejected':int((reject&stable).sum()),'stable_recall':rec(stable),
  'SUN':int(sun.sum()),'SUN_rejected':int((reject&sun).sum()),'SUN_recall':rec(sun),'rejected_nonstable_precision':float((reject&valid&~stable).sum()/reject.sum()) if reject.sum() else None,
  'rejected_mean_steps':float(steps[reject].mean()) if reject.any() else None,'rejected_wall_fraction':float(wall[reject].sum()/wall.sum()) if wall.sum() else None,'rejected_wall_seconds':float(wall[reject].sum()),'rejected_unlabeled':int((reject&~valid).sum())}
def summarize(records):
 keys=('rejection_fraction','stable_recall','SUN_recall','rejected_nonstable_precision','rejected_mean_steps','rejected_wall_fraction')
 out={}
 for key in keys:
  x=np.array([r[key] for r in records if r[key] is not None],float)
  out[key]={'mean':float(x.mean()) if len(x) else None,
            'sample_std':float(x.std(ddof=1)) if len(x)>1 else (0. if len(x)==1 else None)}
 out['stable_rejected_total']=sum(r['stable_rejected'] for r in records); out['SUN_rejected_total']=sum(r['SUN_rejected'] for r in records); out['folds']=len(records)
 return out
def fixed_smact(data,output):
 cache=output/'smact_features.csv'
 if cache.exists():
  f=pd.read_csv(cache); return data.drop(columns=['smact_valid'],errors='ignore').merge(f[['dataset_id','smact_valid']],on='dataset_id',validate='one_to_one')
 if SMACT_SOURCE.exists():
  f=pd.read_csv(SMACT_SOURCE)
  if len(f)==len(data) and not f.dataset_id.duplicated().any() and set(f.dataset_id)==set(data.dataset_id):
   f.to_csv(cache,index=False); print('[SMACT] reuse validated fixed cache',SMACT_SOURCE,flush=True)
   return data.drop(columns=['smact_valid'],errors='ignore').merge(f[['dataset_id','smact_valid']],on='dataset_id',validate='one_to_one')
 from evaluation.metrics.structure import is_smact_valid
 records=[]; started=time.perf_counter()
 for i,row in enumerate(data.itertuples(index=False)):
  t=time.perf_counter(); valid=False; error=''
  try: valid=bool(is_smact_valid(Structure.from_str(str(row.cif_str_unrelaxed).replace('\\n','\n'),fmt='cif')))
  except Exception as e: error=f'{type(e).__name__}: {e}'
  records.append({'dataset_id':row.dataset_id,'smact_valid':int(valid),'smact_seconds':time.perf_counter()-t,'smact_error':error})
  if (i+1)%250==0: print('[SMACT]',i+1,'/',len(data),flush=True)
 f=pd.DataFrame(records); f.to_csv(cache,index=False); print('[SMACT DONE]',f.smact_valid.value_counts().to_dict(),'seconds',time.perf_counter()-started,flush=True)
 return data.drop(columns=['smact_valid'],errors='ignore').merge(f[['dataset_id','smact_valid']],on='dataset_id',validate='one_to_one')
