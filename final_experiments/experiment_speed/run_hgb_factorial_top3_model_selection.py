#!/usr/bin/env python3
"""HGB 4x4 factorial -> frozen top-3 selection -> classifier confirmation."""
from __future__ import annotations

import argparse, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import filter_ablation_lib as lib
import filter_dataset as base

REPO=Path(__file__).resolve().parents[2]
DATA=lib.DATA; FEATURE_CACHE=lib.FEATURE_CACHE
OUTPUT=REPO/'result/final_experiments/experiment_speed/hgb_factorial_top3_model_selection'
INPUTS={
 'I0_geometry':lib.GEOMETRY,
 'I1_composition':lib.COMPOSITION,
 'I2_full':lib.FULL,
 'I3_full_smact':lib.FULL+('smact_valid',),
}
TARGETS=lib.TARGET_EXPERIMENTS
CLASSIFIERS=lib.MODELS

def atomic(value,path): lib.atomic(value,path)
def load_data(run):
 feat=pd.read_csv(FEATURE_CACHE,low_memory=False); parts=[]
 for split in ('train','calibration','development_test'):
  d=base.load_split(DATA,split)
  f=feat.loc[feat.source_split.eq(split),['dataset_id',*lib.FULL,'smact_valid']]
  parts.append(d.merge(f,on='dataset_id',validate='one_to_one'))
 return lib.fixed_smact(pd.concat(parts,ignore_index=True),run)
def select_top3(summary):
 eligible=[]
 for name,r in summary.items():
  stable=r['stable_recall']['mean']; sun=r['SUN_recall']['mean']; precision=r['rejected_nonstable_precision']['mean']
  if stable>=.985 and sun>=.975 and precision>=.90:
   eligible.append((name,r))
 # Greedy ranking with the frozen 0.5 percentage-point practical-equivalence band.
 # Within that band, quality protection takes priority over a negligible cost delta.
 ranked=[]; remaining=list(eligible)
 while remaining:
  best_cost=max(x[1]['rejected_wall_fraction']['mean'] for x in remaining)
  near=[x for x in remaining if x[1]['rejected_wall_fraction']['mean']>=best_cost-.005]
  near.sort(key=lambda x:(-x[1]['SUN_recall']['mean'],-x[1]['stable_recall']['mean'],
                          len(INPUTS[x[0].split('__')[0]])))
  chosen=near[0]; ranked.append(chosen); remaining.remove(chosen)
 eligible=ranked
 if len(eligible)<3: raise RuntimeError(f'only {len(eligible)} factorial groups meet frozen quality floors')
 return [name for name,_ in eligible[:3]], [name for name,_ in eligible]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-root',type=Path,default=OUTPUT); ap.add_argument('--stable-recall',type=float,default=.99); a=ap.parse_args()
 run=a.output_root/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')); run.mkdir(parents=True)
 data=load_data(run); assert set(data.source_seed.unique())==set(lib.SEEDS); valid=data.label_valid
 phase1=[]; phase1_predictions=[]; fold_cache={}; fold_meta={}
 for pos,test_seed in enumerate(lib.SEEDS):
  cal_seed=lib.SEEDS[(pos+1)%len(lib.SEEDS)]
  train=data.loc[valid&~data.source_seed.isin((test_seed,cal_seed))].copy(); cal=data.loc[data.source_seed.eq(cal_seed)&data.label_valid].copy(); test=data.loc[data.source_seed.eq(test_seed)].copy()
  q=float(pd.to_numeric(train.relax_steps,errors='coerce').quantile(.75)); fold_meta[str(test_seed)]={'calibration_seed':cal_seed,'train_seeds':[s for s in lib.SEEDS if s not in (test_seed,cal_seed)],'train_q75_steps':q}
  for input_name,cols in INPUTS.items():
   for target_name,target in TARGETS.items():
    name=f'{input_name}__{target_name}'; rc,rt,seconds=lib.risk_fit(train,cal,test,cols,'HistGradientBoosting',target,q,2026+pos); th=lib.threshold(rc,cal,a.stable_recall)
    if th is None: raise RuntimeError(f'no threshold: {test_seed} {name}')
    rejected=rt>=th['value']; record={'experiment':name,'classifier':'HistGradientBoosting','test_seed':test_seed,**fold_meta[str(test_seed)],'threshold':th,'fit_seconds':seconds,**lib.metrics(test,rejected)}; phase1.append(record)
    fold_cache[(test_seed,name)]=(rc,rt,th,rejected)
    for idx,risk_value,decision in zip(test.index,rt,rejected): phase1_predictions.append({'stage':'factorial','experiment':name,'classifier':'HistGradientBoosting','test_seed':test_seed,'dataset_id':test.at[idx,'dataset_id'],'source_model':test.at[idx,'source_model'],'risk':risk_value,'rejected':bool(decision)})
  print('[PHASE1 FOLD]',test_seed,'done',flush=True)
 summary={name:lib.summarize([x for x in phase1 if x['experiment']==name]) for name in sorted({x['experiment'] for x in phase1})}
 top3,eligible=select_top3(summary)
 selection={'quality_floors':{'stable_recall_mean':.985,'SUN_recall_mean':.975,'rejected_nonstable_precision_mean':.90},'ranking':['rejected_wall_fraction descending','SUN_recall descending if near','stable_recall descending if near','simpler input'],'eligible_ranked':eligible,'selected_top3':top3}
 atomic({'summary':summary,'selection':selection},run/'phase1_factorial_results.json'); print('[TOP3]',top3,flush=True)
 phase3=[]; phase3_predictions=[]
 for pos,test_seed in enumerate(lib.SEEDS):
  cal_seed=lib.SEEDS[(pos+1)%len(lib.SEEDS)]; train=data.loc[valid&~data.source_seed.isin((test_seed,cal_seed))].copy(); cal=data.loc[data.source_seed.eq(cal_seed)&data.label_valid].copy(); test=data.loc[data.source_seed.eq(test_seed)].copy(); q=fold_meta[str(test_seed)]['train_q75_steps']
  for name in top3:
   input_name,target_name=name.split('__'); cols=INPUTS[input_name]; target=TARGETS[target_name]
   for classifier in CLASSIFIERS:
    if classifier=='HistGradientBoosting':
     rc,rt,th,rejected=fold_cache[(test_seed,name)]; seconds=next(x['fit_seconds'] for x in phase1 if x['test_seed']==test_seed and x['experiment']==name)
    else:
     rc,rt,seconds=lib.risk_fit(train,cal,test,cols,classifier,target,q,2026+pos); th=lib.threshold(rc,cal,a.stable_recall)
     if th is None: raise RuntimeError(f'no threshold: {test_seed} {name} {classifier}')
     rejected=rt>=th['value']
    phase3.append({'experiment':name,'classifier':classifier,'test_seed':test_seed,**fold_meta[str(test_seed)],'threshold':th,'fit_seconds':seconds,**lib.metrics(test,rejected)})
    for idx,risk_value,decision in zip(test.index,rt,rejected): phase3_predictions.append({'stage':'classifier_selection','experiment':name,'classifier':classifier,'test_seed':test_seed,'dataset_id':test.at[idx,'dataset_id'],'source_model':test.at[idx,'source_model'],'risk':risk_value,'rejected':bool(decision)})
  print('[PHASE3 FOLD]',test_seed,'done',flush=True)
 model_summary={name:{classifier:lib.summarize([x for x in phase3 if x['experiment']==name and x['classifier']==classifier]) for classifier in CLASSIFIERS} for name in top3}
 protocol={'status':'completed','created_at':datetime.now().astimezone().isoformat(timespec='seconds'),'stages_completed':[1,2,3],'formal_seeds_excluded':[13,37,42,73],'seeds':lib.SEEDS,'outer_cv':'leave-one-seed-out','inner_calibration':'cyclic next seed','calibration_stable_recall_floor':a.stable_recall,'factorial_classifier':'HistGradientBoosting','inputs':{k:list(v) for k,v in INPUTS.items()},'targets':TARGETS,'top3_selection':selection,'classifiers':CLASSIFIERS,'fold_metadata':fold_meta}
 pd.DataFrame(phase1).to_csv(run/'phase1_factorial_fold_metrics.csv',index=False); pd.DataFrame(phase3).to_csv(run/'phase3_classifier_fold_metrics.csv',index=False); pd.DataFrame(phase1_predictions+phase3_predictions).to_csv(run/'predictions.csv',index=False)
 atomic({'protocol':protocol,'factorial_summary':summary,'selection':selection,'classifier_summary':model_summary,'factorial_fold_results':phase1,'classifier_fold_results':phase3},run/'results.json'); atomic(protocol,run/'protocol.json'); print('[DONE]',run,flush=True)
if __name__=='__main__': main()
