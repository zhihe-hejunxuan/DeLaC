#!/usr/bin/env python3
"""Collect a leakage-free seven-generator dataset for universal filtering."""
from __future__ import annotations

import os, sys
import argparse, hashlib, importlib.util, json, os, subprocess, threading, time, traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from pymatgen.core import Structure

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
WORKER=HERE/'persistent_stage1_worker.py'; SUN=REPO/'script/eval_sun_from_csv_cli.py'
CRYS=Path(os.environ.get("DELAC_PYTHON", sys.executable))
METHODS=('scaled_rmsnorm_empirical_gaussian','diffcsp','mattergen_mp20','symat','ftcp','flowprep_scaled_rmsnorm_maf','cdvae')
PYTHONS={
 'scaled_rmsnorm_empirical_gaussian':CRYS,'flowprep_scaled_rmsnorm_maf':CRYS,'ftcp':CRYS,
 'symat':Path(os.environ.get("SYMAT_PYTHON", sys.executable)),
 'cdvae':Path(os.environ.get("CDVAE_PYTHON", sys.executable)),
 'diffcsp':Path(os.environ.get("DIFFCSP_PYTHON", sys.executable)),
 'mattergen_mp20':Path(os.environ.get("MATTERGEN_PYTHON", sys.executable))}
PLAN={'train':{0:300,1:300,7:300},'calibration':{21:100,44:100},'development_test':{123:150}}
DEFAULT_ROOT=REPO/'result/final_experiments/experiment_speed/universal_filter_dataset'

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,indent=2,ensure_ascii=False)); q.replace(p)
def truth(s): return s.fillna(False).astype(str).str.lower().isin({'true','1','yes'})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def load_bench():
 spec=importlib.util.spec_from_file_location('bench',HERE/'relaxation_workers.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
BENCH=load_bench()

def relax_args(gpu):
 return SimpleNamespace(dry_run=False,gpu=gpu,fmax=.05,max_steps=300,crystal_timeout_seconds=3600,serial_retry_timeout_seconds=600,memory_limit_mib=12288,memory_safety_mib=1024,monitor_interval=1.,omp_threads=1,dp_intra_threads=1,dp_inter_threads=1,sharding_strategy='round_robin',neighbor_cutoff=6.,save_interval=1,configuration_name='ordinary_workers_2')

def run_sun(csv,out):
 out.mkdir(parents=True,exist_ok=True); metrics=out/'sun.json'; labels=out/'sun_labels.csv'; log=out/'sun.log'
 t=time.perf_counter()
 with log.open('a') as h: subprocess.run([str(CRYS),'-u',str(SUN),'--csv',str(csv),'--out_json',str(metrics),'--labels_csv',str(labels),'--etot_name','E_tot_pred_dpa_raw'],cwd=REPO,stdout=h,stderr=subprocess.STDOUT,check=True)
 return {'metrics':str(metrics),'labels':str(labels),'sun_count':int(truth(pd.read_csv(labels).SUN).sum()),'seconds':time.perf_counter()-t}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--gpu',default='0'); ap.add_argument('--output-root',type=Path,default=DEFAULT_ROOT); a=ap.parse_args()
 run=a.output_root.resolve()/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')); run.mkdir(parents=True,exist_ok=False)
 protocol={'batch_size':32,'methods':list(METHODS),'plan':{s:{str(k):v for k,v in seeds.items()} for s,seeds in PLAN.items()},'formal_seeds_excluded':[13,37,42,73],'stage_order':['all_stage1','merged_two_worker_stage2','split_sun'],'stage2':{'workers':2,'sharding':'round_robin','model':'DPA-3.1-MPtrj/DeePMD','optimizer':'FIRE+FrechetCellFilter','fmax':.05,'max_steps':300,'timeout_seconds':3600,'serial_retry_seconds':600,'memory_limit_mib':12288}}
 state={'status':'running','phase':'stage1','started_at':now(),'run_root':str(run),'protocol':protocol,'stage1':[]}; atomic(state,run/'pipeline_status.json'); atomic(protocol,run/'protocol.json')
 stop=threading.Event()
 def heartbeat():
  while not stop.wait(30):
   with (run/'controller_events.jsonl').open('a') as h: h.write(json.dumps({'time':now(),'event':'heartbeat','phase':state['phase']})+'\n')
 th=threading.Thread(target=heartbeat,daemon=True); th.start()
 try:
  frames=[]; failures=[]; global_id=0
  for split,seeds in PLAN.items():
   for seed,count in seeds.items():
    for method in METHODS:
     directory=run/'stage1'/split/f'seed_{seed}'/method; log=directory.parent/f'{method}.log'; directory.parent.mkdir(parents=True,exist_ok=True)
     record={'split':split,'seed':seed,'method':method,'requested':count,'started_at':now()}
     try:
      cmd=[str(PYTHONS[method]),'-u',str(WORKER),'--method',method,'--seed',str(seed),'--target',str(count),'--output-dir',str(directory),'--raw-only']
      with log.open('a') as h: cp=subprocess.run(cmd,cwd=REPO,stdout=h,stderr=subprocess.STDOUT,check=False)
      if cp.returncode: raise RuntimeError(f'stage1 exit={cp.returncode}; see {log}')
      summary=json.load(open(directory/'stage1_summary.json')); frame=pd.read_csv(summary['raw_csv'])
      frame['source_split']=split; frame['source_seed']=seed; frame['source_model']=method; frame['source_request_id']=frame.request_id
      frame['dataset_id']=[f'{split}:{seed}:{method}:{int(x)}' for x in frame.request_id]
      frame['request_id']=range(global_id,global_id+len(frame)); global_id+=len(frame)
      frames.append(frame); record.update({'status':'completed','returned':len(frame),'summary':summary,'finished_at':now()})
     except Exception as e:
      record.update({'status':'failed','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'finished_at':now()}); failures.append(record)
     state['stage1'].append(record); atomic(state,run/'pipeline_status.json')
  if failures: raise RuntimeError(f'{len(failures)} stage1 jobs failed; stage2 not started')
  state['phase']='merge_stage1'; atomic(state,run/'pipeline_status.json')
  merged=pd.concat(frames,ignore_index=True); parsed=[]
  for _,row in merged.iterrows():
   try: s=Structure.from_str(str(row.cif_str).replace('\\n','\n'),fmt='cif'); parsed.append(len(s))
   except Exception: parsed.append(None)
  merged['benchmark_num_atoms']=parsed; merged['parseable']=merged.benchmark_num_atoms.notna()
  all_path=run/'dataset'/'all_stage1_candidates.csv'; relax_path=run/'dataset'/'parseable_for_relaxation.csv'; all_path.parent.mkdir(parents=True,exist_ok=True)
  merged.to_csv(all_path,index=False); relaxable=merged.loc[merged.parseable].copy(); relaxable.to_csv(relax_path,index=False)
  hashes=relaxable.cif_str.fillna('').map(lambda x:hashlib.sha256(str(x).encode()).hexdigest()); relaxable['exact_cif_sha256']=hashes
  cross=relaxable.groupby('exact_cif_sha256').source_split.nunique(); leakage=int(cross.gt(1).sum())
  state['merge']={'requested_total':sum(v*len(METHODS) for seeds in PLAN.values() for v in seeds.values()),'returned_total':len(merged),'parseable':len(relaxable),'parse_failures':len(merged)-len(relaxable),'exact_hashes_shared_across_splits':leakage,'all_candidates':str(all_path),'relaxation_input':str(relax_path),'sha256':sha(relax_path)}
  state['phase']='stage2'; atomic(state,run/'pipeline_status.json')
  relaxation=BENCH.run_configuration(relax_args(a.gpu),relax_path,2,run/'stage2')
  state['stage2']=relaxation; state['phase']='split_sun'; atomic(state,run/'pipeline_status.json')
  relaxed=pd.read_csv(relaxation['merged_output']); evaluations={}
  for split in PLAN:
   subset=relaxed.loc[relaxed.source_split.eq(split)].copy().reset_index(drop=True); subset['request_id']=range(len(subset)); p=run/'evaluation'/split/'relaxed.csv'; p.parent.mkdir(parents=True,exist_ok=True); subset.to_csv(p,index=False)
   evaluations[split]={'rows':len(subset),'sun':run_sun(p,p.parent)}; state['evaluations']=evaluations; atomic(state,run/'pipeline_status.json')
  state.update({'status':'completed','phase':'finished','finished_at':now(),'evaluations':evaluations}); atomic(state,run/'pipeline_status.json')
 except BaseException as e:
  state.update({'status':'failed','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'finished_at':now()}); atomic(state,run/'pipeline_status.json'); raise
 finally: stop.set(); th.join(timeout=2)

if __name__=='__main__': main()
