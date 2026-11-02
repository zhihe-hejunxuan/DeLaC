#!/usr/bin/env python3
"""Persistent, timed stage-1 generation with paired raw/filter outputs."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

PROCESS_START = time.perf_counter()
REPO = Path(__file__).resolve().parents[2]
BATCH = 32

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def structures_from_flat(frac, atom, lengths, angles, counts):
    def cpu(x): return x.detach().cpu() if hasattr(x, "detach") else np.asarray(x)
    frac, atom, lengths, angles, counts = map(cpu, (frac, atom, lengths, angles, counts))
    def strip_sample_axis(x):
        while getattr(x,"ndim",0)>1 and x.shape[0]==1: x=x[0]
        return x
    frac, atom, lengths, angles, counts = map(
        strip_sample_axis, (frac, atom, lengths, angles, counts)
    )
    out=[]; start=0
    for i,n in enumerate(counts.tolist()):
        try:
            n=int(n); species=atom[start:start+n]
            if getattr(species,"ndim",1)==2: species=species.argmax(-1)+1
            out.append(Structure(Lattice.from_parameters(*lengths[i].tolist(),*angles[i].tolist()),
                                 species.numpy(),frac[start:start+n].numpy(),coords_are_cartesian=False))
        except Exception: out.append(None)
        start += max(int(n),0)
    return out

def load_sampler(method, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    if method in {"scaled_rmsnorm_empirical_gaussian","flowprep_scaled_rmsnorm_maf"}:
        sys.path.insert(0,str(REPO)); from config.FlowPrep.mp_20_conifg_dict_large_h_rmsnorm import conf
        from dataset import MatDataset; from script.runner_for_flow_prep import RunnerFlowPrep
        r=RunnerFlowPrep(conf,str(REPO/'data/mp_20/new_score_norm.txt'),dataset_name='mp_20')
        ds=MatDataset(str(REPO/'data/mp_20/train.pt'),**conf['data']); r.model.lattice_normalizer=r._get_normalizer(ds)
        r.load_model(str(REPO/'result/FlowPrep/mp_20/large/h_rmsnorm/model_899.pth')); model=r.model.cuda().eval(); model.coordgen.pool=None
        norm=torch.load(REPO/'data/mp_20/norm_flow_feature_h_rmsnorm_ep899.pt',map_location='cpu'); mean,std=norm['mean'].cuda(),norm['std'].cuda()
        rng=np.random.default_rng(seed)
        if method=="scaled_rmsnorm_empirical_gaussian":
            import joblib; g=joblib.load(REPO/'result/final_experiments/latent_sampler_h_rmsnorm/models/empirical_gaussian/model.joblib')
            def latent(n): return torch.from_numpy(rng.multivariate_normal(g['mean'],g['covariance'],size=n)).float().cuda()*std+mean
        else:
            from MAF_script.train_MAF_no_cond import build_maf_no_cond
            flow=build_maf_no_cond(mean.numel(),512,10,2,False,True,True,'relu',0.,False,True).cuda()
            flow.load_state_dict(torch.load(REPO/'result/MAF/no_cond/mp_20/h_rmsnorm/0.0003/512/model_best.pt',map_location='cuda')); flow.eval()
            def latent(n): return flow.sample(n).squeeze(1)*std+mean
        def sample(n):
            c,l,a,ang,f=model.predict_coords(latent(n),conf['model']['noise_start'],conf['model']['noise_end'],conf['model']['num_time_steps'])
            return f,l,a,ang,c
        return sample, lambda x: structures_from_flat(*x)
    if method=="symat":
        root=REPO/'SyMat'; sys.path[:0]=[str(root)]; from config.mp_20_config_dict import conf
        from dataset import MatDataset; from runner import Runner; from paths import resolve_split_paths
        p=resolve_split_paths('mp_20'); r=Runner(conf,p['score_norm_path']); r.model.load_state_dict(torch.load(root/'result/mp_20/model_299.pth',map_location='cpu'))
        ds=MatDataset(p['train_path'],pt_path=p['train_pt_path'],**conf['data']); r.model.lattice_normalizer=r._get_normalizer(ds); model=r.model.cuda().eval()
        def sample(n):
            c,a,l,ang,f=model.generate(n,conf['gen_temp'],conf['model']['noise_start'],conf['model']['noise_end'],conf['model']['num_time_steps'],100,1e-4)
            return f,a,l,ang,c
        return sample, lambda x: structures_from_flat(*x)
    if method=="ftcp":
        root=REPO/'third_party/FTCP'; sys.path.insert(0,str(root))
        from ftcp_recon.experiment import load_cache; from ftcp_recon.model import FTCPVAE
        from ftcp_recon.representation import FTCPCodec,FTCPConfig
        cache=load_cache(root/'artifacts/mp20_niggli_train_scaler.npz'); ck=torch.load(root/'runs/mp20_ablation_B_niggli_properties/best.pt',map_location='cuda',weights_only=False)
        model=FTCPVAE(tuple(ck['shape']),ck['latent_dim'],ck.get('property_dim',0)).cuda().eval(); model.load_state_dict(ck['model'])
        meta=json.loads(str(cache['metadata'])); codec=FTCPCodec(FTCPConfig(**meta['config']),root/'data')
        train=cache['train'].astype(np.int64); cols=cache['property_columns'].astype(str).tolist(); phy=cache['properties']*cache['property_scale']+cache['property_min']
        cand=train[phy[train,cols.index('formation_energy_per_atom')]<0]; order=np.argsort(np.abs(phy[cand,cols.index('band_gap')]-1.5)); idx=cand[order[:10]]
        with torch.inference_mode(): mu,_=model.encode(torch.from_numpy(cache['matrices'][idx]).float().cuda())
        rng=np.random.default_rng(seed); counter=[0]
        def sample(n):
            slots=np.arange(counter[0],counter[0]+n)%10; counter[0]+=n
            z=mu[torch.as_tensor(slots,device='cuda')]+torch.from_numpy(rng.normal(0,.6,size=(n,mu.shape[1]))).float().cuda()
            dec=model.decode(z).detach().cpu().numpy(); return dec*cache['row_scale'][0]+cache['row_min'][0]
        return sample, lambda raw: [codec.decode(x,.5) for x in raw]
    if method=="cdvae":
        root=REPO/'final_experiments/experiment_3/vendor/cdvae-official'; sys.path[:0]=[str(root/'scripts'),str(root)]
        os.environ['PROJECT_ROOT']=str(root)
        # The locally trained CDVAE checkpoint is trusted input.  PyTorch
        # 2.6 changed torch.load's default to weights_only=True, while this
        # Lightning checkpoint legitimately contains pathlib.PosixPath in its
        # Hydra hyperparameters.  Lightning does not expose weights_only here,
        # so use PyTorch's documented compatibility override for this process.
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        from evaluate import generation; from eval_utils import load_model; from types import SimpleNamespace
        model,_,_=load_model(REPO/'result/final_experiments/experiment_3/training/cdvae_mp20',load_data=False); model=model.cuda().eval()
        ld=SimpleNamespace(n_step_each=100,step_lr=1e-4,min_sigma=0,save_traj=False,disable_bar=True)
        def sample(n):
            f,c,a,l,ang,*_=generation(model,ld,1,1,n,10,n); return f,a,l,ang,c
        return sample, lambda x: structures_from_flat(*x)
    if method=="diffcsp":
        root=REPO/'final_experiments/experiment_3/vendor/diffcsp-official'; sys.path[:0]=[str(root/'scripts'),str(root)]; os.environ['PROJECT_ROOT']=str(root)
        from generation import SampleDataset,diffusion,recommand_step_lr; from eval_utils import load_model
        from torch_geometric.data import DataLoader
        model,_,_=load_model(root/'pretrained/mp_gen',load_data=False); model=model.cuda().eval(); lr=recommand_step_lr['gen']['mp_20']
        def sample(n):
            f,a,_,l,ang,c=diffusion(DataLoader(SampleDataset('mp_20',n),batch_size=n),model,lr); return f,a,l,ang,c
        return sample, lambda x: structures_from_flat(*x)
    if method=="mattergen_mp20":
        root=REPO/'final_experiments/experiment_3/vendor/mattergen-official'; sys.path.insert(0,str(root))
        from hydra.utils import instantiate
        from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
        from mattergen.generator import CrystalGenerator,structure_from_model_output
        from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
        g=CrystalGenerator(checkpoint_info=MatterGenCheckpointInfo(model_path=(root/'checkpoints/mp_20_base').resolve(),load_epoch='last'),batch_size=BATCH,num_batches=1,record_trajectories=False,num_atoms_distribution='MP_20'); g.prepare(); g.model.eval()
        def sample(n):
            cfg=g.load_sampling_config(batch_size=n,num_batches=1,num_samples=n); loader=g.get_condition_loader(cfg,None); sampler=instantiate(cfg.sampler_partial)(pl_module=g.model); out=[]
            for cond,mask in loader:
                _,x=sampler.sample(cond,mask); out.append(x)
            return out
        def convert(graphs):
            out=[]
            for x in graphs:
                l,ang=lattice_matrix_to_params_torch(x.cell)
                out.extend(structure_from_model_output(x.pos,x.atomic_numbers,l,ang,x.num_atoms))
            return out
        return sample, convert
    raise ValueError(method)

def evaluate(s):
    if s is None: return False,"parse_error",{}
    n=len(s); abc=np.asarray(s.lattice.abc); dm=np.asarray(s.distance_matrix); md=float(dm[~np.eye(n,dtype=bool)].min()) if n>1 else math.inf
    v=s.volume/n; ar=abc.max()/abc.min(); den=float(s.density); reasons=[]
    if not 3<=v<=80: reasons.append('volume_per_atom')
    if ar>10: reasons.append('lattice_aspect_ratio')
    if np.isfinite(md) and md<.5: reasons.append('min_distance')
    if not .1<=den<=30: reasons.append('density')
    return not reasons,';'.join(reasons),{'n_atoms':n,'volume_per_atom_A3':v,'lattice_aspect_ratio':ar,'min_distance_A':md,'density_g_cm3':den}

def rows(structs,start=0,batch=-1, apply_filter=True):
    out=[]
    for i,s in enumerate(structs,start):
        ok,reason,feat=evaluate(s) if apply_filter else (s is not None,"",{})
        out.append({'candidate_id':i,'generation_batch':batch,'generation_success':s is not None,'cif_str':s.to(fmt='cif') if s else None,'filter_accepted':ok,'rejection_reasons':reason,**feat})
    return out

def write(frame,path): path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--target',type=int,default=1000); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--raw-only',action='store_true'); a=ap.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    setup=time.perf_counter()-PROCESS_START; load0=time.perf_counter(); sample,convert=load_sampler(a.method,a.seed); sync(); load=time.perf_counter()-load0
    allrows=[]; infer=[]; conversion=[]; filt=[]; batch_records=[]; batch_id=0
    remain=a.target
    while remain:
        n=min(BATCH,remain); sync(); t=time.perf_counter(); native=sample(n); sync(); ti=time.perf_counter()-t
        t=time.perf_counter(); ss=convert(native); tc=time.perf_counter()-t
        t=time.perf_counter(); rr=rows(ss,len(allrows),batch_id,False); tr=time.perf_counter()-t
        infer.append(ti); conversion.append(tc+tr); allrows+=rr
        batch_records.append({'batch':batch_id,'phase':'initial','requested':n,'returned':len(ss),'gpu_inference_seconds':ti,'native_to_cif_seconds':tc+tr,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else 0})
        remain-=n; batch_id+=1
    initial=pd.DataFrame(allrows); w=time.perf_counter(); raw=initial.copy(); raw.insert(0,'request_id',range(len(raw))); write(raw,out/'raw_initial_1000.csv'); raw_write=time.perf_counter()-w; raw_done=time.perf_counter()-PROCESS_START
    if a.raw_only:
        result={'status':'completed','mode':'raw_only','method':a.method,'seed':a.seed,'batch_size':BATCH,'initial_requests':a.target,'generated_total':len(raw),'timing_seconds':{'process_setup':setup,'model_load':load,'initial_inference':sum(infer),'initial_native_to_cif':sum(conversion),'raw_write':raw_write,'cold_stage1_raw':raw_done},'batch_records':batch_records,'raw_csv':str(out/'raw_initial_1000.csv')}
        (out/'stage1_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True); return
    t=time.perf_counter()
    # Filtering starts only after the raw-arm endpoint has been recorded.
    filter_rows=[]
    for r in allrows:
        try: s=Structure.from_str(str(r['cif_str']),fmt='cif') if r['cif_str'] else None
        except Exception: s=None
        ok,reason,feat=evaluate(s); r.update({'filter_accepted':ok,'rejection_reasons':reason,**feat}); filter_rows.append(r)
    filt.append(time.perf_counter()-t)
    accepted=[r for r in allrows if r['filter_accepted']]; refill_infer=[]; refill_filter=[]
    while len(accepted)<a.target:
        sync(); t=time.perf_counter(); native=sample(BATCH); sync(); ti=time.perf_counter()-t
        t=time.perf_counter(); ss=convert(native); tc=time.perf_counter()-t
        t=time.perf_counter(); rr=rows(ss,len(allrows),batch_id,True); tf=time.perf_counter()-t
        refill_infer.append(ti); refill_filter.append(tf); conversion.append(tc)
        batch_records.append({'batch':batch_id,'phase':'refill','requested':BATCH,'returned':len(ss),'gpu_inference_seconds':ti,'native_to_cif_seconds':tc,'filter_seconds':tf,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else 0})
        allrows+=rr; accepted += [r for r in rr if r['filter_accepted']]; batch_id+=1
        if len(allrows)>3000: raise RuntimeError('failed to accept 1000 within 3000 candidates')
    selected=pd.DataFrame(accepted[:a.target]); selected.insert(0,'request_id',range(len(selected))); w=time.perf_counter(); write(pd.DataFrame(allrows),out/'all_candidates.csv'); write(selected,out/'filtered_accepted_1000.csv'); write(pd.DataFrame([r for r in allrows if not r['filter_accepted']]),out/'rejected.csv'); write(pd.DataFrame(accepted[a.target:]),out/'accepted_overflow.csv'); filtered_write=time.perf_counter()-w; total=time.perf_counter()-PROCESS_START
    initial_convert=sum(conversion[:len(infer)]); refill_convert=sum(conversion[len(infer):])
    logical_filtered=setup+load+sum(infer)+initial_convert+sum(filt)+sum(refill_infer)+refill_convert+sum(refill_filter)+filtered_write
    result={'status':'completed','method':a.method,'seed':a.seed,'batch_size':BATCH,'initial_requests':a.target,'generated_total':len(allrows),'rejected_total':sum(not r['filter_accepted'] for r in allrows),'overflow':len(accepted[a.target:]),'timing_seconds':{'process_setup':setup,'model_load':load,'initial_inference':sum(infer),'initial_native_to_cif':initial_convert,'initial_filter':sum(filt),'refill_inference':sum(refill_infer),'refill_native_to_cif':refill_convert,'refill_filter':sum(refill_filter),'inference_total_raw':sum(infer),'inference_total_filtered':sum(infer)+sum(refill_infer),'native_to_cif_total':sum(conversion),'raw_write':raw_write,'filtered_write':filtered_write,'cold_stage1_raw':raw_done,'cold_stage1_filtered_logical':logical_filtered,'physical_combined_wall':total},'logical_timing_note':'filtered logical time excludes raw-only CSV write; physical_combined_wall is actual shared-process wall time','batch_records':batch_records,'raw_csv':str(out/'raw_initial_1000.csv'),'filtered_csv':str(out/'filtered_accepted_1000.csv')}
    (out/'stage1_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
