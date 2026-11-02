"""Shared file contracts for the generators used by the paper benchmark."""
from pathlib import Path


def model_resources(method, repo):
    repo=Path(repo)
    if method in ('scaled_rmsnorm_empirical_gaussian','flowprep_scaled_rmsnorm_maf'):
        items={
            'training graphs':repo/'data/mp_20/train.pt',
            'score normalization':repo/'data/mp_20/new_score_norm.txt',
            'autoencoder checkpoint':repo/'result/FlowPrep/mp_20/large/h_rmsnorm/model_899.pth',
            'latent normalization':repo/'data/mp_20/norm_flow_feature_h_rmsnorm_ep899.pt',
        }
        if method=='scaled_rmsnorm_empirical_gaussian':
            items['Gaussian density']=repo/'result/final_experiments/latent_sampler_h_rmsnorm/models/empirical_gaussian/model.joblib'
        else: items['MAF checkpoint']=repo/'result/MAF/no_cond/mp_20/h_rmsnorm/0.0003/512/model_best.pt'
        return items
    if method=='symat':
        items={'SyMat checkpoint':repo/'SyMat/result/mp_20/model_299.pth','score normalization':repo/'data/mp_20/new_score_norm.txt'}
        for split in ('train','val','test'):
            cached=repo/f'SyMat/data/mp_20/{split}.pt'
            items[f'SyMat {split} data']=cached if cached.is_file() else repo/f'data/mp_20/{split}.csv'
        return items
    if method=='ftcp':
        root=repo/'third_party/FTCP'
        return {'FTCP checkpoint':root/'runs/mp20_ablation_B_niggli_properties/best.pt',
                'FTCP preprocessing cache':root/'artifacts/mp20_niggli_train_scaler.npz',
                'element descriptors':root/'data/atom_init.json','element list':root/'data/elements.json'}
    if method=='cdvae':
        root=repo/'result/final_experiments/experiment_3/training/cdvae_mp20'
        items={name:root/name for name in ('epoch=144-step=15370.ckpt','hparams.yaml','lattice_scaler.pt','prop_scaler.pt')}
        items['GemNet calibration']=repo/'final_experiments/experiment_3/vendor/cdvae-official/cdvae/pl_modules/gemnet/gemnet-dT.json'
        return items
    if method=='diffcsp':
        root=repo/'final_experiments/experiment_3/vendor/diffcsp-official/pretrained/mp_gen'
        return {name:root/name for name in ('last.ckpt','hparams.yaml')}
    if method=='mattergen_mp20':
        root=repo/'final_experiments/experiment_3/vendor/mattergen-official/checkpoints/mp_20_base'
        return {'MatterGen checkpoint':root/'checkpoints/last.ckpt','MatterGen configuration':root/'config.yaml',
                'GemNet calibration':repo/'final_experiments/experiment_3/vendor/mattergen-official/mattergen/common/gemnet/gemnet-dT.json'}
    raise ValueError(method)


def missing_model_resources(methods, repo):
    return [f'missing external input {method}/{name}: {path}'
            for method in methods for name,path in model_resources(method,repo).items() if not path.is_file()]
