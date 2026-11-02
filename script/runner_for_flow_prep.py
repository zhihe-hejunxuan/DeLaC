"""Training and decoding entry point for the four paper configurations."""
from pathlib import Path
import os, gc, torch, argparse, sys, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Lattice, Structure, Element
from pymatgen.io.cif import CifWriter
from torch_scatter import scatter
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)
from model.flow_prep import FlowPrep
from model.flow_prep_h_norm import FlowPrepHNorm as FlowPrepHNormLayerNorm
from model.flow_prep_h_rmsnorm import FlowPrepHNorm
from dataset import MatDataset
from utils import StandardScalerTorch
from script.tools import build_structures_for_m3gnet
args = argparse.Namespace(result_dir="result/FlowPrep/", dataset="mp_20", model_scale="large")

def load_config(suffix):
    import importlib
    if suffix not in ("h_rmsnorm", "rmsnorm", "flow_prep_none", "flow_prep_zscore"):
        raise ValueError(suffix)
    return importlib.import_module("config.FlowPrep.mp_20_conifg_dict_large_" + suffix).conf

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunnerFlowPrep:


    def __init__(self, conf, score_norm_path=None, matgen_ckpt=None, device="cuda", dataset_name=None):
        self.conf = conf
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self._dataset_name = dataset_name or "mp_20"
        score_norm = np.loadtxt(score_norm_path) if score_norm_path and os.path.isfile(score_norm_path) else None
        model_conf = dict(conf["model"])
        use_rms = model_conf.pop("use_flow_prep_h_rmsnorm", False)
        use_ln = model_conf.pop("use_flow_prep_h_layernorm", False)
        model_conf.pop("use_flow_prep_plain", None)
        for key in list(model_conf):
            if key.startswith("use_flow_prep_pro"):
                model_conf.pop(key)
        if use_rms:
            self.model = FlowPrepHNorm(**model_conf, h_pre_norm_flag=True, score_norm=score_norm, device=device)
        elif use_ln:
            self.model = FlowPrepHNormLayerNorm(**model_conf, h_pre_norm_flag=True, score_norm=score_norm, device=device)
        else:
            self.model = FlowPrep(**model_conf, score_norm=score_norm, device=device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **conf['optim'])
        self.verbose = conf.get("verbose", 100)


    def _loss_from_ae_dict(self, loss_dict):
        elem_type_num_loss = loss_dict["elem_type_num_loss"]
        elem_type_loss = loss_dict["elem_type_loss"]
        elem_num_loss = loss_dict["elem_num_loss"]
        lattice_loss = loss_dict["lattice_loss"]
        coord_loss = loss_dict["coord_loss"]
        loss = (
            self.conf["elem_type_num_weight"] * elem_type_num_loss
            + self.conf["elem_type_weight"] * elem_type_loss
            + self.conf["elem_num_weight"] * elem_num_loss
            + self.conf["lattice_weight"] * lattice_loss
            + self.conf["coord_weight"] * coord_loss
        )
        comp_dense = loss_dict.get("composition_dense_loss")
        if comp_dense is not None:
            w = self.conf.get("composition_dense_weight", 0.0)
            if w != 0.0:
                loss = loss + w * comp_dense
        aux_loss = loss_dict.get("aux_loss")
        if aux_loss is not None:
            loss = loss + aux_loss
        return loss


    def _get_normalizer(self, dataset):
        normalizer = StandardScalerTorch()
        lengths = torch.tensor([data["graph_arrays"][2] for data in dataset.data_dict_list])
        angles  = torch.tensor([data["graph_arrays"][4] for data in dataset.data_dict_list])
        length_angles = torch.cat((lengths, angles), dim=-1)
        normalizer.fit(length_angles)
        # 放到 device
        normalizer.means, normalizer.stds = normalizer.means.to(self.device), normalizer.stds.to(self.device)
        return normalizer


    def _cleanup(self):
        # 关闭可能存在的进程池等资源（CoordGen）
        try:
            if hasattr(self.model, "coordgen") and hasattr(self.model.coordgen, "pool") and self.model.coordgen.pool is not None:
                self.model.coordgen.pool.close()
                self.model.coordgen.pool.join()
                self.model.coordgen.pool = None
        except Exception as e:
            print(f"[CLEANUP] pool close/join error: {e}")
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_elem_type_num_loss, total_elem_type_loss, total_elem_num_loss, total_lattice_loss, total_coord_loss = 0, 0, 0, 0, 0
        total_loss = 0

        for iter_num, data_batch in enumerate(loader):
            data_batch = data_batch.to('cuda')
            try_num = 0
            while try_num < 3:
                try:
                    self.optimizer.zero_grad()
                    loss_dict = self.model.forward_ae(data_batch, global_step=self.global_step)
                    elem_type_num_loss, elem_type_loss, elem_num_loss, lattice_loss, coord_loss = loss_dict['elem_type_num_loss'], \
                        loss_dict['elem_type_loss'], loss_dict['elem_num_loss'], loss_dict['lattice_loss'], loss_dict['coord_loss']
                    loss = self._loss_from_ae_dict(loss_dict)
                except Exception as e:
                    print(f"[DEBUG] Exception at iter {iter_num}, try {try_num}: {e}")
                    loss = None
                
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    try_num += 1
                    if try_num < 3:
                        print(f"[DEBUG] Retrying iter {iter_num}, try {try_num}")
                    else:
                        print(f"[DEBUG] Skipping iter {iter_num} after 3 tries")
                else:
                    break
            
            if torch.isnan(loss) or torch.isinf(loss) or loss is None:
                print(f"[DEBUG] NaN/Inf loss at iter {iter_num}")
                # self.optimizer.zero_grad()
                return None


            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            total_loss += loss.to('cpu').item()
            total_elem_type_num_loss += elem_type_num_loss.to('cpu').item()
            total_elem_type_loss += elem_type_loss.to('cpu').item()
            total_elem_num_loss += elem_num_loss.to('cpu').item()
            total_lattice_loss += lattice_loss.to('cpu').item()
            total_coord_loss += coord_loss.to('cpu').item()

            if iter_num % self.conf['verbose'] == 0:
                print('Training iteration {} | loss elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f}'.format(iter_num, elem_type_num_loss.to('cpu').item(), elem_type_loss.to('cpu').item(), elem_num_loss.to('cpu').item(), lattice_loss.to('cpu').item(), coord_loss.to('cpu').item()))
        
        iter_num += 1
        return total_loss / iter_num, total_elem_type_num_loss / iter_num, total_elem_type_loss / iter_num, total_elem_num_loss / iter_num, total_lattice_loss / iter_num, total_coord_loss / iter_num


    def valid(self, loader):
        self.model.eval()
        total_elem_type_num_loss, total_elem_type_loss, total_elem_num_loss, total_lattice_loss, total_coord_loss =  0, 0, 0, 0, 0
        total_loss = 0
        total_elem_type_num_num, total_elem_type_num_correct = 0, 0
        total_elem_type_num, total_pos_elem_type_num, total_elem_num_num = 0, 0, 0
        total_elem_type_correct, total_pos_elem_type_correct, total_elem_num_correct = 0, 0, 0
        total_elem_type_match, total_elem_num_match, total_comp_match = 0, 0, 0
        total_comp_match_nogt = 0
        has_nogt = False

        with torch.no_grad():
            for iter_num, data_batch in enumerate(loader):
                data_batch = data_batch.to('cuda')
                loss_dict = self.model.forward_ae(data_batch, eval=True, global_step=0)
                elem_type_num_loss, elem_type_loss, elem_num_loss, lattice_loss, coord_loss = loss_dict['elem_type_num_loss'], \
                    loss_dict['elem_type_loss'], loss_dict['elem_num_loss'], loss_dict['lattice_loss'], loss_dict['coord_loss']
                loss = self._loss_from_ae_dict(loss_dict)

                total_loss += loss.to('cpu').item()
                total_elem_type_num_loss += elem_type_num_loss.to('cpu').item()
                total_elem_type_loss += elem_type_loss.to('cpu').item()
                total_elem_num_loss += elem_num_loss.to('cpu').item()
                total_lattice_loss += lattice_loss.to('cpu').item()
                total_coord_loss += coord_loss.to('cpu').item()

                total_elem_type_num_num += loss_dict['total_elem_type_num_num']
                total_elem_type_num_correct += loss_dict['elem_type_num_correct']
                total_elem_type_num += loss_dict['total_elem_type_num']
                total_pos_elem_type_num += loss_dict['pos_elem_type_num']
                total_elem_num_num += loss_dict['elem_num_num']
                total_elem_type_correct += loss_dict['elem_type_correct']
                total_pos_elem_type_correct += loss_dict['pos_elem_type_correct']
                total_elem_num_correct += loss_dict['elem_num_correct']
                total_elem_type_match += loss_dict['composition_correct'][0]
                total_elem_num_match += loss_dict['composition_correct'][1]
                total_comp_match += loss_dict['composition_correct'][2]
                if 'composition_correct_nogt' in loss_dict:
                    has_nogt = True
                    total_comp_match_nogt += loss_dict['composition_correct_nogt'][2]

        iter_num += 1
        denom = total_elem_type_num_num
        comp_match_nogt_rate = (total_comp_match_nogt / denom) if has_nogt and denom > 0 else float("nan")
        return total_loss / iter_num, \
            total_elem_type_num_loss / iter_num, total_elem_type_loss / iter_num, total_elem_num_loss / iter_num, total_lattice_loss / iter_num, total_coord_loss / iter_num, \
            total_elem_type_num_correct / total_elem_type_num_num, total_elem_type_correct / total_elem_type_num, total_pos_elem_type_correct / total_pos_elem_type_num, \
            total_elem_num_correct / total_elem_num_num, total_elem_type_match / total_elem_type_num_num, total_elem_num_match / total_elem_type_num_num, total_comp_match / total_elem_type_num_num, \
            comp_match_nogt_rate


    def train(self, data_path, val_data_path, out_path):
        dataset = MatDataset(data_path, **self.conf['data'])
        loader = DataLoader(dataset, batch_size=self.conf['batch_size'], shuffle=True)
        normalizer = self._get_normalizer(dataset)
        self.model.lattice_normalizer = normalizer
        val_dataset = MatDataset(val_data_path, **self.conf['data'])
        val_loader = DataLoader(val_dataset, batch_size=self.conf['batch_size'], shuffle=False)

        end_epoch = self.conf['end_epoch']
        start = self.conf['start_epoch']
        print(f"Training for {end_epoch - start + 1} epochs")
        try:
            for epoch in range(start, end_epoch + 1):
                train_returns = self._train_epoch(loader, epoch)
                avg_loss, avg_elem_type_num_loss, avg_elem_type_loss, avg_elem_num_loss, avg_lattice_loss, avg_coord_loss = train_returns
                
                _, _, _, _, _, _, elem_type_num_acc, elem_type_acc, elem_type_recall, elem_num_acc, elem_type_match, elem_num_match, comp_match, comp_match_nogt_train = self.valid(loader)

                avg_val_loss, avg_val_elem_type_num_loss, avg_val_elem_type_loss, avg_val_elem_num_loss, avg_val_lattice_loss,\
                    avg_val_coord_loss, val_elem_type_num_acc, val_elem_type_acc, val_elem_type_recall, val_elem_num_acc, val_elem_type_match, val_elem_num_match, val_comp_match, val_comp_match_nogt = self.valid(val_loader)
                print('Training | Average loss {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} \
                    elem_type_num_acc {:.4f}, elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f} comp_match_nogt {:.4f}'.format(avg_loss, 
                    avg_elem_type_num_loss, avg_elem_type_loss, avg_elem_num_loss, avg_lattice_loss, avg_coord_loss, \
                    elem_type_num_acc, elem_type_acc, elem_type_recall, elem_num_acc, elem_type_match, elem_num_match, comp_match, comp_match_nogt_train), flush=True)
                print('Validation | Average loss {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} \
                    elem_type_num_acc {:.4f}, elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f} comp_match_nogt {:.4f}'.format(avg_val_loss, avg_val_elem_type_num_loss, avg_val_elem_type_loss, avg_val_elem_num_loss, avg_val_lattice_loss, avg_val_coord_loss, \
                    val_elem_type_num_acc, val_elem_type_acc, val_elem_type_recall, val_elem_num_acc, val_elem_type_match, val_elem_num_match, val_comp_match, val_comp_match_nogt))
                
                if out_path is not None:
                    if (epoch + 1) % self.conf['save_interval'] == 0:
                        torch.save(self.model.state_dict(), os.path.join(out_path, 'model_{}.pth'.format(epoch)))
                    
                    file_obj = open(os.path.join(out_path, 'train.txt'), 'a')
                    file_obj.write('Training | Average loss {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} '
                        'elem_type_num_acc {:.4f}, elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f} comp_match_nogt {:.4f}\n'.format(avg_loss, 
                        avg_elem_type_num_loss, avg_elem_type_loss, avg_elem_num_loss, avg_lattice_loss, avg_coord_loss, \
                        elem_type_num_acc, elem_type_acc, elem_type_recall, elem_num_acc, elem_type_match, elem_num_match, comp_match, comp_match_nogt_train))
                    file_obj.close()

                    file_obj = open(os.path.join(out_path, 'val.txt'), 'a')
                    file_obj.write('Validation | Average loss {:.4f} elem_type_num {:.4f} elem_type {:.4f} elem_num {:.4f} lattice {:.4f} coord {:.4f} elem_type_num_acc {:.4f} '
                        'elem_type_acc {:.4f} elem_type_recall {:.4f} elem_num_acc {:.4f} elem_type_match {:.4f} elem_num_match {:.4f} comp_match {:.4f} comp_match_nogt {:.4f}\n'.format(avg_val_loss, avg_val_elem_type_num_loss, \
                        avg_val_elem_type_loss, avg_val_elem_num_loss, avg_val_lattice_loss, avg_coord_loss, val_elem_type_num_acc, val_elem_type_acc, val_elem_type_recall, val_elem_num_acc, val_elem_type_match, val_elem_num_match, val_comp_match, val_comp_match_nogt))
                    file_obj.close()
            print("training finished.")
            return
        finally:
            self._cleanup()


    def load_model(self, model_path):
        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"No model file found at {model_path}")


    def flow_random_gen(
        self,
        train_data_path,
        out_path,
        feature_name,
        condGen_dir,
        save_cif_dir,
        small_batch_size=128,
        random_seed=42,
        feature_pt_path=None,
        ae_checkpoint_path=None,
    ):
        try:
            os.makedirs(save_cif_dir, exist_ok=True)
            dataset = MatDataset(train_data_path, **self.conf['data'])
            normalizer = self._get_normalizer(dataset)
            self.model.lattice_normalizer = normalizer
            cpkt_id = self.conf['end_epoch'] - 1
            ckpt_path = (
                ae_checkpoint_path
                if (ae_checkpoint_path is not None and str(ae_checkpoint_path).strip())
                else os.path.join(out_path, f"model_{cpkt_id}.pth")
            )
            self.load_model(ckpt_path)
            print(f"Loaded AE checkpoint: {ckpt_path}", flush=True)
            self.model.eval()
            self.model.coordgen.pool = None  # 关闭多进程池，避免子进程过多
            # set_global_seed(random_seed)  # 固定随机种子，确保每次生成一致

            if feature_pt_path is not None and str(feature_pt_path).strip():
                feature_path = str(feature_pt_path).strip()
            else:
                feature_path = os.path.join(condGen_dir, feature_name)

            with torch.no_grad():
                loaded = torch.load(feature_path, map_location="cpu")
                if torch.is_tensor(loaded):
                    flow_feature = loaded.to("cuda")
                elif isinstance(loaded, dict) and "flow_feature" in loaded:
                    flow_feature = loaded["flow_feature"].to("cuda")
                else:
                    raise ValueError(
                        f"期望 .pt 为 [N,D] 张量或含 flow_feature 的字典，got keys/type: {type(loaded)}"
                    )
                total = flow_feature.shape[0]

                cif_counter = 0
                for start in range(0, total, small_batch_size):
                    end = min(start + small_batch_size, total)
                    batch_feature = flow_feature[start:end]

                    try:
                        # decoder
                        num_atoms, atom_types, lengths, angles, frac_coords = self.model.predict_coords(
                            flow_f=batch_feature,
                            coord_noise_end=self.conf['model']['noise_end'],
                            coord_noise_start=self.conf['model']['noise_start'],
                            coord_num_diff_steps=self.conf['model']['num_time_steps']
                        )

                        # build structures
                        reconstuction_crys_list = build_structures_for_m3gnet(
                            num_atoms,
                            atom_types,
                            lengths,
                            angles,
                            frac_coords,
                            angles_in_radians=False,
                            zero_based_types=False
                        )

                        # 保存 CIF
                        for structure in reconstuction_crys_list:
                            try:
                                writer = CifWriter(structure)
                                writer.write_file(f"{save_cif_dir}/structure_{cif_counter}.cif")
                                cif_counter += 1
                            except Exception as e:
                                print(f"[Skip CIF save] index={cif_counter}, error={e}")

                    except Exception as e:
                        print(f"[Skip batch] {start}:{end}, error={e}")
                        continue

            print(f"cond test finished. {cif_counter} CIF files saved.")

        finally:
            self._cleanup()



@torch.no_grad()
def export_latents(runner, checkpoint, output):
    """Encode train+validation; fit only dataset-level Z-score statistics."""
    from torch.utils.data import ConcatDataset
    root=Path(_REPO_ROOT)
    train=MatDataset(str(root/'data/mp_20/train.pt'), **runner.conf['data'])
    val=MatDataset(str(root/'data/mp_20/val.pt'), **runner.conf['data'])
    runner.model.lattice_normalizer=runner._get_normalizer(train)
    runner.load_model(str(checkpoint)); runner.model.eval()
    if runner.model.coordgen.pool is not None:
        runner.model.coordgen.pool.close(); runner.model.coordgen.pool.join()
        runner.model.coordgen.pool=None
    zs=[]
    for batch in DataLoader(ConcatDataset([train,val]),batch_size=runner.conf['batch_size'],shuffle=False):
        zs.append(runner.model.build_flow_data(batch.to(runner.device)).cpu())
    z=torch.cat(zs).float()
    if z.shape != (33808,144) or not torch.isfinite(z).all():
        raise ValueError(f'Unexpected train+validation representation shape: {tuple(z.shape)}')
    mean=z.mean(0,keepdim=True); std=z.std(0,keepdim=True)+1e-8
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'flow_feature':(z-mean)/std,'mean':mean.squeeze(0),'std':std.squeeze(0)},output)
    print(output)


def main():
    global args
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['train','build_flow_data','flow_random_gen_maf'],default='train')
    p.add_argument('--config_suffix',choices=['h_rmsnorm','rmsnorm','flow_prep_none','flow_prep_zscore'],default='h_rmsnorm')
    p.add_argument('--dataset',choices=['mp_20'],default='mp_20')
    p.add_argument('--model_scale',choices=['large'],default='large')
    p.add_argument('--result_dir',default=str(Path(_REPO_ROOT)/'result/FlowPrep'))
    p.add_argument('--result_version',default='')
    p.add_argument('--checkpoint',default='')
    p.add_argument('--build_flow_out',default='data/mp_20/norm_flow_feature_h_rmsnorm_ep899.pt')
    p.add_argument('--cpkt_id',type=int,default=899)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--maf_latent_out',default='')
    p.add_argument('--gen_save_cif_dir',default='')
    p.add_argument('--maf_num_samples',type=int,default=1000)
    p.add_argument('--gen_small_batch',type=int,default=128)
    p.add_argument('--skip_maf_sample',action='store_true')
    args=p.parse_args()
    os.chdir(_REPO_ROOT)
    conf=load_config(args.config_suffix)
    version=args.result_version or conf['result_version_tag']
    out=Path(args.result_dir)/'mp_20/large'/version
    checkpoint=Path(args.checkpoint) if args.checkpoint else out/f'model_{args.cpkt_id}.pth'
    set_global_seed(args.seed)
    if args.mode=='train' and out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to replace an existing training run: {out}')
    runner=RunnerFlowPrep(conf,'data/mp_20/new_score_norm.txt',dataset_name='mp_20')
    if args.mode=='train':
        runner.train('data/mp_20/train.pt','data/mp_20/val.pt',str(out))
    elif args.mode=='build_flow_data':
        export_latents(runner,checkpoint,args.build_flow_out)
    else:
        if not args.skip_maf_sample:
            raise ValueError('Sample latents with experiment_2/sampler.py, then use --skip_maf_sample')
        if not args.maf_latent_out or not args.gen_save_cif_dir:
            raise ValueError('Both --maf_latent_out and --gen_save_cif_dir are required')
        dest=Path(args.gen_save_cif_dir)
        if dest.exists() and any(dest.iterdir()): raise FileExistsError(dest)
        runner.flow_random_gen('data/mp_20/train.pt',str(out),Path(args.maf_latent_out).name,
            str(Path(args.maf_latent_out).parent),str(dest),small_batch_size=args.gen_small_batch,
            random_seed=args.seed,feature_pt_path=args.maf_latent_out,ae_checkpoint_path=str(checkpoint))


if __name__=='__main__': main()
