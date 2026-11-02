# flow_prep.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .spherenet import SphereNetEncoder
from .modules import build_mlp
from .coordgen import CoordGen
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from utils.mat_utils import filter_invalid_lattices

class FlowPrep(nn.Module):
    def __init__(
        self,
        enc_backbone_params,
        dec_backbone_params,
        latent_dim: int,
        latent_latt_dim: int,
        num_fc_hidden_layers: int,
        fc_hidden_dim: int,
        max_num_atoms: int,
        max_atomic_num: int,
        num_time_steps: int,
        noise_start: float,
        noise_end: float,
        cutoff: float,
        max_num_neighbors: int,
        coord_loss_type: str = "per_node",
        use_multi_latent: bool = False,
        lattice_scale: bool = True,
        use_gpu: bool = True,
        score_norm=None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device if use_gpu else "cpu")
        self.use_gpu = use_gpu
        self.use_multi_latent = use_multi_latent
        self.lattice_scale = lattice_scale

        # -------- 图编码器（生成 h）--------
        self.encoder_backbone = SphereNetEncoder(**enc_backbone_params)
        enc_out_dim = enc_backbone_params["out_channels"]
        self.enc_out_dim = enc_out_dim                       # <- 后续切分 flow_f 用
        self.latent_dim = latent_dim
        self.latent_latt_dim = latent_latt_dim
        self.flow_feature_dim = enc_out_dim + latent_latt_dim     # <- 供 Flow 侧参考

        # -------- 晶格专属编码器（不经图编码器）---------
        # 输入 6 维 (lengths, angles) -> latent_latt
        self.fc_lattice_enc = build_mlp(6, fc_hidden_dim, num_fc_hidden_layers, latent_latt_dim)

        # -------- 元素/晶格 AE 解码头 --------
        self.fc_elem_type = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_atomic_num)
        self.fc_elem_num = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_num_atoms)
        self.fc_elem_type_num = build_mlp(latent_dim, fc_hidden_dim, num_fc_hidden_layers, max_num_atoms)
        self.elem_emb = nn.Embedding(max_atomic_num, latent_dim)
        self.fc_lattice = build_mlp(latent_latt_dim, fc_hidden_dim, num_fc_hidden_layers, 6)

        # -------- 将 h 映射到元素解码的 latent 空间（AE 模式）--------
        self.h_norm_for_comp = nn.LayerNorm(self.enc_out_dim)
        self.h_to_latent_comp = build_mlp(self.enc_out_dim, fc_hidden_dim, 2, latent_dim)
        
        self.h_norm_for_pos = nn.LayerNorm(self.enc_out_dim)
        self.h_to_latent_pos = nn.Linear(self.enc_out_dim, latent_dim)


        # -------- 坐标分支（完整保留，默认冻结）--------
        self.coordgen = CoordGen(
            dec_backbone_params,
            latent_dim,
            num_fc_hidden_layers,
            fc_hidden_dim,
            num_time_steps,
            noise_start,
            noise_end,
            cutoff,
            max_num_neighbors,
            loss_type=coord_loss_type,
            use_gpu=use_gpu,
            score_norm=score_norm,
        )
        self.num_time_steps = num_time_steps
        self.noise_start = noise_start
        self.noise_end = noise_end

        # 设备放置
        if use_gpu:
            for m in [
                self.encoder_backbone,
                self.fc_lattice_enc,     # <- 记得放设备
                self.fc_elem_type,
                self.fc_elem_num,
                self.fc_elem_type_num,
                self.fc_lattice,
                self.elem_emb,
                self.h_norm_for_comp,
                self.h_to_latent_comp,
                self.h_norm_for_pos,
                self.h_to_latent_pos,
                self.coordgen,
            ]:
                m.to(self.device)

        # 运行时成员
        self.max_atomic_num = max_atomic_num
        self.max_num_atoms = max_num_atoms
        # self.elem_emb = None
        self.lattice_normalizer = None

    # -------------------------- 工具 --------------------------

    def __topk_mask(self, input, k):
        sorted, _ = torch.sort(input, dim=-1, descending=True)
        thres = sorted[torch.arange(input.shape[0], device=input.device), k-1].view(-1, 1)
        return (input >= thres).long().to(input.device)
    

    def __match_composition(self, elem_type_topk, target_elem_type, elem_num_pred, target_elem_num, num_elem_per_mat):
        idx = 0
        elem_type_match_num, elem_num_match_num, match_num = 0, 0, 0
        for i in range(len(num_elem_per_mat)):
            idx2 = idx + num_elem_per_mat[i].long()
            elem_type_match = (elem_type_topk[i] == target_elem_type[i]).min(-1).values
            elem_num_match = (elem_num_pred[idx : idx2] == target_elem_num[idx : idx2]).min(-1).values
            elem_type_match_num += elem_type_match
            elem_num_match_num += elem_num_match
            match_num += elem_type_match * elem_num_match
            idx = idx2
        return elem_type_match_num, elem_num_match_num, match_num

    # -------------------------- AE 前向与损失 --------------------------

    def forward_ae(self, data_batch, eval: bool = False, global_step: int = 0):
        """
        - 解码元素/晶格/pos
        """
        h = self.encoder_backbone(data_batch)  # [B, enc_out_dim]
        h_norm_c = self.h_norm_for_comp(h)
        h_norm_p = self.h_norm_for_pos(h)
        latent_comp = self.h_to_latent_comp(h_norm_c)  # [B, latent_dim]
        latent_pos = self.h_to_latent_pos(h_norm_p)  # [B, latent_dim]

        lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.transform(lengths_angles)
        latent_latt = self.fc_lattice_enc(lengths_angles)               # [B, latent_dim]
        pred_lengths_angles = self.fc_lattice(latent_latt)              # [B, 6]

        pred_elem_type_num = self.fc_elem_type_num(latent_comp) # [256, 20] 
        elem_type_num = pred_elem_type_num.argmax(dim=-1) + 1 # [256]
        pred_elem_type = self.fc_elem_type(latent_comp) # [256, 100]
        elem_type_topk = self.__topk_mask(pred_elem_type, elem_type_num).to('cuda') # select top k elem type -> [256, 100] 1 or 0

        elem_types = data_batch.atom_types - 1
        elem_types_one_hot = F.one_hot(elem_types, num_classes=self.max_atomic_num)
        elem_num = scatter(elem_types_one_hot, data_batch.batch, dim=0, reduce='sum')
        nonzero_mask = elem_num > 0
        target_elem_type = nonzero_mask.long()

        num_elem_per_mat = target_elem_type.sum(dim=-1)
        latent_repeat = torch.repeat_interleave(latent_comp, num_elem_per_mat, dim=0)
        elem_type_index = torch.nonzero(elem_num)
        batch_index, elem_type_batch = elem_type_index[:, 0], elem_type_index[:, 1]
        pred_elem_num = self.fc_elem_num(latent_repeat * self.elem_emb(elem_type_batch))
        elem_num_pred = pred_elem_num.argmax(dim=-1)
        target_elem_num = torch.masked_select(elem_num, nonzero_mask) - 1


        loss_dict = {}
        elem_type_num_loss = F.cross_entropy(pred_elem_type_num, num_elem_per_mat - 1, reduction='mean')
        elem_type_loss = F.binary_cross_entropy_with_logits(pred_elem_type, target_elem_type.float(), reduction='mean')
        try:
            # tmp_target_elem_num = target_elem_num.cpu()
            # tmp_target_shape = target_elem_num.shape
    
            # # pred 相关
            # tmp_pred_shape = pred_elem_num.shape    # [N, C]
            # tmp_pred_size1 = pred_elem_num.size(1)  # 类别数
            
            # # batch_index 相关
            # tmp_batch_index = batch_index.detach().cpu()
            # tmp_batch_index_shape = batch_index.shape
            elem_num_loss = scatter(F.cross_entropy(pred_elem_num, target_elem_num, reduction='none'), batch_index, reduce='mean').mean()
        except:
            # 如果数据量很大，可以只看前几个
            # print("target min/max:", tmp_target_elem_num.min().item(), tmp_target_elem_num.max().item())
            # print("pred shape:", tmp_pred_shape)
            # print("batch_index min/max:", tmp_batch_index.min().item(), tmp_batch_index.max().item())
            # print("tmp_target_elem_num[:10]:", tmp_target_elem_num[:10])
            # print("tmp_pred_shape:", tmp_pred_shape, "tmp_pred_size1:", tmp_pred_size1)
            # print("tmp_batch_index[:10]:", tmp_batch_index[:10])
            elem_num_loss = torch.tensor(0.0, device=pred_elem_num.device)

        loss_dict['elem_type_num_loss'] = elem_type_num_loss
        loss_dict['elem_type_loss'] = elem_type_loss
        loss_dict['elem_num_loss'] = elem_num_loss

        if eval:
            loss_dict['total_elem_type_num_num'] = len(elem_type_num)
            loss_dict['elem_type_num_correct'] = elem_type_num.eq(num_elem_per_mat).sum().cpu().detach().item()

            loss_dict['total_elem_type_num'] = target_elem_type.shape[0] * target_elem_type.shape[1]
            loss_dict['pos_elem_type_num'] = nonzero_mask.sum().cpu().detach().item()
            loss_dict['elem_type_correct'] = elem_type_topk.eq(target_elem_type).sum().cpu().detach().item()
            loss_dict['pos_elem_type_correct'] = torch.logical_and(elem_type_topk.eq(target_elem_type), nonzero_mask).sum().cpu().detach().item()

            loss_dict['elem_num_num'] = len(target_elem_num)
            loss_dict['elem_num_correct'] = pred_elem_num.argmax(dim=-1).eq(target_elem_num).sum().cpu().detach().item()

            loss_dict['composition_correct'] = self.__match_composition(elem_type_topk, target_elem_type, elem_num_pred, target_elem_num, num_elem_per_mat)       
        target_lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            target_lengths_angles = self.lattice_normalizer.transform(target_lengths_angles)
        loss_dict['lattice_loss'] = F.mse_loss(pred_lengths_angles, target_lengths_angles)
        loss_dict['coord_loss'] = self.coordgen(latent_pos, data_batch.num_atoms, data_batch.atom_types-1, data_batch.frac_coords, data_batch.lengths, data_batch.angles, data_batch.batch)

        return loss_dict


    @torch.no_grad()
    def reconstuction_crys(
        self,
        data_batch
    ):
        h = self.encoder_backbone(data_batch)  # [B, enc_out_dim]
        lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.transform(lengths_angles)
        latent_latt = self.fc_lattice_enc(lengths_angles)               # [B, latent_dim]
        num_atoms, atom_types, lengths, angles = self.ae_decode_from_h(h, latent_latt)
        h_norm_p = self.h_norm_for_pos(h)
        latent_pos = self.h_to_latent_pos(h_norm_p)  # [B, latent_dim]

        frac_coords = self.coordgen.generate(
            latents=latent_pos,
            num_atoms=num_atoms,
            atom_types=atom_types - 1,
            lengths=lengths,
            angles=angles,
            noise_start=self.noise_start,
            noise_end=self.noise_end,
            num_gen_steps=self.num_time_steps
        )
        return num_atoms, atom_types, lengths, angles, frac_coords
        


    @torch.no_grad()
    def ae_decode_from_h(self, h, latent_latt, temperature_elem: float = 1.0):
        """
        仅用 (h, latent_latt) 通过 AE 解码器生成 (atom_types, num_atoms, lengths, angles)。
        """
        h_norm_c = self.h_norm_for_comp(h)
        latent_comp = self.h_to_latent_comp(h_norm_c)  # [B, latent_dim]

        pred_elem_type_num = self.fc_elem_type_num(latent_comp)
        elem_type_num = pred_elem_type_num.argmax(dim=-1) + 1
        pred_elem_type = self.fc_elem_type(latent_comp)
        elem_type_topk = self.__topk_mask(pred_elem_type, elem_type_num)

        num_elem_per_mat = elem_type_topk.sum(dim=-1)
        latent_repeat = torch.repeat_interleave(latent_comp, num_elem_per_mat, dim=0)

        elem_type_index = torch.nonzero(elem_type_topk)
        batch_index, elem_type_batch = elem_type_index[:, 0], elem_type_index[:, 1]

        pred_elem_num = self.fc_elem_num(latent_repeat * self.elem_emb(elem_type_batch)).argmax(dim=-1) + 1

        atom_types = torch.repeat_interleave(elem_type_batch + 1, pred_elem_num, dim=0)
        num_atoms = scatter(pred_elem_num, batch_index, dim=0, reduce='sum')

        lengths_angles = self.fc_lattice(latent_latt)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.inverse_transform(lengths_angles)
        lengths, angles = lengths_angles[:, :3], lengths_angles[:, 3:]

        if self.lattice_scale:
            lengths = lengths * num_atoms.view(-1, 1).float() ** (1.0 / 3.0)

        return num_atoms, atom_types, lengths, angles

    @torch.no_grad()
    def predict_coords(
        self,
        flow_f: torch.Tensor,                     # 期望 shape [B, enc_out_dim + latent_latt_dim]
        coord_noise_start: float = 0.01,
        coord_noise_end: float = 10.0,
        coord_num_diff_steps: int = 50
    ):
        """
        从 flow_f 生成完整结构：
          1) 切分 flow_f -> (h, latent_latt)
          2) AE 解码 (C,L)
          3) 采样 latent_pos（μ/σ 条件化或先验）
          4) CoordGen Langevin 生成 frac_coords
        """
        h = flow_f[:, : self.enc_out_dim]
        latent_latt = flow_f[:, self.enc_out_dim :]

        num_atoms, atom_types, lengths, angles = self.ae_decode_from_h(h, latent_latt)
        
        # latent_pos
        h_norm_p = self.h_norm_for_pos(h)
        latent_pos = self.h_to_latent_pos(h_norm_p)  # [B, latent_dim]

        num_atoms, atom_types, lengths, angles, latent_pos = filter_invalid_lattices(
            num_atoms, atom_types, lengths, angles, latent_pos, eps=1e-6
        )

        frac_coords = self.coordgen.generate(
            latents=latent_pos,
            num_atoms=num_atoms,
            atom_types=atom_types - 1,
            lengths=lengths,
            angles=angles,
            noise_start=coord_noise_start,
            noise_end=coord_noise_end,
            num_gen_steps=coord_num_diff_steps
        )
        return num_atoms, atom_types, lengths, angles, frac_coords
    
    @torch.no_grad()
    def build_flow_data(self, data_batch):
        h = self.encoder_backbone(data_batch)  # [B, enc_out_dim]
        lengths_angles = torch.cat([data_batch.scaled_lengths, data_batch.angles], dim=-1)
        if self.lattice_normalizer is not None:
            lengths_angles = self.lattice_normalizer.transform(lengths_angles)
        latent_latt = self.fc_lattice_enc(lengths_angles)               # [B, latent_latt_dim]
        flow_f = torch.cat([h, latent_latt], dim=-1)  # [B, enc_out_dim + latent_latt_dim]
        return flow_f
        




