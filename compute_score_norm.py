import sys
# sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
# import numpy as np
# import torch
# from tqdm import tqdm

# from dataset import MatDataset
# from config.FlowPrep.mp_20_config_dict import conf
# from model.coordgen import CoordGen


# @torch.no_grad()
# def compute_gt_score(
#     dataloaders,
#     conf_model,
#     device="cuda"
# ):

#     # 初始化 CoordGen
#     coordgen = CoordGen(
#         backbone_params=conf_model['dec_backbone_params'], 
#         latent_dim=conf_model['latent_dim'],
#         num_fc_hidden_layers=conf_model['num_fc_hidden_layers'],
#         fc_hidden_dim=conf_model['fc_hidden_dim'],
#         num_time_steps=conf_model['num_time_steps'],
#         noise_start=conf_model['noise_start'],
#         noise_end=conf_model['noise_end'],
#         cutoff=conf_model['cutoff'],
#         max_num_neighbors=conf_model['max_num_neighbors'],
#         score_norm=np.ones(conf_model['num_time_steps'], dtype=np.float32)
#     ).to(device)
#     coordgen.eval()

#     # 存储每个 step 的累计平方和 & 计数
#     sum_sq = torch.zeros(conf_model['num_time_steps'], device=device)
#     count = torch.zeros(conf_model['num_time_steps'], device=device)

#     for loader in dataloaders:
#         for data in tqdm(loader, desc="Computing gt_scores"):
#             data = data.to(device)

#             for step in range(conf_model['num_time_steps']):
#                 gt_scores = coordgen.forward_no_norm(
#                     current_step=step,
#                     num_atoms=data.num_atoms,
#                     gt_frac_coords=data.frac_coords,
#                     lengths=data.lengths,
#                     angles=data.angles,
#                     batch=data.batch
#                 )  # [N_nodes]

#                 # 计算每个 step 下的平方和
#                 sum_sq[step] += (gt_scores ** 2).sum()
#                 count[step] += gt_scores.numel()

#     # 计算 RMS 作为 score_norm
#     score_norm = torch.sqrt(sum_sq / count)
#     return score_norm.cpu().numpy()


# if __name__ == "__main__":
#     import argparse
#     from torch_geometric.loader import DataLoader

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_path", type=str, default="data/alex_mp/train.csv")
#     parser.add_argument("--val_path", type=str, default="data/alex_mp/val.csv")
#     # parser.add_argument("--train_path", type=str, default="data/alex_mp/train.pt")
#     # parser.add_argument("--val_path", type=str, default="data/alex_mp/val.pt")
#     parser.add_argument("--output", type=str, default="data/alex_mp/score_norm.txt")
#     args = parser.parse_args()

#     # 分别加载 train/val 数据
#     val_dataset   = MatDataset(args.val_path, **conf["data"])
#     train_dataset = MatDataset(args.train_path, **conf["data"])
    
#     print(len(train_dataset), len(val_dataset))  
#     loaders = [
#         DataLoader(train_dataset, batch_size=256, shuffle=False),
#         DataLoader(val_dataset, batch_size=256, shuffle=False),
#     ]

#     score_norm = compute_gt_score(
#         loaders,
#         conf_model=conf["model"]
#     )

#     # 保存结果
#     np.savetxt(args.output, score_norm, fmt="%.10e")
#     print(f"Saved score_norm to {args.output}")

# =================== mp_20 ===================

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import torch
from tqdm import tqdm

from dataset import MatDataset
from config.FlowPrep.mp_20_conifg_dict_large_h_rmsnorm import conf
from model.coordgen import CoordGen


@torch.no_grad()
def compute_gt_score(
    dataloaders,
    conf_model,
    device="cuda"
):

    # 初始化 CoordGen
    coordgen = CoordGen(
        backbone_params=conf_model['dec_backbone_params'], 
        latent_dim=conf_model['latent_dim'],
        num_fc_hidden_layers=conf_model['num_fc_hidden_layers'],
        fc_hidden_dim=conf_model['fc_hidden_dim'],
        num_time_steps=conf_model['num_time_steps'],
        noise_start=conf_model['noise_start'],
        noise_end=conf_model['noise_end'],
        cutoff=conf_model['cutoff'],
        max_num_neighbors=conf_model['max_num_neighbors'],
        score_norm=np.ones(conf_model['num_time_steps'], dtype=np.float32)
    ).to(device)
    coordgen.eval()

    # 存储每个 step 的累计平方和 & 计数
    sum_sq = torch.zeros(conf_model['num_time_steps'], device=device)
    count = torch.zeros(conf_model['num_time_steps'], device=device)

    for loader in dataloaders:
        for data in tqdm(loader, desc="Computing gt_scores"):
            data = data.to(device)

            for step in range(conf_model['num_time_steps']):
                gt_scores = coordgen.forward_no_norm(
                    current_step=step,
                    num_atoms=data.num_atoms,
                    gt_frac_coords=data.frac_coords,
                    lengths=data.lengths,
                    angles=data.angles,
                    batch=data.batch
                )  # [N_nodes]

                # 计算每个 step 下的平方和
                sum_sq[step] += (gt_scores ** 2).sum()
                count[step] += gt_scores.numel()

    # 计算 RMS 作为 score_norm
    score_norm = torch.sqrt(sum_sq / count)
    return score_norm.cpu().numpy()


if __name__ == "__main__":
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mp_20")
    parser.add_argument("--train_path", type=str, default="data/mp_20/train.csv")
    parser.add_argument("--val_path", type=str, default="data/mp_20/val.csv")
    parser.add_argument("--test_path", type=str, default="data/mp_20/test.csv")
    parser.add_argument("--output", type=str, default="data/mp_20/new_score_norm.txt")
    args = parser.parse_args()

    # 分别加载 train/val/test 数据
    train_dataset = MatDataset(args.train_path, **conf["data"])
    val_dataset   = MatDataset(args.val_path, **conf["data"])
    test_dataset  = MatDataset(args.test_path, **conf["data"])

    loaders = [
        DataLoader(train_dataset, batch_size=256, shuffle=False),
        DataLoader(val_dataset, batch_size=256, shuffle=False),
        DataLoader(test_dataset, batch_size=256, shuffle=False),
    ]

    score_norm = compute_gt_score(
        loaders,
        conf_model=conf["model"]
    )

    # 保存结果
    np.savetxt(args.output, score_norm, fmt="%.10e")
    print(f"Saved score_norm to {args.output}")



