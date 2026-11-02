# train_MAF_no_cond.py
import os
import sys
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from script.flow_dataset import FlowFeatureDataset  # 你的数据集类

import argparse
import torch.nn.functional as F
from nflows.flows.autoregressive import MaskedAutoregressiveFlow


def get_activation(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return F.relu
    if name == "elu":
        return F.elu
    if name in ("leaky_relu", "lrelu"):
        return F.leaky_relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        # nflows 里可以直接传函数，这里用 torch.tanh
        return torch.tanh
    # 默认 relu
    return F.relu


def build_maf_no_cond(
    x_dim: int,
    hidden_features: int = 512,
    num_layers: int = 10,
    num_blocks_per_layer: int = 2,
    use_residual_blocks: bool = True,
    use_random_masks: bool = False,
    use_random_permutations: bool = False,
    activation: str = "relu",
    dropout_probability: float = 0.0,
    bn_within_layers: bool = False,
    bn_between_layers: bool = True,
):
    act_fn = get_activation(activation)
    flow = MaskedAutoregressiveFlow(
        features=x_dim,
        hidden_features=hidden_features,
        num_layers=num_layers,
        num_blocks_per_layer=num_blocks_per_layer,
        use_residual_blocks=use_residual_blocks,
        use_random_masks=use_random_masks,
        use_random_permutations=use_random_permutations,
        activation=act_fn,
        dropout_probability=dropout_probability,
        batch_norm_within_layers=bn_within_layers,
        batch_norm_between_layers=bn_between_layers,
    )
    return flow


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    parser = argparse.ArgumentParser()
    # 训练超参
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--dataset_name", type=str, default="mp_20")
    parser.add_argument(
        "--pt_path",
        type=str,
        default="",
        help="若指定则直接用该 .pt（须为 normalize_flow_feature 后的 norm_flow_feature.pt）；否则用默认 data/<dataset_name>/norm_flow_feature.pt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="权重保存目录；空则 result/MAF/no_cond/<dataset>/<lr>/<batch_size>",
    )

    # MAF 架构超参
    parser.add_argument("--hidden_features", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_blocks_per_layer", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu",
                        help="relu|elu|leaky_relu|gelu|tanh")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_blocks", action="store_true", default=False)
    parser.add_argument("--random_masks", action="store_true", default=True)
    parser.add_argument("--random_permutations", action="store_true", default=True)
    parser.add_argument("--bn_within_layers", action="store_true", default=False)
    parser.add_argument("--bn_between_layers", action="store_true", default=True)
    parser.set_defaults(residual_blocks=False, random_masks=True, random_permutations=True,
                        bn_within_layers=False, bn_between_layers=True)

    args = parser.parse_args()
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # 数据：须为 script/flow_dataset.normalize_flow_feature 输出的 norm_flow_feature.pt（含 Z-score 后的 flow_feature 与 Ef_norm）
    if args.pt_path.strip():
        pt_path = args.pt_path.strip()
    elif args.dataset_name == "mp_20_perovskite":
        pt_path = "data/mp_20/norm_perovskite_flow_feature.pt"
    else:
        pt_path = f"data/{args.dataset_name}/norm_flow_feature.pt"
    print("data path:", pt_path)
    dataset = FlowFeatureDataset(pt_path, ef_type="norm")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("num batches:", len(train_loader))
    print("num data:", len(dataset))

    # 自动推断特征维度（也可以像你的NSF脚本一样手写 x_dim=144）
    with torch.no_grad():
        x_dim = dataset[0][0].shape[-1]

    # 构建 MAF（非条件）
    flow = build_maf_no_cond(
        x_dim=x_dim,
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        num_blocks_per_layer=args.num_blocks_per_layer,
        use_residual_blocks=args.residual_blocks,
        use_random_masks=args.random_masks,
        use_random_permutations=args.random_permutations,
        activation=args.activation,
        dropout_probability=args.dropout,
        bn_within_layers=args.bn_within_layers,
        bn_between_layers=args.bn_between_layers,
    ).to(device)

    # 优化器 & 余弦退火
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.epochs * steps_per_epoch,
        eta_min=0.0
    )

    # 模型保存路径
    if args.save_dir.strip():
        save_dir = args.save_dir.strip()
    else:
        save_dir = f"result/MAF/no_cond/{args.dataset_name}/{args.lr}/{args.batch_size}"
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir:", save_dir, flush=True)
    best_loss = float("inf")
    best_path = os.path.join(save_dir, "model_best.pt")
    last_path = os.path.join(save_dir, "model.pt")

    print("-----------------------begin training------------------------")
    for epoch in range(1, args.epochs + 1):
        flow.train()
        start_time = time.time()
        total_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)  # [B, x_dim], float32
            log_prob = flow.log_prob(x)
            loss = -log_prob.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        current_lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}: loss={avg_loss:.6f}, lr={current_lr:.6e}", flush=True)
        print(f"Time for epoch {epoch}: {elapsed:.2f}s", flush=True)

        # 保存最近 & 最优
        if (epoch % 10 == 0) or (epoch == args.epochs):
            torch.save(flow.state_dict(), last_path)
            print(f"Model saved to {last_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(flow.state_dict(), best_path)
            print(f"Best model updated: {best_path}")


if __name__ == "__main__":
    main()



















