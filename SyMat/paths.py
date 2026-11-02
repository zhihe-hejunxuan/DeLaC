"""SyMat 数据 / score_norm 路径：mp_20 默认对齐父项目 crystal_gen_flow。"""
from __future__ import annotations

import os
from pathlib import Path

_SYMAT_ROOT = Path(__file__).resolve().parent
_PARENT_ROOT = _SYMAT_ROOT.parent
_PARENT_MP20 = _PARENT_ROOT / "data" / "mp_20"
_LOCAL_MP20 = _SYMAT_ROOT / "data" / "mp_20"

# 父项目 AE(large) 使用的 score_norm；与 compute_score_norm.py 默认产出一致
PARENT_SCORE_NORM = _PARENT_MP20 / "new_score_norm.txt"
PARENT_SCORE_NORM_SMALL = _PARENT_MP20 / "score_norm.txt"


def resolve_split_paths(dataset: str, data_root: str | None = None):
    """返回 (train_path, val_path, test_path, score_norm_path, pt_dir).

    mp_20:
      - CSV / score_norm 读父项目 data/mp_20（不改父目录）
      - 预处理 .pt 写到 SyMat/data/mp_20/（本地缓存）
    其它数据集仍用 SyMat/data/<dataset>/。
    """
    if dataset == "mp_20":
        root = Path(data_root) if data_root else _PARENT_MP20
        pt_dir = _LOCAL_MP20
        pt_dir.mkdir(parents=True, exist_ok=True)

        def _csv_or_local_pt(split: str) -> tuple[str, str | None]:
            local_pt = pt_dir / f"{split}.pt"
            if local_pt.is_file():
                return str(local_pt), None
            csv_path = root / f"{split}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"缺少 {csv_path}")
            return str(csv_path), str(local_pt)

        train_path, train_pt = _csv_or_local_pt("train")
        val_path, val_pt = _csv_or_local_pt("val")
        test_path, test_pt = _csv_or_local_pt("test")

        score_norm = PARENT_SCORE_NORM
        if not score_norm.is_file():
            score_norm = PARENT_SCORE_NORM_SMALL
        if not score_norm.is_file():
            raise FileNotFoundError(
                f"父项目缺少 score_norm: {PARENT_SCORE_NORM} 或 {PARENT_SCORE_NORM_SMALL}"
            )

        return {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "train_pt_path": train_pt,
            "val_pt_path": val_pt,
            "test_pt_path": test_pt,
            "score_norm_path": str(score_norm),
            "data_root": str(root),
            "pt_dir": str(pt_dir),
        }

    # perov_5 / carbon_24：沿用 SyMat 自带数据
    root = Path(data_root) if data_root else (_SYMAT_ROOT / "data" / dataset)

    def _pick(split: str) -> str:
        pt = root / f"{split}.pt"
        if pt.is_file():
            return str(pt)
        csv = root / f"{split}.csv"
        if csv.is_file():
            return str(csv)
        raise FileNotFoundError(f"缺少 {pt} 或 {csv}")

    score = root / "score_norm.txt"
    if not score.is_file():
        raise FileNotFoundError(f"缺少 {score}")

    return {
        "train_path": _pick("train"),
        "val_path": _pick("val"),
        "test_path": _pick("test"),
        "train_pt_path": None,
        "val_pt_path": None,
        "test_pt_path": None,
        "score_norm_path": str(score),
        "data_root": str(root),
        "pt_dir": str(root),
    }
