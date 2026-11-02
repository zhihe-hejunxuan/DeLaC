#!/usr/bin/env python3
"""CLI：CIF 目录或含 cif_str 的 CSV → 写入 DPA raw/corr 能量。

环境：neqV（deepmd.calculator.DP）
默认势：mlff_choosing/dpa-3.1-mptrj.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Energy_evaluation"))

from energy_cal_dpa import add_Etot_dpa_from_csv, add_Etot_dpa_from_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", type=str, default="", help="CIF 目录（与 --csv_in 二选一）")
    ap.add_argument("--csv_in", type=str, default="", help="已有 cif_str 的 CSV")
    ap.add_argument("--csv_out", type=str, required=True, help="写出带能量的 CSV")
    ap.add_argument(
        "--model",
        type=str,
        default=str(_REPO / "mlff_choosing/dpa-3.1-mptrj.pth"),
    )
    ap.add_argument("--cif_name", type=str, default="cif_str")
    args = ap.parse_args()

    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.cif_dir.strip():
        add_Etot_dpa_from_dir(
            dir_path=args.cif_dir.strip(),
            data_path=str(out),
            model_path=args.model,
        )
    elif args.csv_in.strip():
        # 就地写到 csv_out：先复制再算
        import shutil

        src = Path(args.csv_in.strip())
        if src.resolve() != out.resolve():
            shutil.copy2(src, out)
        add_Etot_dpa_from_csv(
            data_path=str(out),
            cif_name=args.cif_name,
            model_path=args.model,
        )
    else:
        raise SystemExit("需要 --cif_dir 或 --csv_in")

    print(f"[DONE] DPA energies -> {out}", flush=True)


if __name__ == "__main__":
    main()
