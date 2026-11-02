#!/usr/bin/env python3
"""CLI：对 CSV 跑 SUN（避免 evaluate_sun_from_csv.py 文件末尾硬编码）。

环境：CrysFV
  - etot=E_tot_pred_dpa_corr → 必须 --no_double_correction
  - etot=E_tot_pred_dpa_raw  → 默认开 MP2020 一次
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Energy_evaluation"))

from evaluate_sun_from_csv import evaluate_sun  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, default="")
    ap.add_argument("--etot_name", type=str, default="E_tot_pred_dpa_raw")
    ap.add_argument("--cif_name", type=str, default="cif_str")
    ap.add_argument(
        "--no_double_correction",
        action="store_true",
        help="能量已是 corr 时必须打开，禁止二次 MP2020",
    )
    args = ap.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    labels = args.labels_csv or None

    scheme = "no_double_correction" if args.no_double_correction else None
    print(
        f"[INFO] SUN csv={args.csv} etot={args.etot_name} "
        f"correction={'OFF(no double)' if scheme else 'ON(MP2020 once)'}"
    )
    metrics = evaluate_sun(
        args.csv,
        str(out_json),
        etot_name=args.etot_name,
        cif_name=args.cif_name,
        save_labels_csv=labels,
        energy_correction_scheme=scheme,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
