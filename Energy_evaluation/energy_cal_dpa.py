from io import StringIO
import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from ase.io import read as ase_read
from deepmd.calculator import DP
import sys

from evaluation.utils.vasprunlike import VasprunLike



def add_Etot_dpa_from_csv(data_path, cif_name="cif_str", model_path="mlff_choosing/dpa-3.1-mptrj.pth"):
    df = pd.read_csv(data_path)

    print(f"[INFO] Loaded {len(df)} structures")

    # 初始化 DPA 模型与校准方案
    calc = DP(model=model_path)
    compat = MaterialsProject2020Compatibility()  # ✅ MP2020 校准器

    e_pred_raw = []
    e_pred_corr = []
    if "E_tot_pred_dpa_corr" in df:
        return

    for idx, row in df.iterrows():
        try:
            cif_str = row[cif_name].replace("\\n", "\n")
            atoms = ase_read(StringIO(cif_str), format="cif")
            atoms.calc = calc
            E_tot_pred = atoms.get_potential_energy()

            # ---------- 构造 pymatgen 结构 ----------
            s = Structure.from_str(cif_str, fmt="cif")

            # ---------- 转换为 ComputedStructureEntry ----------
            vasprun_like = VasprunLike(structure=s, energy=E_tot_pred)
            entry = vasprun_like.get_computed_entry(
                inc_structure=True, energy_correction_scheme=compat
            )
            E_tot_corr = entry.energy  # 修正后的总能量

            e_pred_raw.append(E_tot_pred)
            e_pred_corr.append(E_tot_corr)

            if idx % 50 == 0:
                print(f"[INFO] {idx+1}/{len(df)} done | raw={E_tot_pred:.3f}, corr={E_tot_corr:.3f}")

        except Exception as e:
            print(f"[WARN] Failed at index {idx}: {e}")
            e_pred_raw.append(None)
            e_pred_corr.append(None)

    # ---------- 保存结果 ----------
    df["E_tot_pred_dpa_corr"] = e_pred_corr
    df['E_tot_pred_dpa_raw'] = e_pred_raw

    df.to_csv(data_path, index=False)

def add_Etot_dpa_from_dir(dir_path, data_path, model_path="mlff_choosing/dpa-3.1-mptrj.pth"):
    from pathlib import Path
    cif_dir = Path(dir_path)
    # 初始化 DPA 模型与校准方案
    calc = DP(model=model_path)
    compat = MaterialsProject2020Compatibility()  # ✅ MP2020 校准器

    cif_strs = []
    candidate_ids = []
    e_pred_raw = []
    e_pred_corr = []
    idx = 0
    for cif_file in cif_dir.glob("*.cif"):
        try:
            s = Structure.from_file(cif_file)
            s_str = s.to(fmt="cif")
            atoms = ase_read(StringIO(s_str), format="cif")

            atoms.calc = calc
            E_tot_pred = atoms.get_potential_energy()

            # ---------- 转换为 ComputedStructureEntry ----------
            vasprun_like = VasprunLike(structure=s, energy=E_tot_pred)
            entry = vasprun_like.get_computed_entry(
                inc_structure=True, energy_correction_scheme=compat
            )
            E_tot_corr = entry.energy  # 修正后的总能量

            e_pred_raw.append(E_tot_pred)
            e_pred_corr.append(E_tot_corr)
            cif_strs.append(s.to(fmt="cif"))
            candidate_ids.append(cif_file.stem)

            idx += 1
            if idx % 50 == 0:
                print(f"[INFO] {idx+1} done | raw={E_tot_pred:.3f}, corr={E_tot_corr:.3f}")

        except Exception as e:
            print(f"[WARN] Failed at index {idx}: {e}")

    # ---------- 保存结果 ----------

    df = pd.DataFrame()
    df['id'] = candidate_ids
    df['candidate_id'] = candidate_ids
    df['cif_str'] = cif_strs
    df["E_tot_pred_dpa_corr"] = e_pred_corr
    df['E_tot_pred_dpa_raw'] = e_pred_raw
    df.to_csv(data_path, index=False)



