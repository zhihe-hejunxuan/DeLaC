#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_sun_from_csv.py
使用 CSV 中的结构 (path) 和 E_tot (总能量)，计算 SUN 指标
"""
import json
import os, sys
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_analyzer import oxide_type

from evaluation.metrics.evaluator import MetricsEvaluator
from evaluation.reference.reference_dataset import ReferenceDataset
from evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher
from evaluation.reference.presets import ReferenceMP2020Correction

reference = None

def has_forbidden_elements(structure) -> bool:
    """
    20260207 appended: 检查结构中是否含有禁止元素
    https://github.com/microsoft/mattergen/blob/main/data-release/mp-20/README.md
    """
    for el in structure.composition.elements:
        Z = Element(el.symbol).Z
        if el.symbol in {"Tc", "Pm", "Ne", "He"}:
            return True
        if Z >= 84:
            return True
    return False

def load_structures_and_etot(csv_path, etot_name, cif_name="cif_str"):
    """从 CSV 读取结构和能量，并按 MatterGen MP-20 协议过滤禁止元素。"""
    df = pd.read_csv(csv_path)
    count = 1
    structures, energies = [], []
    valid_indices = []  # 记录成功加载的原始行索引
    filtered_count = 0
    for i, row in df.iterrows():
        # if count > 100:
        #     break
        count += 1
        try:
            cif_value = row.get(cif_name, None)
            if not isinstance(cif_value, str) or not cif_value.strip():
                raise ValueError("missing CIF string")
            cif_str = cif_value.replace("\\n", "\n")
            etot = float(row.get(etot_name, None))
            if not np.isfinite(etot):
                raise ValueError(f"non-finite energy: {etot}")
            s = Structure.from_str(cif_str, fmt="cif")
            matrix = np.asarray(s.lattice.matrix, dtype=float)
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) <= 1e-8:
                raise ValueError("non-finite or singular lattice")
            if np.linalg.cond(matrix) > 1e8:
                raise ValueError("ill-conditioned lattice")
            # MetricsEvaluator 后续会调用 oxide_type/LLL。提前逐结构触发并捕获
            # 数值退化，避免一个坏 CIF 终止整个 seed；该行仍留在 SUN/all 分母中。
            oxide_type(s)
            if has_forbidden_elements(s) or any(
                el.symbol in {"Xe", "Kr"} for el in s.composition.elements
            ):
                filtered_count += 1
                continue
            structures.append(s)
            energies.append(etot)
            valid_indices.append(i)  # 记录有效的行索引
        except Exception as e:
            print(f"出错: {e}")
            continue
    if filtered_count > 0:
        print(f"已按 MP-20 元素范围过滤 {filtered_count} 个结构")
    return structures, energies, valid_indices, df


def evaluate_sun(csv_path, save_as, etot_name, cif_name="cif_str", save_labels_csv=None, energy_correction_scheme=None):
    """使用 MetricsEvaluator 计算所有指标，包括 SUN
    
    Args:
        csv_path: 输入的 CSV 文件路径
        save_as: 保存整体指标结果的 JSON 文件路径
        etot_name: CSV 中总能量列的名称
        save_labels_csv: 可选，保存每个材料标签的 CSV 文件路径
    """
    global reference
    if reference is None:
        reference = ReferenceMP2020Correction()
    structures, energies, valid_indices, original_df = load_structures_and_etot(csv_path, etot_name, cif_name)
    class NoCorrection:
        def get_adjustments(self, entry):
            return []
        def process_entry(self, entry):
            return entry
        def process_entries(self, entries):
            return entries
    if not structures:
        raise RuntimeError("没有可用的结构数据！")
    if energy_correction_scheme is None:
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=structures,
            energies=energies,
            reference=reference,  # 默认用 MP2020 corrected reference
            structure_matcher=DefaultDisorderedStructureMatcher(),
            # energy_correction_scheme=NoCorrection(),  # 不使用 Compatibility 修正
        )
    else:
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=structures,
            energies=energies,
            reference=reference,  # 默认用 MP2020 corrected reference
            structure_matcher=DefaultDisorderedStructureMatcher(),
            energy_correction_scheme=NoCorrection(),
        )

    # 检查可用的指标
    from evaluation.metrics.energy import FracStableStructures
    available_metric_names = [m.name for m in evaluator.available_metrics]
    print(f"可用指标数量: {len(available_metric_names)}")
    print(f"可用指标列表: {available_metric_names}")
    
    # 检查是否有 EnergyMetricsCapability
    from evaluation.metrics.energy import EnergyMetricsCapability
    has_energy_capability = EnergyMetricsCapability in evaluator.available_capability_types
    print(f"是否有 EnergyMetricsCapability: {has_energy_capability}")
    
    # 检查 frac_stable_structures 是否在可用指标中
    frac_stable_in_available = FracStableStructures in evaluator.available_metrics
    if frac_stable_in_available:
        print("✓ FracStableStructures 在可用指标列表中")
    else:
        print("✗ FracStableStructures 不在可用指标列表中")
        if not has_energy_capability:
            print("  原因: 缺少 EnergyMetricsCapability")

    results = evaluator.compute_metrics(
        metrics="all",
        save_as=save_as,
        pretty_print=True,
    )
    
    # 确保 frac_stable_structures 在结果中
    if "frac_stable_structures" not in results:
        print("\n警告: frac_stable_structures 不在结果中！")
        if frac_stable_in_available:
            print("尝试手动计算 frac_stable_structures...")
            try:
                frac_stable = evaluator.compute_metric(FracStableStructures)
                results["frac_stable_structures"] = frac_stable
                print(f"✓ 成功手动计算 frac_stable_structures: {frac_stable}")
                # 如果成功计算，也更新保存的文件
                if save_as:
                    import json
                    from pathlib import Path
                    save_path = Path(save_as).resolve()
                    if save_path.exists():
                        with open(save_path, "r") as f:
                            saved_data = json.load(f)
                        saved_data["frac_stable_structures"] = {
                            "value": frac_stable,
                            "description": evaluator._get_metric(FracStableStructures).description
                        }
                        with open(save_path, "w") as f:
                            json.dump(saved_data, f, indent=4)
                        print(f"✓ 已更新保存的文件: {save_path}")
            except Exception as e:
                print(f"✗ 无法手动计算 frac_stable_structures: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("无法计算 frac_stable_structures: 该指标不在可用指标列表中")
            if not has_energy_capability:
                print("  可能的原因: 缺少 EnergyMetricsCapability（参考数据集中缺少某些终端系统）")
    else:
        print(f"\n✓ frac_stable_structures 已在结果中: {results['frac_stable_structures']}")
    
    print(f"加载了 {len(structures)} 个结构用于评估")
    
    # 提取每个材料的 stable、novel、unique 标签
    if save_labels_csv:
        try:
            # 获取每个材料的标签（检查能力是否可用）
            is_stable = None
            is_unique = None
            is_novel = None
            
            try:
                is_stable = evaluator.is_stable
            except Exception as e:
                print(f"警告: 无法获取 is_stable: {e}")
            
            try:
                is_unique = evaluator.is_unique
            except Exception as e:
                print(f"警告: 无法获取 is_unique: {e}")
            
            try:
                is_novel = evaluator.is_novel
            except Exception as e:
                print(f"警告: 无法获取 is_novel: {e}")
            
            if is_stable is None and is_unique is None and is_novel is None:
                print("✗ 无法提取任何标签，跳过保存")
            else:
                # 创建标签 DataFrame
                labels_data = []
                for idx, orig_idx in enumerate(valid_indices):
                    label_dict = {'original_index': orig_idx}
                    if is_stable is not None:
                        label_dict['stable'] = bool(is_stable[idx])
                    if is_unique is not None:
                        label_dict['unique'] = bool(is_unique[idx])
                    if is_novel is not None:
                        label_dict['novel'] = bool(is_novel[idx])
                    
                    # 计算 SUN（需要所有三个标签都可用）
                    if is_stable is not None and is_unique is not None and is_novel is not None:
                        label_dict['SUN'] = bool(is_stable[idx] & is_unique[idx] & is_novel[idx])
                    
                    labels_data.append(label_dict)
                
                labels_df = pd.DataFrame(labels_data)
                
                # 只保留有效行的原始数据
                valid_df = original_df.iloc[valid_indices].copy().reset_index(drop=True)
                
                # 合并标签（排除 original_index 列）
                label_cols = [col for col in labels_df.columns if col != 'original_index']
                result_df = pd.concat([valid_df, labels_df[label_cols]], axis=1)
                
                # 保存到 CSV
                result_df.to_csv(save_labels_csv, index=False)
                print(f"\n✓ 已保存每个材料的标签到: {save_labels_csv}")
                print(f"  总计 {len(result_df)} 个材料")
                if is_stable is not None:
                    print(f"  Stable: {is_stable.sum()}")
                if is_novel is not None:
                    print(f"  Novel: {is_novel.sum()}")
                if is_unique is not None:
                    print(f"  Unique: {is_unique.sum()}")
                if is_stable is not None and is_unique is not None and is_novel is not None:
                    sun_count = (is_stable & is_unique & is_novel).sum()
                    print(f"  SUN (stable & unique & novel): {sun_count}")
        except Exception as e:
            print(f"\n✗ 提取标签时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return results

