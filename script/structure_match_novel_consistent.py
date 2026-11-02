#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与 `MetricsEvaluator` / `StructureMetricsCapability.is_novel` 使用的匹配器一致的拆分接口。

- 默认 matcher：`DefaultDisorderedStructureMatcher`（与 `evaluation/evaluate.py` 中默认一致）
- 组成判断：复刻 `DisorderedStructureMatcher.fit` 中的组成与有序/无序分支
- **组成（直接计数）**：`composition_match_cif_element_counts` — 仅比较单胞内元素种类与各元素
  数量（occupancy 求和），不依赖 `StructureMatcher`
- **comp_match**：`comp_match` — 与 `runner_for_flow_prep.py` 训练/验证中 `comp_match` 完全一致，
  即元素种类 + 各元素原子数完全匹配；在 Structure 层面等价于
  `composition_match_cif_element_counts(normalized=False)`
- 晶格：在会调用 `StructureMatcher.fit`（即 `super().fit`）的路径上，使用与 pymatgen
  `StructureMatcher` 相同的 `_preprocess` + `_get_supercells`（内部为 `Lattice.find_all_mappings`，
  容差为 `ltol` / `angle_tol`）
- 原子位置：在同一预处理后的结构上调用与 `StructureMatcher.fit` 相同的 `_match`，
  用 `stol` 判定（与 `fit` 中 `match[0] <= stol` 一致）
- **novel**：与 `StructureMetricsCapability.is_novel` 一致，底层使用    
  `get_dataset_matcher` + `DatasetMatcher`（`get_matches` → `structure_matcher.fit`）

建议在 conda 环境 CrysFV 或 neqV 下运行（均已含 pymatgen）。

用法示例::

    conda run -n CrysFV python script/structure_match_novel_consistent.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

# 项目根目录加入 path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from evaluation.reference.reference_dataset import ReferenceDataset
from evaluation.utils.dataset_matcher import get_dataset_matcher
from evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    try_make_structure_disordered,
)

FitPathKind = Literal[
    "identity",
    "ordered_early",
    "composition_fail",
    "no_disorder_path",
    "super",
]


def _strip_oxidation_states(structure: Structure) -> Structure:
    """pymatgen 的 remove_oxidation_states() 原地修改并常返回 None，不可链式赋值。"""
    s = structure.copy()
    s.remove_oxidation_states()
    return s


def _disordered_fit_path(
    matcher: DisorderedStructureMatcher,
    structure_1: Structure,
    structure_2: Structure,
) -> Tuple[FitPathKind, Optional[Structure], Optional[Structure]]:
    """
    与 `DisorderedStructureMatcher.fit` 相同的分支，返回是否会进入 `super().fit`
    以及传入 `StructureMatcher.fit` 的一对结构（若为 super 路径）。
    """
    structure_1_nooxi = _strip_oxidation_states(structure_1)
    structure_2_nooxi = _strip_oxidation_states(structure_2)

    if structure_1_nooxi == structure_2_nooxi:
        return "identity", None, None

    if structure_1_nooxi.is_ordered and structure_2_nooxi.is_ordered:
        if (
            structure_2_nooxi.composition.reduced_formula
            != structure_1_nooxi.composition.reduced_formula
        ):
            return "composition_fail", None, None
        if matcher.ordered_structurematcher.fit(structure_1_nooxi, structure_2_nooxi):
            return "ordered_early", None, None
        structure_1_nooxi, can_be_disordered_1 = try_make_structure_disordered(
            structure=structure_1_nooxi,
            relative_radius_difference_threshold=matcher.relative_radius_difference_threshold,
            electronegativity_difference_threshold=matcher.electronegativity_difference_threshold,
        )
        if can_be_disordered_1:
            return "super", structure_1_nooxi, structure_2_nooxi
        return "no_disorder_path", None, None

    if not structure_1_nooxi.composition.fractional_composition.almost_equals(
        structure_2_nooxi.composition.fractional_composition,
        atol=matcher.reduced_formula_atol,
        rtol=matcher.reduced_formula_rtol,
    ):
        return "composition_fail", None, None
    return "super", structure_1_nooxi, structure_2_nooxi


def composition_match(
    structure_1: Structure,
    structure_2: Structure,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> bool:
    """
    仅元素组成（及与 `DisorderedStructureMatcher` 一致的有序 reduced formula /
    无序 fractional composition 规则），不比较晶格与坐标。
    """
    matcher = matcher or DefaultDisorderedStructureMatcher()
    s1 = _strip_oxidation_states(structure_1)
    s2 = _strip_oxidation_states(structure_2)
    if s1 == s2:
        return True
    if s1.is_ordered and s2.is_ordered:
        return s1.composition.reduced_formula == s2.composition.reduced_formula
    return s1.composition.fractional_composition.almost_equals(
        s2.composition.fractional_composition,
        atol=matcher.reduced_formula_atol,
        rtol=matcher.reduced_formula_rtol,
    )


def _element_count_dict_unit_cell(structure: Structure) -> Dict[str, float]:
    """
    单胞内各元素符号 -> 该元素总 occupation（原子当量之和）。
    先去掉氧化态，再按 site 累加。
    """
    s = _strip_oxidation_states(structure)
    totals: Dict[str, float] = {}
    for site in s:
        for species, occ in site.species.items():
            sym = species.symbol
            totals[sym] = totals.get(sym, 0.0) + float(occ)
    return totals


def _element_ratio_dict(counts: Dict[str, float]) -> Dict[str, float]:
    """归一化为摩尔分数（化学计量比），键按字母序。"""
    tot = sum(counts.values())
    if tot <= 0:
        return {}
    return {k: counts[k] / tot for k in sorted(counts.keys())}


def composition_match_cif_element_counts(
    structure_1: Structure,
    structure_2: Structure,
    *,
    normalized: bool = False,
    atol: float = 1e-5,
) -> bool:
    """
    直接根据两个 **Structure**（可由 CIF 解析得到）比较元素组成：

    - **种类**：出现的元素符号集合是否一致；
    - **数量**：各元素在 **当前单胞** 中的总 occupation 是否一致。

    与 `composition_match`（DisorderedStructureMatcher / reduced_formula 规则）**无关**，
    仅做「单胞内元素计数」层面的比较。

    Args:
        structure_1, structure_2: 通常为 `Structure.from_str(cif_str, fmt="cif")` 的结果。
        normalized: 若为 False（默认），要求两结构单胞内 **各元素计数完全一致**（总原子当量也一致）。
            若为 True，则只要求 **化学计量比** 相同（允许超胞倍数不同，例如 Fe2O3 vs Fe4O6）。
        atol: occupation 浮点比较容差（无序占位等）。

    Returns:
        是否视为组成一致。
    """
    d1 = _element_count_dict_unit_cell(structure_1)
    d2 = _element_count_dict_unit_cell(structure_2)
    if normalized:
        r1 = _element_ratio_dict(d1)
        r2 = _element_ratio_dict(d2)
        if set(r1.keys()) != set(r2.keys()):
            return False
        for k in r1:
            if not np.isclose(r1[k], r2[k], rtol=1e-6, atol=atol):
                return False
        return True
    if set(d1.keys()) != set(d2.keys()):
        return False
    for k in d1:
        if not np.isclose(d1[k], d2[k], rtol=0.0, atol=atol):
            return False
    return True


def composition_match_cif_element_counts_from_str(
    cif_str_1: str,
    cif_str_2: str,
    *,
    normalized: bool = False,
    atol: float = 1e-5,
) -> bool:
    """`composition_match_cif_element_counts` 的 CIF 字符串便捷封装。"""
    s1 = Structure.from_str(cif_str_1, fmt="cif")
    s2 = Structure.from_str(cif_str_2, fmt="cif")
    return composition_match_cif_element_counts(
        s1, s2, normalized=normalized, atol=atol
    )


def comp_match(
    structure_1: Structure,
    structure_2: Structure,
    *,
    atol: float = 1e-5,
) -> bool:
    """
    与 `runner_for_flow_prep.py` 训练/验证中的 **comp_match** 完全一致。

    FlowPrep 中 comp_match 定义为：元素种类集合相同 且 各元素原子数完全相同
    （即 `__match_composition` 返回的 match_num，elem_type_match ∧ elem_num_match）。
    在 Structure 层面，等价于单胞内元素种类 + 各元素 occupation 总和完全一致。

    本函数即 `composition_match_cif_element_counts(normalized=False)` 的语义等价实现，
    用于重构评估（目标结构 vs 重构结构）时与训练日志中的 comp_match 口径对齐。

    Args:
        structure_1, structure_2: 两个 pymatgen Structure（通常来自 CIF 解析）
        atol: occupation 浮点比较容差

    Returns:
        是否组成完全匹配（与训练 comp_match 一致）
    """
    return composition_match_cif_element_counts(
        structure_1, structure_2, normalized=False, atol=atol
    )


def lattice_match(
    structure_1: Structure,
    structure_2: Structure,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> bool:
    """
    晶格是否在 `ltol` / `angle_tol` 下可与参考晶格对齐（与 pymatgen `StructureMatcher`
    中 `_get_lattices` / `_get_supercells` 一致）。身份匹配或有序子匹配器已整体成功时视为 True。
    """
    matcher = matcher or DefaultDisorderedStructureMatcher()
    kind, s1, s2 = _disordered_fit_path(matcher, structure_1, structure_2)
    if kind == "identity":
        return True
    if kind == "ordered_early":
        return True
    if kind == "composition_fail":
        return False
    if kind == "no_disorder_path":
        return False
    assert s1 is not None and s2 is not None
    s1p, s2p = matcher._process_species([s1, s2])
    # 与 StructureMatcher.fit 一致：comparator 的 composition hash
    hash_match = matcher._comparator.get_hash(s1p.composition) == matcher._comparator.get_hash(
        s2p.composition
    )
    if not matcher._subset and not hash_match:
        return False
    s1p, s2p, fu, s1_supercell = matcher._preprocess(s1p, s2p)
    gen = matcher._get_supercells(s1p, s2p, fu, s1_supercell)
    try:
        next(gen)
        return True
    except StopIteration:
        return False


def atomic_position_match(
    structure_1: Structure,
    structure_2: Structure,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> bool:
    """
    与 `DisorderedStructureMatcher.fit` / `full_match` **数值上完全一致**（即整体验证，
    与 `is_novel` 所用 `fit` 相同）。

    说明：pymatgen 在 `StructureMatcher.fit` 内将晶格搜索与原子坐标对齐耦合在 `_strict_match`
    中，无法在不调用私有 API 的情况下得到「与官方 fit 完全一致且仅含坐标」的单独布尔量。
    本函数在 super 路径上复现 `StructureMatcher.fit`（`_process_species` → composition hash →
    `_preprocess` → `_match` → `stol`）。

    若 `composition_match` 与 `lattice_match` 均为 True 而 `atomic_position_match` 为 False，
    说明在已通过组成与晶格候选后，坐标/叠合仍超出 `stol`（或 comparator hash 未通过）。
    """
    matcher = matcher or DefaultDisorderedStructureMatcher()
    kind, s1, s2 = _disordered_fit_path(matcher, structure_1, structure_2)
    if kind == "identity":
        return True
    if kind == "ordered_early":
        return True
    if kind == "composition_fail":
        return False
    if kind == "no_disorder_path":
        return False
    assert s1 is not None and s2 is not None
    s1p, s2p = matcher._process_species([s1, s2])
    hash_match = matcher._comparator.get_hash(s1p.composition) == matcher._comparator.get_hash(
        s2p.composition
    )
    if not matcher._subset and not hash_match:
        return False
    s1p, s2p, fu, s1_supercell = matcher._preprocess(s1p, s2p)
    match = matcher._match(
        s1p, s2p, fu, s1_supercell, break_on_match=True
    )
    if match is None:
        return False
    return match[0] <= matcher.stol


def full_match(
    structure_1: Structure,
    structure_2: Structure,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> bool:
    """与 `DisorderedStructureMatcher.fit` 完全一致（即与 is_novel 所用 `fit` 一致）。"""
    matcher = matcher or DefaultDisorderedStructureMatcher()
    return matcher.fit(structure_1, structure_2)


def _reference_with_material_ids(reference: ReferenceDataset) -> ReferenceDataset:
    """与 `StructureMetricsCapability.ensure_reference_dataset_has_material_ids` 思路一致。"""
    entries = [deepcopy(e) for e in reference]
    for i, e in enumerate(entries):
        if "material_id" not in e.data:
            e.data["material_id"] = str(i)
    return ReferenceDataset.from_entries(reference.name, entries)


def _single_structure_dataset(structure: Structure, name: str = "candidate") -> ReferenceDataset:
    """构造单条候选，与 `BaseMetricsCapability.dataset` 中 `entry_id=0` 对齐。"""
    entry = ComputedStructureEntry(structure, 0.0)
    entry.entry_id = 0
    entry.data.setdefault("material_id", "candidate_0")
    return ReferenceDataset.from_entries(name, [entry])


def structure_is_novel(
    structure: Structure,
    reference_dataset: ReferenceDataset,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> bool:
    """
    与 `StructureMetricsCapability.is_novel` 一致：候选结构在 reference 中 **不存在**
    `structure_matcher.fit` 意义下的匹配时返回 True。

    底层调用 `evaluation.utils.dataset_matcher.get_dataset_matcher` 与
    `DatasetMatcher(dataset, reference)`（与 `matches_in_reference` 相同），
    按 `OrderedDatasetMatcher` / `DisorderedDatasetMatcher` 分组（reduced_formula / chemsys）。
    """
    matcher = matcher or DefaultDisorderedStructureMatcher()
    ref = _reference_with_material_ids(reference_dataset)
    if len(ref) == 0:
        return True
    all_ordered = structure.is_ordered and ref.is_ordered
    dataset_matcher = get_dataset_matcher(all_ordered, matcher)
    candidate = _single_structure_dataset(structure)
    matches = dataset_matcher(candidate, ref)
    # matches: entry_id -> [reference material_ids]
    return len(matches.get(0, [])) == 0


def structures_are_novel(
    structures: Sequence[Structure],
    reference_dataset: ReferenceDataset,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> List[bool]:
    """
    批量 novel，与 `StructureMetricsCapability.is_novel` 的逐条定义一致。

    实现上构造 `ReferenceDataset.from_entries`，为每条候选设置 `entry_id=0..n-1`，
    再调用一次 `DatasetMatcher`（与 evaluator 中批量匹配逻辑一致）。
    """
    matcher = matcher or DefaultDisorderedStructureMatcher()
    ref = _reference_with_material_ids(reference_dataset)
    if len(ref) == 0:
        return [True] * len(structures)
    if len(structures) == 0:
        return []
    all_ordered = all(s.is_ordered for s in structures) and ref.is_ordered
    dataset_matcher = get_dataset_matcher(all_ordered, matcher)
    entries: List[ComputedStructureEntry] = []
    for i, structure in enumerate(structures):
        e = ComputedStructureEntry(structure, 0.0)
        e.entry_id = i
        e.data.setdefault("material_id", f"candidate_{i}")
        entries.append(e)
    candidate = ReferenceDataset.from_entries("candidates", entries)
    matches = dataset_matcher(candidate, ref)
    return [len(matches.get(i, [])) == 0 for i in range(len(structures))]


def match_breakdown(
    structure_1: Structure,
    structure_2: Structure,
    matcher: Optional[DisorderedStructureMatcher] = None,
) -> dict:
    """
    返回组成 / 晶格 / 原子位置（super 路径）及完整 `fit`，便于调试。
    若 `composition` 与 `lattice` 均为 True 而 `full_fit` 为 False，通常表示在 super 路径下
    晶格或坐标未同时满足（本模块中 `atomic_positions` 与 `full_fit` 在 super 路径应对齐）。

    - **comp_match**：与 `runner_for_flow_prep.py` 训练 comp_match 一致（元素种类+各元素数完全匹配）
    """
    matcher = matcher or DefaultDisorderedStructureMatcher()
    c = composition_match(structure_1, structure_2, matcher)
    cm = comp_match(structure_1, structure_2)
    l_ = lattice_match(structure_1, structure_2, matcher)
    ap = atomic_position_match(structure_1, structure_2, matcher)
    f = matcher.fit(structure_1, structure_2)
    return {
        "composition": c,
        "comp_match": cm,
        "lattice": l_,
        "atomic_positions": ap,
        "full_fit": f,
        "path_kind": _disordered_fit_path(matcher, structure_1, structure_2)[0],
    }


def _demo():
    from pymatgen.core import Lattice

    # 简单自检：同一结构
    lat = Lattice.cubic(3.0)
    s = Structure(lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    m = DefaultDisorderedStructureMatcher()
    assert full_match(s, s.copy(), m)
    assert match_breakdown(s, s.copy(), m)["full_fit"]

    # novel：与自身参考库应非 novel；空参考应 novel
    ref_same = ReferenceDataset.from_entries(
        "ref",
        [ComputedStructureEntry(s.copy(), 0.0)],
    )
    ref_same = _reference_with_material_ids(ref_same)
    assert not structure_is_novel(s, ref_same, m)
    empty_ref = ReferenceDataset.from_entries("empty", [])
    assert structure_is_novel(s, empty_ref, m)
    assert structures_are_novel([s, s.copy()], ref_same, m) == [False, False]

    assert composition_match_cif_element_counts(s, s.copy())
    assert composition_match_cif_element_counts_from_str(
        s.to(fmt="cif"), s.to(fmt="cif")
    )

    print("structure_match_novel_consistent: demo OK")
    print("  composition_match:", composition_match(s, s.copy(), m))
    print("  comp_match (与训练 comp_match 一致):", comp_match(s, s.copy()))
    print("  lattice_match:", lattice_match(s, s.copy(), m))
    print("  atomic_position_match:", atomic_position_match(s, s.copy(), m))
    print("  full_match:", full_match(s, s.copy(), m))
    print("  structure_is_novel (vs self in ref):", structure_is_novel(s, ref_same, m))


if __name__ == "__main__":
    _demo()
